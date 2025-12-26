# adapter_train.py
#
# Train a small adapter that behaves ~like identity on REMAIN speakers'
# embeddings. Forget speaker (27) is not used during training.
#
# At inference, the same adapter module will:
#   - act nearly as identity for remain speakers
#   - add random noise to embeddings for the forget speaker (via a flag)
#
# You will use this together with adapter_eval_speakers.py.

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from resemblyzer import VoiceEncoder, preprocess_wav


# ---------------- CONFIG ----------------

DATA_ROOT = Path("data/libritts_8spk")

# Metadata file that contains ALL 8 speakers (including 27)
# Change this name if your file is different.
FULL_META_FILE = DATA_ROOT / "metadata_clean.csv"

FORGET_SPK = "27"            # forget speaker ID (string)
MAX_UTTS_PER_SPK = 30        # max utterances per speaker for adapter training
BATCH_SIZE = 64
NUM_EPOCHS = 50              # identity mapping is easy; 30 is fine
LR = 1e-3

# This is used only for the forget branch at inference (not during training),
# but we save it in case you want to record it in logs.
FORGET_NOISE_ALPHA = 2.0


# ---------------- DATA LOADING ----------------

def load_metadata(meta_path: Path):
    """Parse Coqui-style metadata file into list of (wav_path, text, speaker_id)."""
    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                print(f"[WARN] Skipping malformed line {i}: {line}")
                continue
            rel_path, text, spk = parts[0], parts[1], parts[2]
            wav_path = DATA_ROOT / rel_path
            if not wav_path.is_file():
                print(f"[WARN] Missing wav file (line {i}): {wav_path}")
                continue
            entries.append((wav_path, text, spk))
    return entries


def build_paths_per_speaker(entries, max_utts_per_spk=MAX_UTTS_PER_SPK):
    by_spk = {}
    for wav_path, text, spk in entries:
        by_spk.setdefault(spk, []).append(wav_path)

    spk_to_paths = {}
    for spk, paths in by_spk.items():
        if len(paths) <= max_utts_per_spk:
            chosen = paths
        else:
            chosen = random.sample(paths, max_utts_per_spk)
        spk_to_paths[spk] = chosen
    return spk_to_paths


# ---------------- MODEL DEFINITIONS ----------------

class EmbeddingDataset(Dataset):
    """Holds embeddings (REMAIN speakers only)."""

    def __init__(self, emb_list):
        self.emb_list = emb_list

    def __len__(self):
        return len(self.emb_list)

    def __getitem__(self, idx):
        emb = self.emb_list[idx]
        return torch.from_numpy(emb).float()


class AdapterMLP(nn.Module):
    """
    Simple MLP adapter: emb_dim -> hidden -> emb_dim, with residual + LayerNorm.

    At inference we will call:
        adapter(x, is_forget=False/True)

    During training we ONLY use is_forget=False (remain speakers).
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 512, forget_noise_alpha: float = 2.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.forget_noise_alpha = forget_noise_alpha

    def _identity_branch(self, x: torch.Tensor) -> torch.Tensor:
        # identity-like mapping for remain speakers
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        out = self.ln(out + x)   # residual + layernorm
        return out

    def _forget_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Random-noise branch for forget speaker embeddings.

        x: [B, D]
        We:
          - normalize each embedding
          - add random noise (unit-norm) scaled by alpha
          - renormalize
        """
        # unit normalization
        x_norm = x.norm(dim=-1, keepdim=True) + 1e-8
        x_unit = x / x_norm

        noise = torch.randn_like(x_unit)
        noise_norm = noise.norm(dim=-1, keepdim=True) + 1e-8
        noise_unit = noise / noise_norm

        v = x_unit + self.forget_noise_alpha * noise_unit
        v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
        v_unit = v / v_norm
        return v_unit

    def forward(self, x: torch.Tensor, is_forget: bool = False) -> torch.Tensor:
        """
        x: [B, D]
        is_forget:
          - False -> identity-like MLP (remain speakers)
          - True  -> random scrambling (forget speaker)
        """
        if is_forget:
            return self._forget_branch(x)
        else:
            return self._identity_branch(x)


# ---------------- TRAINING SCRIPT ----------------

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if not FULL_META_FILE.is_file():
        raise FileNotFoundError(f"Full metadata file not found: {FULL_META_FILE}")

    print("[INFO] Loading metadata from:", FULL_META_FILE)
    entries = load_metadata(FULL_META_FILE)
    print(f"[INFO] Loaded {len(entries)} usable entries.")

    speakers = sorted({spk for _, _, spk in entries})
    print("[INFO] Speakers in full metadata:", speakers)
    if FORGET_SPK not in speakers:
        print(f"[WARN] Forget speaker {FORGET_SPK} is not in this metadata!")

    spk_to_paths = build_paths_per_speaker(entries, MAX_UTTS_PER_SPK)
    for spk, paths in spk_to_paths.items():
        print(f"[INFO] Speaker {spk}: using {len(paths)} utterances")

    print("\n[INFO] Loading VoiceEncoder (Resemblyzer)...")
    encoder = VoiceEncoder()

    # Embeddings for REMAIN speakers only (skip forget speaker 27)
    remain_embs = []

    print("\n[INFO] Computing embeddings for REMAIN speakers...")
    for spk, paths in spk_to_paths.items():
        if spk == FORGET_SPK:
            continue  # skip forget speaker in adapter training
        for p in paths:
            try:
                wav = preprocess_wav(str(p))
                emb = encoder.embed_utterance(wav)  # np.array [D]
                remain_embs.append(emb)
            except Exception as e:
                print(f"[WARN] Failed to embed {p}: {e}")

    if not remain_embs:
        raise RuntimeError("No remain-speaker embeddings computed; aborting.")

    emb_dim = remain_embs[0].shape[0]
    print(f"[INFO] Embedding dimension: {emb_dim}")
    print(f"[INFO] Total REMAIN embeddings: {len(remain_embs)}")

    dataset = EmbeddingDataset(remain_embs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    adapter = AdapterMLP(
        emb_dim=emb_dim,
        hidden_dim=512,
        forget_noise_alpha=FORGET_NOISE_ALPHA,
    ).to(device)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)

    print("\n[INFO] Training adapter (identity on REMAIN speakers)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for emb_batch in dataloader:
            emb_batch = emb_batch.to(device)  # [B, D]
            optimizer.zero_grad()

            # We use ONLY the identity branch during training
            out = adapter(emb_batch, is_forget=False)
            loss = F.mse_loss(out, emb_batch)  # identity loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | identity_loss={avg_loss:.6f}")

    ckpt_path = "adapter_unlearning.pth"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "emb_dim": emb_dim,
            "forget_speaker": FORGET_SPK,
            "forget_noise_alpha": FORGET_NOISE_ALPHA,
        },
        ckpt_path,
    )
    print(f"\n[OK] Saved adapter to {ckpt_path}")


if __name__ == "__main__":
    main()
