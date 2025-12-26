# adapter_eval_speakers.py
#
# Evaluate the effect of the adapter on top of **M_ALL**.
#
# For each speaker:
#   - Real embeddings are NEVER changed.
#   - Generated audio comes from M_ALL (eval_M_ALL).
#
# We report:
#   M_ALL before = cosine(real_emb, gen_emb)          # no adapter
#   M_ALL after  =
#       remain speakers: cosine(real_emb, gen_emb)    # unchanged, no adapter
#       forget speaker 27: cosine(real_emb, adapter(gen_emb, is_forget=True))
#
# So only the forget speaker is affected by the adapter; remain speakers stay the same.

import os
import re
from pathlib import Path
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------- CONFIG ----------

DATA_ROOT = Path("data/libritts_8spk")

# Metadata with all 8 speakers (same used for training M_ALL)
META_ALL = DATA_ROOT / "metadata_clean.csv"   # change if your file name differs

REAL_CLIPS_PER_SPK = 5
FORGET_SPK = "27"

# Generated files from M_ALL inference (directory should already exist)
EVAL_ALL_DIR = Path("eval_M_ALL")

ADAPTER_CKPT = Path("adapter_unlearning.pth")


# ---------- UTILS ----------

def load_metadata(meta_path: Path):
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


def build_real_refs(entries, clips_per_spk=REAL_CLIPS_PER_SPK):
    by_spk = {}
    for wav_path, text, spk in entries:
        by_spk.setdefault(spk, []).append(wav_path)

    real_refs = {}
    for spk, paths in by_spk.items():
        if len(paths) <= clips_per_spk:
            chosen = paths
        else:
            chosen = random.sample(paths, clips_per_spk)
        real_refs[spk] = chosen
    return real_refs


def embed_wavs(encoder, wav_paths):
    embs = []
    for p in wav_paths:
        try:
            wav_pre = preprocess_wav(str(p))
            emb = encoder.embed_utterance(wav_pre)
            embs.append(emb)
        except Exception as e:
            print(f"[WARN] Failed to embed {p}: {e}")
    if not embs:
        return None
    return np.stack(embs, axis=0).mean(axis=0)


def collect_generated(eval_dir: Path, tag: str):
    """
    Collect generated wav paths per speaker ID by looking for 'spk<digits>'
    and ensuring `tag` is in the filename.
    """
    gen = {}
    if not eval_dir.is_dir():
        print(f"[WARN] eval dir missing: {eval_dir}")
        return gen

    spk_regex = re.compile(r"spk(\d+)")
    count = 0

    for fname in os.listdir(eval_dir):
        if not fname.lower().endswith(".wav"):
            continue
        if tag.lower() not in fname.lower():
            continue

        m = spk_regex.search(fname)
        if not m:
            print(f"[WARN] could not find speaker id in filename: {fname}")
            continue

        spk_id = m.group(1)
        gen.setdefault(spk_id, []).append(eval_dir / fname)
        count += 1

    print(f"[INFO] Found {count} generated files in {eval_dir} with tag '{tag}'.")
    return gen


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------- ADAPTER MODEL (must match adapter_train.py) ----------

class AdapterMLP(nn.Module):
    """
    Same architecture as adapter_train.py.

    In THIS eval script we will:
      - NOT use it for remain speakers (they stay unchanged)
      - Use only the forget branch for speaker 27.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 512, forget_noise_alpha: float = 2.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.forget_noise_alpha = forget_noise_alpha

    def _identity_branch(self, x: torch.Tensor) -> torch.Tensor:
        # Not used in this script, but kept for checkpoint compatibility.
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        out = self.ln(out + x)
        return out

    def _forget_branch(self, x: torch.Tensor) -> torch.Tensor:
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
        if is_forget:
            return self._forget_branch(x)
        else:
            # For remain speakers we won't actually call this;
            # they keep their original embeddings.
            return x


# ---------- MAIN EVAL ----------

def main():
    random.seed(42)

    if not META_ALL.is_file():
        raise FileNotFoundError(f"Metadata file not found: {META_ALL}")
    if not ADAPTER_CKPT.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {ADAPTER_CKPT}")

    print("[INFO] Loading metadata from:", META_ALL)
    meta_entries = load_metadata(META_ALL)
    print(f"[INFO] Loaded {len(meta_entries)} usable metadata rows.")

    speakers = sorted({spk for _, _, spk in meta_entries})
    print("Speakers in metadata:", speakers)

    # Real reference clips per speaker (never modified)
    real_refs = build_real_refs(meta_entries, REAL_CLIPS_PER_SPK)

    print("\n[INFO] Loading speaker encoder (Resemblyzer)...")
    encoder = VoiceEncoder()

    # Real embeddings
    spk_real_emb = {}
    for spk, paths in real_refs.items():
        print(f"[REAL] Speaker {spk}, {len(paths)} clips")
        emb = embed_wavs(encoder, paths)
        if emb is None:
            print(f"  [WARN] no valid real clips for speaker {spk}")
            continue
        spk_real_emb[spk] = emb

    # Generated audio from M_ALL
    gen_all = collect_generated(EVAL_ALL_DIR, "M_ALL")

    # Load adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[INFO] Using device:", device)

    ckpt = torch.load(ADAPTER_CKPT, map_location=device)
    emb_dim = ckpt["emb_dim"]
    forget_alpha = ckpt.get("forget_noise_alpha", 2.0)
    adapter = AdapterMLP(emb_dim=emb_dim, hidden_dim=512, forget_noise_alpha=forget_alpha).to(device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()

    def adapt_forget(np_emb: np.ndarray) -> np.ndarray:
        """Apply adapter's forget branch to a single embedding."""
        with torch.no_grad():
            x = torch.from_numpy(np_emb).float().to(device).unsqueeze(0)  # [1, D]
            y = adapter(x, is_forget=True)  # [1, D]
        return y.cpu().numpy()[0]

    rows = []

    for spk in speakers:
        real_emb = spk_real_emb.get(spk)
        if real_emb is None:
            continue

        all_paths = gen_all.get(spk, [])

        def get_sim_before(paths):
            if not paths:
                return None
            gen_emb = embed_wavs(encoder, paths)
            if gen_emb is None:
                return None
            # BEFORE: no adapter anywhere
            return cosine(real_emb, gen_emb)

        def get_sim_after(paths, speaker_id: str):
            """
            AFTER (M_ALL + adapter):
              - remain speakers: real vs gen_emb (unchanged)
              - forget speaker:  real vs adapter_forget(gen_emb)
            """
            if not paths:
                return None
            gen_emb = embed_wavs(encoder, paths)
            if gen_emb is None:
                return None

            if speaker_id == FORGET_SPK:
                gen_mod = adapt_forget(gen_emb)
            else:
                gen_mod = gen_emb  # no adapter for remain speakers

            return cosine(real_emb, gen_mod)

        sim_all_before = get_sim_before(all_paths)
        sim_all_after = get_sim_after(all_paths, spk)

        rows.append({
            "speaker_id": spk,
            "is_forget_speaker": (spk == FORGET_SPK),
            "sim_all_before": sim_all_before,
            "sim_all_after": sim_all_after,
        })

    print("\n=== Adapter effect on M_ALL similarity (real vs gen / adapter(gen)) ===")
    print("speaker | forget? | M_ALL before | M_ALL after | Δ (after - before)")
    for r in rows:
        b = r["sim_all_before"]
        a = r["sim_all_after"]
        if b is None or a is None:
            delta = None
        else:
            delta = a - b
        print(
            f"{r['speaker_id']:>7} | "
            f"{'YES' if r['is_forget_speaker'] else 'no ':>6} | "
            f"{b if b is not None else '  - ':>12} | "
            f"{a if a is not None else '  - ':>11} | "
            f"{delta if delta is not None else '  - '}"
        )

    out_csv = "adapter_effect_M_ALL.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "is_forget_speaker",
                "sim_all_before",
                "sim_all_after",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[OK] Saved adapter effect results to {out_csv}")


if __name__ == "__main__":
    main()
