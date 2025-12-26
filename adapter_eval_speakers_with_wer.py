# adapter_eval_speakers.py
#
# Evaluate adapter on top of M_ALL:
#   - Cosine similarity:
#       BEFORE: cos(E_real, E_all)
#       AFTER:
#         remain speakers -> cos(E_real, E_all)  (no adapter)
#         forget speaker  -> cos(E_real, adapter(E_all, is_forget=True))
#   - WER (Whisper + jiwer) for M_ALL (same before/after since audio is unchanged)
#
# Output: adapter_effect_M_ALL.csv
#
# IMPORTANT: Ensure TEST_SENTENCES matches infer_vits_all.py.

import os
import re
import random
import csv
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from jiwer import wer

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper

# ---------- CONFIG ----------

DATA_ROOT = Path("data/libritts_8spk")

META_ALL = DATA_ROOT / "metadata_clean.csv"
EVAL_ALL_DIR = Path("eval_M_ALL")
ADAPTER_CKPT = Path("adapter_unlearning.pth")

FORGET_SPK = "27"
REAL_CLIPS_PER_SPK = 5

OUT_CSV = "adapter_effect_M_ALL.csv"

TEST_SENTENCES = [
    "This is a test sentence for our unlearning experiment.",
    "The quick brown fox jumps over the lazy dog.",
    # Add more if you used more prompts
]


# ---------- UTILS: METADATA & EMBEDDINGS ----------

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


def embed_wavs(encoder: VoiceEncoder, wav_paths):
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


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------- UTILS: GENERATED FILES & WER ----------

def collect_generated(eval_dir: Path, tag: str):
    """
    Collect generated wav paths per speaker and utt index.
    Expects filenames like: {tag}_spk<speaker>_idx<...>_utt<utt_idx>.wav
    """
    gen = {}
    if not eval_dir.is_dir():
        print(f"[WARN] eval dir missing: {eval_dir}")
        return gen

    spk_regex = re.compile(r"spk(\d+)")
    utt_regex = re.compile(r"utt(\d+)")

    count = 0
    for fname in os.listdir(eval_dir):
        if not fname.lower().endswith(".wav"):
            continue
        if tag.lower() not in fname.lower():
            continue

        m_spk = spk_regex.search(fname)
        m_utt = utt_regex.search(fname)
        if not m_spk or not m_utt:
            print(f"[WARN] could not parse speaker or utt index in filename: {fname}")
            continue

        spk_id = m_spk.group(1)
        utt_idx = int(m_utt.group(1))

        gen.setdefault(spk_id, []).append((eval_dir / fname, utt_idx))
        count += 1

    print(f"[INFO] Found {count} generated files in {eval_dir} with tag '{tag}'.")
    return gen


def compute_wer_per_speaker(asr_model, gen_map):
    """
    gen_map: dict[spk] -> list of (wav_path, utt_idx)
    Returns: dict[spk] -> avg WER for M_ALL.
    """
    spk_to_wers = {}
    for spk, items in gen_map.items():
        wers = []
        for wav_path, utt_idx in items:
            if utt_idx < 0 or utt_idx >= len(TEST_SENTENCES):
                print(f"[WARN] utt_idx {utt_idx} out of range for {wav_path}, skipping.")
                continue
            ref_text = TEST_SENTENCES[utt_idx].strip().lower()
            try:
                result = asr_model.transcribe(str(wav_path), language="en", fp16=False)
                hyp_text = result["text"].strip().lower()
                w = wer(ref_text, hyp_text)
                wers.append(w)
            except Exception as e:
                print(f"[WARN] ASR failed on {wav_path}: {e}")
                continue

        if wers:
            spk_to_wers[spk] = float(np.mean(wers))
        else:
            spk_to_wers[spk] = None

    return spk_to_wers


# ---------- ADAPTER MODEL ----------

class AdapterMLP(nn.Module):
    """
    Same as in adapter_train.py.
    Here:
      - remain speakers: we DO NOT apply adapter (identity at eval level)
      - forget speaker: we call adapter(x, is_forget=True) to scramble embedding
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 512, forget_noise_alpha: float = 2.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.forget_noise_alpha = forget_noise_alpha

    def _identity_branch(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        out = self.ln(out + x)
        return out

    def _forget_branch(self, x: torch.Tensor) -> torch.Tensor:
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
            # not actually used for remain speakers in this script
            return x


# ---------- MAIN ----------

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

    # Real refs
    real_refs = build_real_refs(meta_entries, REAL_CLIPS_PER_SPK)

    print("\n[INFO] Loading speaker encoder (Resemblyzer)...")
    spk_encoder = VoiceEncoder()

    spk_real_emb = {}
    for spk, paths in real_refs.items():
        print(f"[REAL] Speaker {spk}, {len(paths)} clips")
        emb = embed_wavs(spk_encoder, paths)
        if emb is None:
            print(f"  [WARN] no valid real clips for speaker {spk}")
            continue
        spk_real_emb[spk] = emb

    # Generated audio from M_ALL
    gen_all = collect_generated(EVAL_ALL_DIR, "M_ALL")

    # ASR model for WER (M_ALL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[INFO] Loading Whisper ASR model on device:", device)
    asr_model = whisper.load_model("small", device=device)

    print("\n[INFO] Computing WER per speaker for M_ALL...")
    wer_all = compute_wer_per_speaker(asr_model, gen_all)

    # Adapter
    device_torch = torch.device(device)
    ckpt = torch.load(ADAPTER_CKPT, map_location=device_torch)
    emb_dim = ckpt["emb_dim"]
    forget_alpha = ckpt.get("forget_noise_alpha", 2.0)

    adapter = AdapterMLP(emb_dim=emb_dim, hidden_dim=512, forget_noise_alpha=forget_alpha).to(device_torch)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()

    def adapt_forget(np_emb: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np_emb).float().to(device_torch).unsqueeze(0)
            y = adapter(x, is_forget=True)
        return y.cpu().numpy()[0]

    # Helper to get cos sims
    def get_sim_before(spk: str):
        real_emb = spk_real_emb.get(spk)
        if real_emb is None:
            return None
        items = gen_all.get(spk, [])
        if not items:
            return None
        paths = [p for (p, _) in items]
        gen_emb = embed_wavs(spk_encoder, paths)
        if gen_emb is None:
            return None
        return cosine(real_emb, gen_emb)

    def get_sim_after(spk: str):
        real_emb = spk_real_emb.get(spk)
        if real_emb is None:
            return None
        items = gen_all.get(spk, [])
        if not items:
            return None
        paths = [p for (p, _) in items]
        gen_emb = embed_wavs(spk_encoder, paths)
        if gen_emb is None:
            return None

        if spk == FORGET_SPK:
            gen_mod = adapt_forget(gen_emb)
        else:
            gen_mod = gen_emb  # unchanged for remain speakers

        return cosine(real_emb, gen_mod)

    rows = []
    for spk in speakers:
        sim_before = get_sim_before(spk)
        sim_after = get_sim_after(spk)
        if sim_before is None or sim_after is None:
            delta = None
        else:
            delta = sim_after - sim_before

        rows.append({
            "speaker_id": spk,
            "is_forget_speaker": (spk == FORGET_SPK),
            "sim_all_before": sim_before,
            "sim_all_after": sim_after,
            "wer_M_ALL": wer_all.get(spk),
        })

    print("\n=== Adapter effect on M_ALL similarity (real vs gen / adapter(gen)) ===")
    print("speaker | forget? | M_ALL before | M_ALL after | Δ (after - before) | wer_M_ALL")
    for r in rows:
        print(
            f"{r['speaker_id']:>7} | "
            f"{'YES' if r['is_forget_speaker'] else 'no ':>6} | "
            f"{r['sim_all_before'] if r['sim_all_before'] is not None else '  - ':>12} | "
            f"{r['sim_all_after'] if r['sim_all_after'] is not None else '  - ':>11} | "
            f"{(r['sim_all_after'] - r['sim_all_before']) if (r['sim_all_after'] is not None and r['sim_all_before'] is not None) else '  - ':>18} | "
            f"{r['wer_M_ALL'] if r['wer_M_ALL'] is not None else '  - ':>9}"
        )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "is_forget_speaker",
                "sim_all_before",
                "sim_all_after",
                "wer_M_ALL",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[OK] Saved adapter effect results (with WER) to {OUT_CSV}")


if __name__ == "__main__":
    main()
