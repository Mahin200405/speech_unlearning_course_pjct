# speaker_eval_all_remain_unlearned.py
#
# Compare speaker similarity (real vs generated) for:
#  - M_ALL
#  - M_REMAIN
#  - M_UNLEARNED
#
# Uses Resemblyzer to compute embeddings.

import os
import re
from pathlib import Path
import random
import csv

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


# ---- CONFIG ----

DATA_ROOT = Path("data/libritts_8spk")

# Use the metadata that now includes ALL speakers (including 27).
# You said you updated metadata_remain.csv to include 27, so we use that.
META_ALL = DATA_ROOT / "metadata_clean.csv"

REAL_CLIPS_PER_SPK = 5
FORGET_SPK = "27"  # forget speaker id as string

EVAL_ALL_DIR = Path("eval_M_ALL")
EVAL_REMAIN_DIR = Path("eval_M_REMAIN")
EVAL_UNL_DIR = Path("eval_M_UNLEARNED")


def load_metadata(meta_path):
    """Parse Coqui-style metadata file into list of (audio_path, text, speaker_id)."""
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


def load_wav(path):
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def embed_wavs(encoder, wav_paths, target_sr=16000):
    embs = []
    for p in wav_paths:
        try:
            # let Resemblyzer handle loading & resampling
            wav_pre = preprocess_wav(str(p))
            emb = encoder.embed_utterance(wav_pre)
            embs.append(emb)
        except Exception as e:
            print(f"[WARN] Failed to embed {p}: {e}")
    if not embs:
        return None
    return np.stack(embs, axis=0).mean(axis=0)


def collect_generated(eval_dir: Path, prefix_substring: str):
    """
    Collect generated wav paths per speaker ID by looking for 'spk<digits>' anywhere in the filename
    and requiring prefix_substring somewhere in the name.
    Example filenames:
      M_ALL_spk201_idx0_utt0.wav
      M_REMAIN_spk201_idx0_utt0.wav
      M_UNLEARNED_spk201_idx0_utt0.wav
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
        if prefix_substring.lower() not in fname.lower():
            continue

        m = spk_regex.search(fname)
        if not m:
            print(f"[WARN] could not find speaker id in filename: {fname}")
            continue

        spk_id = m.group(1)
        gen.setdefault(spk_id, []).append(eval_dir / fname)
        count += 1

    print(f"[INFO] Found {count} generated files in {eval_dir} with tag '{prefix_substring}'.")
    return gen


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    random.seed(42)

    if not META_ALL.is_file():
        raise FileNotFoundError(f"Metadata file not found: {META_ALL}")

    print("[INFO] Loading metadata from:", META_ALL)
    meta_entries = load_metadata(META_ALL)
    print(f"[INFO] Loaded {len(meta_entries)} usable metadata rows.")

    speakers = sorted({spk for _, _, spk in meta_entries})
    print("Speakers in metadata:", speakers)

    # 1) Choose real reference clips per speaker
    real_refs = build_real_refs(meta_entries, REAL_CLIPS_PER_SPK)

    # 2) Init speaker encoder
    print("\n[INFO] Loading speaker encoder (Resemblyzer)...")
    encoder = VoiceEncoder()

    # 3) Real reference embeddings
    spk_real_emb = {}
    for spk, paths in real_refs.items():
        print(f"[REAL] Speaker {spk}, {len(paths)} clips")
        emb = embed_wavs(encoder, paths)
        if emb is None:
            print(f"  [WARN] no valid real clips for speaker {spk}")
            continue
        spk_real_emb[spk] = emb

    # 4) Generated audio from each model
    gen_all = collect_generated(EVAL_ALL_DIR, "M_ALL")
    gen_remain = collect_generated(EVAL_REMAIN_DIR, "M_REMAIN")
    gen_unl = collect_generated(EVAL_UNL_DIR, "M_UNLEARNED")

    # 5) Compute similarities
    rows = []
    for spk in speakers:
        real_emb = spk_real_emb.get(spk)
        if real_emb is None:
            continue

        all_paths = gen_all.get(spk, [])
        remain_paths = gen_remain.get(spk, [])
        unlearn_paths = gen_unl.get(spk, [])

        sim_all = sim_remain = sim_unl = None

        if all_paths:
            emb_all = embed_wavs(encoder, all_paths)
            if emb_all is not None:
                sim_all = cosine(real_emb, emb_all)

        if remain_paths:
            emb_rem = embed_wavs(encoder, remain_paths)
            if emb_rem is not None:
                sim_remain = cosine(real_emb, emb_rem)

        if unlearn_paths:
            emb_unl = embed_wavs(encoder, unlearn_paths)
            if emb_unl is not None:
                sim_unl = cosine(real_emb, emb_unl)

        rows.append({
            "speaker_id": spk,
            "is_forget_speaker": (spk == FORGET_SPK),
            "sim_real_M_ALL": sim_all,
            "sim_real_M_REMAIN": sim_remain,
            "sim_real_M_UNLEARNED": sim_unl,
        })

    # 6) Print table
    print("\n=== Speaker similarity (real vs generated) ===")
    print("speaker_id | forget? | sim_real_M_ALL | sim_real_M_REMAIN | sim_real_M_UNLEARNED")
    for r in rows:
        print(
            f"{r['speaker_id']:>9} | "
            f"{'YES' if r['is_forget_speaker'] else 'no ':>6} | "
            f"{r['sim_real_M_ALL'] if r['sim_real_M_ALL'] is not None else '  - ':>15} | "
            f"{r['sim_real_M_REMAIN'] if r['sim_real_M_REMAIN'] is not None else '  - ':>18} | "
            f"{r['sim_real_M_UNLEARNED'] if r['sim_real_M_UNLEARNED'] is not None else '  - ':>22}"
        )

    # 7) Save CSV
    out_csv = "speaker_similarity_all_remain_unlearned.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "is_forget_speaker",
                "sim_real_M_ALL",
                "sim_real_M_REMAIN",
                "sim_real_M_UNLEARNED",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[OK] Saved similarity results to {out_csv}")


if __name__ == "__main__":
    main()
