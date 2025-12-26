# speaker_eval_all_remain.py

import os
from pathlib import Path
import random
import csv
import re 

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


# ---- CONFIG ----
DATA_ROOT = Path("data/libritts_8spk")
META_ALL = DATA_ROOT / "metadata_clean.csv"  # we only need remain speakers here
REAL_CLIPS_PER_SPK = 5  # number of real utterances per speaker for reference

EVAL_ALL_DIR = Path("eval_M_ALL")
EVAL_REMAIN_DIR = Path("eval_M_REMAIN")
FORGET_SPK = "27"  # forget speaker id as string (used later)


def load_metadata(meta_path):
    """Parse Coqui-style metadata file into list of (audio_path, text, speaker_id)."""
    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            rel_path, text, spk = parts[0], parts[1], parts[2]
            wav_path = DATA_ROOT / rel_path
            if not wav_path.is_file():
                continue
            entries.append((wav_path, text, spk))
    return entries


def build_real_refs(entries, clips_per_spk=REAL_CLIPS_PER_SPK):
    """Select a few real samples per speaker."""
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
    """Load wav as mono float32 numpy array."""
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def embed_wavs(encoder, wav_paths, target_sr=16000):
    embs = []
    for p in wav_paths:
        wav, sr = load_wav(p)
        # Resemblyzer expects 16k mono
        if sr != target_sr:
            # simple resample via librosa would be nicer, but to avoid extra deps:
            # let resemblyzer handle it via preprocess_wav
            wav_pre = preprocess_wav(str(p))
            emb = encoder.embed_utterance(wav_pre)
        else:
            wav_pre = preprocess_wav(str(p))
            emb = encoder.embed_utterance(wav_pre)
        embs.append(emb)
    if not embs:
        return None
    return np.stack(embs, axis=0).mean(axis=0)  # average embedding


def main():
    random.seed(42)

    # 1) Load metadata for remain speakers (same speakers as M_REMAIN)
    meta_entries = load_metadata(META_ALL)
    speakers = sorted({spk for _, _, spk in meta_entries})
    print("Speakers in metadata_remain.csv:", speakers)

    # 2) Choose real reference clips per speaker
    real_refs = build_real_refs(meta_entries, REAL_CLIPS_PER_SPK)

    # 3) Init speaker encoder
    print("Loading speaker encoder (Resemblyzer)...")
    encoder = VoiceEncoder()

    # 4) Compute reference embedding for each speaker (real audio)
    spk_real_emb = {}
    for spk, paths in real_refs.items():
        print(f"[REAL] Speaker {spk}, {len(paths)} clips")
        emb = embed_wavs(encoder, paths)
        if emb is None:
            print(f"  [WARN] no valid real clips for speaker {spk}")
            continue
        spk_real_emb[spk] = emb

    # 5) Helper to list generated files and infer speaker_id from filename

    def collect_generated(eval_dir: Path, prefix_substring: str):
        """
        Collect generated wav paths per speaker ID by looking for 'spk<digits>' anywhere in the filename.
        Example expected patterns:
        M_ALL_spk201_idx0_utt0.wav
        M_all_speaker_spk201_utt0.wav
        We only require that:
        1) filename contains prefix_substring (e.g. 'M_ALL'),
        2) filename contains 'spk<digits>'.
        """
        gen = {}
        if not eval_dir.is_dir():
            print(f"[WARN] eval dir missing: {eval_dir}")
            return gen

        spk_regex = re.compile(r"spk(\d+)")

        for fname in os.listdir(eval_dir):
            if not fname.lower().endswith(".wav"):
                continue
            if prefix_substring.lower() not in fname.lower():
                continue

            m = spk_regex.search(fname)
            if not m:
                print(f"[WARN] could not find speaker id in filename: {fname}")
                continue

            spk_id = m.group(1)  # digits inside 'spkXXXX'
            gen.setdefault(spk_id, []).append(eval_dir / fname)

        print(f"[INFO] Found {sum(len(v) for v in gen.values())} generated files in {eval_dir}")
        return gen


    gen_all = collect_generated(EVAL_ALL_DIR, "M_ALL")
    gen_remain = collect_generated(EVAL_REMAIN_DIR, "M_REMAIN")

    # 6) For each speaker, embed generated audio and compute cosine similarity
    def cosine(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    rows = []
    for spk in speakers:
        real_emb = spk_real_emb.get(spk)
        if real_emb is None:
            continue

        # M_ALL similarity (if we have generated samples)
        all_paths = gen_all.get(spk, [])
        remain_paths = gen_remain.get(spk, [])

        if all_paths:
            emb_all = embed_wavs(encoder, all_paths)
            sim_all = cosine(real_emb, emb_all)
        else:
            sim_all = None

        if remain_paths:
            emb_rem = embed_wavs(encoder, remain_paths)
            sim_rem = cosine(real_emb, emb_rem)
        else:
            sim_rem = None

        rows.append({
            "speaker_id": spk,
            "is_forget_speaker": (spk == FORGET_SPK),
            "sim_real_M_ALL": sim_all,
            "sim_real_M_REMAIN": sim_rem,
        })

    # 7) Print table and save to CSV
    print("\n=== Speaker similarity (real vs generated) ===")
    print("speaker_id | forget? | sim_real_M_ALL | sim_real_M_REMAIN")
    for r in rows:
        print(
            f"{r['speaker_id']:>9} | "
            f"{'YES' if r['is_forget_speaker'] else 'no ':>6} | "
            f"{r['sim_real_M_ALL'] if r['sim_real_M_ALL'] is not None else ' - ':>15} | "
            f"{r['sim_real_M_REMAIN'] if r['sim_real_M_REMAIN'] is not None else ' - ':>17}"
        )

    out_csv = "speaker_similarity_all_vs_remain.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker_id", "is_forget_speaker", "sim_real_M_ALL", "sim_real_M_REMAIN"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[OK] Saved similarity results to {out_csv}")


if __name__ == "__main__":
    main()
