# eval_M_unlearned.py
#
# Compare M_ALL, M_REMAIN, and M_UNLEARNED:
#   - Cosine similarity between real vs generated embeddings (Resemblyzer)
#   - WER of generated audio vs ground-truth prompts (Whisper + jiwer)
#
# Output CSV (same name as before, with extra WER columns):
#   speaker_similarity_all_remain_unlearned.csv
#
# IMPORTANT: Make sure TEST_SENTENCES matches the texts used in infer_vits_all.py.

import os
import re
import random
import csv
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from jiwer import wer

import torch
import whisper

# ---------------- CONFIG ----------------

DATA_ROOT = Path("data/libritts_8spk")

# Metadata with ALL 8 speakers (same as used for M_ALL)
META_ALL = DATA_ROOT / "metadata_clean.csv"

# Forget speaker ID
FORGET_SPK = "27"

# Real references
REAL_CLIPS_PER_SPK = 5

# Generated audio directories
EVAL_ALL_DIR = Path("eval_M_ALL")
EVAL_REMAIN_DIR = Path("eval_M_REMAIN")
EVAL_UNL_DIR = Path("eval_M_UNLEARNED")

# Output CSV
OUT_CSV = "speaker_similarity_all_remain_unlearned.csv"

# Test sentences used for generation in infer_vits_all.py
# >>> MAKE SURE THIS LIST MATCHES THAT SCRIPT <<<
TEST_SENTENCES = [
    "This is a test sentence for our unlearning experiment.",
    "The quick brown fox jumps over the lazy dog.",
    # Add more here if you used more prompts in infer_vits_all.py
]


# ---------------- UTILS: METADATA & EMBEDDINGS ----------------

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


# ---------------- UTILS: GENERATED FILES & WER ----------------

def collect_generated(eval_dir: Path, tag: str):
    """
    Collect generated wav paths per speaker and utterance index.
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


def compute_wer_per_speaker(asr_model, gen_map, model_tag: str):
    """
    gen_map: dict[spk] -> list of (wav_path, utt_idx)
    Returns: dict[spk] -> avg WER for that speaker for this model.
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
                print(f"[WARN] ASR failed on {wav_path} ({model_tag}): {e}")
                continue

        if wers:
            spk_to_wers[spk] = float(np.mean(wers))
        else:
            spk_to_wers[spk] = None

    return spk_to_wers


# ---------------- MAIN ----------------

def main():
    random.seed(42)

    # ----- Load metadata -----
    if not META_ALL.is_file():
        raise FileNotFoundError(f"Metadata file not found: {META_ALL}")

    print("[INFO] Loading metadata from:", META_ALL)
    meta_entries = load_metadata(META_ALL)
    print(f"[INFO] Loaded {len(meta_entries)} usable metadata rows.")

    speakers = sorted({spk for _, _, spk in meta_entries})
    print("Speakers in metadata:", speakers)

    # ----- Real reference clips -----
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

    # ----- Generated files -----
    gen_all = collect_generated(EVAL_ALL_DIR, "M_ALL")
    gen_remain = collect_generated(EVAL_REMAIN_DIR, "M_REMAIN")
    gen_unl = collect_generated(EVAL_UNL_DIR, "M_UNL")

    # ----- ASR model for WER -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[INFO] Loading Whisper ASR model on device:", device)
    asr_model = whisper.load_model("small", device=device)

    print("\n[INFO] Computing WER per speaker for each model...")
    wer_all = compute_wer_per_speaker(asr_model, gen_all, "M_ALL")
    wer_remain = compute_wer_per_speaker(asr_model, gen_remain, "M_REMAIN")
    wer_unl = compute_wer_per_speaker(asr_model, gen_unl, "M_UNL")

    # ----- Cosine similarities -----
    def get_sim(spk, gen_map):
        real_emb = spk_real_emb.get(spk)
        if real_emb is None:
            return None
        items = gen_map.get(spk, [])
        if not items:
            return None
        wav_paths = [p for (p, _) in items]
        gen_emb = embed_wavs(spk_encoder, wav_paths)
        if gen_emb is None:
            return None
        return cosine(real_emb, gen_emb)

    rows = []
    for spk in speakers:
        sim_all = get_sim(spk, gen_all)
        sim_remain = get_sim(spk, gen_remain)
        sim_unl = get_sim(spk, gen_unl)

        row = {
            "speaker_id": spk,
            "is_forget_speaker": (spk == FORGET_SPK),
            "sim_real_M_ALL": sim_all,
            "sim_real_M_REMAIN": sim_remain,
            "sim_real_M_UNLEARNED": sim_unl,
            "wer_M_ALL": wer_all.get(spk),
            "wer_M_REMAIN": wer_remain.get(spk),
            "wer_M_UNLEARNED": wer_unl.get(spk),
        }
        rows.append(row)

    # ----- Print table -----
    print("\n=== Speaker similarity + WER (real vs generated) ===")
    print("speaker_id | forget? | sim_M_ALL | sim_M_REMAIN | sim_M_UNL | wer_M_ALL | wer_M_REMAIN | wer_M_UNL")
    for r in rows:
        print(
            f"{r['speaker_id']:>9} | "
            f"{'YES' if r['is_forget_speaker'] else 'no ':>6} | "
            f"{r['sim_real_M_ALL'] if r['sim_real_M_ALL'] is not None else '  - ':>9} | "
            f"{r['sim_real_M_REMAIN'] if r['sim_real_M_REMAIN'] is not None else '  - ':>12} | "
            f"{r['sim_real_M_UNLEARNED'] if r['sim_real_M_UNLEARNED'] is not None else '  - ':>10} | "
            f"{r['wer_M_ALL'] if r['wer_M_ALL'] is not None else '  - ':>9} | "
            f"{r['wer_M_REMAIN'] if r['wer_M_REMAIN'] is not None else '  - ':>13} | "
            f"{r['wer_M_UNLEARNED'] if r['wer_M_UNLEARNED'] is not None else '  - ':>9}"
        )

    # ----- Save CSV -----
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "is_forget_speaker",
                "sim_real_M_ALL",
                "sim_real_M_REMAIN",
                "sim_real_M_UNLEARNED",
                "wer_M_ALL",
                "wer_M_REMAIN",
                "wer_M_UNLEARNED",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[OK] Saved similarity + WER results to {OUT_CSV}")


if __name__ == "__main__":
    main()
