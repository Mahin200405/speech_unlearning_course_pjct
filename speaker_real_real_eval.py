# speaker_real_real_eval.py

import os
from pathlib import Path
import random
import csv

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

# ==== CONFIG =====

DATA_ROOT = Path("data/libritts_8spk")

# IMPORTANT: set this to the metadata file that contains ALL 8 speakers
# e.g. "metadata_clean2.csv" or whatever your full meta file is called.
META_ALL = DATA_ROOT / "metadata_clean.csv"

CLIPS_PER_SPK = 10      # how many real clips per speaker to use (max)
FORGET_SPK = "27"       # forget speaker id as string


def load_metadata(meta_path: Path):
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


def build_clip_lists(entries, clips_per_spk=CLIPS_PER_SPK):
    """Group entries by speaker and select up to clips_per_spk real utterances."""
    by_spk = {}
    for wav_path, text, spk in entries:
        by_spk.setdefault(spk, []).append(wav_path)

    spk_to_paths = {}
    for spk, paths in by_spk.items():
        if len(paths) <= clips_per_spk:
            chosen = paths
        else:
            chosen = random.sample(paths, clips_per_spk)
        spk_to_paths[spk] = chosen
    return spk_to_paths


def embed_single_clip(encoder: VoiceEncoder, wav_path: Path):
    """Embed a single utterance using Resemblyzer."""
    wav = preprocess_wav(str(wav_path))  # handles resampling etc.
    emb = encoder.embed_utterance(wav)
    return emb


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    random.seed(42)

    if not META_ALL.is_file():
        raise FileNotFoundError(f"Metadata file not found: {META_ALL}")

    print("[INFO] Loading metadata from:", META_ALL)
    entries = load_metadata(META_ALL)
    print(f"[INFO] Loaded {len(entries)} usable entries from metadata.")

    # All speakers present in this file
    speakers = sorted({spk for _, _, spk in entries})
    print("Speakers in full metadata:", speakers)

    # Build real clip lists per speaker
    spk_to_paths = build_clip_lists(entries, CLIPS_PER_SPK)
    for spk, paths in spk_to_paths.items():
        print(f"[INFO] Speaker {spk}: using {len(paths)} real clips")

    # Initialize speaker encoder
    print("\n[INFO] Loading Resemblyzer VoiceEncoder...")
    encoder = VoiceEncoder()

    # Compute embeddings for each real clip
    print("\n[INFO] Embedding real clips...")
    spk_clip_embs = {}  # spk -> list of embeddings
    for spk, paths in spk_to_paths.items():
        emb_list = []
        for p in paths:
            try:
                emb = embed_single_clip(encoder, p)
                emb_list.append(emb)
            except Exception as e:
                print(f"[WARN] Failed to embed {p}: {e}")
        if emb_list:
            spk_clip_embs[spk] = emb_list

    # Compute a centroid (mean embedding) for each speaker
    spk_centroid = {}
    for spk, emb_list in spk_clip_embs.items():
        spk_centroid[spk] = np.stack(emb_list, axis=0).mean(axis=0)

    # Build speaker list actually present (with embeddings)
    spk_list = sorted(spk_centroid.keys())
    print("\n[INFO] Speakers with computed centroids:", spk_list)

    # Compute similarity matrix between speaker centroids
    n = len(spk_list)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    for i, spk_i in enumerate(spk_list):
        for j, spk_j in enumerate(spk_list):
            sim_matrix[i, j] = cosine(spk_centroid[spk_i], spk_centroid[spk_j])

    # Print table
    print("\n=== Real vs Real: Speaker centroid cosine similarities ===")
    header = "spk_i\\spk_j | " + " ".join(f"{s:>6}" for s in spk_list)
    print(header)
    print("-" * len(header))
    for i, spk_i in enumerate(spk_list):
        row_vals = " ".join(f"{sim_matrix[i, j]:6.3f}" for j in range(n))
        print(f"{spk_i:>10} | {row_vals}")

    # Summaries: same-speaker vs different-speaker statistics
    same_sims = []
    diff_sims = []
    for i in range(n):
        for j in range(n):
            if i == j:
                same_sims.append(sim_matrix[i, j])
            else:
                diff_sims.append(sim_matrix[i, j])

    print("\n=== Summary stats ===")
    print(f"Same-speaker sims (diagonal): mean={np.mean(same_sims):.3f}, "
          f"min={np.min(same_sims):.3f}, max={np.max(same_sims):.3f}")
    print(f"Diff-speaker sims (off-diagonal): mean={np.mean(diff_sims):.3f}, "
          f"min={np.min(diff_sims):.3f}, max={np.max(diff_sims):.3f}")

    # Focus on forget speaker 27 vs others (if present)
    if FORGET_SPK in spk_list:
        idx_27 = spk_list.index(FORGET_SPK)
        print(f"\n=== Similarity of forget speaker {FORGET_SPK} vs others (real vs real) ===")
        for j, spk_j in enumerate(spk_list):
            print(f"27 vs {spk_j}: {sim_matrix[idx_27, j]:.3f}")
    else:
        print(f"\n[WARN] Forget speaker {FORGET_SPK} not present in centroid list.")

    # Save matrix to CSV for your report
    out_csv = "speaker_real_real_similarity_matrix.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_i", "speaker_j", "cosine_sim"])
        for i, spk_i in enumerate(spk_list):
            for j, spk_j in enumerate(spk_list):
                writer.writerow([spk_i, spk_j, sim_matrix[i, j]])

    print(f"\n[OK] Saved similarity matrix to {out_csv}")


if __name__ == "__main__":
    main()
