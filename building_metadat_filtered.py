#we filter the metadata further by removing long samples and those that cant be loaded 

# filter_metadata_by_duration.py
import os
import csv

import soundfile as sf

INPUT_METADATA = os.path.join("data", "libritts_8spk", "metadata.csv")
OUTPUT_METADATA = os.path.join("data", "libritts_8spk", "metadata_clean.csv")

# target max duration in seconds (slightly below our 20s model limit)
MAX_DURATION_SEC = 20.0


def main():
    if not os.path.isfile(INPUT_METADATA):
        raise FileNotFoundError(f"Input metadata not found: {INPUT_METADATA}")

    kept = 0
    dropped = 0

    root = os.path.join("data", "libritts_8spk")

    with open(INPUT_METADATA, "r", encoding="utf-8") as fin, \
         open(OUTPUT_METADATA, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin, delimiter="|")
        writer = csv.writer(fout, delimiter="|")

        header = next(reader)
        writer.writerow(header)  # audio_file|text|speaker_name

        for row in reader:
            if len(row) < 3:
                dropped += 1
                continue

            audio_rel, text, speaker = row
            audio_path = os.path.join(root, audio_rel)

            if not os.path.isfile(audio_path):
                dropped += 1
                continue

            try:
                data, sr = sf.read(audio_path)
            except Exception as e:
                print(f"[WARN] Failed to read {audio_path}: {e}")
                dropped += 1
                continue

            # duration in seconds
            if data.ndim > 1:
                # take first channel if stereo
                data = data[:, 0]
            duration = len(data) / float(sr)

            if duration > MAX_DURATION_SEC:
                # too long, drop it
                dropped += 1
                continue

            # keep it
            writer.writerow([audio_rel, text, speaker])
            kept += 1

    print(f"[OK] Wrote cleaned metadata to: {OUTPUT_METADATA}")
    print(f"    Kept samples   : {kept}")
    print(f"    Dropped samples: {dropped}")


if __name__ == "__main__":
    main()
