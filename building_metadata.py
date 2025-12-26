# build_metadata_anywhere.py - robust version for LibriTTS-R with .normalized/.original files
import os

def main():
    # Root of your mini LibriTTS subset
    # IMPORTANT: this folder should contain your 8 speaker folders somewhere below it
    root = os.path.join("data", "libritts_8spk")

    if not os.path.isdir(root):
        print(f"[ERROR] Dataset root does not exist: {root}")
        return

    rows = []

    print(f"[INFO] Walking under: {root}")
    num_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root):
        num_dirs += 1
        # Uncomment this if you want to see where it's looking:
        # print("DIR:", dirpath)

        for fname in filenames:
            # we only care about the .normalized transcript files
            if not fname.endswith(".normalized.txt"):
                continue

            norm_path = os.path.join(dirpath, fname)

            # read the text (single line)
            with open(norm_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:
                    continue

            # corresponding wav file: same prefix, .wav extension
            base = fname[: -len(".normalized.txt")]  # strip suffix
            wav_name = base + ".wav"
            wav_path = os.path.join(dirpath, wav_name)
            if not os.path.exists(wav_path):
                print(f"[WARN] Missing wav for {norm_path}")
                continue

            # speaker_id = folder one level above this dir
            # e.g. data/libritts_8spk/train-clean-100/8468/286673/8468_...normalized
            # or data/libritts_8spk/8468/286673/...
            speaker_dir = os.path.basename(os.path.dirname(dirpath))
            speaker_id = speaker_dir

            # make path relative to dataset root (data/libritts_8spk)
            rel_wav_path = os.path.relpath(wav_path, root)

            # avoid '|' in text because we'll use '|' as separator
            text = text.replace("|", " ")

            rows.append((rel_wav_path, text, speaker_id))

    out_path = os.path.join(root, "metadata.csv")
    os.makedirs(root, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("audio_file|text|speaker_name\n")
        for audio_file, text, speaker_name in rows:
            out_f.write(f"{audio_file}|{text}|{speaker_name}\n")

    print(f"[INFO] Walked {num_dirs} directories under {root}")
    print(f"[OK] Wrote {len(rows)} entries to {out_path}")

if __name__ == "__main__":
    main()
