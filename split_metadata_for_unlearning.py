# split_metadata_for_unlearning.py
import os

from TTS.tts.configs.vits_config import VitsConfig

# ---- SET THIS TO YOUR FORGET SPEAKER ID ----
FORGET_SPK = "27"  # keep as string, e.g. "27", "201", etc.


def main():
    # 1. Load M_ALL config to find the original metadata file
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = VitsConfig()
    cfg.load_json(config_path)
    print(f"[INFO] Loaded config from {config_path}")

    if not cfg.datasets or len(cfg.datasets) == 0:
        raise ValueError("No datasets defined in config.")

    dataset_cfg = cfg.datasets[0]
    dataset_root = dataset_cfg.path
    base_meta = dataset_cfg.meta_file_train

    in_meta_path = os.path.join(dataset_root, base_meta)
    print(f"[INFO] Using ORIGINAL metadata file: {in_meta_path}")

    if not os.path.isfile(in_meta_path):
        raise FileNotFoundError(f"Original metadata file not found: {in_meta_path}")

    # 2. Prepare output paths
    meta_remain_path = os.path.join(dataset_root, "metadata_remain.csv")
    meta_forget_path = os.path.join(dataset_root, "metadata_forget.csv")

    remain_count = 0
    forget_count = 0
    total_lines = 0

    # 3. Read original metadata line-by-line and split by FORGET_SPK
    with open(in_meta_path, "r", encoding="utf-8") as fin, \
         open(meta_remain_path, "w", encoding="utf-8") as fout_remain, \
         open(meta_forget_path, "w", encoding="utf-8") as fout_forget:

        for i, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            total_lines += 1

            parts = line.split("|")
            # Expect at least: audio_file|text|speaker_id
            if len(parts) < 3:
                print(f"[WARN] Skipping malformed line {i}: {line}")
                continue

            spk = parts[2].strip()  # speaker id as string

            if spk == FORGET_SPK:
                fout_forget.write(line + "\n")
                forget_count += 1
            else:
                fout_remain.write(line + "\n")
                remain_count += 1

    print(f"[INFO] Total lines read     : {total_lines}")
    print(f"[INFO] Remain samples written: {remain_count} -> {meta_remain_path}")
    print(f"[INFO] Forget samples written: {forget_count} -> {meta_forget_path}")
    print(f"[INFO] Forget speaker ID: {FORGET_SPK}")


if __name__ == "__main__":
    main()
