# tweak_vits_config_for_clean.py
# this file is to make the config file point to the newly formed cleaned metadata file
import os
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig

def main():
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = VitsConfig()
    cfg.load_json(config_path)

    # Use cleaned metadata
    dataset_root = "data/libritts_8spk"
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train="metadata_clean.csv",
        path=dataset_root,
        language="en",
    )
    cfg.datasets = [dataset_config]

    # Keep the GPU-friendly settings we previously tuned
    sr = cfg.audio.sample_rate if cfg.audio and cfg.audio.sample_rate else 24000
    cfg.batch_size = 4
    cfg.eval_batch_size = 2
    cfg.batch_group_size = 0
    cfg.num_loader_workers = 0
    cfg.num_eval_loader_workers = 0
    cfg.min_audio_len = 1
    cfg.max_audio_len = int(sr * 20.0)  # ~20s
    cfg.min_text_len = 1
    cfg.max_text_len = 300
    cfg.mixed_precision = True

    # Optionally bump epochs high so training doesn't auto-stop too early
    cfg.epochs = 1000

    cfg.save_json(config_path)
    print("Updated config saved to:", config_path)
    print("Using meta_file_train = metadata_clean.csv")

if __name__ == "__main__":
    main()
