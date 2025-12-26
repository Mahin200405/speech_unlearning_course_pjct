# tweak_vits_config_4060.py  (or vits_config_tweak.py)
import os
from TTS.tts.configs.vits_config import VitsConfig

def main():
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = VitsConfig()
    cfg.load_json(config_path)

    # --- Keep it GPU-friendly, but not overly restrictive ---
    cfg.batch_size = 4          # keep small for 8 GB GPU
    cfg.eval_batch_size = 2
    cfg.batch_group_size = 0

    cfg.num_loader_workers = 0  # safer on Windows
    cfg.num_eval_loader_workers = 0

    # Set max audio length to ~20 seconds
    # Use whatever sample rate is in the config (LibriTTS-R is usually 24k)
    sr = cfg.audio.sample_rate if cfg.audio and cfg.audio.sample_rate else 24000
    cfg.min_audio_len = 1
    cfg.max_audio_len = int(sr * 20.0)   # 20 seconds -> e.g. 480000 samples at 24k

    # Text length limits can stay conservative
    cfg.min_text_len = 1
    cfg.max_text_len = 300

    # Mixed precision helps memory
    cfg.mixed_precision = True

    cfg.save_json(config_path)
    print("Updated config saved to:", config_path)
    print(f"sample_rate = {sr}, max_audio_len = {cfg.max_audio_len} samples (~20s)")

if __name__ == "__main__":
    main()
