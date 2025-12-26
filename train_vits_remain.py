# train_vits_remain.py
import os

import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs


def main():
    print("=== train_vits_remain: START ===")

    # ---- 1. Load config (same base config as M_ALL) ----
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    print("Config path:", config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = VitsConfig()
    config.load_json(config_path)
    print(f"Loaded config from {config_path}")

    # ---- 2. Use the same dataset config but swap metadata file ----
    if not config.datasets or len(config.datasets) == 0:
        raise ValueError("No datasets defined in config.")
    dataset_config = config.datasets[0]

    # change ONLY the metadata file, keep path/formatter etc. the same
    print("Original meta_file_train:", dataset_config.meta_file_train)
    dataset_config.meta_file_train = "metadata_remain.csv"
    print("Using meta_file_train    :", dataset_config.meta_file_train)

    print("Dataset config:")
    print("  path           :", dataset_config.path)
    print("  meta_file_train:", dataset_config.meta_file_train)
    print("  formatter      :", dataset_config.formatter)

    # ---- 3. Load ONLY training samples (no eval split) ----
    print("Loading training samples for M_REMAIN...")
    train_samples, _ = load_tts_samples(
        dataset_config,
        eval_split=False,
        eval_split_max_size=0,
        eval_split_size=0,
    )
    eval_samples = []

    print(f"Loaded {len(train_samples)} train samples for M_REMAIN.")

    if len(train_samples) == 0:
        print(" [!] No train samples found. Exiting.")
        return

    # ---- 4. Disable evaluation in the config ----
    config.run_eval = False
    config.test_delay_epochs = -1

    # ---- 5. Change output path so we don't overwrite M_ALL ----
    remain_run_root = os.path.join("runs", "vits_libritts_8spk_remain")
    os.makedirs(remain_run_root, exist_ok=True)
    config.output_path = remain_run_root
    print("Output path (M_REMAIN):", config.output_path)

    # ---- 6. Initialize VITS model ----
    print("Initializing VITS model for M_REMAIN...")
    model = Vits.init_from_config(config, samples=train_samples)
    print("Initialized VITS model for M_REMAIN.")

    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)

    # ---- 7. Create Trainer and start training ----
    print("Creating Trainer...")
    trainer = Trainer(
        TrainerArgs(),          # default trainer args
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    print("Starting training (trainer.fit) for M_REMAIN...")

    trainer.fit()

    print("=== train_vits_remain: FINISHED NORMALLY ===")


if __name__ == "__main__":
    main()
