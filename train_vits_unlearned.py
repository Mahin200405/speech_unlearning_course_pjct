# train_vits_unlearned.py
import os

import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs


def main():
    print("=== train_vits_unlearned (M_UNLEARNED): START ===")

    # ---- 1. Load M_ALL config ----
    base_config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    print("Config path (from M_ALL):", base_config_path)
    if not os.path.isfile(base_config_path):
        raise FileNotFoundError(f"Config not found: {base_config_path}")

    config = VitsConfig()
    config.load_json(base_config_path)
    print(f"Loaded config from {base_config_path}")

    # ---- 2. Point dataset to metadata_remain.csv (forget speaker removed) ----
    if not config.datasets or len(config.datasets) == 0:
        raise ValueError("No datasets defined in config.")
    dataset_config = config.datasets[0]

    print("Original meta_file_train:", dataset_config.meta_file_train)
    dataset_config.meta_file_train = "metadata_remain.csv"
    print("Using meta_file_train    :", dataset_config.meta_file_train)

    print("Dataset config:")
    print("  path           :", dataset_config.path)
    print("  meta_file_train:", dataset_config.meta_file_train)
    print("  formatter      :", dataset_config.formatter)

    # ---- 3. Load ONLY training samples (no eval split) ----
    print("Loading training samples for M_UNLEARNED (remain speakers only)...")
    train_samples, _ = load_tts_samples(
        dataset_config,
        eval_split=False,
        eval_split_max_size=0,
        eval_split_size=0,
    )
    eval_samples = []

    print(f"Loaded {len(train_samples)} train samples for M_UNLEARNED.")

    if len(train_samples) == 0:
        print(" [!] No train samples found. Exiting.")
        return

    # ---- 4. Disable evaluation ----
    config.run_eval = False
    config.test_delay_epochs = -1

    # ---- 5. Set new output path for M_UNLEARNED ----
    unlearn_run_root = os.path.join("runs", "vits_libritts_8spk_unlearned")
    os.makedirs(unlearn_run_root, exist_ok=True)
    config.output_path = unlearn_run_root
    print("Output path (M_UNLEARNED):", config.output_path)

    # ---- 6. Initialize model (same architecture as M_ALL) ----
    print("Initializing VITS model for M_UNLEARNED...")
    model = Vits.init_from_config(config, samples=train_samples)
    print("Initialized VITS model for M_UNLEARNED.")

    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)

    # ---- 7. Restore weights from M_ALL checkpoint (this is the key unlearning step) ----
    # >>>>> CHANGE THIS to the checkpoint path of your M_ALL model (e.g. checkpoint_50000.pth) <<<<<
    restore_path = r"runs\vits_libritts_8spk_all\vits_libritts_8spk_all-December-17-2025_10+35PM-0000000\checkpoint_50000.pth"

    if not os.path.isfile(restore_path):
        raise FileNotFoundError(f"Restore checkpoint (M_ALL) not found: {restore_path}")
    print("Restoring from M_ALL checkpoint:", restore_path)

    trainer_args = TrainerArgs(
        restore_path=restore_path
    )

    # ---- 8. Trainer + fine-tune ----
    print("Creating Trainer for M_UNLEARNED...")
    trainer = Trainer(
        trainer_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("Starting fine-tuning (unlearning) on remain speakers...")
    trainer.fit()

    print("=== train_vits_unlearned: FINISHED NORMALLY ===")


if __name__ == "__main__":
    main()
