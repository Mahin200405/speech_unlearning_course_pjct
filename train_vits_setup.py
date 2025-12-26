# train_vits_setup.py
import os

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import VitsAudioConfig

def main():
    # ---- 1. Dataset config (using our coqui-style metadata.csv) ----
    dataset_root = "data/libritts_8spk"

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train="metadata.csv",
        path=dataset_root,
        language="en",
    )

    print("Loading samples from:", dataset_root)
    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=False)
    print(f"Loaded {len(train_samples)} samples from metadata.csv")

    # Get unique speaker names from the samples
    speakers = sorted({s["speaker_name"] for s in train_samples})
    print("Detected speakers:", speakers)
    num_speakers = len(speakers)
    print("Number of speakers:", num_speakers)

    # ---- 2. Audio config for VITS ----
    # LibriTTS-R is 24kHz; VITS defaults are often 22.05kHz, but 24k is also OK.
    # If your wavs are 24k, set sample_rate=24000. If they are 22050, set 22050.
    audio_config = VitsAudioConfig(
        sample_rate=24000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    # ---- 3. VITS training config (NO TRAINING YET) ----
    output_path = os.path.join("runs", "vits_libritts_8spk_all")
    os.makedirs(output_path, exist_ok=True)

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_libritts_8spk_all",
        output_path=output_path,

        # training params (you can tweak later)
        batch_size=16,
        eval_batch_size=8,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        save_step=5000,
        print_step=25,
        print_eval=False,
        mixed_precision=True,

        # text processing
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,     # keep False for now to reduce complexity

        # dataset list
        datasets=[dataset_config],

        # multi-speaker setup
        use_speaker_embedding=True,
        num_speakers=num_speakers,
    )

    # ---- 4. Save config to JSON so we can reuse it for real training later ----
    config_path = os.path.join(output_path, "config.json")
    config.save_json(config_path)

    print("\nSaved VITS config to:", config_path)
    print("You now have a training config ready for VITS.")

if __name__ == "__main__":
    main()
