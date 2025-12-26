# check_dataset_loading.py

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

def main():
    # Root of your custom dataset
    # This folder should contain: metadata.csv  and  train-clean-100/<spk>/<chapter>/*.wav
    dataset_root = "data/libritts_8spk"

    # Dataset config for the generic "coqui" formatter
    dataset_config = BaseDatasetConfig(
        formatter="coqui",           # use our own metadata.csv format
        meta_file_train="metadata.csv",
        path=dataset_root,
        language="en"
    )

    print("Loading samples from:", dataset_root)
    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=False)

    print(f"Loaded {len(train_samples)} samples.")

    # Print a few example entries to confirm everything looks right
    for i, s in enumerate(train_samples[:5]):
        print(f"\nSample {i}:")
        print("  text        :", s["text"])
        print("  audio_file  :", s["audio_file"])
        print("  speaker_name:", s.get("speaker_name"))

if __name__ == "__main__":
    main()
