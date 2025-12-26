# build_speakers_json.py
import os
import json

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples


def main():
    dataset_root = "data/libritts_8spk"
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train="metadata.csv",
        path=dataset_root,
        language="en",
    )

    train_samples, _ = load_tts_samples(dataset_config, eval_split=False)

    speakers = sorted({s["speaker_name"] for s in train_samples})
    print("Detected speakers:", speakers)

    # Create a simple mapping: speaker_name -> index
    mapping = {spk: i for i, spk in enumerate(speakers)}
    print("Speaker mapping:", mapping)

    # Write it into the main runs folder
    root_json = os.path.join("runs", "vits_libritts_8spk_all", "speakers.json")
    os.makedirs(os.path.dirname(root_json), exist_ok=True)
    with open(root_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print("Wrote:", root_json)

    # Also write into the specific run folder (so Synthesizer can find it there too)
    run_dir = r"runs\vits_libritts_8spk_all\vits_libritts_8spk_all-December-14-2025_02+16PM-0000000"
    run_json = os.path.join(run_dir, "speakers.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print("Wrote:", run_json)


if __name__ == "__main__":
    main()
