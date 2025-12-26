# infer_vits_all.py

import os
from pathlib import Path

import torch
import soundfile as sf

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.synthesizer import Synthesizer

import numpy as np

def find_latest_checkpoint(run_dir: str) -> str:
    p = Path(run_dir)
    cand = sorted(p.glob("*.pth"))
    if not cand:
        raise FileNotFoundError(f"No .pth checkpoints found in {run_dir}")
    return str(cand[-1])


def main():
    # ---- 1. Paths ----
    exp_dir = r"runs\vits_libritts_8spk_all\vits_libritts_8spk_all-December-17-2025_07+04AM-0000000"

    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    checkpoint_path = find_latest_checkpoint(exp_dir)

    print("Using config:    ", config_path)
    print("Using checkpoint:", checkpoint_path)

    # ---- 2. Get speaker list from metadata ----
    dataset_root = "data/libritts_8spk"
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train="metadata.csv",
        path=dataset_root,
        language="en",
    )

    train_samples, _ = load_tts_samples(dataset_config, eval_split=False)
    speakers = sorted({s["speaker_name"] for s in train_samples})
    print("Speakers detected from metadata:", speakers)

    # Build an index mapping: 0,1,2,... -> speaker_id string
    speaker_list = list(speakers)
    print("Speaker index mapping:")
    for idx, spk in enumerate(speaker_list):
        print(f"  idx {idx} -> speaker_id {spk}")

    # ---- 3. Build Synthesizer ----
    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)

    synthesizer = Synthesizer(
        tts_checkpoint=checkpoint_path,
        tts_config_path=config_path,
        use_cuda=use_cuda,
    )

    # ---- 4. Define some test texts ----
    test_texts = [
        "This is a test sentence for our unlearning experiment.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    out_root = Path("samples_M_all")
    out_root.mkdir(parents=True,exist_ok=True)

    # ---- 5. Generate audio for each speaker index ----
    for idx, spk in enumerate(speaker_list):
        for i, text in enumerate(test_texts):
            print(f"Generating for speaker_idx {idx} (speaker_id {spk}), text {i}...")
            wav = synthesizer.tts(text=text, speaker_idx=idx)

            # If the text was split into sentences, tts() may return a list.
            if isinstance(wav, list):
                if len(wav) == 0:
                    raise ValueError("TTS returned an empty list of chunks.")
                # For our use case, it's always a single sentence, so just take the first chunk.
                wav = wav[0]

            # Convert to numpy array and flatten to 1D
            wav = np.asarray(wav).reshape(-1)


    print("\nDone. Check the 'samples_M_all' folder for generated audio.")

if __name__ == "__main__":
    main()
