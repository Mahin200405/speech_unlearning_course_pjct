# eval_M_all.py
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.synthesizer import Synthesizer


def main():
    # 1) Paths
    # a) Config used for training M_ALL (multi-speaker)
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")

    # b) CHECKPOINT: set this to your 50k-step multi-speaker checkpoint
    #    e.g. "runs\\vits_libritts_8spk_all\\vits_libritts_8spk_all-December-18-2025_10+30PM-0000000\\checkpoint_50000.pth"
    checkpoint_path = "runs\\vits_libritts_8spk_all\\vits_libritts_8spk_all-December-17-2025_10+35PM-0000000\\checkpoint_50000.pth"

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Using config:", config_path)
    print("Using checkpoint:", checkpoint_path)

    # 2) Load config just to reuse metadata info
    cfg = VitsConfig()
    cfg.load_json(config_path)

    dataset_root = "data/libritts_8spk"
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train=cfg.datasets[0].meta_file_train,
        path=dataset_root,
        language="en",
    )

    # 3) Get speaker list from metadata
    train_samples, _ = load_tts_samples(dataset_config, eval_split=False)
    speakers = sorted({s["speaker_name"] for s in train_samples})
    speaker_list = list(speakers)

    print("Speakers in metadata:", speaker_list)
    print("Speaker index mapping:")
    for idx, spk in enumerate(speaker_list):
        print(f"  idx {idx} -> speaker_id {spk}")

    # 4) Build Synthesizer
    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)

    synthesizer = Synthesizer(
        tts_checkpoint=checkpoint_path,
        tts_config_path=config_path,
        use_cuda=use_cuda,
    )

    # 5) Fixed test texts (we'll reuse these for M_REMAIN + unlearned model)
    test_texts = [
        "This is a test sentence for our unlearning experiment.",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
    ]

    out_root = Path("eval_M_ALL")
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output directory:", out_root.resolve())

    # 6) Generate
    for idx, spk in enumerate(speaker_list):
        for i, text in enumerate(test_texts):
            print(f"\n[GEN] speaker_idx {idx} (speaker_id {spk}), utt {i}")
            wav = synthesizer.tts(text=text, speaker_idx=idx)

            # Make sure it's a 1D float32 numpy array
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if wav.size == 0:
                print("  [WARN] Empty waveform, skipping.")
                continue

            out_path = out_root / f"M_ALL_spk{spk}_idx{idx}_utt{i}.wav"
            sf.write(str(out_path), wav, synthesizer.output_sample_rate)
            print("  -> saved:", out_path)

    print("\nDone. Check the 'eval_M_ALL' folder.")


if __name__ == "__main__":
    main()
