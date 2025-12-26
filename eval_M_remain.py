# eval_M_remain.py
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
    # Config used for training M_REMAIN (we based it on M_ALL config)
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")

    # IMPORTANT: set this to your 50k-step M_REMAIN checkpoint
    checkpoint_path = r"runs\\vits_libritts_8spk_remain\\vits_libritts_8spk_all-December-19-2025_12+34AM-0000000\\checkpoint_50000.pth"

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Using M_REMAIN config:", config_path)
    print("Using M_REMAIN checkpoint:", checkpoint_path)

    # 2) Load config, dataset info (metadata_remain.csv)
    cfg = VitsConfig()
    cfg.load_json(config_path)

    if not cfg.datasets or len(cfg.datasets) == 0:
        raise ValueError("No datasets defined in config.")

    dataset_cfg = cfg.datasets[0]
    dataset_root = dataset_cfg.path

    dataset_config = BaseDatasetConfig(
        formatter=dataset_cfg.formatter,
        meta_file_train=dataset_cfg.meta_file_train,
        path=dataset_root,
        language="en",
    )

    # 3) Speakers from metadata_remain.csv
    train_samples, _ = load_tts_samples(dataset_config, eval_split=False)
    speakers = sorted({s["speaker_name"] for s in train_samples})
    speaker_list = list(speakers)

    print("Speakers in REMAIN metadata:", speaker_list)
    print("Speaker index mapping (M_REMAIN):")
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

    # 5) Same test texts as M_ALL (for fair comparison)
    test_texts = [
        "This is a test sentence for our unlearning experiment.",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
    ]

    out_root = Path("eval_M_REMAIN")
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output directory:", out_root.resolve())

    # 6) Generate audio
    for idx, spk in enumerate(speaker_list):
        for i, text in enumerate(test_texts):
            print(f"\n[GEN REMAIN] speaker_idx {idx} (speaker_id {spk}), utt {i}")
            wav = synthesizer.tts(text=text, speaker_idx=idx)

            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if wav.size == 0:
                print("  [WARN] Empty waveform, skipping.")
                continue

            out_path = out_root / f"M_REMAIN_spk{spk}_idx{idx}_utt{i}.wav"
            sf.write(str(out_path), wav, synthesizer.output_sample_rate)
            print("  -> saved:", out_path)

    print("\nDone. Check the 'eval_M_REMAIN' folder.")


if __name__ == "__main__":
    main()
