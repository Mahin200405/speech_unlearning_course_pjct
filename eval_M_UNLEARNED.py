# eval_M_unlearned.py
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
    # 1) Use same config as M_ALL (architecture, speakers, etc.)
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")

    # 2) M_UNLEARNED checkpoint (from your unlearn run, e.g. step 50000)
    checkpoint_path = r"runs\\vits_libritts_8spk_unlearned\\vits_libritts_8spk_all-December-19-2025_03+54PM-0000000\\checkpoint_50000.pth"

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Using config (from M_ALL):", config_path)
    print("Using M_UNLEARNED checkpoint:", checkpoint_path)

    # 3) Load config just to get dataset root etc.
    cfg = VitsConfig()
    cfg.load_json(config_path)

    dataset_root = "data/libritts_8spk"

    # IMPORTANT: use the same metadata file that now includes all 8 speakers
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train="metadata_clean.csv",  # this should now have 27 + others
        path=dataset_root,
        language="en",
    )

    # 4) Speakers from metadata
    train_samples, _ = load_tts_samples(dataset_config, eval_split=False)
    speakers = sorted({s["speaker_name"] for s in train_samples})
    speaker_list = list(speakers)

    print("Speakers in metadata_remain.csv:", speaker_list)
    print("Speaker index mapping (M_UNLEARNED):")
    for idx, spk in enumerate(speaker_list):
        print(f"  idx {idx} -> speaker_id {spk}")

    # 5) Build Synthesizer with M_UNLEARNED weights
    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)

    synthesizer = Synthesizer(
        tts_checkpoint=checkpoint_path,
        tts_config_path=config_path,
        use_cuda=use_cuda,
    )

    # 6) Same test texts as before (for fair comparison)
    test_texts = [
        "This is a test sentence for our unlearning experiment.",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
    ]

    out_root = Path("eval_M_UNLEARNED")
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output directory:", out_root.resolve())

    # 7) Generate audio for each speaker + text
    for idx, spk in enumerate(speaker_list):
        for i, text in enumerate(test_texts):
            print(f"\n[GEN UNLEARNED] speaker_idx {idx} (speaker_id {spk}), utt {i}")
            wav = synthesizer.tts(text=text, speaker_idx=idx)

            # wav is a 1D numpy array (Resemblyzer style script already handled this)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if wav.size == 0:
                print("  [WARN] Empty waveform, skipping.")
                continue

            out_path = out_root / f"M_UNLEARNED_spk{spk}_idx{idx}_utt{i}.wav"
            sf.write(str(out_path), wav, synthesizer.output_sample_rate)
            print("  -> saved:", out_path)

    print("\nDone. Check the 'eval_M_UNLEARNED' folder.")


if __name__ == "__main__":
    main()
