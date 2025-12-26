# infer_M_all_eval.py
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.synthesizer import Synthesizer


def find_latest_checkpoint(run_dir: str) -> str:
    p = Path(run_dir)
    cand = sorted(p.glob("*.pth"))
    if not cand:
        raise FileNotFoundError(f"No .pth checkpoints found in {run_dir}")
    latest = cand[-1]
    print("Using checkpoint:", latest)
    return str(latest)


def main():
    # ---------- 1. PATHS ----------
    exp_dir = r"runs\vits_libritts_8spk_remain\vits_libritts_8spk_all-December-14-2025_02+16PM-0000000"
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    checkpoint_path = find_latest_checkpoint(exp_dir)

    print("Config path:", config_path)

    # ---------- 2. SPEAKERS FROM METADATA ----------
    dataset_root = "data/libritts_8spk"
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        meta_file_train="metadata_clean.csv",  # or metadata_clean2.csv if that's what you used
        path=dataset_root,
        language="en",
    )

    train_samples, _ = load_tts_samples(dataset_config, eval_split=False)
    speakers = sorted({s["speaker_name"] for s in train_samples})
    speaker_list = list(speakers)

    print("Speakers detected from metadata:", speakers)
    print("Speaker index mapping:")
    for idx, spk in enumerate(speaker_list):
        print(f"  idx {idx} -> speaker_id {spk}")

    # ---------- 3. BUILD SYNTHESIZER ----------
    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)

    synthesizer = Synthesizer(
        tts_checkpoint=checkpoint_path,
        tts_config_path=config_path,
        use_cuda=use_cuda,
    )

    # ---------- 4. FIXED TEXTS ----------
    test_texts = [
        "This is a test sentence for our unlearning experiment.",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
    ]

    out_root = Path("eval_M_u1")
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output directory:", out_root.resolve())

    # ---------- 5. GENERATE ----------
    for idx, spk in enumerate(speaker_list):
        for i, text in enumerate(test_texts):
            print(f"\nGenerating for speaker_idx {idx} (speaker_id {spk}), text {i}...")
            wav = synthesizer.tts(text=text, speaker_idx=idx)

            # Convert whatever we got into a 1D numpy array
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if wav.size == 0:
                print("[WARN] Empty waveform, skipping.")
                continue

            out_path = out_root / f"M_ALL_spk{spk}_idx{idx}_utt{i}.wav"
            sf.write(str(out_path), wav, synthesizer.output_sample_rate)
            print("  -> saved to", out_path)

    print("\nDone. Check the 'eval_M_ALL' folder.")


if __name__ == "__main__":
    main()
