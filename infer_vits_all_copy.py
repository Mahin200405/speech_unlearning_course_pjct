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
    cand = sorted(p.glob("*0.pth"))
    if not cand:
        raise FileNotFoundError(f"No .pth checkpoints found in {run_dir}")
    print(cand)
    return str(cand[-1])


def main():
    # ---- 1. Paths ----
    exp_dir = r"runs\vits_libritts_8spk_all\vits_libritts_8spk_all-December-14-2025_02+16PM-0000000"
    config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
    checkpoint_path = find_latest_checkpoint(exp_dir)

    print("CWD:", os.getcwd())
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

    # ---- 4. Test texts ----
    test_texts = [
        "This is a test sentence for our unlearning experiment.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    out_root = Path("samples_M_u1")
    out_root.mkdir(parents=True, exist_ok=True)
    print("Output directory:", out_root.resolve())

    # ---- 5. Generate audio for each speaker index ----
    for idx, spk in enumerate(speaker_list):
        for i, text in enumerate(test_texts):
            print(f"\n[DEBUG] Generating for speaker_idx {idx} (speaker_id {spk}), text {i}...")
            wav = synthesizer.tts(text=text, speaker_idx=idx)

            print("[DEBUG] Raw wav type:", type(wav))

            # Whatever we get (list of floats, numpy array, etc.), convert to 1D numpy array
            wav = np.asarray(wav, dtype=np.float32)
            print("[DEBUG] After np.asarray, shape:", wav.shape, "dtype:", wav.dtype)

            # Flatten to 1D: (N,) regardless of original shape
            wav = wav.reshape(-1)
            print("[DEBUG] After reshape(-1), shape:", wav.shape)

            if wav.size == 0:
                print("[ERROR] Wav has zero size after processing, skipping.")
                continue


            out_path = out_root / f"M_all_spk{spk}_idx{idx}_utt{i}.wav"
            try:
                sf.write(str(out_path), wav, synthesizer.output_sample_rate)
                print("[DEBUG] Wrote file:", out_path)
            except Exception as e:
                print("[ERROR] Failed to write file:", out_path, "Error:", repr(e))

    # ---- 6. List files in output dir ----
    print("\nFinal contents of output directory:")
    try:
        files = list(out_root.iterdir())
        for f in files:
            print("  -", f.name)
        if not files:
            print("  (no files)")
    except Exception as e:
        print("[ERROR] Could not list output directory:", repr(e))

    print("\nDone. Check the 'samples_M_all' folder (path above).")


if __name__ == "__main__":
    main()
