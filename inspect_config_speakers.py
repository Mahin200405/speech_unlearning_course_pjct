# inspect_config_speakers.py
import os
from TTS.tts.configs.vits_config import VitsConfig

config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")
cfg = VitsConfig()
cfg.load_json(config_path)

print("num_speakers:", getattr(cfg.model_args, "num_speakers", None))
print("use_speaker_embedding:", getattr(cfg.model_args, "use_speaker_embedding", None))

# Also print distinct speakers seen in the dataset config, just to sanity check
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

dataset_root = "data/libritts_8spk"
dataset_config = BaseDatasetConfig(
    formatter="coqui",
    meta_file_train=cfg.datasets[0].meta_file_train,
    path=dataset_root,
    language="en",
)

train_samples, _ = load_tts_samples(dataset_config, eval_split=False)
speakers = sorted({s["speaker_name"] for s in train_samples})
print("speakers in metadata:", speakers)
print("num distinct speakers in metadata:", len(speakers))
