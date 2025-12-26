#My dum ass did not change the number of speakers in the config file to 8, so it trained as a single speaker model
#using this file to change that

from TTS.tts.configs.vits_config import VitsConfig
import os

config_path = os.path.join("runs", "vits_libritts_8spk_all", "config.json")

cfg = VitsConfig()
cfg.load_json(config_path)

cfg.model_args.num_speakers = 8          # 👈 we have 8 speakers in metadata
cfg.model_args.use_speaker_embedding = True

cfg.save_json(config_path)
print("Saved multi-speaker config to:", config_path)
print("num_speakers:", cfg.model_args.num_speakers)
print("use_speaker_embedding:", cfg.model_args.use_speaker_embedding)
