# TTS Unlearning Experiment Plan (Quick Notes)

## Dataset
- Corpus: LibriTTS-R (SLR141)
- Splits: train_clean_100 (train), dev_clean/test_clean for eval (optional)
- Sampling rate: 24 kHz
- Language: English

## Speakers
- Total speakers for experiment: 6–8 from train_clean_100
- 1 speaker = forget speaker (Sf)
- 5–7 speakers = remain speakers (Sr)

- Forget Speaker(Sf):
    - 27

- Remain Speakers (Sr):
    - 233
    - 201
    - 229
    - 254
    - 446
    - 911
    - 1743


## Models
- M_all (Student model)

    - The model M_ALL is stored in runs\vits_libritts_8spk_all\vits_libritts_8spk_all-December-17-2025_10+35AM-0000000
    - This model has been trained on the 8 chosen speakers for 50000 steps (~315 epochs (each epoch has 156 steps))

- M_remain (Teacher Model)
    - The model M_ALL is stored in runs\vits_libritts_8spk_remain\vits_libritts_8spk_all-December-17-2025_10+35AM-0000000
    - This model has been trained on the 8 chosen speakers for 50000 steps (~315 epochs (each epoch has 156 steps))


