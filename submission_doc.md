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

All of these models have been trained on the 8 chosen speakers for 50000 steps (~315 epochs (each epoch has 156 steps))


- M_all (Student model)
    - Please find the generated audio clips for 27(forget) and a random retain speaker [here](https://drive.google.com/drive/folders/1Kde9O2n2YUJUGmkb-2a69xozL5jN52Ia?usp=drive_link)

- M_remain (Teacher Model)
    - Please find the generated audio clips for 27(forget) and a random retain speaker [here](https://drive.google.com/drive/folders/19B8Xz-7xCJR-eClLWpE8bPeajFLP_YiQ?usp=drive_link)

- M_unlearned (via fine tuning)
    - Please find the generated audio clips for 27(forget) and a random retain speaker [here](https://drive.google.com/drive/folders/1iBy-h-RMGre0jTmxWHj9JD-194N86oDY?usp=drive_link)

- M_All + Adapter
    - Please find the generated audio clips for 27(forget) and a random retain speaker [here](https://drive.google.com/drive/folders/1T649ayzUqXKH_bNP5hYXZUwes5yIDZy1?usp=drive_link)

### Test Sentences
    1. This is a test sentence for our unlearning experiment
    2. The quick brown fox jumped over the lazy dog


## Explanation of Adapter

The adapter is a small, separate module that sits after the TTS system and after the speaker encoder: the TTS model generates audio, a fixed speaker encoder (e.g. Resemblyzer) turns that audio into a speaker embedding vector 
𝑒 ∈ 𝑅<sup>d</sup>
, and the adapter optionally modifies only the generated embeddings of the forget speaker, leaving all real embeddings and all remain-speaker embeddings unchanged. Concretely, for remain speakers the adapter behaves like an identity mapping (so their cosine similarity to their real voice is exactly the same as before), while for the forget speaker it injects controlled random noise and renormalises the vector, pushing it to a different direction in the embedding space so that cosine similarity with the true voice drops a lot; the strength of this perturbation is governed by a scalar hyperparameter 
𝛼. Because it operates purely in embedding space and not on waveforms or TTS weights, it is cheap, model-agnostic, and can be turned on or off during evaluation / deployment.

