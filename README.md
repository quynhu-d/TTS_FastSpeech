# Text to Speech with FastSpeech model
> This repository includes FastSpeech model implementation.

## Directory Layout
    .
    ├── data                    # dataset and collator for LJ
    ├── featurizer              # melspec
    ├── model                   # FastSpeech, Vocoder, Grapheme Aligner
    ├── trainer                 # training functions
    ├── Clean_TTS.ipynb         # overfit on one batch
    └── requirements.txt

## Utils
### Cloning
    !git clone github://///
    !cd TTS_FastSpeech
    !pip install -r ./requirements
### Data
    !wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    !tar -xjf LJSpeech-1.1.tar.bz2 -C OUT_PATH

## Training
Training is performed with `train.py` from trainer directory, configurations can be set with `TrainConfig`, `MelSpectrogramConfig` and `FastSpeechConfig`
    
    train_config = TrainConfig()
    mel_config = MelSpectrogramConfig()
    fconfig = FastSpeechConfig()
    train(train_config, mel_config, fconfig)

## Configs
    TrainConfig

## Inference (Vocoder)
