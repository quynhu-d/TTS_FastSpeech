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

## Cloning
    !git clone https://github.com/quynhu-d/TTS_FastSpeech
    %cd TTS_FastSpeech
    !pip install -r ./requirements
## Data
    !wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    !tar -xjf LJSpeech-1.1.tar.bz2 -C OUT_PATH
> Remember to pass `OUT_PATH` to `lj_path` in `TrainConfig()`.

## Training
Training is performed with `train.py` from trainer directory, configurations can be set with `TrainConfig`, `MelSpectrogramConfig` and `FastSpeechConfig`

    from model import FastSpeechConfig
    from featurizer import MelSpectrogramConfig
    from trainer import TrainConfig, train

    train_config = TrainConfig()
    mel_config = MelSpectrogramConfig()
    fconfig = FastSpeechConfig()
    train(train_config, mel_config, fconfig)


## Audio logging (Vocoder)
Vocoder is to be initialized from waveglow repository and to be passed to train function if audio tracking is needed.
    
    from model import Vocoder

    !git clone https://github.com/NVIDIA/waveglow.git
    !pip install googledrivedownloader
    from google_drive_downloader import GoogleDriveDownloader as gdd
    gdd.download_file_from_google_drive(
        file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
        dest_path='./waveglow_256channels_universal_v5.pt'
    )

    %cd waveglow
    vocoder = Vocoder()
    %cd ..

    train(train_config, mel_config, fconfig, vocoder)

## Inference
TODO