# Text to Speech with FastSpeech model
> This repository includes FastSpeech model implementation.

## Directory Layout
    .
    ├── data                    # dataset and collator for LJ
    ├── featurizer              # melspec
    ├── model                   # FastSpeech, Vocoder, Grapheme Aligner
    ├── trainer                 # training functions
    ├── tester                  # test function
    ├── Clean_TTS.ipynb         # example of model training and testing
    └── requirements.txt

## Report
For training and testing runs see `Clean_TTS.ipynb`. [Report](https://wandb.ai/quynhu_d/TTS_FastSpeech/reports/TTS-FastSpeech--VmlldzoxMzAzOTIz) and [loggings](https://wandb.ai/quynhu_d/TTS_FastSpeech?workspace=user-quynhu_d) located in `wandb` project.

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
`Vocoder` is to be initialized from `waveglow` repository and to be passed to train function if audio tracking is needed.
    
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
For inference use `test.py` in `test` directory. `Vocoder` is to be created as in section above. Set `FastSpeechConfig` and `model_path` to load trained model.

    transcript = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    fconfig = FastSpeechConfig()
    vocoder = Vocoder()    # import as stated above
    model_path = '/saved/01-01-70-00-00/model.pth'
    wavs_pred = test(vocoder, fconfig, model_path, transcript)