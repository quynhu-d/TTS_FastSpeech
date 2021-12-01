import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from data import LJSpeechDataset, LJSpeechCollator
from featurizer import MelSpectrogramConfig, MelSpectrogram
from model import GraphemeAligner
from model import FastSpeech
from model import FastSpeechConfig
from model import Vocoder


def main(lj_path='.'):
    train_dataloader = DataLoader(LJSpeechDataset(lj_path), batch_size=20, collate_fn=LJSpeechCollator())
    mel_config = MelSpectrogramConfig()
    featurizer = MelSpectrogram(mel_config)
    model_config = FastSpeechConfig()
    model = FastSpeech(51, model_config)
    aligner = GraphemeAligner()
    vocoder = Vocoder()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4, (.9, .98))
    n_epochs = 30000
    wandb.init(name='overfit_batch')
    for i in range(n_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch.mel = featurizer(batch.waveform)
            durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)
            durations *= batch.mel.size(-1)
            batch.durations = durations.round()
            output = model(batch)
            dp_loss = F.mse_loss(batch.duration_preds, batch.durations)
            sz_diff = np.abs(batch.mel.size(-1) - output.size(-1))
            if batch.mel.size(-1) > output.size(-1):
                print('padding batch')
                output = F.pad(output, (0, sz_diff, 0, 0, 0, 0), "constant", MelSpectrogramConfig.pad_value)
            else:
                print('padding mel')
                batch.mel = F.pad(batch.mel, (0, sz_diff, 0, 0, 0, 0), "constant", MelSpectrogramConfig.pad_value)
            mel_loss = F.mse_loss(output, batch.mel)
            loss = mel_loss + dp_loss
            loss.backward()
            optimizer.step()
            wav = vocoder(output)

            idx = np.random.randint(batch.mel.shape[0])
            wandb.log({
                'loss': loss,
                'mel_loss': mel_loss,
                'dp_loss': dp_loss,
                'mel': wandb.Image(batch.mel[idx]),
                'mel_pred': wandb.Image(output[idx]),
                'audio': wandb.Audio(batch.waveform[idx]),
                'audio_pred': wandb.Audio(wav[idx]),
                'step': i
            })
            break    # one batch


if __name__ == '__main__':
    main()
