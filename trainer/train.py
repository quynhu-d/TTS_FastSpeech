import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from data import LJSpeechDataset, LJSpeechCollator
from featurizer import MelSpectrogramConfig, MelSpectrogram
from model import FastSpeech, FastSpeechConfig
from model import GraphemeAligner, Vocoder
from trainer.trainer_config import TrainConfig
import errno
import os
from typing import Tuple


def train(
        train_config: TrainConfig,
        mel_config: MelSpectrogramConfig,
        fconfig: FastSpeechConfig,
        vocoder: Vocoder = None,    # pass mel2wav model to log audio
        model_cp_path: str = None,
        wandb_resume: Tuple[str, str] = (None, None)    # resume mode and run id
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on', device)
    try:
        train_dataloader = DataLoader(
            LJSpeechDataset(train_config.lj_path), batch_size=train_config.batch_size, collate_fn=LJSpeechCollator()
        )
    except errno:
        raise "No dataset found at %s" % train_config.lj_path

    if not os.path.exists(train_config.save_dir):
        os.mkdir(train_config.save_dir)
    import time
    model_path = train_config.save_dir + '/%s/' % (time.strftime("%d-%m-%I-%M-%S"))
    os.mkdir(model_path)
    print('Saving model at %s' % model_path)

    featurizer = MelSpectrogram(mel_config).to(device)
    model = FastSpeech(1000, fconfig).to(device)
    if model_path is not None:
        print('Loading model checkpoint')
        model.load_state_dict(torch.load(model_cp_path, map_location=device))
    aligner = GraphemeAligner().to(device)
    if vocoder is not None:
        vocoder = vocoder.to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), train_config.lr, (.9, .98))
    total_config = {**train_config.__dict__, **mel_config.__dict__, **fconfig.__dict__}
    wandb.init(
        name=train_config.wandb_name, project=train_config.wandb_project,
        config=total_config, resume=wandb_resume[0], id=wandb_resume[1]
    )
    min_loss = None
    for i in range(train_config.n_epochs):
        for j, batch in enumerate(train_dataloader):
            batch.to(device)
            optimizer.zero_grad()
            batch.mel = featurizer(batch.waveform)
            durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)
            durations *= batch.mel.size(-1)
            batch.durations = durations.round().to(device)
            # batch.to(device)
            output = model(batch)
            batch.mel_pred = output.to(device)
            batch.duration_preds = batch.duration_preds.to(device)  # batch.duration_preds
            # print(batch.token_lengths)
            if batch.duration_preds.size() != batch.durations.size():
                print('Batch skipped')
                print(batch.transcript)
                print('Predicted duration length:%d, aligner length: %d' %
                      (batch.duration_preds.size(1), batch.durations.size(1)))
                continue
            # print(batch.duration_preds, batch.durations)
            # print(batch.duration_preds.sum(1), batch.mel.size(-1), batch.durations.sum(1))
            dp_loss = F.mse_loss(batch.duration_preds, batch.durations)
            # print(output.size(), batch.mel.size())
            sz_diff = np.abs(batch.mel.size(-1) - output.size(-1))
            if batch.mel.size(-1) > output.size(-1):
                # print('padding batch')
                output = F.pad(output, (0, sz_diff, 0, 0, 0, 0), "constant", MelSpectrogramConfig.pad_value)
            else:
                # print('padding mel')
                batch.mel = F.pad(batch.mel, (0, sz_diff, 0, 0, 0, 0), "constant", MelSpectrogramConfig.pad_value)
            # print(batch.mel.shape, output.shape)
            mel_loss = F.mse_loss(output, batch.mel)
            loss = mel_loss + dp_loss
            if (i == 0) and (j == 0):
                min_loss = loss.item()
            if loss.item() <= min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), model_path + '/best_loss_model.pth')
            torch.save(model.state_dict(), model_path + '/checkpoint_model.pth')
            loss.backward()
            optimizer.step()

            # getting audio
            # batch_idx = np.random.randint(len(train_dataloader))
            # batch = iter(train_dataloader).next()
            if (vocoder is not None) and (i % train_config.display_every == 0):
                wav = vocoder.inference(batch.mel_pred)
                idx = np.random.randint(batch.mel.shape[0])
                wandb.log({
                    'loss': loss.item(),
                    'mel_loss': mel_loss.item(),
                    'dp_loss': dp_loss.item(),
                    'mel': wandb.Image(batch.mel[idx]),
                    'mel_pred': wandb.Image(output[idx]),
                    'audio': wandb.Audio(batch.waveform[idx].detach().cpu().numpy(),
                                         sample_rate=MelSpectrogramConfig.sr),
                    'audio_pred': wandb.Audio(wav[idx].detach().cpu().numpy(), sample_rate=MelSpectrogramConfig.sr),
                    'step': j,
                    'epoch': i
                })
            else:
                idx = np.random.randint(batch.mel.shape[0])
                wandb.log({
                    'loss': loss.item(),
                    'mel_loss': mel_loss.item(),
                    'dp_loss': dp_loss.item(),
                    'mel': wandb.Image(batch.mel[idx]),
                    'mel_pred': wandb.Image(batch.mel_pred[idx]),
                    # 'audio': wandb.Audio(batch.waveform[idx].detach().cpu().numpy(),
                    #                         sample_rate=MelSpectrogramConfig.sr),
                    'step': j,
                    'epoch': i
                })
    return model


if __name__ == '__main__':
    train_config = TrainConfig()
    mel_config = MelSpectrogramConfig()
    fconfig = FastSpeechConfig()
    train(train_config, mel_config, fconfig)
