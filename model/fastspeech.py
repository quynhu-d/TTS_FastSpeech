import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential
from model.aligner import GraphemeAligner
from typing import Union, Tuple
from configs.fastspeech_config import FastSpeechConfig
from torch.nn.utils.rnn import pad_sequence
from featurizer.mel_config import MelSpectrogramConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout_rate: float = .1):
        """
        :param d_model: embedding dimension
        :param max_seq_len: maximum sequence length
        :param dropout_rate: dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = np.exp(torch.arange(0, d_model, 2) / d_model * (-np.log(10000)))
        pe[:, ::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # print(pe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x of shape (BATCH_SZ x SEQ_LEN x EMB_DIM)
        return self.dropout(x + self.pe[:x.size(1)])


class SelfAttention(nn.Module):
    def __init__(self, config: FastSpeechConfig = FastSpeechConfig()):
        """
        params in config:
            d_model: embedding size
            d_k: hidden size
        """
        super(SelfAttention, self).__init__()
        self.wq = nn.Linear(config.d_model, config.d_k)
        self.wk = nn.Linear(config.d_model, config.d_k)
        self.wv = nn.Linear(config.d_model, config.d_v)
        self.d_k = config.d_k

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # print(q.shape, k.shape, v.shape)
        z = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k), -1), v)
        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, config: FastSpeechConfig = FastSpeechConfig()):
        """
        params in config:
            n_heads
            d_model
            d_v
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = config.n_heads
        self.att_heads = nn.ModuleList([SelfAttention(config) for _ in range(config.n_heads)])
        self.w_out = nn.Linear(config.n_heads * config.d_v, config.d_model)

    def forward(self, x):
        z = [head(x) for head in self.att_heads]
        z = torch.concat(z, -1)
        return self.w_out(z)


class FFTBlock(nn.Module):
    def __init__(self, config: FastSpeechConfig):
        super(FFTBlock, self).__init__()
        self.pre_ln = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.att_ln = nn.LayerNorm(config.d_model)
        self.conv = Sequential(
            nn.Conv1d(config.d_model, config.conv_hid_sz, config.kernel_sz, padding='same'),
            nn.ReLU(),
            nn.Conv1d(config.conv_hid_sz, config.d_model, config.kernel_sz, padding='same'),
            nn.ReLU()
        )
        self.conv_ln = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x1 = self.pre_ln(x)    # Pre-LN https://arxiv.org/pdf/2002.04745.pdf
        x2 = self.attention(x1)
        assert x2.size() == x.size()
        x3 = x + x2
        x4 = self.att_ln(x3)
        x5 = self.conv(x4.transpose(-1, -2)).transpose(-1, -2)
        out = self.conv_ln(x5 + x3)
        return out


class DurationPredictor(nn.Module):
    def __init__(self, config: FastSpeechConfig = FastSpeechConfig()):
        super(DurationPredictor, self).__init__()
        self.conv1 = Sequential(
            nn.Conv1d(config.d_model, config.conv_hid_sz, config.kernel_sz, padding='same'), nn.ReLU()
        )
        self.ln_1 = nn.LayerNorm(config.conv_hid_sz)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.conv2 = Sequential(
            nn.Conv1d(config.conv_hid_sz, config.d_model, config.kernel_sz, padding='same'), nn.ReLU()
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.out = nn.Linear(config.d_model, 1)

    def forward(self, x):
        x = self.conv1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.dropout1(self.ln_1(x))
        x = self.conv2(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.dropout2(self.ln_2(x))
        return self.out(x)


class LengthRegulator(nn.Module):
    def __init__(self, config: FastSpeechConfig, alpha: float = 1.0):
        super(LengthRegulator, self).__init__()
        self.alpha = alpha
        self.dur_pred = DurationPredictor(config)

    def forward(self, x, durations=None):
        duration_preds = self.dur_pred(x).exp()    # lengths >= 0
        duration_preds = (duration_preds * self.alpha).long()
        if durations is None:
            durations = duration_preds
        h_mel = []
        for h_pho, duration in (x, durations):
            h_mel.append(torch.repeat_interleave(h_pho, duration, 0))
        return pad_sequence(h_mel, batch_first=True, padding_value=MelSpectrogramConfig().pad_value), duration_preds


class FastSpeech(nn.Module):
    def __init__(self, n_vocab, config: FastSpeechConfig):
        super(FastSpeech, self).__init__()
        self.d_model = config.d_model
        self.emb_encoder = nn.Embedding(n_vocab, config.d_model)
        self.phoneme_pe = PositionalEncoding(config.d_model, config.max_phoneme_len, config.dropout_rate)
        self.phoneme_fft = Sequential(*[FFTBlock(config) for _ in range(config.n_enc)])
        self.len_reg = LengthRegulator(config)
        self.mel_pe = PositionalEncoding(config.d_model, config.max_mel_len, config.dropout_rate)
        self.mel_fft = Sequential(*[FFTBlock(config) for _ in range(config.n_dec)])
        self.linear = nn.Linear(config.d_model, config.n_mels)

    def forward(self, batch):
        phoneme_emb = self.emb_encoder(batch.tokens) * np.sqrt(self.d_model)
        phoneme_emb = self.phoneme_pe(phoneme_emb)
        enc_out = self.phoneme_fft(phoneme_emb)
        mel_emb, _ = self.len_reg(enc_out, batch.durations)
        mel_emb = self.mel_pe(mel_emb)
        dec_out = self.mel_fft(mel_emb)
        out = self.linear(dec_out)
        return out


# a = torch.randn(2, 3, 4)
# a_ = PositionalEncoding(d_model=4, max_seq_len=3)(a)
# print(a_.shape)
# print(SelfAttention(4, 2, 2)(a_).shape)
# print(MultiHeadAttention(4, 2, 2, 8)(a_).shape)
# print(FFTBlock(4, 2, 2, 2)(a).shape)
# print(DurationPredictor(4, 2, 3)(a).shape)