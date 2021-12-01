from dataclasses import dataclass
from typing import Union, Tuple


@dataclass
class FastSpeechConfig:
    d_model: int = 384
    conv_hid_sz: int = 1536
    kernel_sz: Union[Tuple, int] = 3
    d_k: int = 384
    d_v: int = 384
    dropout_rate: float = .1
    n_enc: int = 6
    n_dec: int = 6
    n_heads: int = 2
    max_phoneme_len: int = 5000
    max_mel_len: int = 5000
    n_mels: int = 80
