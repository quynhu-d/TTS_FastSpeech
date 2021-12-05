from dataclasses import dataclass
from typing import Union, Tuple


@dataclass
class TrainConfig:
    wandb_project: str = 'TTS_FastSpeech'
    wandb_name: str = 'default'
    lj_path: str = '.'
    n_epochs: int = 50
    lr: float = 3e-4
    val_split: float = None
    batch_size: int = 20
