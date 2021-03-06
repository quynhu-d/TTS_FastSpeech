from dataclasses import dataclass


@dataclass
class TrainConfig:
    wandb_project: str = 'TTS_FastSpeech'
    wandb_name: str = 'default'
    display_every: int = 100    # audio logging step

    lj_path: str = '.'
    batch_size: int = 3
    val_split: float = None

    n_epochs: int = 50
    lr: float = 3e-4
    save_dir: str = 'saved/'

