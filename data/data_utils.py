from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    duration_preds: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform.to(device)
        self.waveform_length.to(device)
        self.tokens.to(device)
        self.token_lengths.to(device)
        if self.durations:
            self.durations.to(device)
