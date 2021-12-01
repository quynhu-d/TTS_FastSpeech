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
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)
        if self.durations is not None:
            self.durations = self.durations.to(device)
        if self.duration_preds is not None:
            self.duration_preds = self.duration_preds.to(device)
