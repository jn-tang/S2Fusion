from typing import Union
import torch
from torch import nn
import torch.nn.functional as F

from model.utils import latest_checkpoint



class PhaseEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 27, 
                 window: int = 120, 
                 duration: float = 4.0,
                 pretrained_ckpt: str = None) -> None:
        super().__init__()
        self.window = window
        self.out_channels = out_channels
        # constants
        self.register_buffer('args', torch.linspace(-duration, duration, window))
        self.register_buffer('freqs', torch.fft.rfftfreq(window)[1:] * window / duration)

        intermediate_channels = in_channels // 2
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.LayerNorm(window),
            nn.ELU(),
            nn.Conv1d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        )
        self.fc = nn.ModuleList(nn.Linear(window, 2) for _ in range(out_channels))

        self.reconstruct = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=intermediate_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.LayerNorm(window),
            nn.ELU(),
            nn.Conv1d(in_channels=intermediate_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        )

        if not pretrained_ckpt is None:
            self.load_state_dict(torch.load(pretrained_ckpt))
    

    def FFT(self, x):
        y = x.permute(0, 2, 1)  # (B,T,C) -> (B,C,T)
        feature = self.feature_extraction(y)
        rfft = torch.fft.rfft(feature, dim=-1)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]
        power = spectrum ** 2
        freq = torch.sum(self.freqs * power, dim=-1) / torch.sum(power, dim=-1)
        amp = 2 * torch.sqrt(torch.sum(power, dim=-1)) / self.window
        offset = rfft.real[:, :, 0] / self.window
        
        p = torch.empty((feature.shape[0], self.out_channels), dtype=feature.dtype, device=feature.device)
        for i in range(self.out_channels):
            v = self.fc[i](feature[:, i])
            p[:, i] = torch.atan2(v[:, 1], v[:, 0]) / torch.pi

        return freq, amp, offset, p
    

    def inverseFFT(self, freq, amp, offset, p):
        freq, amp, offset, p = map(lambda x: x.unsqueeze(dim=-1), (freq, amp, offset, p))
        z = (amp * torch.sin(torch.pi * (freq * self.args + p)) + offset).permute(0, 2, 1)
        return z


    def encode(self, x):
        freq, amp, offset, p = self.FFT(x)
        return self.inverseFFT(freq, amp, offset, p)
    

    def decode(self, z):
        z = z.permute(0, 2, 1)
        return self.reconstruct(z).permute(0, 2, 1)
    

    def form_input(self, data):
        head_pos = F.pad(data['jpos'][:, :, 15], (0, 0, 1, 0))
        head_vel = head_pos[:, 1:] - head_pos[:, :-1]
        bs, tlen, *_ = head_vel.shape
        x = torch.cat([data['acc'].reshape(bs, tlen, -1), head_vel.reshape(bs, tlen, -1)], dim=-1)
        return x
        
    
    def forward(self, data):
        x = self.form_input(data)
        z = self.encode(x)
        x_hat = self.decode(z)
        return nn.functional.mse_loss(x_hat, x)
    

    def validate(self, data):
        x = self.form_input(data)
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


    def load(self, ckpt_path: Union[str, None]) -> None:
        if not ckpt_path is None:
            checkpoint = latest_checkpoint(ckpt_path, hint='model')
            if not checkpoint is None:
                self.load_state_dict(torch.load(checkpoint))