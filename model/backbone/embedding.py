import torch
from torch import nn
import math


class TimeEmbedding(nn.Module):
    def __init__(self, channel: int, d_embedding: int, activation: str = "SiLU"):
        super().__init__()
        act = getattr(nn, activation)
        self.emb = nn.Sequential(
            nn.Linear(channel, d_embedding),
            act(),
            nn.Linear(d_embedding, d_embedding)
        )
    
    def forward(self, t):
        return self.emb(t)
    

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0, period: int = 10000):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.period = period
    
    def forward(self, x):
        dim_half = self.dim // 2
        exp = -math.log(self.period) * torch.arange(start=0, end=dim_half, dtype=x.dtype, device=x.device) / (dim_half - 1)
        emb = torch.exp(exp)
        emb = self.scale * (x[:, None] * emb[None, :])
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


def sinusoidal_position_embedding(timesteps: torch.Tensor, dim_embedding: int, scale: float = 1.0, period: int = 10000):
    return SinusoidalPositionEmbedding(dim_embedding, scale, period)(timesteps)
    