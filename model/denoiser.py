from typing import List, Optional
import torch
from torch import nn
from omegaconf import DictConfig
from omegaconf import DictConfig

from model.backbone import (
    TransformerEncoderLayer, 
    TransformerEncoder,
    SkipConnectTransformerEncoder,
    SinusoidalPositionEmbedding,
    positional
)


class ConditionalLatentDenoiser(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: List[int] = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "GELU",
                 position_embedding: str = "PositionEmbeddingLearned1D"
                 ):
        super().__init__()
        _, self.latent_dim = latent_dim
        act = getattr(nn, activation)
        embedding = getattr(positional, position_embedding)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            act(), 
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.cond_embedding = nn.Sequential(
            nn.Linear(nfeats, self.latent_dim),
            act(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.query_pos = embedding(self.latent_dim)
        self.mem_pos = embedding(self.latent_dim)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation
        )
        self.encoder = SkipConnectTransformerEncoder(encoder_layer, num_layers)



    def forward(self, x, t, cond=None, **kwargs):
        bs, tlen, *_ = x.shape
        x = x.permute(1, 0, 2)  # (B,T,C) -> (T,B,C)
        
        t = t.expand(bs).clone().to(x.dtype)
        latent_embeddings = self.time_embedding(t).unsqueeze(0)
        if not cond is None:
            cond = cond.permute(1, 0, 2)
            cond = self.cond_embedding(cond)
            latent_embeddings = torch.cat([latent_embeddings, cond], dim=0)

        # NOTE: missing linear transform x
        xseq = torch.cat([x, latent_embeddings], axis=0)
        xseq = self.query_pos(xseq)
        tokens = self.encoder(xseq)
        return tokens[:tlen].permute(1, 0, 2)
    



class ConditionalDenoiser(nn.Module):
    def __init__(self,
                 nfeats: int,
                 cond_dim: int,
                 dmodel: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4, 
                 dropout: float = 0.1,
                 activation: str = 'GELU',
                 conditioning: str = 'concat',
                 position_embedding: str = 'PositionEmbeddingLearned1D',
                 tokenizer: Optional[DictConfig] = None
                 ) -> None:
        super().__init__()
        act = getattr(nn, activation)
        embedding = getattr(positional, position_embedding)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(dmodel),
            nn.Linear(dmodel, dmodel),
            act(),
            nn.Linear(dmodel, dmodel)
        )
        
        self.conditioning = conditioning
        if conditioning == 'concat':
            self.in_proj = nn.Linear(nfeats + cond_dim, dmodel)  # since we concat condition and noise
        elif conditioning == 'token':
            self.in_proj = nn.Linear(nfeats, dmodel)
            self.cond_tokenizer = ConditionTokenizer(cond_dim, token_dim=dmodel, **tokenizer)
        else:
            raise ValueError(f'not supported conditioning type: {conditioning}')

        self.out_proj = nn.Linear(dmodel, nfeats)
        self.query_pos_embedding = embedding(dmodel)

        encoder_layer = TransformerEncoderLayer(
            dmodel, 
            num_heads,
            ff_size,
            dropout,
            activation
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.nfeats = nfeats


    def forward(self, x: torch.Tensor, cond: torch.Tensor, timestep: torch.LongTensor, env: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        bs, *_ = x.shape
        x = x.permute(1, 0, 2)  # (B,T,C) -> (T,B,C)
        cond = cond.permute(1, 0, 2)

        emb = self.time_embedding(timestep).unsqueeze(dim=0)
        num_additional_tokens = 1
        if not env is None:
            env = env.unsqueeze(dim=0)
            emb = torch.cat([emb, env], dim=0)
            num_additional_tokens += 1

        if self.conditioning == 'concat':
            x = self.in_proj(torch.cat([x, cond], dim=-1))
        else:
            cond = self.cond_tokenizer(cond, key_padding_mask)
            emb = torch.cat([emb, cond], dim=0)
            num_additional_tokens += 1
            x = self.in_proj(x)

        aug_mask = torch.cat([torch.zeros((bs, num_additional_tokens), device=key_padding_mask.device).bool(), key_padding_mask], dim=-1)

        xseq = torch.cat([emb, x], dim=0)
        xseq = self.query_pos_embedding(xseq)  # add positional embedding
        xseq = self.encoder(xseq, key_padding_mask=aug_mask)[num_additional_tokens:]
        output = self.out_proj(xseq)
        output[key_padding_mask.T] = 0  # zero out padded sequence
        return output.permute(1, 0, 2)  # (T,B,C) -> (B,T,C)
    

class ConditionTokenizer(nn.Module):
    def __init__(self, 
                 nfeats: int,
                 dmodel: int = 128,
                 token_dim: int = 256,
                 ff_size: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = 'GELU',
                 position_embedding: str = 'PositionEmbeddingLearned1D',
                 pooling: str = 'mean') -> None:
        super().__init__()
        embedding = getattr(positional, position_embedding)
        
        self.in_proj = nn.Linear(nfeats, dmodel)
        self.out_proj = nn.Linear(dmodel, token_dim)
        self.pos_embedding = embedding(dmodel)
        encoder_layer = TransformerEncoderLayer(dmodel=dmodel, nhead=num_heads, dim_ff=ff_size, dropout=dropout, activation=activation)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = getattr(torch, pooling)

    
    def forward(self, 
                cond: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.in_proj(cond)
        x = self.pos_embedding(x)
        x = self.encoder(x, key_padding_mask=key_padding_mask)
        x = self.pool(x, dim=0, keepdims=True)
        x = self.out_proj(x)

        return x
