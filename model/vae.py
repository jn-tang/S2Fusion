from typing import List, Dict, Iterable, Union, Optional

import torch
from torch import nn, Tensor, BoolTensor
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig

from model.backbone import positional
from model.backbone import (
    SkipConnectTransformerEncoder,
    SkipConnectTransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer)
from model import PhaseEmbedding
from model.utils import rotation_6d_to_axis_angle, latest_checkpoint
from model.SMPL import SMPLH



class MotionVAE(nn.Module):
    """
    VAE-like conditional generative model
    """
    def __init__(self,
                 nfeats: int,
                 nconds: int,
                 latent_dim: List[int] = [1, 256],
                 ff_size: int = 512,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = 'GELU',
                 position_embedding: str = 'PositionEmbeddingLearned1D',
                 phase_embed: Optional[DictConfig] = None,
                 **kwargs) -> None:
        super().__init__()
        self.latent_size, self.latent_dim = latent_dim
        embedding = getattr(positional, position_embedding) 
        self.query_pos_encoder = embedding(self.latent_dim)
        self.query_pos_decoder = embedding(self.latent_dim)

        encoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipConnectTransformerDecoder(encoder_layer, num_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation
        )
        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipConnectTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self.emb = nn.Linear(nfeats, self.latent_dim)
        self.cond_emb = nn.Linear(nconds, self.latent_dim)
        self.linear = nn.Linear(self.latent_dim, nfeats)

        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))

        if not phase_embed is None:
            self.phase_embed = PhaseEmbedding(**phase_embed)
        else:
            self.phase_embed = False


    def sample(self, data: Dict[str, Tensor]) -> Tensor:
        cond = self.get_condition(data)
        bs, *_ = cond.shape
        mask = data['mask']
        latent_size, latent_dim = self.latent_size, self.latent_dim
        z = torch.randn(bs, latent_size, latent_dim, dtype=cond.dtype, device=cond.device)
        return self.decode(z, cond, mask)


    def forward(self, data: Dict[str, Tensor], lambdas: Iterable[float], smpl_path: str, *args, **kw) -> Tensor:
        # retrieve scaling factor for loss functions
        lam_kl, lam_recon, lam_fk = lambdas
        
        bs, tlen, *_ = data['poses'].shape
        cond = self.get_condition(data)

        x = data['poses'].reshape(bs, tlen, -1)
        mask = data['mask']
        
        z, dist = self.encode(x, cond, mask)
        x_hat = self.decode(z, cond, mask)

        # kl divergence
        dist_rf = torch.distributions.Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.scale))
        loss = lam_kl * torch.distributions.kl_divergence(dist, dist_rf).mean()
        # reconstruction loss
        loss += lam_recon * F.mse_loss(x_hat, x)
        # fk loss
        smpl = SMPLH(smpl_path).to(device=x.device)
        aa_gt = rotation_6d_to_axis_angle(x.view(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        aa_hat = rotation_6d_to_axis_angle(x_hat.view(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        loss += lam_fk * F.mse_loss(smpl(aa_hat)[1][:, :22], smpl(aa_gt)[1][:, :22])

        return loss
    

    def validate(self, 
                 data: Dict[str, Tensor], 
                 smpl_path: Optional[str] = None,
                 *args, **kw) -> Dict[str, Tensor]:
        x_hat = self.sample(data)
        bs, tlen, *_ = x_hat.shape
        smpl = SMPLH(smpl_path).to(x_hat.device)
        aa_hat = rotation_6d_to_axis_angle(x_hat.view(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        joints_hat = smpl(aa_hat)[1][:, :22].reshape(bs, tlen, -1, 3)

        trans_hat = data['jpos'][:, :, 15] - joints_hat[:, :, 15]
        joints_hat += trans_hat.unsqueeze(dim=2)

        return {
            'poses': x_hat,
            'jpos': joints_hat
        }


    def encode(self, x: Tensor, cond: Tensor, mask: BoolTensor) -> Tensor:
        device = x.device
        bs, *_ = x.shape
        
        x = self.emb(x)
        cond = self.cond_emb(cond)
        # (B,T,C) -> (T,B,C)
        x = x.permute(1, 0, 2)
        cond = cond.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None], (1, bs, 1))
        dist_mask = torch.zeros((bs, dist.shape[0]), dtype=torch.bool, device=device)
        aug_mask = torch.cat((dist_mask, mask), dim=1)

        xseq = torch.cat((dist, x), dim=0)
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(
            tgt=cond,
            memory=xseq,
            tgt_key_padding_mask=mask,
            memory_key_padding_mask=aug_mask
        )[:dist.shape[0]]

        mu = dist[:self.latent_size]
        logvar = dist[self.latent_size:]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist


    def decode(self, z: Tensor, c: Tensor, mask: BoolTensor) -> Tensor:
        queries = self.cond_emb(c).permute(1, 0, 2)  # (B,T,C) -> (T,B,C)
        queries = self.query_pos_decoder(queries)
        output = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=mask
        ).squeeze(0)

        output = self.linear(output)
        feats = output.permute(1, 0, 2)
        feats[mask] = 0
        return feats
    
    
    def load(self, ckpt_path: Union[str, None]) -> None:
        if not ckpt_path is None:
            checkpoint = latest_checkpoint(ckpt_path, hint='model')
            if not checkpoint is None:
                self.load_state_dict(torch.load(checkpoint))


    def get_condition(self, data: Dict[str, torch.Tensor]):
        bs, tlen, *_ = data['poses'].shape
        cond = torch.cat([
                data['poses'][:, :, [15, 18, 19]].reshape(bs, tlen, -1), 
                data['angular'][:, :, [15, 18, 19]].reshape(bs, tlen, -1), 
                data['jpos'][:, :, [15, 18, 19]].reshape(bs, tlen, -1), 
                data['linear'][:, :, [15, 18, 19]].reshape(bs, tlen, -1)
            ], dim=-1)
        
        if self.phase_embed:
            phase = self.phase_embed.encode(cond)
            cond = torch.cat([cond, phase], dim=-1)
        
        return cond