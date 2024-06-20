from typing import Optional, Dict, Iterable, Union
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from omegaconf.dictconfig import DictConfig
from pytorch3d.structures import join_pointclouds_as_batch, join_pointclouds_as_scene, Pointclouds
from ema_pytorch import EMA

from model import (
    ConditionalDenoiser,
    MotionVAE,
    PhaseEmbedding,
    PointNet2Encoder,
)
from model.SMPL import SMPLH
from model.conditioning import create_conditioning_from_conf
from model.utils import (
    rotation_6d_to_axis_angle, 
    latest_checkpoint
)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 denoiser: DictConfig,
                 scheduler: DictConfig,
                 phase_embedding: Optional[DictConfig] = None,
                 scene_encoder: Optional[DictConfig] = None,
                 head_only: bool = False) -> None:
        super().__init__()
        self.denoiser = ConditionalDenoiser(**denoiser)
        self.scheduler = DDPMScheduler(**scheduler)
        self.num_steps = self.scheduler.config.num_train_timesteps
        self.nfeats = self.denoiser.nfeats
        if not phase_embedding is None:
            self.phase_embed = PhaseEmbedding(**phase_embedding)
        else:
            self.phase_embed = False
        
        if not scene_encoder is None:
            self.scene_encoder = PointNet2Encoder(**scene_encoder)
        else:
            self.scene_encoder = False
        
        self.head_only = head_only


    def forward(self, data: Dict[str, torch.Tensor], lambdas: Iterable[float], smpl_path: Optional[str] = None) -> torch.Tensor:
        # form condition tensor
        head_pos = data['jpos'][:, :, 15]
        bs, tlen, *_ = head_pos.shape
        cond = self.get_condition(data)
        
        x = data['poses'].reshape(bs, tlen, -1)
        t = torch.randint(0, self.num_steps, (bs,), device=x.device).long()
        env = self.encode_scene(data['scene'][0]) if self.scene_encoder else None
        mask = data['mask']
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        x_t = self.scheduler.add_noise(x, noise, t)
        pred = self.denoiser(x_t, cond, t, env=env, key_padding_mask=mask)
        lam_simple, lam_geometric = lambdas
        smpl = SMPLH(smpl_path).to(x.device)
        loss = lam_simple * F.mse_loss(pred, x)
        aa_hat = rotation_6d_to_axis_angle(pred.view(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        aa_gt = rotation_6d_to_axis_angle(x.view(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        loss += lam_geometric * F.mse_loss(smpl(aa_hat)[1][:, :22], smpl(aa_gt)[1][:, :22])
        return loss
    

    def validate(self, 
                 data: Dict[str, torch.Tensor], 
                 *, 
                 x_init: Optional[torch.Tensor] = None,
                 start_from: int = 1000,
                 conditioning: Optional[DictConfig] = None,
                 smpl_path: str = None,
                 **kwargs) -> torch.Tensor:
        head_pos = data['jpos'][:, :, 15]
        bs, tlen, *_ = head_pos.shape

        # form condition signal
        cond = self.get_condition(data)
        
        mask = data['mask']
        env = self.encode_scene(data['scene']) if self.scene_encoder else None
        conditioning = create_conditioning_from_conf(self, smpl_path, conditioning).to(cond.device) if not conditioning is None else None
        self.scheduler.set_timesteps(timesteps=list(range(start_from - 1, -1, -1)))
        timesteps = self.scheduler.timesteps

        if x_init is None:
            x_init = torch.randn((bs, tlen, self.nfeats), dtype=cond.dtype, device=cond.device) * self.scheduler.init_noise_sigma
        else:
            noise = torch.randn(x_init.shape, dtype=x_init.dtype, device=x_init.device)
            x_init = self.scheduler.add_noise(x_init, noise, start_from)
        
        x_hat = x_init

        # diffusion process
        with torch.no_grad():
            for t in timesteps:
                time = t.reshape((1,)).repeat((bs,)).to(cond.device)
                pred = self.denoiser(x_hat, cond, time, env=env, key_padding_mask=mask)  # predict clean sample
                if not conditioning is None:
                    pred = conditioning(pred, data, t)
                x_hat = self.scheduler.step(pred, t, x_hat).prev_sample  # noise it back

        # compute metrics
        smpl = SMPLH(smpl_path).to(x_hat.device)
        aa_hat = rotation_6d_to_axis_angle(x_hat.view(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        joints_hat = smpl(aa_hat)[1][:, :22].reshape(bs, tlen, -1, 3)
        trans_hat = data['jpos'][:, :, 15] - joints_hat[:, :, 15]
        joints_hat += trans_hat.unsqueeze(dim=2)
        
        return {
            'poses': x_hat,
            'jpos': joints_hat
        }
    
    
    def load(self, ckpt_path: Union[str, None]) -> None:
        if not ckpt_path is None:
            checkpoint = latest_checkpoint(ckpt_path, hint='model')
            if not checkpoint is None:
                self.load_state_dict(torch.load(checkpoint))


    def encode_scene(self, scene: Pointclouds) -> torch.Tensor:
        points = scene.points_padded()
        bs, p, _ = points.shape
        points = torch.cat([points, torch.zeros(bs, p, 6, device=points.device, dtype=points.dtype)], dim=-1)
        feature = self.scene_encoder(points.permute(0, 2, 1))
        # feature = self.scene_encoder(points)
        return feature
    

    def get_condition(self, data: Dict[str, torch.Tensor]):
        bs, tlen, *_ = data['poses'].shape
        if self.head_only:
            cond = torch.cat([
                data['poses'][:, :, [15]].reshape(bs, tlen, -1),
                data['angular'][:, :, [15]].reshape(bs, tlen, -1),
                data['jpos'][:, :, [15]].reshape(bs, tlen, -1),
                data['linear'][:, :, [15]].reshape(bs, tlen, -1)
            ], dim=-1)
        else:
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



# Full pipeline
class S2Diff:
    def __init__(self, 
                 ema: Optional[DictConfig] = None,
                 vae: Optional[DictConfig] = None, 
                 max_points: int = 32768, 
                 *args, **kwargs):
        model = GaussianDiffusion(*args, **kwargs)
        self.diffusion = model
        self.ema_enabled = not ema is None
        if not vae is None:
            self.vae = MotionVAE(**vae)
        else:
            self.vae = None
        if self.ema_enabled:
            self.ema = EMA(model, **ema)

        self.scene = None
        self.max_points = max_points


    def to(self, device):
        self.diffusion = self.diffusion.to(device)
        if self.ema_enabled:
            self.ema = self.ema.to(device)
        return self

    def validate(self, data: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        if (scene := data.get('scene', False)):
            if not self.scene is None:
                scene = join_pointclouds_as_batch([join_pointclouds_as_scene(list(pcds)) for pcds in zip(scene, self.scene)])
            self.scene = scene
            # subsample scene for inference
            subsampled = scene.subsample(self.max_points)
            data['scene'] = subsampled

        if self.ema_enabled:
            return self.ema.model.validate(data, *args, **kwargs)

        if not self.vae is None:
            x_init = self.vae.sample(data)
        else:
            x_init = None
        return self.diffusion.validate(data, x_init=x_init, *args, **kwargs)

    def eval(self):
        self.diffusion.eval()
        if self.ema_enabled:
            self.ema.eval()
        return self
    
    def load(self, ckpt_path: str):
        self.diffusion.load_state_dict(torch.load(latest_checkpoint(ckpt_path, hint='model')))
        if self.ema_enabled:
            self.ema.load_state_dict(torch.load(latest_checkpoint(ckpt_path, hint='ema')))
        if not self.vae is None:
            self.vae.load_state_dict(torch.load(latest_checkpoint(ckpt_path, hint='vae')))

    def clear(self):
        self.scene = None
