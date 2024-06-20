from typing import Sequence, Callable, Optional, Dict, List
import copy
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import torch
from torch.func import grad
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import ball_query
from model.SMPL import SMPLH
from model.utils import rotation_6d_to_matrix, matrix_to_rotation_6d


class Condition(nn.Module):
    def __init__(self, diffusion: nn.Module, smpl_path: str, loss_fn: Optional[ListConfig] = None):
        super().__init__()
        self.smpl = SMPLH(smpl_path)
        if not loss_fn is None:
            self.loss_fn = create_loss_from_conf(diffusion, loss_fn)

    def _loss(self, x: torch.Tensor, data: Dict[str, torch.Tensor]):
        bs, tlen, *_ = x.shape
        rotmat = rotation_6d_to_matrix(x.reshape(bs, tlen, -1, 6)).reshape(bs * tlen, -1, 3, 3)
        _, jpos_fk = self.smpl.forward_kinematic(rotmat)
        jpos_fk = jpos_fk.reshape(bs, tlen, -1, 3)
        trans = data['jpos'][:, :, 15] - jpos_fk[:, :, 15]
        jpos_fk += trans.unsqueeze(dim=2)
        
        return self.loss_fn(x, jpos_fk, data)
    

class LossGuidedPosterior(Condition):
    def __init__(self, diffusion: nn.Module, smpl_path: str, loss_fn: Optional[ListConfig] = None, scale: float = 0.5):
        super().__init__(diffusion, smpl_path, loss_fn)
        self.scale = scale
    
    def forward(self, x: torch.Tensor, data: Dict[str, torch.Tensor], *args, **kwargs):
        g = grad(self._loss, argnums=0)(x, data)
        return x - self.scale * g
    

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x, *args, **kwargs):
        return x
    

def create_conditioning_from_conf(diffusion: nn.Module, smpl_path: str, conf: DictConfig):
    cls = conf['cls']
    try:
        return globals()[cls](diffusion, smpl_path, **conf.args)
    except KeyError:
        return Identity()


class CombinedGeometricLoss(nn.Module):
    def __init__(self, *loss_fn: Sequence[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.loss_funcs = nn.ModuleList(loss_fn)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = 0
        for loss_fn in self.loss_funcs:
            loss += loss_fn(*args, **kwargs)
        return loss
    

class PhaseLoss(nn.Module):
    upper_body = [15, 18, 19]
    lower_body = [0, 7, 8]

    def __init__(self, 
                 diffusion: nn.Module, 
                 freq_scale: float = 1.0,
                 amp_scale: float = 1.0,
                 phase_scale: float = 1.0,
                 fps: int = 30) -> torch.Tensor:
        super().__init__()
        self.phase_enc = copy.deepcopy(diffusion.phase_embed)
        self.imu = diffusion.imu
        self.freq_scale = freq_scale
        self.amp_scale = amp_scale
        self.phase_scale = phase_scale
        self.fps = fps
        
    def forward(self, poses, joints, data):
        freq_up, amp_up, phase_up = self.extract_phase(poses, joints, self.upper_body)
        freq_lo, amp_lo, phase_lo = self.extract_phase(poses, joints, self.lower_body)

        return self.amp_scale * F.mse_loss(amp_lo, amp_up) + self.phase_scale * F.mse_loss(phase_lo, phase_up)
    
    def extract_phase(self, poses, joints, idxs):
        bs, tlen, *_ = poses.shape
        poses = poses.view(bs, tlen, -1, 6)[:, :, idxs]
        rotmat = rotation_6d_to_matrix(poses)
        joints = joints[:, :, idxs]
        angular = matrix_to_rotation_6d(torch.cat([rotmat[:, 0].unsqueeze(dim=1), torch.linalg.pinv(rotmat[:, :-1]) @ rotmat[:, 1:]], dim=1))
        if self.imu:
            anchor = joints[:, :, 0]
            acc = (joints[:, :-2] + joints[:, 2:] - 2 * joints[:, 1:-1]) * (self.fps ** 2)
            acc = F.pad(acc, (0, 0, 0, 0, 1, 1))
            signal = torch.cat([
                poses.reshape(bs, tlen, -1),
                angular.reshape(bs, tlen, -1),
                anchor,
                acc.reshape(bs, tlen, -1)
            ], dim=-1)
        else:
            linear = torch.cat([joints[:, [0]], joints[:, 1:] - joints[:, :-1]], dim=1)
            signal = torch.cat([
                poses.reshape(bs, tlen, -1),
                angular.reshape(bs, tlen, -1),
                joints.reshape(bs, tlen, -1),
                linear.reshape(bs, tlen, -1)
            ], dim=-1)
        
        freq, amp, _, phase = self.phase_enc.FFT(signal)
        phase = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
        return freq, amp, phase

                    

# In order to use this module in PyTorch >= 2.0, please modify ball_query function in PyTorch3D to override setup_context method
class ScenePenetrationLoss(nn.Module):
    def __init__(self, _: nn.Module, contact_idxs: List[int], k: int = 4, radius: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.contact_idxs = contact_idxs
        self.k = k
        self.radius = radius
        self.scale = scale

    def forward(self, poses, joints, data):
        scene = data['scene']
        bs, tlen, *_ = joints.shape
        positions = joints[:, :, self.contact_idxs]
        offset = data['offset'][:, [0]]
        positions = positions.reshape(bs, -1, 3)
        _, idx, nn = ball_query(positions, scene.points_padded() - offset, lengths2=scene.num_points_per_cloud(), K=tlen*self.k, radius=self.radius)
        
        dist = (self.radius - torch.norm(positions.unsqueeze(dim=-2) - nn, dim=-1))[idx > 0].sum()
        # avoid division by zero
        num_points = (idx > 0).sum()
        denum = 1.0 / num_points if num_points > 0.0 else 0.0
        return self.scale * dist * denum



def create_loss_from_conf(diffusion: nn.Module, conf: ListConfig) -> CombinedGeometricLoss:
    loss_funcs = []
    for loss_cfg in conf:
        cls = loss_cfg['cls']
        loss_func = globals()[cls](diffusion, **loss_cfg.args)
        loss_funcs.append(loss_func)
    return CombinedGeometricLoss(*loss_funcs)
