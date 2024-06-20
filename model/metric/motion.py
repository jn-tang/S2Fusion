from typing import Optional, List
import torchmetrics
import torch

from model.utils import (
    rotation_6d_to_matrix,
    rotation_6d_to_axis_angle
)
from model.SMPL import SMPLH


class MPJPE(torchmetrics.Metric):
    def __init__(self, select_idx: Optional[List[int]] = None, in_meters: bool = True, align_idx: Optional[int] = None):
        super().__init__()
        self.select_idx = select_idx
        self.in_meters = in_meters
        self.align_idx = align_idx
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('mpjpe', default=torch.tensor([0.0]), dist_reduce_fx='sum')

    def compute(self):
        scale = 1000.0 if self.in_meters else 1.0
        return self.mpjpe / self.count * scale

    def update(self, source, target):
        jpos_hat = source['jpos']
        jpos_gt = target['jpos']
        mask = target['mask']
        bs, tlen, *_ = jpos_hat.shape
        jpos_hat = jpos_hat.view(bs, tlen, -1, 3)
        jpos_gt = jpos_gt.view(bs, tlen, -1, 3)

        self.count += 1
       
        if not self.align_idx is None:
            start_gt = jpos_gt[:, :, self.align_idx].unsqueeze(dim=2).clone()
            start_gt[:, :, :, 1] = 0
            start_hat = jpos_hat[:, :, self.align_idx].unsqueeze(dim=2).clone()
            start_hat[:, :, :, 1] = 0
            jpos_gt -= start_gt
            jpos_hat -= start_hat 

        if not self.select_idx is None:
            jpos_hat = jpos_hat[:, :, self.select_idx]
            jpos_gt = jpos_gt[:, :, self.select_idx]
 
        
        self.mpjpe += torch.norm(jpos_hat[~mask] - jpos_gt[~mask], dim=-1).mean()



class ForwardKinematics(torchmetrics.Metric):
    def __init__(self, smpl_path: str, in_meters: bool = True, with_trans: bool = False):
        super().__init__()
        self.in_meters = in_meters
        self.with_trans = with_trans
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fk', default=torch.tensor([0.0]), dist_reduce_fx='sum')
        self.smpl = SMPLH(smpl_path)

    def compute(self):
        scale = 1000.0 if self.in_meters else 1.0
        return self.fk / self.count * scale
    
    def update(self, source, target):
        poses_hat = source['poses']
        aa_gt = target['poses']
        bs, tlen, *_ = poses_hat.shape
        
        self.count += 1
        aa_hat = rotation_6d_to_axis_angle(poses_hat.reshape(bs, tlen, -1, 6)).reshape(bs * tlen, -1)
        jpos_hat = self.smpl(aa_hat)[1].reshape(bs, tlen, -1, 3)[:, :, :22]
        jpos_gt = self.smpl(aa_gt.view(bs * tlen, -1))[1].reshape(bs, tlen, -1, 3)[:, :, :22]
        if self.with_trans:
            jpos_hat += source['trans'].unsqueeze(dim=2)
            jpos_gt += target['trans'].unsqueeze(dim=2)
        
        self.fk += torch.norm(jpos_hat - jpos_gt, dim=-1).mean()



class Accel(torchmetrics.Metric):
    def __init__(self, select_idx: Optional[List[int]] = None, in_meters: bool = True):
        super().__init__()
        self.select_idx = select_idx
        self.in_meters = in_meters
        self.add_state('count', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('accel', torch.tensor([0.0]), dist_reduce_fx='sum')

    def compute(self):
        scale = 1000.0 if self.in_meters else 1.0
        return self.accel / self.count * scale

    def update(self, source, target):
        pos_hat = source['jpos']
        pos_gt = target['jpos']

        bs, tlen, *_ = pos_hat.shape
        pos_hat = pos_hat.view(bs, tlen, -1, 3)
        pos_gt = pos_gt.view(bs, tlen, -1, 3)

        self.count += 1
        
        if not self.select_idx is None:
            pos_hat = pos_hat[:, :, self.select_idx]
            pos_gt = pos_gt[:, :, self.select_idx]
        
        acc_hat = pos_hat[:, 2:] - 2 * pos_hat[:, 1:-1] + pos_hat[:, :-2]
        acc_gt = pos_gt[:, 2:] - 2 * pos_gt[:, 1:-1] + pos_gt[:, :-2]

        self.accel += torch.norm(acc_hat - acc_gt, dim=-1).mean()



class FootSkating(torchmetrics.Metric):
    foot_idx = [7, 8, 10, 11]  # indices corresponding to left/right ankles and left/right toes
    heights = [0.08, 0.08, 0.04, 0.04]     # contact threshold of ankles and toes
    def __init__(self, in_meters: bool = True):
        super().__init__()
        self.add_state('count', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fs', torch.tensor([0.0]), dist_reduce_fx='sum')
        self.in_meters = in_meters

    def compute(self):
        scale = 1000.0 if self.in_meters else 1.0
        return self.fs / self.count * scale

    def update(self, source, *args, **kwargs):
        jpos = source['jpos']
        bs, tlen, *_ = jpos.shape
        jpos = jpos.view(bs, tlen, -1, 3)
        self.count += 1

        stats = 0
        for idx, h in zip(self.foot_idx, self.heights):
            disp = (jpos[:, 1:, idx, [0, 2]] - jpos[:, :-1, idx, [0, 2]]).norm(dim=-1)
            mask = jpos[:, :-1, idx, 1] < h
            stat = torch.abs(disp * (2 - 2 ** (jpos[:, :-1, idx, 1] / h)))[mask].sum() / (bs * tlen)
            stats += stat
        self.fs += stats / 4.0


class RotationAngle(torchmetrics.Metric):
    def __init__(self, select_idx: Optional[List[int]] = None, in_radius: bool = True):
        super().__init__()
        self.select_idx = select_idx
        self.in_radius = in_radius
        self.add_state('count', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('angle', torch.tensor([0.0]), dist_reduce_fx='sum')

    def compute(self):
        scale = 1.0 if self.in_radius else 180.0
        return self.angle / self.count * scale
    
    def update(self, source, data):
        pred = source['poses']
        bs, tlen, *_ = pred.shape
        gt = data['poses']
        self.count += 1

        rot_pred = rotation_6d_to_matrix(pred.view(bs, tlen, -1, 6))
        rot_gt = rotation_6d_to_matrix(gt.view(bs, tlen, -1, 6))
        
        if not self.select_idx is None:
            rot_pred = rot_pred[:, :, self.select_idx]
            rot_gt = rot_gt[:, :, self.select_idx]

        diff = rot_pred @ torch.linalg.inv(rot_gt)
        trace = torch.vmap(torch.vmap(torch.vmap(torch.trace)))
        angle = ((trace(diff) - 1) / 2).arccos()  # angle = arccos (0.5 * (tr(R) - 1)), where R is the difference of rotation matrices
        self.angle += angle.mean()


class Translation(torchmetrics.Metric):
    def __init__(self, in_meters: bool = True):
        super().__init__()
        self.in_meters = in_meters
        self.add_state('count', torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('global_pos', torch.tensor([0.0]), dist_reduce_fx='sum')

    def compute(self):
        scale = 1000.0 if self.in_meters else 1.0
        return self.global_pos / self.count * scale
    
    def update(self, source, target):
        pred = source['trans']
        gt = target['trans']
        mask = target['mask']

        self.count += 1

        self.global_pos += torch.norm(pred[~mask] - gt[~mask], p=2, dim=-1).mean()
