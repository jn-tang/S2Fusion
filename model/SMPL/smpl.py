# Simply wraps SMPLH model
from typing import Tuple

import numpy as np
import torch
from torch import nn
import os.path
from typing import Optional

from model.SMPL.lbs import lbs, vertices2joints
from model.utils import singleton



@singleton
class SMPLH(nn.Module):
    """A simple SMPL-H wrapper.
    """

    # SMPL joint names
    joint_names = ['pelvis',                                # 0
                   'left leg root', 'right leg root',       # 1, 2
                   'lowerback',                             # 3
                   'left knee', 'right knee',               # 4, 5
                   'upperback',                             # 6
                   'left ankle', 'right ankle',             # 7, 8
                   'thorax',                                # 9
                   'left toes', 'right toes',               # 10, 11
                   'lowerneck',                             # 12
                   'left clavicle', 'right clavicle',       # 13, 14 
                   'upperneck',                             # 15
                   'left armroot', 'right armroot',         # 16, 17
                   'left elbow', 'right elbow',             # 18, 19
                   'left wrist', 'right wrist']             # 20, 21
                #    'left hand', 'right hand']

    joints = {k: v for v, k in enumerate(joint_names)}
    num_joints = len(joint_names)

    def __init__(self, smpl_path, gender='neutral', dmpl_path=None, dtype=torch.float32) -> None:
        super().__init__()

        self.dtype = dtype
        self.use_dmpl = bool(dmpl_path)

        # load SMPL parameters
        model_path = os.path.expanduser(os.path.join(smpl_path, gender, 'model.npz'))
        smpl_dict = np.load(model_path, encoding='latin1')
        self.register_buffer('v_template', torch.tensor(smpl_dict['v_template'][None], dtype=dtype))
        self.register_buffer('shapedirs', torch.tensor(smpl_dict['shapedirs'], dtype=dtype))
        self.register_buffer('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype))
        self.register_buffer('weights', torch.tensor(smpl_dict['weights'], dtype=dtype))
        self.register_buffer('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32))

        J_template = vertices2joints(self.J_regressor[:self.num_joints], self.v_template)
        self.register_buffer('J_template', J_template)

        posedirs = torch.tensor(smpl_dict['posedirs'], dtype=dtype)
        self.register_buffer('posedirs', posedirs.reshape((posedirs.shape[0] * 3, -1)).T)

        parents = smpl_dict['kintree_table'][0]
        self.register_buffer('parents_lbs', torch.tensor(parents, dtype=torch.int64))
        parents[0] = -1
        self.register_buffer('parents', torch.tensor(parents[:self.num_joints], dtype=torch.int32))

        if self.use_dmpl:
            dmpl_path = os.path.join(dmpl_path, gender, 'model.npz')
            dmpldirs = np.load(dmpl_path)['eigvec']
            self.register_buffer('dmpldirs', torch.tensor(dmpldirs, dtype=dtype))
            self.register_buffer('init_betas', torch.zeros((1, 24), dtype=dtype))
        else:
            self.register_buffer('init_betas', torch.zeros((1, 16), dtype=dtype))

        self.register_buffer('init_pose_hands', torch.zeros((1, 90), dtype=dtype))
        self.register_buffer('init_transl', torch.zeros((1, 3), dtype=dtype))

    def forward(self, 
                body_pose: torch.Tensor, 
                transl: Optional[torch.Tensor] = None, 
                betas: Optional[torch.Tensor] = None, 
                expand_beta: bool = True) -> Tuple[torch.Tensor]:
        batch_size = body_pose.shape[0]

        hand_pose = self.init_pose_hands.expand(batch_size, -1)
        full_pose = torch.cat([body_pose, hand_pose], dim=-1)

        v_template = self.v_template.expand(batch_size, -1, -1)
        if transl is None:
            transl = self.init_transl.expand(batch_size, -1)
        if expand_beta or betas is None:
            betas = self.init_betas.expand(batch_size, -1) if betas is None else betas.expand(batch_size, -1)
        shapedirs = self.shapedirs if not self.use_dmpl else torch.cat([self.shapedirs, self.dmpldirs], dim=-1)

        verts, Jtr = lbs(betas=betas, pose=full_pose, v_template=v_template,
                         shapedirs=shapedirs, posedirs=self.posedirs,
                         J_regressor=self.J_regressor, parents=self.parents_lbs,
                         lbs_weights=self.weights, dtype=self.dtype)

        Jtr = Jtr + transl.unsqueeze(dim=1)
        verts = verts + transl.unsqueeze(dim=1)

        return verts, Jtr

    def forward_kinematic(self, rot_mats: torch.Tensor) -> Tuple[torch.Tensor]:
        bs = rot_mats.shape[0]
        rot_global = [rot_mats[:, 0]]
        for i in range(1, len(self.parents)):
            rot_global.append(torch.bmm(rot_global[self.parents[i]], rot_mats[:, i]))
        rot_global = torch.stack(rot_global, dim=1)
        jpos = torch.vmap(torch.vmap(torch.mv))(rot_global, self.J_template.repeat(bs, 1, 1))
        return rot_global, jpos
