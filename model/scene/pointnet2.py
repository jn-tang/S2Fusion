from typing import Optional, Sequence

import torch
from model.scene.pointnet2_utils import PointNetSetAbstraction

    
class PointNet2Encoder(torch.nn.Module):
    def __init__(self,
                 npoints: Sequence[int] = [1024, 256, 64, 16],
                 in_channels: Sequence[int] = [9, 64, 128, 256],
                 nsample: int = 32,
                 ckpt: Optional[str] = None) -> None:
        super(PointNet2Encoder, self).__init__()
        
        np1, np2, np3, np4 = npoints
        c1, c2, c3, c4 = in_channels
        self.sa1 = PointNetSetAbstraction(np1, 0.1, nsample, c1 + 3, [nsample, nsample, nsample * 2], False)
        self.sa2 = PointNetSetAbstraction(np2, 0.2, nsample, c2 + 3, [nsample * 2, nsample * 2, nsample * 4], False)
        self.sa3 = PointNetSetAbstraction(np3, 0.4, nsample, c3 + 3, [nsample * 4, nsample * 4, nsample * 8], False)
        self.sa4 = PointNetSetAbstraction(np4, 0.8, nsample, c4 + 3, [nsample * 8, nsample * 8, nsample * 16], False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

        if not ckpt is None:
            self.load_state_dict(torch.load(ckpt), strict=False)
            

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        return self.mlp(l4_points.mean(dim=-1))
