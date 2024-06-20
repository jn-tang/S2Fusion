from typing import Dict
import os
import glob
from itertools import islice
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.io import load_ply
from pytorch3d.transforms import Rotate

from dataset.base import MotionDataset

def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class CIRCLE(MotionDataset):
    def __init__(self,
                 fpath: str,
                 train: bool = True,
                 fps: int = 30,
                 depth_fps: int = 1,
                 duration: float = 4.0,
                 load_scene: bool = True,
                 num_points: int = 8192,
                 align: bool = True) -> None:
        
        self.window_len = int(fps * duration)
        self.align = align
        self.train = train
        self.num_points = num_points
        self.depth_ratio = fps // depth_fps
        self.load_scene = load_scene

        if load_scene:
            verts, _ = load_ply(os.path.expanduser(os.path.join(fpath, 'circle.ply')))
            rotate = Rotate(torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T)
            self.scene = Pointclouds([rotate.transform_points(verts).to('cuda')])

        self.filemap = dict(
            enumerate(glob.glob(os.path.join(os.path.expanduser(fpath), 'train' if train else 'test', '*')))
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(os.path.join(self.filemap[idx], 'motion.pt'))
        offset = data['offset']
        motion, _ = self.get_motion(data)

        if self.load_scene:
            scene = []
    
            for trans in batched(motion['trans'], self.window_len):
                pcd = []
                for t in trans[::self.depth_ratio]:
                    bbox = torch.stack([t + torch.tensor([-1.0, -1.0, -1.0]) + offset, t + torch.tensor([1.0, 1.0, 1.0]) + offset]).to('cuda')
                    idxs = self.scene.inside_box(bbox)
                    pcd.append(self.scene.points_packed()[idxs])
                pcd = torch.cat(pcd, dim=0).unique(dim=0)
                if pcd.shape[0] == 0:
                    pcd = torch.zeros(1, 3, device='cuda')
                scene.append(pcd)

            scene = Pointclouds(scene).subsample(self.num_points)
            motion['scene'] = scene
        
        if self.align:
            motion = self.align_motion(motion)

        return motion
    
