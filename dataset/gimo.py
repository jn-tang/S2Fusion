from typing import Dict
import os
import glob
from itertools import islice
import torch
from pytorch3d.structures import Pointclouds

from dataset.base import MotionDataset



def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class GIMO(MotionDataset):
    # Hard-wired constants
    fps = 30
    depth_fps = 2

    def __init__(self, 
                 fpath: str, 
                 train: bool = True, 
                 duration: float = 4.0, 
                 load_scene: bool = True,
                 num_points: int = 8192,
                 align: bool = True) -> None:
        
        self.window_len = int(self.fps * duration)
        self.align = align
        self.train = train
        self.num_points = num_points
        self.load_scene = load_scene
        self.filemap = dict(
            enumerate(glob.glob(os.path.join(os.path.expanduser(fpath), 'train' if train else 'test', '*')))
        )
        scenes = glob.glob(os.path.join(os.path.expanduser(fpath), 'scene', '*.pt'))
        self.scenemap = dict(zip([scene.split('/')[-1][:-3] for scene in scenes], scenes))

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(os.path.join(self.filemap[idx], 'motion.pt'))
        motion, _ = self.get_motion(data)
        if self.load_scene:
            scene = data['scene']
            if isinstance(self.scenemap[scene], str):
                pcd = torch.load(self.scenemap[scene])
                self.scenemap[scene] = Pointclouds([pcd]).to('cuda')
            scene = self.scenemap[scene]
            offset = motion['offset']
            
            pcds = []
    
            for trans in batched(motion['trans'], self.window_len):
                pcd = []
                for t in trans[::self.fps // self.depth_fps]:
                    bbox = torch.cat([t + torch.tensor([-1.0, -1.0, -1.0]) + offset, t + torch.tensor([1.0, 1.0, 1.0]) + offset]).to('cuda')
                    idxs = scene.inside_box(bbox)
                    pcd.append(scene.points_packed()[idxs])
                pcd = torch.cat(pcd, dim=0).unique(dim=0)
                if pcd.shape[0] == 0:
                    pcd = torch.zeros(1, 3, device='cuda')
                pcds.append(pcd)
    
            pcds = Pointclouds(pcds).subsample(self.num_points)
    
            motion['scene'] = pcds
        
        if self.align:
            motion = self.align_motion(motion)
            
        return motion
    
