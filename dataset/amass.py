from typing import Dict, List

import os
from glob import glob
from functools import reduce
from operator import add
import torch

from .base import MotionDataset


class AMASS(MotionDataset):
    def __init__(self, 
                 fpath: str, 
                 splits: List[str],
                 framerate: int = 30, 
                 duration: float = 4.0,
                 train: bool = True,
                 align: bool = True) -> None:
        
        self.window_len = int(framerate * duration)
        self.framerate = framerate
        fpath = os.path.expanduser(fpath)
        filenames = reduce(add, [glob(os.path.join(fpath, split, '*/*.pt')) for split in splits])
        self.filemap = {k: v for (k, v) in enumerate(filenames)}
        self.train = train
        self.align = align

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.filemap[idx])
        motion, _ = self.get_motion(data)
        if self.align:
            motion = self.align_motion(motion)
        return motion