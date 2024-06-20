from typing import Dict, Tuple
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

from model.utils import axis_angle_to_matrix, matrix_to_rotation_6d, axis_angle_to_rotation_6d


class MotionDataset(Dataset):
    @abstractmethod
    def __init__(self):
        # All children instances has to define `window_len` and `filemap` fields
        pass

    def __len__(self) -> int:
        return len(self.filemap)
    
    def align_motion(self, data):
        head_start = data['jpos'][0, 15].unsqueeze(dim=0).clone()
        head_start[:, 1] = 0
        data['trans'][~data['mask']] -= head_start
        data['jpos'][~data['mask']] -= head_start.unsqueeze(dim=1)
        if 'offset' in data:
            data['offset'] += head_start
            data['offset'] = data['offset'].repeat(data['jpos'].shape[0], 1)
        return data
    
    def get_motion(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
        tlen = data['poses'].shape[0]

        if self.train and tlen > self.window_len:
            sample_start = np.random.randint(0, tlen - self.window_len + 1)
            time_slice = slice(sample_start, sample_start + self.window_len)
        else:
            sample_start = 0
            time_slice = slice(None)

        poses = data['poses'][time_slice].reshape(-1, 22, 3)
        trans = data['trans'][time_slice]
        jpos = data['jpos'][time_slice]
        acc = data['acc'][time_slice]
        mask = data['mask'][time_slice]

        rot_mat = axis_angle_to_matrix(poses)
        angular_vel = torch.cat([rot_mat[0].unsqueeze(dim=0), torch.inverse(rot_mat[:-1]) @ rot_mat[1:]], dim=0)
        linear_vel = torch.cat([jpos[0].unsqueeze(dim=0), jpos[1:] - jpos[:-1]], dim=0)

        motion = {
            'poses': axis_angle_to_rotation_6d(poses),
            'trans': trans,
            'jpos': jpos,
            'angular': matrix_to_rotation_6d(angular_vel),
            'linear': linear_vel,
            'acc': acc[:, [2, 0, 1]],
            'mask': mask,
        }

        if 'offset' in data:
            motion['offset'] = data['offset'].unsqueeze(dim=0)

        return motion, sample_start
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass