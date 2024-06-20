import torchmetrics
import torch
import torch.nn.functional as F


class SignalReconstruction(torchmetrics.Metric):
    def __init__(self, _):
        super().__init__()
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('mse', default=torch.tensor([0.0]), dist_reduce_fx='sum')

    def compute(self):
        return self.mse / self.count
    
    def update(self, source, target):
        head_pos = F.pad(target['jpos'][:, :, 15], (0, 0, 1, 0))
        head_vel = head_pos[:, 1:] - head_pos[:, :-1]
        bs, tlen, *_ = head_vel.shape
        x = torch.cat([target['acc'].reshape(bs, tlen, -1), head_vel.reshape(bs, tlen, -1)], dim=-1)

        self.count += 1
        self.mse += F.mse_loss(source, x)