from typing import Union, Dict, List, Callable, Any, Optional

from omegaconf import OmegaConf, DictConfig
import argparse
import os
import math
from itertools import starmap, zip_longest
import collections
import datetime
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate
from pytorch3d.structures import Pointclouds
from torchmetrics import MetricCollection, Metric
from ema_pytorch import EMA

import model as mdl
from model import metric
import dataset


def ddp_setup(rank: int, world_size: int, port: str) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def create_model(cfg: DictConfig):
    model_cls = getattr(mdl, cfg.pop('cls'))
    model = model_cls(**cfg)
    return model

# Create a closure for saving sequences
def save_for_vis(pred, gt):
    pose_hat = pred['poses']
    trans_hat = pred['trans']
    pose_gt = gt['poses']
    trans_gt = gt['trans']
    mask = gt['mask']
    if not os.path.exists('./visualization'):
        os.makedirs('./visualization')
    for i in range(pose_hat.shape[0]):
        if (~mask[i]).sum() <= 0:
            continue
        save_for_vis.cnt += 1
        torch.save({
            'pose_hat': pose_hat[i][~mask[i]],
            'trans_hat': trans_hat[i][~mask[i]],
            'pose_gt': pose_gt[i][~mask[i]],
            'trans_gt': trans_gt[i][~mask[i]],
            'scene': Pointclouds([gt['scene'][i].points_list()[0] - gt['offset'][i, 0]])
        }, f'./visualization/sequence{save_for_vis.cnt}.pt')


save_for_vis.cnt = 0


def transfer(data: Any, dtype: torch.dtype = torch.float32, device: torch.device = 'cpu'):
    container_type = type(data)
    if isinstance(data, torch.Tensor):
        if torch.is_floating_point(data):
            return data.to(dtype=dtype, device=device)
        return data.to(device=device)
    elif isinstance(data, Pointclouds):
        # You have to modify pytorch3d/common/datatypes.py line 28 to make it work
        return data.to(device=device)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return container_type({key: transfer(data[key], dtype=dtype, device=device) for key in data})
        except TypeError:
            return {key: transfer(data[key], dtype=dtype, device=device) for key in data}
    elif isinstance(data, collections.abc.Sequence):
        try:
            return container_type([transfer(d, dtype=dtype, device=device) for d in data])
        except TypeError:
            return [transfer(d, dtype=dtype, device=device) for d in data]
    else:
        return data
    

# define collate function for point clouds
def collate_point_cloud_fn(batch, *, collate_fn_map):
    device = batch[0].device
    points = zip_longest(*[pcd.points_list() for pcd in batch], fillvalue=torch.zeros((0, 3), device=device))
    return list(starmap(lambda *args: Pointclouds(list(args)), points))

# update default collate registry
torch.utils.data._utils.collate.default_collate_fn_map.update({Pointclouds: collate_point_cloud_fn})


def collate_with_padding(window_length) -> Callable:
    """Collate samples of variance length by padding.

    Args:
        window_length (int): window length during inference time.
    """
    def pad_to_length(sample: Dict[str, torch.Tensor], pad: int) -> Dict[str, torch.Tensor]:
        scene = sample.pop('scene', None)
        data =  {k: torch.cat([v, torch.zeros((pad - v.shape[0],) + v.shape[1:], 
                                             device=v.device, dtype=v.dtype)
                                 if not v.dtype is torch.bool
                                 else torch.ones((pad - v.shape[0],) + v.shape[1:],
                                                  device=v.device, dtype=v.dtype)])
                                for k, v in sample.items()}
        if not scene is None:
            data.update({'scene': scene})
        return data

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        seqlens = [next(iter(sample.values())).shape[0] for sample in batch]
        padded_len = math.ceil(max(seqlens) / window_length) * window_length
        batch = [pad_to_length(sample, padded_len) for sample in batch]
        return default_collate(batch)

    return _collate


class Evaluator:
    def __init__(self,
                 model: torch.nn.Module,
                 testloader: DataLoader,
                 metrics: Metric,
                 seqlen: int,
                 device: Union[str, int]) -> None:
        model.eval()
        self.device = device
        self.metrics = metrics.to(device)
        self.model = model.to(device)
        self.testloader = testloader
        self.seqlen = seqlen

    def evaluate(self, vis: bool = False, *args, **kw) -> None:
        for data in self.testloader:
            data = transfer(data, device=self.device)
            # Split whole sequence into small chunks, but not for scene,
            # because we have already chunk the scene for each windows.
            # This workaround is ugly, but I have no other idea about this,
            # sorry for the misdesign if any confusion occurs.
            data = {k: v.split(self.seqlen, dim=1) if k != 'scene' else v for k, v in data.items()}
            chunks = [dict(zip(data, val)) for val in zip(*data.values())] 
            # The training data differs from testing data, so we need to 
            # split the testing data into small chunks to feed into model
            offset = 0
            for chunk in chunks:
                # align chunks 
                head_start = chunk['jpos'][:, 0, 15].unsqueeze(dim=1).clone()
                head_start[:, :, 1] = 0
                head_start = head_start.repeat(1, self.seqlen, 1)
                mask = ~chunk['mask']
                chunk['trans'][mask] -= head_start[mask]
                chunk['jpos'][mask] -= head_start.unsqueeze(dim=2)[mask]
                if not chunk.get('offset', None) is None:
                    offset = head_start + offset
                    chunk['offset'] = chunk['offset'] + offset
                
                output = self.model.validate(chunk, *args, **kw)
                self.metrics.update(output, chunk)
                if vis:
                    save_for_vis(output, chunk)
            try:
                # Clear all cached scene information
                self.model.clear()
            except Exception:
                pass
        print(self.metrics.compute())


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 trainloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 save_every: int = 100,
                 ckpt_path: str = '',
                 ema: DictConfig = None) -> None:
        model.train()
        self.gpu_id = gpu_id
        model = model.to(gpu_id)
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.save_every = save_every
        self.ckpt_path = os.path.expanduser(ckpt_path)
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.ema = EMA(model, **ema) if not ema is None else None

    def train_step(self, epoch: int, *args, **kw) -> None:
        self.trainloader.sampler.set_epoch(epoch)
        loss_acc = []
        for data in self.trainloader:
            data = transfer(data, device=self.gpu_id)
            self.optimizer.zero_grad()
            loss = self.model(data, *args, **kw)
            loss.backward()
            self.optimizer.step()
            if not self.ema is None:
                self.ema.update()
            loss_acc.append(loss.detach().cpu().item())
        print(f'[{datetime.datetime.now()}]\t[GPU{self.gpu_id}\tEpoch {epoch}]\tLoss: {sum(loss_acc) / len(loss_acc)}')

    def train(self, num_epochs: int, *args, **kw) -> None:
        for epoch in range(num_epochs):
            self.train_step(epoch, *args, **kw)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                if not os.path.exists(self.ckpt_path):
                    os.makedirs(self.ckpt_path)
                if not self.ema is None:
                    ema_save_path = os.path.join(self.ckpt_path, f'ema-{epoch}.pt')
                    torch.save(self.ema.state_dict(), ema_save_path)
                    print(f'Saving EMA at {ema_save_path}')
                save_path = os.path.join(self.ckpt_path, f'model-{epoch}.pt')
                torch.save(self.model.module.state_dict(), save_path)
                print(f'Saving model at {save_path}')


def train(rank: int, world_size: int, port: str, model_cfg: DictConfig, train_cfg: DictConfig) -> None:
    # intialize multiprocessing
    ddp_setup(rank, world_size, port)
    # create and load model
    model = create_model(model_cfg)
    ckpt_path = train_cfg.pop('ckpt_path')
    model.load(ckpt_path)  # continued training
    dist.barrier()
    # training related settings
    # optimizer
    opt_cfg = train_cfg.pop('optimizer')
    optim_cls = getattr(torch.optim, opt_cfg.pop('cls'))
    optim = optim_cls(model.parameters(), **opt_cfg)

    # dataset
    dataset_cfg = train_cfg.pop('dataset')
    dataset_cls = getattr(dataset, dataset_cfg.pop('cls'))
    batch_size = dataset_cfg.pop('batch_size')
    seqlen = dataset_cfg.pop('seqlen')
    dset = dataset_cls(**dataset_cfg)
    trainloader = DataLoader(
        dset, 
        batch_size, 
        shuffle=False,
        sampler=DistributedSampler(dset),
        collate_fn=collate_with_padding(seqlen))

    # others
    epochs = train_cfg.pop('epochs')
    save_every = train_cfg.pop('save_every')
    ema = train_cfg.pop('ema', None)
    trainer = Trainer(model, trainloader, optim, rank, save_every, ckpt_path, ema)
    trainer.train(epochs, **train_cfg)
    dist.destroy_process_group()


def test(model_cfg: DictConfig, test_cfg: DictConfig, vis: bool) -> None:
    # create and load model
    model = create_model(model_cfg)
    model.load(test_cfg.pop('ckpt_path'))
    # validation/test related settings
    # test dataset
    dataset_cfg = test_cfg.pop('dataset')
    dataset_cls = getattr(dataset, dataset_cfg.pop('cls'))
    batch_size = dataset_cfg.pop('batch_size')
    seqlen = dataset_cfg.pop('seqlen')
    dset = dataset_cls(**dataset_cfg)
    testloader = DataLoader(
        dset, 
        batch_size,
        pin_memory=False,
        shuffle=False,
        collate_fn=collate_with_padding(seqlen))

    # metrics
    metric_cfg = test_cfg.pop('metrics')
    metrics = [getattr(metric, cfg['cls'])(**cfg['args']) for cfg in metric_cfg]
    metric_ = MetricCollection(metrics)

    evaluator = Evaluator(model, testloader, metric_, seqlen, test_cfg.pop('device', 'cpu'))
    evaluator.evaluate(**test_cfg, vis=vis)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model configuration file'
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task configuration file, either train of test'
    )

    parser.add_argument(
        '--port',
        type=str,
        default='8964',
        help='Rendezvous port number of DDP models'
    )

    parser.add_argument(
        '--vis',
        action='store_true',
        help='Save temporary file for visualization'
    )
    
    args = parser.parse_args()
    model_cfg = OmegaConf.load(args.model)
    task_cfg = OmegaConf.load(args.task)
    stage = args.task.split('/')[-2]
    world_size = torch.cuda.device_count()
    if stage == 'train':
        mp.spawn(train, args=(world_size, args.port, model_cfg, task_cfg), nprocs=world_size)
    elif stage == 'test':
        test(model_cfg, task_cfg, args.vis)
    else:
        raise Exception('Not supported stage')

if __name__ == '__main__':
    main()
