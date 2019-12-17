# coding=utf-8
# pylint: skip-file
# pylint: disable-all

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from BaiduNet import BaiduNet9P

SEED=1234
ROOT_DIR="/home/gw/data"


def setup(rank, size, gpus, backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12346'
    print("using gpus: ", ",".join(gpus))
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)

def cleanup():
    dist.destroy_process_group()


def main()

    args
    setup(rank, size, gpus, backend)

    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank*n, (rank+1)*n))

    model = BaiduNet9P()
    ddp_model = DDP(model, device_ids=device_ids)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    
    train_set = datasets.CIFAR10(root=ROOT_DIR,
            train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
    test_set = datasets.CIFAR10(root=ROOT_DIR,
            train=False, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))

    train_sampler = \
        torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
            train_set, 


