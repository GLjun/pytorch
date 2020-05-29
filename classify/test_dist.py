# coding=utf-8
# pylint: skip-file
# pylint: disable-all

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from vgg import VGG16

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    #initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    #setting the seed to make sure that models created in two processes start
    #from same random weights and biases
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)

    device_cnt = torch.cuda.device_count()
    #print("use %d gpus" % (device_cnt))
    n = device_cnt // world_size


    device_ids = list(range(rank*n, (rank+1)*n))

    model = VGG16().to(device_ids[0])

    ddp_model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=device_ids)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    optimizer.zero_grad()
    inputs = torch.randn(64*n, 3, 224, 224).to(device_ids[0])
    labels = torch.randn(64*n).to(device_ids[0])

    for epoch in range(0, 2):
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    cleanup()

def run_func(func, world_size):
    mp.spawn(func, args=(world_size,), nprocs=world_size, join=True)

if __name__== "__main__":
    run_func(main, torch.cuda.device_count())
    

