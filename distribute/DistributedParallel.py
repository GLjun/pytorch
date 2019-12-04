#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8

import os
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
DEVICE_COUNT=torch.cuda.device_count()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    #initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    #setting the seed to make sure that models created in two processes start
    #from same random weights and biases
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    setup(rank, world_size)
    
    n = DEVICE_COUNT // world_size;
    print("using %d GPUs for %d world" % (n, world_size))
    # rank r using [r*n, (r+1)*n) GPUs
    device_ids = list(range(rank*n, (rank+1)*n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])

    # output_device defaults to device_ids[0]
    ddp_model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    n = DEVICE_COUNT // world_size;
    print("using %d GPUs for %d world" % (n, world_size))
    # rank r using [r*n, (r+1)*n) GPUs
    device_ids = list(range(rank*n, (rank+1)*n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])

    # output_device defaults to device_ids[0]
    ddp_model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    print("checkpoint path: ", CHECKPOINT_PATH)

    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # use a barrier() to make sure that other processes loads the model
    # after process 0 saves it
    dist.barrier()

    # configure map_location properly
    rank0_devices = [ x - rank*n for x in device_ids ]
    device_pairs = zip(rank0_devices, device_ids)

    map_location = { 'cuda:%d' % x : 'cuda:%d' % y for x, y in device_pairs }

    print("map location: ", map_location)

    ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location));

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()

    optimizer.step()



    

