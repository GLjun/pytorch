# coding=utf-8
# pylint: skip-file
# pylint: disable-all

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    print("run in %d of %d" % (rank, size))
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ initialize the distributed environment """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    
