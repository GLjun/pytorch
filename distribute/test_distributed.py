# coding=utf-8
# pylint: skip-file
# pylint: disable-all

import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp

def run(rank, size):
    # if master(rank==0) process finished too early, e.g. before other process
    # call init_process_group, some exception will be
    # reported
    #if rank == 0:
    #    time.sleep(1) 
    #time.sleep(rank)
    print("run in %d of %d" % (rank, size))
    #pass

def init_process(rank, size, fn, backend='gloo'):
    """ initialize the distributed environment """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=size)
    # make sure master process will not end before other processes call the
    # init_process_group function
    dist.barrier()
    fn(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        #p = Process(target=run, args=(rank,size))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    #mp.spawn(init_process, args=(size,run), nprocs=size, join=True)
    
