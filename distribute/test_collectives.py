# coding=utf-8
# pylint: skip-file

# pylint: disable-all

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

def run(rank, size):
    pass

def p2p_test_block(rank, size):
    tensor = torch.tensor(rank, dtype=torch.int32)
    if rank%2 == 0: 
        dst = rank+1
        dist.send(tensor=tensor, dst=dst)
    else:
        src = rank-1
        dist.recv(tensor=tensor, src=src)

    print('Rank %d value %d' % (rank, tensor.item()))


def p2p_test_non_block(rank, size):
    tensor = torch.tensor(rank, dtype=torch.int32)
    req = None
    if rank%2 == 0: 
        dst = rank+1
        req = dist.isend(tensor=tensor, dst=dst)
    else:
        src = rank-1
        req = dist.irecv(tensor=tensor, src=src)

    req.wait()

    print('Rank %d value %d' % (rank, tensor.item()))

def all_reduce_test(rank, size):
    #group = dist
    tensor = torch.ones(1)
    group = dist.new_group([0,1,4,5])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print("rank ", rank, " value ", tensor[0])

def gather_test(rank, size):
    dst = 0
    tensor = torch.ones(1)
    tensor_list = [ torch.zeros(1) for i in range(size) ]
    gather_list = tensor_list if rank==dst else None
    dist.gather(tensor, gather_list, dst)
    #if rank==dst :
    #    dist.gather(tensor, dst=dst, gather_list=tensor_list)
    #else:
    #    dist.gather(tensor, dst=dst)
    print("rank ", rank, " tensorlist ", tensor_list)


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    fn(rank, size)

if __name__ == "__main__":
    size = 8
    processes = []
    func = gather_test
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
