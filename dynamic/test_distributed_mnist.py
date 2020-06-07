# coding=utf-8
# pylint: skip-file

# pylint: disable-all

import os
import argparse
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from progress import accuracy
from random import Random
from torch.multiprocessing import Process
from test import Net

from torch.utils.tensorboard import SummaryWriter
writer = None #SummaryWriter('runs/')


NUM_AVE = 1

parser = argparse.ArgumentParser(description='Pytorch Distribute \
Trainning')

parser.add_argument('--num_ave', default=1, type=int, help="average \
number")



class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=(0.7, 0.2, 0.1), seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [ x for x in range(0, data_len) ]
        rng.shuffle(indexes)

        parts = [ int(data_len*x) for x in sizes[0:-1] ]
        # set the last part to rest data
        remain = data_len
        for x in parts:
            data_len -= x
        parts.append(remain)

        accum = 0
        for x in parts:
            self.partitions.append(indexes[accum:accum+x])
            accum += x

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def paritition_dataset():
    dataset = datasets.MNIST('/home/gw/data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    testset = datasets.MNIST('/home/gw/data', 
            train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    size = dist.get_world_size()
    batch = 128/float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())

    print("rank %d partitionsize %d batchsize %d" %
            (dist.get_rank(),len(partition), batch))
    train_set = torch.utils.data.DataLoader(
            partition,
            batch_size = int(batch),
            shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size = int(batch),
            shuffle=False)

    return train_set, test_loader, batch


def average_gradients(model, i):
    if i%NUM_AVE==0:
        size = dist.get_world_size()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= float(size)

def average_parameter(model):
    size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= float(size)

def test(model, criterion, epoch, test_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    loss_acc1 = torch.zeros(2).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        loss_acc1[0] = test_loss / len(test_loader)
        loss_acc1[1] = 100.0*correct / total
        #print('rank ', dist.get_rank(), ' test loss ', loss_acc1[0].item(), 
        #        ' test acc1 ', loss_acc1[1].item())
        dist.reduce(tensor=loss_acc1,
                dst=0,
                op=dist.ReduceOp.SUM)
        loss_acc1.div_(dist.get_world_size()*1.0)
        return loss_acc1[0].item(), loss_acc1[1].item()

def run(rank, size):
    torch.manual_seed(1234)
    train_set, test_set, batch = paritition_dataset()
    device = torch.device("cuda:{}".format(rank))
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), 
            lr = 0.01, momentum=0.5)

    num_batches = math.ceil(len(train_set.dataset)/float(batch))

    criterion = nn.NLLLoss()

    if rank==0:
        writer = SummaryWriter()

    for epoch in range(10):
        epoch_loss = 0.0
        running_loss = torch.zeros(1).to(device)
        #running_loss = 0.0
        # clear the grad at start
        for i, data in enumerate(train_set, 0):
            inputs, labels = data
        #for data, target in train_set:
        #    data = data.to(device)
        #    target = target.to(device)
        #    optimizer.zero_grad()
        #    output = model(data)
        #    loss = F.nll_loss(output, target)
            #print("data to cuda")
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print("data to cuda done")
            #print("zero grad")
            optimizer.zero_grad()
            #print("calculate output")
            output = model(inputs)
            #print("calculate loss")
            loss = criterion(output, labels)

            #print("calculate backward")
            loss.backward()
            #print("calculate average")
            average_parameter(model)
            #average_gradients(model, i)
            #print("calculate step")
            optimizer.step()

            epoch_loss += loss.item()
            running_loss.add_(loss.item())

            if (i+1)%100 == 0:
                test_loss, test_acc1 = test(model, criterion, epoch, test_set,
                        device)
                #print('write ', rank)
                #dist.reduce(tensor=running_loss, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    #writer.add_scalar('loss num ave {}'.format(NUM_AVE), 
                    writer.add_scalar('test loss ',
                            test_loss,
                            epoch*len(train_set)+i)
                    writer.add_scalar('test acc1 ',
                            test_acc1,
                            epoch*len(train_set)+i)
                #running_loss.zero_()
            
            #acc1, acc5 = accuracy(output, target, topk=(1,5))
            #print("rank %d acc1 %f acc2 %f", dist.get_rank(), 
            #        acc1, acc5)
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ',
                    epoch_loss/num_batches)

def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12346'

    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    fn(rank, size)

if __name__ == "__main__":
    size = 4
    processes = []
    func = run
    args = parser.parse_args()
    print("AVE_NUM ", NUM_AVE)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, func, "nccl"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
