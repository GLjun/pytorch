# coding=utf-8
# pylint: skip-file

# pylint: disable-all

import os
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp


import torchvision.datasets as datasets
import torchvision.transforms as transfoms
import torchvision.models as models

from random import Random
from torch.utils.tensorboard import SummaryWriter

import partition
import progress

#SYN_FRQ = 1
#GPUS = []
SEED = 1234
#
#WRITER = None

BATCH_CNT = 0
SYN_CNT = 0
best_acc = 0.0

parser = argparse.ArgumentParser(description='Dynamic')

parser.add_argument('--syn_frq', default=1, type=int, 
    help='synchronize frequency')

parser.add_argument('--gpus', default=None, type=str, 
        nargs='*', help='gpu list')

parser.add_argument('--nproc', default=1, type=int, 
        help='number of processes, default is set to len(gpus)')

parser.add_argument('--batch', default=256, type=int, 
        help='batch size, default is 256')

parser.add_argument('--data', type=str,
        help='mnist or cifar10')

def init_process(rank, size, gpus,  backend):
    print("gpus ", gpus)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12346'
    print(",".join(gpus))
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)

    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    print("** rank %d/%d finished initializing process" % (rank, size))

def cleanup():
    dist.destroy_process_group()

def synchronize_gradients(model, SYN_FRQ):
    global BATCH_CNT
    global SYN_CNT
    if BATCH_CNT%SYN_FRQ==0:
        size = dist.get_world_size()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= float(size)
        SYN_CNT += 1
        
    BATCH_CNT += 1

def test(model, criterion, epoch, test_loader, device):
    global best_acc
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
        print('rank ', dist.get_rank(), ' test loss ', loss_acc1[0].item(), 
                ' test acc1 ', loss_acc1[1].item())
        dist.reduce(tensor=loss_acc1,
                dst=0,
                op=dist.ReduceOp.SUM)
        loss_acc1.div_(dist.get_world_size()*1.0)
        return loss_acc1[0].item(), loss_acc1[1].item()



def paralllel_run(rank, size, data, global_batch, gpus, syn_frq, backend='gloo'):
    init_process(rank, size, gpus, backend)
    torch.manual_seed(SEED)

    device = torch.device('cuda:{}'.format(rank))
    
    train_loader, test_loader, batch = partition.paritition_dataset(
            data, global_batch, SEED)

    model = models.mobilenet_v2(num_classes=10).to(device)

    optimizer = optim.SGD(model.parameters(),
            lr = 0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    num_batches = math.ceil(len(train_loader.dataset) / float(batch))

    # rank 0 logs data
    if rank == 0:
        writer = SummaryWriter()

    c1 = 0.001
    c2 = 0.001
    w = 1.0
    pbest = {}
    pbest_loss = 100000.0

    rng = Random()

    for epoch in range(5):
        epoch_loss = 0.0
        loss_acc1 = torch.zeros(2).to(device)

        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            #synchronize_gradients(model, syn_frq)

            ################################
            for name, param in model.named_parameters():
                #grad = param.grad.data
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= float(size)
                if len(pbest) > 0:
                    r1 = rng.random()
                    r2 = rng.random()
                    param.grad.data += c1*r1*(pbest[name]-param.data)
            ################################

            optimizer.step()
            #for param in model.parameters():
            #    param.data += 0.001 * param.grad.data

            epoch_loss += loss.item()

            loss_acc1[0].add_(loss.item())
            loss_acc1[1] = progress.accuracy(outputs, labels)[0]

            ################################

            if loss.item() < pbest_loss:
                pbest_loss = loss.item()
                for name, param in model.named_parameters():
                    #print(name)
                    pbest[name] = torch.tensor(param.data)

            ###############################

            #if (i+1)%100 == 0:
            if True:
                dist.reduce(tensor=loss_acc1,
                        dst=0, op=dist.ReduceOp.SUM)
                loss_acc1.div_(size*1.0)
                if rank == 0:
                    writer.add_scalar('loss ', 
                            loss_acc1[0].item()/1,
                            epoch*len(train_loader) + i)
                    writer.add_scalar('acc1 ',
                            loss_acc1[1].item()/1,
                            epoch*len(train_loader) + i)

                loss_acc1.zero_()
        print('Rank ', rank, ', epoch ', epoch, ': ',
                epoch_loss/num_batches, 
                ' SYN_CNT ', SYN_CNT,
                ' BATCH_CNT ', BATCH_CNT)
        test_loss, test_acc1 = test(model, criterion, epoch, test_loader, device)
        if rank==0:
            print('after reduce rank ', rank,' test_loss ', test_loss, ' test acc1 ', test_acc1)
            writer.add_scalar('test loss ', 
                    test_loss,
                    epoch)
            writer.add_scalar('test acc1 ', 
                    test_acc1,
                    epoch)
            
    cleanup()


def main():

    args = parser.parse_args()
    
    syn_frq = args.syn_frq;
    print("** synchronize frequency: ", syn_frq)

    world_size = args.nproc
    print('** processes number: ', world_size)

    if args.gpus is not None:
        gpus = args.gpus
        world_size = len(gpus)
        print("** using gpus: ", gpus, 
            " and ", world_size, "processses")

    mp.spawn(paralllel_run, 
            args=(world_size, args.data, args.batch, gpus, syn_frq, 'nccl'),
            nprocs=world_size, join=True)

if __name__=='__main__':
    main()
