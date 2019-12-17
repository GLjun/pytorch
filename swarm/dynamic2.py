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

from BaiduNet import BaiduNet9P

import partition
import progress

SEED = 1234

best_acc = 0.0
c1 = 1.5
c2 = 1.5
w = 1.0
pbest = {}
pbest_loss = 100000.0
rng = None
record_frq = 50
batch_cnt = 0
syn_frq = 1
syn_begin = 0
batch_syn_cnt = 0


parser = argparse.ArgumentParser(description='Swarm')

parser.add_argument('--gpus', default=None, type=str, 
        nargs='*', help='gpu list')

parser.add_argument('--lr', default=1.01, type=float, help='learning rate')

parser.add_argument('--nproc', default=1, type=int, 
        help='number of processes, default is set to len(gpus)')

parser.add_argument('--batch', default=256, type=int, 
        help='batch size, default is 256')

parser.add_argument('--syn_frq', default=1, type=int, 
        help='synchronize gradients frequency')

parser.add_argument('--syn_begin', default=0, type=int, 
        help='synchronize gradients start batch')

parser.add_argument('--data', type=str,
        help='mnist or cifar10')


def init_process(rank, size, gpus,  backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '22346'
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)

    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    print("** rank %d/%d finished initializing process" % (rank, size))

def cleanup():
    dist.destroy_process_group()

def synchronize_gradients(model, p, frq=1, begin=0):
    global batch_cnt
    global batch_syn_cnt
    #rank = dist.get_rank()
    if batch_cnt >= begin:
        if batch_cnt % frq == 0:
            size = dist.get_world_size()
            for param in model.parameters():
                #param.grad.data *= p[rank].item()
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= float(size)
            batch_syn_cnt +=1 
    else:
        size = dist.get_world_size()
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= float(size)

    batch_cnt += 1

def update_weights(model, lr):
    for param in model.parameters():
        param.data -= lr*param.grad.data

def swarm_synchronize_gradients(model):
    global rng
    global pbest
    size = dist.get_world_size()
    for name, param in model.named_parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= float(size)
        if len(pbest) > 0:
            r1 = rng.random()
            r2 = rng.random()
            #w_norm = param.data.norm()
            #g_norm = param.grad.data.norm()
            #g_pro = g_norm.item()/w_norm.item()
            #param.grad.data -= c1*r1*g_pro*(pbest[name]-param.data)
            #param.grad.data.div_(2.0)
            #param.grad.data += c1*r1*(param.data-pbest[name])
            param.grad.data.mul_(c2*r2)
            param.grad.data += c1*r1*(param.data - pbest[name])

def std_swarm_synchronize_gradients(model, p):
    global rng
    global pbest
    rank = dist.get_rank()
    _, pred = p.max(0)
    size = dist.get_world_size()
    for name, param in model.named_parameters():
        #dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        #param.grad.data /= float(size)
        if len(pbest) > 0:
            r1 = rng.random()
            r2 = rng.random()
            #w_norm = param.data.norm()
            #g_norm = param.grad.data.norm()
            #g_pro = g_norm.item()/w_norm.item()
            #param.grad.data -= c1*r1*g_pro*(pbest[name]-param.data)
            #param.grad.data.div_(2.0)
            #param.grad.data += c1*r1*(param.data-pbest[name])
            dist.broadcast(param.grad.data, int(pred.item()))
            param.grad.data.mul_(c2*r2)
            param.grad.data += c1*r1*(param.data - pbest[name])
        else:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= float(size)


def record_pbest(model, loss):
    global pbest_loss
    global pbest
    if loss < pbest_loss:
        pbest_loss = loss
        for name, param in model.named_parameters():
            pbest[name] = torch.tensor(param.data)

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



def paralllel_run(rank, size, data, global_batch, syn_frequency,
        syn_begin_batch, lr, gpus, backend='gloo'):
    global rng
    global syn_frq
    global batch_cnt
    global syn_begin
    global batch_syn_cnt
    syn_frq = syn_frequency
    syn_begin = syn_begin_batch

    print("synchronize frequency: ", syn_frq, " syn_begin ", syn_begin)

    init_process(rank, size, gpus, backend)
    torch.manual_seed(SEED)

    device = torch.device('cuda:{}'.format(rank))
    
    train_loader, test_loader, batch = partition.paritition_dataset(
            data, global_batch, SEED)

    #model = models.mobilenet_v2(num_classes=10).to(device)
    model = BaiduNet9P().to(device)

    optimizer = optim.SGD(model.parameters(),lr=lr)
            #momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    num_batches = math.ceil(len(train_loader.dataset) / float(batch))

    if rank == 0:
        writer = SummaryWriter()

    if rng is None:
        rng = Random()

    for epoch in range(100):
        epoch_loss = 0.0
        loss_acc1 = torch.zeros(2).to(device)
        loss_accum = torch.zeros(size).to(device)
        loss_accum_list = [ torch.zeros(1).to(device) for x in range(size)]

        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            #dist.all_gather(loss_accum_list, loss)
            #for k in range(size):
            #    loss_accum[k] = loss_accum_list[k].item()

            #p = torch.nn.functional.softmax(loss_accum, 0)
            synchronize_gradients(model, None, syn_frq, syn_begin)

            #swarm_synchronize_gradients(model)
            #std_swarm_synchronize_gradients(model, p)


            optimizer.step()
            #update_weights(model, lr)

            epoch_loss += loss.item()

            loss_acc1[0].add_(loss.item())
            loss_acc1[1] = progress.accuracy(outputs, labels)[0]

            record_pbest(model, loss.item())

            if (i+1)%record_frq == 0:
                dist.reduce(tensor=loss_acc1,
                        dst=0, op=dist.ReduceOp.SUM)
                loss_acc1.div_(size * 1.0)
                if rank == 0:
                    writer.add_scalar('loss ', 
                            loss_acc1[0].item()/record_frq,
                            epoch*len(train_loader) + i)
                    writer.add_scalar('acc1 ',
                            loss_acc1[1].item()/record_frq,
                            epoch*len(train_loader) + i)

                loss_acc1.zero_()
        print('Rank %d epoch %d loss %f best_loss %f batch_cnt %d syn_frq %d \
                batch_syn_cnt %d' % \
                (rank, epoch, epoch_loss/num_batches, pbest_loss, batch_cnt,
                    syn_frq, batch_syn_cnt))
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

    world_size = args.nproc
    print('** processes number: ', world_size)
    print('** learning rate: ', args.lr)

    if args.gpus is not None:
        gpus = args.gpus
        world_size = len(gpus)
        print("** using gpus: ", gpus, 
            " and ", world_size, "processses")

    mp.spawn(paralllel_run, 
            args=(world_size, args.data, args.batch, args.syn_frq,
                args.syn_begin, args.lr, gpus, 'nccl'),
            nprocs=world_size, join=True)

if __name__=='__main__':
    main()
