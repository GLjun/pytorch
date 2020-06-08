# coding=utf-8

import argparse
import os
import time
import warnings
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from vgg_opt_outplace import VGG16OPO
from resnet import ResNet50
from sgd_opt_outplace_merge import SGDOPO

parser = argparse.ArgumentParser(description='Weight Average SGD')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--no-details', action="store_true",
                    help='do not print details. ')
parser.add_argument('--ave-weight', action="store_true",
                    help='average weight instead average gradient. ')
parser.add_argument('--gpus', default=None, type=str, 
                    nargs='*', help='gpu list')
parser.add_argument('--classes', default=1000, type=int,
                    help='num of classes.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:12345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--adjlr', dest='adjlr', action='store_true',
                    help='adjusted learning rate')
parser.add_argument('--logdir', default=None, type=str,
                    help='log dir')

train_dir = "/home/gw/data/imagenet_10/train"
val_dir = "/home/gw/data/imagenet_10/val"

best_acc1 = 0.0



def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(args.gpus)
        print("using gpus : ", args.gpus)

    gpu_count = torch.cuda.device_count()
    args.world_size  = gpu_count * 1 # one process per gpu
    print("==>world size %d , total %d gpus" % (args.world_size, args.world_size))

    num_classes = args.classes
    print("==>classes '{}'".format(num_classes))

    mp.spawn(wa_sgd_run, nprocs=args.world_size, args=(args,), join=True)


def wa_sgd_run(rank, args):
    global best_acc1

    if rank == 0:
        print("async wa_sgd ", "average weight" if args.ave_weight else "average \
        gradients")
        if args.adjlr:
            writer = SummaryWriter(log_dir=args.logdir, comment='async_fwd_inplace_wa_sgd_ave_w_vgg16_adjlr{:.3f}_m{:.2f}'.format(args.lr, args.momentum) if args.ave_weight else
                'async_fwd_inplace_wa_sgd_ave_g_vgg16_adjlr{:.3f}_m{:.2f}'.format(args.lr, args.momentum))
        else:
            writer = SummaryWriter(log_dir=args.logdir, comment='async_fwd_inplace_wa_sgd_ave_w_vgg16_lr{:.3f}_m{:.2f}'.format(args.lr, args.momentum) if args.ave_weight else
                'async_fwd_inplace_wa_sgd_ave_g_vgg16_lr{:.3f}_m{:.2f}'.format(args.lr, args.momentum))

    #init process group
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(args.gpus)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=rank)

    args.batch_size = int(args.batch_size/args.world_size)
    
    device = torch.device("cuda:{}".format(rank))
    print("batchsize ", args.batch_size, " rank ", rank, " device ", device)

    model = VGG16OPO(num_classes=args.classes).to(device)

    #model = ResNet50(num_classes=args.classes).to(device)
    #model = models.resnet50(num_classes=args.classes).to(device)
    #model = models.vgg16_bn(num_classes=args.classes).to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)

    #model = VGG16(num_classes=args.classes).cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = optim.SGD(model.parameters(), args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_alpha = SGDOPO(model, args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, 
            alpha=1.0/dist.get_world_size())
    #optimizer = optim.SGD(model.parameters(), args.lr)
    #optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum)

    #open cudnn benchmark
    #cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    acc_red = torch.zeros(1).to(device)
    acum_time = 0.0
    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.adjlr:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        batch_time = train(device, train_loader, model, criterion, optimizer, 
                optimizer_alpha, epoch, args)
        acum_time += batch_time

        # evaluate on validation set
        acc1 = validate(device, val_loader, model, criterion, args)

        # average acc1
        acc_red[0] = acc1
        dist.reduce(tensor=acc_red, dst=0, op=dist.ReduceOp.SUM)
        acc_red.div_(args.world_size*1.0)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        #print("final best acc of epoch %d : %f" % (epoch, best_acc1))
        if rank == 0:
            print("==> acc1 ", acc_red[0].item())
            writer.add_scalar('test acc1 ', acc_red[0].item(), epoch)
            writer.add_scalar('acc1 over time(0.1s)', acc_red[0].item(),
                    int(acum_time*10))

def average_gradients(model, size_f):
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size_f

def average_weights(model, size_f):
    for param in model.parameters():
        if param.data is not None:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size_f
    
def train(device, train_loader, model, criterion, optimizer, optimizer_alpha, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    torch.cuda.set_device(device)
    #print("world size ", args.world_size, " device ", device)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #images = images.cuda(device)
        #target = target.cuda(device)
        #print("rank ", dist.get_rank(), " device ", device)
        images = images.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)
        #print("get data finished")

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        #optimizer_alpha.zero_grad()
        loss.backward()
        #if optimizer_alpha.cnt < 0:
        #    #if dist.get_rank() == 0:
        #        #print("%d average gradients" % (optimizer_alpha.cnt))
        #    average_gradients(model, float(args.world_size))
        #optimizer_alpha.step()
            
            

        # compute gradient and do SGD step
        
        #optimizer.zero_grad()
        #if args.ave_weight and epoch > 0:
        #    average_weights(model, float(args.world_size))
        #loss.backward()
        #if not args.ave_weight or (args.ave_weight and epoch == 0):
        #    average_gradients(model, float(args.world_size))
        #
        #optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_details and i % args.print_freq == 0:
            progress.display(i)
        #dist.barrier()
    return batch_time.sum


def validate(device, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    print("begin validate device ", device)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_details and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
