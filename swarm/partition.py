# coding=utf-8
# pylint: skip-file
# pylint: disable-all

import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from random import Random


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

def paritition_dataset(data, global_batch, seed):
    dataset = None
    testset = None
    
    if data == "mnist":
        dataset = datasets.MNIST('/home/gw/programming/pytorch/data', 
                train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        testset = datasets.MNIST('/home/gw/programming/pytorch/data', 
                train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    elif data == "cifar10":
        dataset = datasets.CIFAR10(root='/home/gw/programming/pytorch/data',
                train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))
        testset = datasets.CIFAR10(root='/home/gw/programming/pytorch/data',
                train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))
    else:
        print("no such data set: ", data)
        exit(1)

    size = dist.get_world_size()
    batch = global_batch/float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())

    print("rank %d partitionsize %d batchsize %d" %
            (dist.get_rank(),len(partition), batch))
    train_loader = torch.utils.data.DataLoader(
            partition,
            batch_size = int(batch),
            shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size = int(batch),
            shuffle=False)

    return train_loader, test_loader, int(batch)


def paritition_all_dataset(data, global_batch, seed):
    dataset = None
    testset = None
    
    if data == "mnist":
        dataset = datasets.MNIST('/home/gw/programming/pytorch/data', 
                train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        testset = datasets.MNIST('/home/gw/programming/pytorch/data', 
                train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    elif data == "cifar10":
        dataset = datasets.CIFAR10(root='/home/gw/programming/pytorch/data',
                train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))
        testset = datasets.CIFAR10(root='/home/gw/programming/pytorch/data',
                train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]))
    else:
        print("no such data set: ", data)
        exit(1)

    size = dist.get_world_size()
    batch = global_batch/float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    #partition = partition.use(dist.get_rank())

    #print("rank %d partitionsize %d batchsize %d" %
    #        (dist.get_rank(),len(partition), batch))
    dataloaders = []
    for r in range(size):
        train_loader = torch.utils.data.DataLoader(
                partition.use(r),
                batch_size = int(batch),
                shuffle=True)
        dataloaders.append(train_loader)
        
    testloaders = torch.utils.data.DataLoader(
            testset,
            batch_size = int(batch),
            shuffle=False)

    return dataloaders, testloaders, int(batch)


