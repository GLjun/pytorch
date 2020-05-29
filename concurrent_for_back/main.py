# coding=utf-8
# pylint: skip-file

# pylint: disable-all

import torch
import torch.optim as optim
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from BaiduNet import BaiduNet9P

device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device2 = 'cuda:1' if torch.cuda.is_available() else 'cpu'


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/gw/data', 
        train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/gw/data',
        train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, 
        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net1 = BaiduNet9P().to(device1)
net2 = BaiduNet9P().to(device2)

criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=0.1, momentum=0.9, 
        weight_decay=5e-4)

criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.SGD(net2.parameters(), lr=0.1, momentum=0.9, 
        weight_decay=5e-4)

s1=torch.cuda.Stream(device2)
s2=torch.cuda.Stream(device2)

def train_sync(epoch):
    print('train_syn for back')
    
    net1.train()
    i = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device1), targets.to(device1)
        torch.cuda.synchronize(device1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = net1(inputs)
        loss = criterion1(outputs, targets)
        loss.backward()
        optimizer1.step()
        end.record()
        torch.cuda.synchronize(device1)
        time = start.elapsed_time(end)
        print('time in train_sync : ', time)
        if i >= 5:
            break
        i += 1

def train_conn(epoch):
    print('train_conn for back')
    
    net1.train()
    i = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # first iteration
        inputs, targets = inputs.to(device2), targets.to(device2)
        outputs = net2(inputs)
        loss = criterion2(outputs, targets)

        torch.cuda.synchronize(device2)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.stream(s1):
            outputs_1 = net2(inputs)
            loss_1 = criterion2(outputs_1, targets)

        with torch.cuda.stream(s2):
            loss.backward()
            optimizer2.step()
        end.record()
        torch.cuda.synchronize(device2)
        time = start.elapsed_time(end)
        print('time in train_conn : ', time)

        if i >= 5:
            break
        i += 1

train_sync(1)
train_conn(1)

