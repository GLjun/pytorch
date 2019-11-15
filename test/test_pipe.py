#!/usr/bin/env python
# coding=utf-8
# pylint: disable-all
# pylint: skip-file

import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data/', 
        train=True, download=False, transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
        download=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
        shuffle=False, num_workers=2)

classes= ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
        'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

#def imshow(img):
#    img = img/2 + 0.5
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()
#
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
#imshow(torchvision.utils.make_grid(images))
#
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

net2 = Net()
print(net)
print(net2)

device = torch.device("cuda:7")
print("move net to gpu:7")
net.to(device)
net2.to(device)
print(net)
print(net2)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    running_loss2 = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        if i==0:
            print(type(inputs))
            print(type(labels))

        cuinputs = inputs.to(device)
        culabels = labels.to(device)

        #cuinputs2 = cuinputs
        #culabels2 = culabels


        #cuinputs = inputs.to(device)

        optimizer.zero_grad()
        optimizer2.zero_grad()

        outputs = net(cuinputs)
        outputs2 = net2(cuinputs)
        loss = criterion(outputs, culabels)
        loss2 = criterion2(outputs2, culabels)
        loss.backward()
        loss2.backward()
        optimizer.step()
        optimizer2.step()



        running_loss += loss.item()
        running_loss2 += loss2.item()
        if i%200 == 199:
            print('[%d, %5d] loss: %.3f loss2: %.3f' % 
                    (epoch+1, i+1, running_loss/200,
                        running_loss2/200))
            running_loss = 0.0
            running_loss2 = 0.0

print('Finished Training')

PATH = './test_cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



testnet = Net()
testnet.load_state_dict(torch.load(PATH))

outputs = testnet(images)

v, predicted = torch.max(outputs, 1)
print(v)
print(predicted)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for
    j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = testnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print('Accuracy on 10000 test images: %d %%' % (100*correct/total))


