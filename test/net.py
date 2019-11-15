#!/usr/bin/env python
# coding=utf-8
# pylint: skip-file
# pylint: disable-all

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

param = list(net.parameters())
print(len(param))
print(param[0].size())

data = torch.randn(1,1,32,32)
out = net(data)
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

out = net(data)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print(loss.grad_fn)

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
print('conv1.bias.data before backward')
print(net.conv1.bias.data)

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
out = net(data)
loss = criterion(out, target)
loss.backward()
optimizer.step()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
print('conv1.bias.data after backward')
print(net.conv1.bias.data)

