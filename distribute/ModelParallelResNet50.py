#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000

class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)
        #super(ModelParallelResNet50, self).__init__(
        #        Bottleneck, 
        #        [3, 4, 6, 3],
        #        num_classes = num_classes,
        #        *args, **kwargs)

        self.seq1 = nn.Sequential(
                self.conv1, 
                self.bn1, 
                self.relu, 
                self.maxpool,
                self.layer1,
                self.layer2).to('cuda:0')

        self.seq2 = nn.Sequential(
                self.layer3,
                self.layer4,
                self.avgpool).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))

model = ModelParallelResNet50()
print(model)

num_batches = 3
batch_size = 120
image_h = 128
image_w = 128

def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size).random_(0,
            num_classes).view(batch_size, 1)

    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 3, image_h, image_w)
        labels = torch.zeros(batch_size, num_classes).scatter_(1,
                one_hot_indices, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))
        labels = labels.to(outputs.device)

        loss_fn(outputs, labels).backward()
        optimizer.step()

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit



stmt = "train(model)"
setup = "model=ModelParallelResNet50();"
num_repeat = 10

mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat = num_repeat, globals = globals())
print("model parallel run for repeat: ", mp_run_times)
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + "model = \
        models.resnet50(num_classes=num_classes).to('cuda:0')"

rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat = num_repeat, globals = globals())
print("model in torchvision run for repeat: ", rn_run_times)
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)



def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds, 
            align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

print("save MP and RN to mp_vs_rn.png")
plot(
    [mp_mean, rn_mean],
    [mp_std, rn_std],
    ['Model Parallel', 'Single GPU'],
    'mp_vs_rn.png')


#class PipelineParallelResNet50(ModelParallelResNet50):



