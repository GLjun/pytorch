#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8

import torch
import torch.nn as nn
import torchvision


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 192, 5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    model=AlexNet()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)



