#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8

import torch
import torch.nn as nn
import torchvision

def conv2d3x3_BN_ReLU(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))

class VGGNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, has_softmax=False):
        super(VGGNet, self).__init__()
        self.stage1 = self.make_layers(3,   64,  blocks[0])
        self.stage2 = self.make_layers(64,  128, blocks[1])
        self.stage3 = self.make_layers(128, 256, blocks[2])
        self.stage4 = self.make_layers(256, 512, blocks[3])
        self.stage5 = self.make_layers(512, 512, blocks[4])

        self.fc_stage = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes))

        self.softmax = None
        if has_softmax :
            self.softmax = nn.Softmax(dim=1)



    def make_layers(self, in_channels, out_channels, block):
        layers = []
        layers.append(conv2d3x3_BN_ReLU(in_channels, out_channels))
        for i in range(1, block):
            layers.append(conv2d3x3_BN_ReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = x.view(x.size(0), -1)
        x = self.fc_stage(x)
        if self.softmax is not None :
            x = self.softmax(x)

        return x

def VGG16():
    blocks = [2, 2, 3, 3, 3]
    return VGGNet(blocks)

def VGG19():
    blocks = [2, 2, 4, 4, 4]
    return VGGNet(blocks)


if __name__ == '__main__':
    model = VGG16()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
        

