#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8

import torch
import torch.nn as nn
import torchvision

#print("Pytorch :", torch.__version__)
#print("Torchvision :", torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']

def Conv1(Ci, Co, stride=2):
    return nn.Sequential(
            nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=7,
                stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(Co),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class Bottleneck(nn.Module):
    def __init__(self, Ci, Co, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
                nn.Conv2d(Ci, Co, 1, stride=1, bias=False),
                nn.BatchNorm2d(Co),
                nn.ReLU(inplace=True),
                #downsample using stride
                nn.Conv2d(Co, Co, 3, stride=stride, padding=1, 
                    bias=False),
                nn.BatchNorm2d(Co),
                nn.ReLU(inplace=True),
                nn.Conv2d(Co, self.expansion*Co, 1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion*Co))

        if self.downsampling:
            self.downsample = nn.Sequential(
                    nn.Conv2d(Ci, self.expansion*Co, 1, stride=stride,
                        bias=False),
                    nn.BatchNorm2d(self.expansion*Co))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        
        self.expansion = expansion

        self.conv1 = Conv1(3, 64);

        self.layer1 = self.make_layer(64, 64, blocks[0], 1);
        self.layer2 = self.make_layer(256, 128, blocks[1], 2);
        self.layer3 = self.make_layer(512, 256, blocks[2], 2);
        self.layer4 = self.make_layer(1024, 512, blocks[3], 2);

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def make_layer(self, Ci, Co, block, stride):
        layers = []
        layers.append(Bottleneck(Ci, Co, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(self.expansion*Co, Co))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=1000):
    return ResNet([3, 4, 6, 3], num_classes)

def ResNet101(num_classes=1000):
    return ResNet([3, 4, 23, 3], num_classes)

def ResNet152(num_classes=1000):
    return ResNet([3, 8, 36, 3], num_classes)

if __name__=='__main__':
    model = ResNet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
