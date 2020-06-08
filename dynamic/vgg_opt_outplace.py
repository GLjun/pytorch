#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from sgd_opt_outplace import SGDOPO

def conv2d3x3_BN_ReLU(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))

class VGGNetOPO(nn.Module):
    def __init__(self, blocks, num_classes=1000, has_softmax=False):
        super(VGGNetOPO, self).__init__()
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
        
def VGG16OPO(num_classes=1000):
    blocks = [2, 2, 3, 3, 3]
    return VGGNetOPO(blocks, num_classes=num_classes)

def VGG19OPO(num_classes=1000):
    blocks = [2, 2, 4, 4, 4]
    return VGGNetOPO(blocks, num_classes=num_classes)

def print_children(module, prefix=" "):
    for n, m in module.named_modules():
        #print(n)
        if len(m.modules()) > 0:
            print(n)
        print_children(m, prefix+" ")


if __name__ == '__main__':
    model = VGG16OPO()
    cnt = 0
    cnt1 = 0
    for param in model.parameters():
        if param.data is not None and param.requires_grad:
            print(param.data.size())
            cnt += 1
    
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01, weight_decay=0.0001)
    input = torch.randn(1, 3, 224, 224)

    out = model(input)
    print(out.shape)
    #out.backward()
    lossfunc = torch.nn.NLLLoss()
    #label = torch.zeros_like(out, dtype=torch.long)
    
    #label[0][0] = 1
    #print(label)
    #print(label.shape)
    label = torch.zeros(1, dtype=torch.long)
    loss = lossfunc(out, label)
    loss.backward()

    for param in model.parameters():
        if param.grad is not None:
            cnt1 += 1
    print("cnt ", cnt, " cnt1 ", cnt1)
    print(model.parameters()[0].data.device)

    

    #for k, v in model.named_modules():
    #    print(k, " ", v)
    #    print("======")
    #print_children(model)
    #nn = []
    #mm = []
    #pp = []
    #for k, v in model.named_children():
    #    for n,  m in v.named_children():
    #        nn.append(n)
    #        mm.append(m)
    #        for pn, p in m.named_parameters():
    #            if p.data is not None and p.requires_grad:
    #                print("p==>", pn)
    #                pp.append(p)
    #        for pn, p in m.named_buffers():
    #            print("b==>", pn)
    #        print("")
    #print(nn)
    #print(len(mm))
    #print(len(pp))

    #input = torch.randn(1, 3, 224, 224)
    #out = model(input)
    #print(out.shape)
        

