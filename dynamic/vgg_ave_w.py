#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist

def conv2d3x3_BN_ReLU(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))

class VGGNetAW(nn.Module):
    def __init__(self, blocks, num_classes=1000, has_softmax=False):
        super(VGGNetAW, self).__init__()
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

        self.cnt = 0



    def make_layers(self, in_channels, out_channels, block):
        layers = []
        layers.append(conv2d3x3_BN_ReLU(in_channels, out_channels))
        for i in range(1, block):
            layers.append(conv2d3x3_BN_ReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x): 
        #rank = dist.get_rank()
        size = dist.get_world_size()
        factor = 1.0/float(size)

        
        if self.cnt >= 102 and self.training:
            #if rank == 0:
            #    print("==> train reduce weight")
            wait_list = []
            x = self.stage1(x)
            for param in self.stage1.parameters():
                #if rank == 0:
                    #print("cnt>=100 requires_grad ", param.requires_grad)
                if param.data is not None and param.requires_grad:
                    param.data.mul_(factor)
                    red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                            async_op=True)
                    wait_list.append(red)
            x = self.stage2(x)
            for param in self.stage2.parameters():
                if param.data is not None and param.requires_grad:
                    param.data.mul_(factor)
                    red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                            async_op=True)
                    wait_list.append(red)
            x = self.stage3(x)
            for param in self.stage3.parameters():
                if param.data is not None and param.requires_grad:
                    param.data.mul_(factor)
                    red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                            async_op=True)
                    wait_list.append(red)
            x = self.stage4(x)
            for param in self.stage4.parameters():
                if param.data is not None and param.requires_grad:
                    param.data.mul_(factor)
                    red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                            async_op=True)
                    wait_list.append(red)
            x = self.stage5(x)
            for param in self.stage5.parameters():
                if param.data is not None and param.requires_grad:
                    param.data.mul_(factor)
                    red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                            async_op=True)
                    wait_list.append(red)
#
            x = x.view(x.size(0), -1)
            x = self.fc_stage(x)
            for param in self.fc_stage.parameters():
                if param.data is not None and param.requires_grad:
                    param.data.mul_(factor)
                    red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                            async_op=True)
                    wait_list.append(red)
            if self.softmax is not None :
                x = self.softmax(x)
                for param in self.softmax.parameters():
                    if param.data is not None and param.requires_grad:
                        param.data.mul_(factor)
                        red = dist.all_reduce(param.data, dist.ReduceOp.SUM,
                                async_op=True)
                        wait_list.append(red)
            

            for red in reversed(wait_list):
                if not red.is_completed():
                    red.wait()
        else:
            #if rank == 0 and self.training:
            #    print("%d traing not reduce weight " % (self.cnt))
            #if rank == 0 and not self.training:
            #    print("%d val not reduce weight" % (self.cnt))
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)

            x = x.view(x.size(0), -1)
            x = self.fc_stage(x)
            if self.softmax is not None :
                x = self.softmax(x)
        self.cnt += 1


        return x
        
def VGG16AW(num_classes=1000):
    blocks = [2, 2, 3, 3, 3]
    return VGGNetAW(blocks, num_classes=num_classes)

def VGG19AW(num_classes=1000):
    blocks = [2, 2, 4, 4, 4]
    return VGGNetAW(blocks, num_classes=num_classes)


if __name__ == '__main__':
    model = VGG16AW()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
        

