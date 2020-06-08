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

class VGGNetAWO(nn.Module):
    def __init__(self, blocks, num_classes=1000, has_softmax=False):
        super(VGGNetAWO, self).__init__()
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
        
        

        #for key, param in self.stage1.named_buffers():
        #    if param.data is not None and param.requires_grad:
        #        self.stage1.register_buffer(key, param.data.clone().detach().requires_grad_(False))
        #for key, param in self.stage2.named_buffers():
        #    if param.data is not None and param.requires_grad:
        #        self.stage2.register_buffer(key, param.data.clone().detach().requires_grad_(False))
        #for key, param in self.stage3.named_buffers():
        #    if param.data is not None and param.requires_grad:
        #        self.stage3.register_buffer(key, param.data.clone().detach().requires_grad_(False))
        #for key, param in self.stage4.named_buffers():
        #    if param.data is not None and param.requires_grad:
        #        self.stage4.register_buffer(key, param.data.clone().detach().requires_grad_(False))
        #for key, param in self.stage5.named_buffers():
        #    if param.data is not None and param.requires_grad:
        #        self.stage5.register_buffer(key, param.data.clone().detach().requires_grad_(False))
        #for key, param in self.fc_stage.named_buffers():
        #    if param.data is not None and param.requires_grad:
        #        self.fc_stage.register_buffer(key, param.data.clone().detach().requires_grad_(False))
        
        self.cnt = 0
    
    def gather_kv(self, kv):
        for key, param in self.named_parameters():
            if param.data is not None and param.requires_grad:
                #print("key ", key, " ", hash(param.data), " ", hash(param))
                kv[param] = param.data.clone().detach().requires_grad_(False)
        self.kv = kv
        print("vgg ke len ", len(self.kv))



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

        is_async = True
        wait_list = []
        
        if self.cnt >= 0 and self.training:
            
            for key, param in self.fc_stage.named_parameters():
                if param.data is not None and param.requires_grad:
                    #print("key ", key, " ", hash(param.data), " ", hash(param))
                    buf = self.kv[param]
                    #buf.copy_(param.data)
                    red = dist.all_reduce(buf, dist.ReduceOp.SUM,
                        async_op=is_async)
                    wait_list.append(red)
            x = self.stage1(x)
            
            for param in self.stage5.parameters():
                if param.data is not None and param.requires_grad:
                    buf = self.kv[param]
                    #buf.copy_(param.data)
                    red = dist.all_reduce(buf, dist.ReduceOp.SUM,
                        async_op=is_async)
                    wait_list.append(red)
            x = self.stage2(x)
            
            for param in self.stage4.parameters():
                if param.data is not None and param.requires_grad:
                    buf = self.kv[param]
                    #buf.copy_(param.data)
                    red = dist.all_reduce(buf, dist.ReduceOp.SUM,
                        async_op=is_async)
                    wait_list.append(red)
            x = self.stage3(x)

            for param in self.stage3.parameters():
                if param.data is not None and param.requires_grad:
                    buf = self.kv[param]
                    #buf.copy_(param.data)
                    red = dist.all_reduce(buf, dist.ReduceOp.SUM,
                        async_op=is_async)
                    wait_list.append(red)
            x = self.stage4(x)

            for param in self.stage2.parameters():
                if param.data is not None and param.requires_grad:
                    buf = self.kv[param]
                    #buf.copy_(param.data)
                    red = dist.all_reduce(buf, dist.ReduceOp.SUM,
                        async_op=is_async)
                    wait_list.append(red)
            x = self.stage5(x)

            for param in self.stage1.parameters():
                if param.data is not None and param.requires_grad:
                    buf = self.kv[param]
                    #buf.copy_(param.data)
                    red = dist.all_reduce(buf, dist.ReduceOp.SUM,
                        async_op=is_async)
                    wait_list.append(red)
#
            x = x.view(x.size(0), -1)
            x = self.fc_stage(x)
            if self.softmax is not None :
                x = self.softmax(x)

            if is_async:
                for red in wait_list:
                    if red is not None and not red.is_completed():
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
        
def VGG16AWO(num_classes=1000):
    blocks = [2, 2, 3, 3, 3]
    return VGGNetAWO(blocks, num_classes=num_classes)

def VGG19AWO(num_classes=1000):
    blocks = [2, 2, 4, 4, 4]
    return VGGNetAWO(blocks, num_classes=num_classes)


if __name__ == '__main__':
    model = VGG16AWO()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
        

