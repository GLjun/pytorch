#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8


import torch
import os
from resnet import ResNet50

#os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def test_stream_standalone():
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    A = torch.randn(1000, 1000)
    B = torch.randn(1000, 1000)
    C = torch.randn(1000, 1000)
    D = torch.randn(1000, 1000)
    A.to(device)
    B.to(device)
    C.to(device)
    D.to(device)
    C = torch.mm(A, B)
    D = torch.mm(A, B)

def test_stream_seq(A, B, C, D):
    torch.cuda.synchronize()
    for i in range(1000):
        C += A
    for i in range(1000):
        D += B
    torch.cuda.synchronize()


def test_stream(s1, s2, A, B, C, D):
    torch.cuda.synchronize()
    #with torch.cuda.stream(s1):
    for i in range(1000):
        C += A
    with torch.cuda.stream(s1):
        for i in range(1000):
            D += A
    torch.cuda.synchronize()


if __name__=='__main__':
    print("allocate ")
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    A = torch.randn(500, 500)
    B = torch.randn(500, 500)
    C = torch.randn(500, 500)
    D = torch.randn(500, 500)
    A.cuda()
    B.cuda()
    C.cuda()
    D.cuda()
    torch.cuda.synchronize()
    print("profilling ")
    test_stream_seq(A, B, C, D)
    test_stream(s1, s2, A, B, C, D)
    #with torch.cuda.profiler.profile():
    #    test_stream_seq(A, B, C, D)
    #    test_stream(s1, s2, A, B, C, D)
        #with torch.autograd.profiler.emit_nvtx():
        #    test_stream_seq(A, B, C, D)
        #    test_stream(s1, s2, A, B, C, D)
            
        
#if __name__=='__main__':
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", device)
    model = ResNet50()
    model.to(device)
    input = torch.randn(1,3,224,224)
    cuinput=input.to(device)
    with torch.cuda.profiler.profile():
        model(cuinput)
        with torch.autograd.profiler.emit_nvtx():
            model(cuinput)
