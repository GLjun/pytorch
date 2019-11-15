#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8


import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

s1 = torch.cuda.Stream(device)
s2 = torch.cuda.Stream(device)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    A = torch.randn(1000, 1000)
    B = torch.randn(1000, 1000)
    A.to(device)
    B.to(device)
    AB = torch.randn(1000, 1000, device=device)
    C = torch.randn(1000, 1000, device=device)
    D = torch.randn(1000, 1000, device=device)
    print("warming up")
    AB = torch.mm(A, B)
    C = torch.mm(A,B)
    D = torch.mm(A,B)
    
    
    start_event1 = torch.cuda.Event(enable_timing=True)
    end_event1 = torch.cuda.Event(enable_timing=True)
    print("single")
    start_event1.record()
    
    for i in range(0, 10):
        C = torch.mm(A, B)
        D = torch.mm(A, B)
    end_event1.record()
    torch.cuda.synchronize()
    time1 = start_event1.elapsed_time(end_event1)
    
    print(time1)
    
    start_event2 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)
    print("two stream")
    start_event2.record()
    with torch.cuda.stream(s1):
        for i in range(0, 10):
            C = torch.mm(A, B)
    
    with torch.cuda.stream(s2):
        for i in range(0, 10):
            D = torch.mm(A, B)
    
    end_event2.record()
    torch.cuda.synchronize()
    
    time2 = start_event2.elapsed_time(end_event2)
    
    print(time2)

print(prof)
