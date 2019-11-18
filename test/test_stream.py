#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8

import torch
import time


cuda = torch.device('cuda')

s = torch.cuda.Stream()

#number of streams
NS=4

M=2000
N=1000
K=3000
repeat=10

streams = [ torch.cuda.Stream() for _ in range(NS) ]
A = [torch.ones(M, K).cuda() for _ in range(NS)]
B = [torch.ones(K, N).cuda() for _ in range(NS)]
C = [torch.ones(M, N).cuda() for _ in range(NS)]
D = [torch.ones(M, N).cuda() for _ in range(NS)]

def serial_mm():
    for i in range(repeat):
        for j in range(NS):
            C[j] = torch.mm(A[j], B[j])
            D[j] = torch.add(C[j], C[j])

def parallel_mm():
    for i in range(repeat):
        for j in range(NS):
            with torch.cuda.stream(streams[j]):
                C[j] = torch.mm(A[j], B[j])
                D[j] = torch.add(C[j], C[j])
    


torch.cuda.synchronize()

with torch.cuda.profiler.profile():
    #wamming up
    for i in range(repeat):
        for j in range(NS):
            C[j] = torch.mm(A[j], B[j])
            D[j] = torch.add(C[j], C[j])
    
    torch.cuda.synchronize()
    se = time.time()
    for i in range(repeat):
        for j in range(NS):
            C[j] = torch.mm(A[j], B[j])
            D[j] = torch.add(C[j], C[j])
    torch.cuda.synchronize()
    ee = time.time()
    print("serial time: ", ee-se)
    
    se = time.time()
    for i in range(repeat):
        for j in range(NS):
            with torch.cuda.stream(streams[j]):
                C[j] = torch.mm(A[j], B[j])
                D[j] = torch.add(C[j], C[j])
    torch.cuda.synchronize()
    ee = time.time()
    print("parallel time: ", ee-se)
    
    se = time.time()
    serial_mm()
    torch.cuda.synchronize()
    ee = time.time()
    print("serial time: ", ee-se)
    
    se = time.time()
    parallel_mm()
    torch.cuda.synchronize()
    ee = time.time()
    print("parallel time: ", ee-se)
