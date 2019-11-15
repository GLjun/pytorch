#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8

import torch
import time


cuda = torch.device('cuda')

s = torch.cuda.Stream()

#number of streams
N=5

M=500 
N=100
K=400
repeat=1000

streams = [ torch.cuda.Stream() for _ in range(N) ]
A = [torch.ones(M, K).cuda() for _ in range(N)]
B = [torch.ones(K, N).cuda() for _ in range(N)]
C = [torch.ones(M, N).cuda() for _ in range(N)]
D = [torch.ones(M, K).cuda() for _ in range(N)]

def serial_mm():
    for i in range(repeat):
        for j in range(N):
            C[j] = torch.mm(A[j], B[j])
            D[j] = torch.add(A[j], A[j])

def parallel_mm():
    for i in range(repeat):
        for j in range(N):
            with torch.cuda.stream(streams[j]):
                C[j] = torch.mm(A[j], B[j])
                D[j] = torch.add(A[j], A[j])
    


torch.cuda.synchronize()

#wamming up
for i in range(repeat):
    for j in range(N):
        C[j] = torch.mm(A[j], B[j])

torch.cuda.synchronize()
se = time.time()
for i in range(repeat):
    for j in range(N):
        C[j] = torch.mm(A[j], B[j])
        D[j] = torch.add(A[j], A[j])
torch.cuda.synchronize()
ee = time.time()
print("serial time: ", ee-se)

se = time.time()
for i in range(repeat):
    for j in range(N):
        with torch.cuda.stream(streams[j]):
            C[j] = torch.mm(A[j], B[j])
            D[j] = torch.add(A[j], A[j])
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
