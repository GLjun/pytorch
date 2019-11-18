#!/usr/bin/env python
# pylint: disable-all

# pylint: skip-file 
# coding=utf-8


import torch
import os
import ctypes

_cudart = ctypes.CDLL('libcudart.so')

def prof_start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)

def test_stream_serial(A, B, C, D):
    for i in range(1000):
        C = torch.mm(A, A)
    for i in range(1000):
        D = torch.mm(B, B)
    torch.cuda.synchronize()


def test_stream_parallel(s1,  A, B, C, D):
    for i in range(1000):
        C = torch.mm(A, A)
    with torch.cuda.stream(s1):
        for i in range(1000):
            D = torch.mm(B, B)
    torch.cuda.synchronize()

#if __name__ == '__main__':
prof_start()
print("alloc")
s1 = torch.cuda.Stream()
A = torch.randn(500, 500)
B = torch.randn(500, 500)
C = torch.randn(500, 500)
D = torch.randn(500, 500)
A.cuda()
B.cuda()
C.cuda()
D.cuda()
torch.cuda.synchronize()

#warmming up
test_stream_serial(A, B, C, D)
test_stream_parallel(s1, A, B, C, D)

#testing
test_stream_serial(A, B, C, D)
test_stream_parallel(s1, A, B, C, D)


prof_stop()

        
