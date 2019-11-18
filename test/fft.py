#!/usr/bin/env python
# coding=utf-8
# pylint: disable-all
# pylint: skip-file

import numpy as np
import torch
import time

k = 4
cuda = torch.device('cuda')
streams = [torch.cuda.Stream(cuda) for i in range(k)]

x = torch.tensor(np.random.rand(100, 2048, 2048).astype(np.float32))
xs = x.to(cuda)
Xgs = []

torch.cuda.synchronize()
st = time.time()
# Start timing here
for _ in range(1000):
    for i, s in enumerate(streams):
        with torch.cuda.stream(s):
            Xg = torch.rfft(xs[i], 2)
    #Xgs.append(Xg)

torch.cuda.synchronize()
et = time.time()
print("warmming up time: ", et-st)

torch.cuda.synchronize()
st = time.time()
# Start timing here
for _ in range(1000):
    for i, s in enumerate(streams):
        with torch.cuda.stream(s):
            Xg = torch.rfft(xs[i], 2)
    #Xgs.append(Xg)

torch.cuda.synchronize()
et = time.time()
print("serial time: ", et-st)

st = time.time()
# Start timing here
for _ in range(1000):
    for i, s in enumerate(streams):
        Xg = torch.rfft(xs[i], 2)
    #Xgs.append(Xg)

torch.cuda.synchronize()
et = time.time()
print("parallel time: ", et-st)
# Indent below to run explicitly in serial
torch.cuda.synchronize()
