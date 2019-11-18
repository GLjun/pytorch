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


import torch
import time

k = 5
w = 2000
tensors = [torch.ones(w, w).cuda() for _ in range(k)]
output = [torch.zeros(w, w).cuda() for _ in range(k)]
streams = [torch.cuda.Stream() for _ in range(k)]

print("starting...")
#prof_start()

se = torch.cuda.Event(enable_timing=True)
ee = torch.cuda.Event(enable_timing=True)
se1 = torch.cuda.Event(enable_timing=True)
ee1 = torch.cuda.Event(enable_timing=True)

#with torch.cuda.profiler.profile():

print("warmming up")
torch.cuda.synchronize()
ts = time.time()
se.record()
for i in range(10):
    for j in range(k):
        torch.mm(tensors[j], tensors[j], out=output[j])
ee.record()
torch.cuda.synchronize()
te = time.time()
print("time : ", te-ts)
time1 = se.elapsed_time(ee)
print("warmming up: ", time1)

torch.cuda.synchronize()
ts = time.time()
se.record()
for i in range(10):
    for j in range(k):
        torch.mm(tensors[j], tensors[j], out=output[j])
ee.record()
torch.cuda.synchronize()
te = time.time()
print("time : ", te-ts)
time1 = se.elapsed_time(ee)
print("serial: ", time1)

ts = time.time()
se1.record()
for i in range(10):
    for idx in range(k):
        with torch.cuda.stream(streams[idx]):
            torch.mm(tensors[idx], tensors[idx], out=output[idx])

ee1.record()
torch.cuda.synchronize()
te = time.time()
print("time : ", te-ts)
time2 = se1.elapsed_time(ee1)
print("inner stream: ", time2)

ts = time.time()
se1.record()
for idx in range(k):
    for i in range(10):
        with torch.cuda.stream(streams[idx]):
            torch.mm(tensors[idx], tensors[idx], out=output[idx])

ee1.record()
torch.cuda.synchronize()
te = time.time()
print("time : ", te-ts)
time2 = se1.elapsed_time(ee1)
print("outer stream: ", time2)
#for idx in range(k):
#    with torch.cuda.stream(streams[idx]):
#        for i in range(2000):
#            torch.mm(tensors[idx], tensors[idx], out=output[idx])
for idx in range(k):
    torch.cuda.current_stream().wait_stream(streams[idx])

#prof_stop()
print("done")
