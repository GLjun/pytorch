#!/bin/bash
time python dynamic.py --syn_frq 1 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
time python dynamic.py --syn_frq 2 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
time python dynamic.py --syn_frq 4 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
time python dynamic.py --syn_frq 8 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
time python dynamic.py --syn_frq 16 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
time python dynamic.py --syn_frq 32 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
time python dynamic.py --syn_frq 100 --gpus 2 3 --nproc 2 --batch 256 --data cifar10
