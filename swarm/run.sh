#!/bin/bash

python dynamic.py --gpus  2 3 4 5 --lr 0.001 --nproc 4 --batch 1024 --syn_frq 4 --syn_begin 0 --data cifar10

python dynamic2.py --gpus  2 3 4 5 --lr 0.001 --nproc 4 --batch 1024 --syn_frq 1 --syn_begin 0 --data cifar10

python dynamic.py --gpus  2 3 4 5 --lr 0.001 --nproc 4 --batch 1024 --syn_frq 2 --syn_begin 0 --data cifar10

