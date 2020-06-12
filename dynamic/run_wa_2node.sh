#!/bin/bash
python wa_sgd_opt_outplace.py --classes 10 -b 768 --ave-weight --lr 0.01 --sgd SGDOPO_BW --momentum 0.0 --print-freq 30 --dist-url 'tcp://192.168.212.84:12345' --world-size 2 --epochs 30 --no-details
#python main.py -a resnet50 --epochs 30 --multiprocessing-distributed --world-size 2 --rank 0 -b 384 --workers 8 --lr 0.01 --momentum 0.0 --classes 10 --dist-url 'tcp://192.168.212.84:12345' -p 30 /home/gw/data/imagenet_10
