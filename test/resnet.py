#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_dir = '/home/gw/tmp/imagenet/train'

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,]))

train_sampler

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, sample=


