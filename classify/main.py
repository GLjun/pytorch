#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8

import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet import ResNet50
from progress import *

def main():

    gpus=[4,5,6,7]
    print("GPUs :", gpus)
    print("prepare data")
    normalize = transforms.Normalize( 
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225])

    train_tfs = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    val_tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    train_ds = datasets.ImageFolder(
            '/home/gw/tmp/imagenet/train',
            train_tfs);
    val_ds = datasets.ImageFolder(
            '/home/gw/tmp/imagenet/val',
            val_tfs);

    train_ld = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=256,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    val_ld = torch.utils.data.DataLoader(
            val_ds,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    print("construct model")
    model = ResNet50()
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpus[0])

    criterion = nn.CrossEntropyLoss().cuda(gpus[0])
    optimizer = torch.optim.SGD(
            model.parameters(), 0.256,
            momentum=0.875,
            weight_decay=3.0517578125e-05)

    model.train()
    print("begin trainning")
    for epoch in range(0, 50):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
                len(train_ld),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
        
        end = time.time()
        for i, (images, labels) in enumerate(train_ld):
            data_time.update(time.time()-end)
            #images = images.cuda(gpus[0], non_blocking=True)
            labels = labels.cuda(gpus[0], non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # measure accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time()-end)
            end = time.time()

            if i % 10 == 0 :
                progress.display(i)




if __name__ == '__main__':
    main()
