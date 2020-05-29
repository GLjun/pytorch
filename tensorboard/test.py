# coding=utf-8
# pylint: skip-file

# pylint: disable-all

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for i in range(100):
    writer.add_scalar('train/loss', i*1.0, i*100,
            comment='hello', filename_suffix='hh')
