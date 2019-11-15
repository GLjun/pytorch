#!/usr/bin/env python
# pylint: skip-file

# pylint: disable-all
# coding=utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size),
        batch_size = batch_size, 
        shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        print("\tIn Model: input size", inputs.size(),
                "output size", outputs.size())
        return outputs

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Le's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    inputs = data.to(device)
    print("begin model")
    outputs = model(inputs)
    print("Outside: input size", inputs.size(),
            "output_size", outputs.size())


