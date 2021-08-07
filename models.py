#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(32 * 5 * 5, 120),
                                        nn.Linear(120, 84),
                                        nn.Linear(84, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.classifier(x)
        return x
