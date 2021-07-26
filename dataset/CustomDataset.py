# -*- coding: utf-8 -*-

import os
from torch.utils.data.dataset import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, annotations_file, sample_dir, sample_size, tensor_shape,
                 transform=None,
                 target_transform=None):
        self.annotations_file = annotations_file
        self.labels = self.read_label_file()
        self.sample_dir = sample_dir
        self.transform = transform
        self.target_transform = target_transform
        self.sample_size = sample_size
        self.tensor_shape = tensor_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.sample_dir,
                                   self.labels[idx][0])

        sample_tensor = self.transform(sample_path, self.sample_size,
                                       self.tensor_shape)
        label = self.labels[idx][1]

        label = torch.tensor(self.target_transform(label))
        return sample_tensor, label

    def read_label_file(self):
        label_list = list()
        with open(self.annotations_file, 'r') as h:
            while True:
                line = h.readline()
                if not line:
                    break
                label_list.append(
                    line.strip(" ").strip("\r").strip("\n").split(","))
            return label_list
