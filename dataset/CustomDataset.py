# -*- coding: utf-8 -*-

import os
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, annotations_file, sample_dir, sample_size,
                 tensor_shape):
        self.annotations_file = annotations_file
        self.labels = self.read_label_file()
        self.sample_dir = sample_dir
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

    def target_transform(self, label):
        label_num_mapping = {"neg": 0, "pos": 1}
        return label_num_mapping[label]

    def transform(self, sample_path, sample_size, tensor_shape):
        with open(sample_path, "rb") as h:
            content = h.read()
            content = np.frombuffer(content, dtype=np.uint8, offset=0)
            if content.size < sample_size:
                padding_len = sample_size - content.size
                content = np.hstack(
                    (content, np.zeros(padding_len, dtype=np.uint8)))
            elif content.size > sample_size:
                content = content[0:sample_size]

            return torch.reshape(torch.tensor(content).type(torch.float),
                                 tensor_shape)
