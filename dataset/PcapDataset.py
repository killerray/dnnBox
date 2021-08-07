#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 14:13
# @File    : PcapDataset.py

# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import json
import glob
from typing import List, Dict, Tuple


class PcapDataset(Dataset):
    def __init__(self, sample_dir, sample_size,
                 tensor_shape, direction, min_seq_len, slices):
        self.sample_dir = sample_dir
        self.sample_size = sample_size
        self.tensor_shape = tensor_shape
        self.direction = direction
        self.min_seq_len = min_seq_len
        self.sample_slices = slices
        self._sample_list = self._create_sample_list()

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, idx):

        sample_tensor = self.transform(self._sample_list[idx],
                                       self.sample_size,
                                       self.tensor_shape)

        label = torch.tensor(self.target_transform(self._sample_list[idx]))
        return sample_tensor, label

    def _create_sample_list(self) -> List[Dict]:
        res_list = list()
        path_list = glob.glob("%s\\*.json" % self.sample_dir)
        for filepath in path_list:
            with open(filepath, 'r') as f:
                json_body = json.load(f)
                for session in json_body:
                    is_ok = self._filter_sample(session, self.direction,
                                                self.min_seq_len)
                    if is_ok:
                        res_list.append(session)
        return res_list

    def _filter_sample(self, session: Dict, direction: str, min_item_num: int):
        raw = session.get(direction, [])
        if len(raw) == 0 or len(raw) < min_item_num:
            return False
        else:
            return True

    def stat_sample_num(self):
        sample_num_dict = dict()
        for session in self._sample_list:
            if session['pcapFileName'] in sample_num_dict:
                sample_num_dict[session['pcapFileName']] += 1
            else:
                sample_num_dict[session['pcapFileName']] = 1
        print(sample_num_dict)

    def target_transform(self, session):
        label_num_mapping = {"neg": 0, "pos": 1}
        if "neg" in session['pcapFileName']:
            return label_num_mapping['neg']
        elif "pos" in session['pcapFileName']:
            return label_num_mapping['pos']
        else:
            return label_num_mapping['neg']

    def _cut_sample(self, session: Dict, direction: str, slices) -> List[str]:
        raw = session.get(direction, [])
        return raw[slices]

    def _padding_nparray_item(self, payload_narr_list: List[np.ndarray],
                              embedding_size) -> List[np.ndarray]:
        res_list = list()
        for payload_narr in payload_narr_list:
            if payload_narr.size < embedding_size:
                padding_len = embedding_size - payload_narr.size
                res_list.append(np.hstack(
                    (payload_narr, np.zeros(padding_len, dtype=np.uint8))))
            else:
                res_list.append(payload_narr[0:embedding_size])
        return res_list

    def transform(self, session: Dict, sample_size: Tuple[int, int],
                  tensor_shape) -> torch.Tensor:
        payload_list = self._cut_sample(session, self.direction,
                                        self.sample_slices)
        payload_narr_list = self._padding_nparray_item(
            [np.frombuffer(bytes.fromhex(payload), dtype=np.uint8) for payload
             in payload_list], sample_size[1])

        data = np.concatenate(payload_narr_list, axis=0)

        return torch.reshape(torch.tensor(data).type(torch.float),
                             tensor_shape)


if __name__ == "__main__":
    pcapDataset = PcapDataset("./", (3, 40 * 40), (3, 1, 40, 40), "Raw", 9,
                              slice(6, 9))
    data, label = pcapDataset.__getitem__(0)
    print("sample num", pcapDataset.__len__())

    print(label)
