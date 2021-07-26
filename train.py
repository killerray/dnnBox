#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np


def train_loop(dataloader, model, loss_fn, optimizer, is_rnn):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        if is_rnn:
            X = X.squeeze(1)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, is_rnn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if is_rnn:
                X = X.squeeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")


def transform_label_2_num(label):
    label_num_mapping = {"neg": 0, "pos": 1}
    return label_num_mapping[label]


def sample_transform(sample_path, sample_size, tensor_shape):
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
