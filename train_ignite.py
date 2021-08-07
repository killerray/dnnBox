#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 17:55
# @File    : train_ignite.py

import torch
from ignite.metrics import Accuracy, Precision, Recall


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
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


def test_loop(dataloader, model, loss_fn):
    # 实例化相关metrics的计算对象
    test_acc = Accuracy()
    test_recall = Recall()
    test_precision = Precision()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 一个batch进行计算迭代
            test_acc.update((pred, y))
            test_recall.update((pred, y))
            test_precision.update((pred, y))

    test_loss /= num_batches
    correct /= size

    # 计算一个epoch的accuray、recall、precision
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f}, "
          f"ignite acc: {(100 * total_acc):>0.1f}%\n")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
