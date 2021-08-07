#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torchmetrics


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
    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='none', num_classes=10)
    test_precision = torchmetrics.Precision(average='none', num_classes=10)
    test_auc = torchmetrics.AUROC(average="macro", num_classes=10)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 一个batch进行计算迭代
            test_acc(pred.argmax(1), y)
            test_auc.update(pred, y)
            test_recall(pred.argmax(1), y)
            test_precision(pred.argmax(1), y)

    test_loss /= num_batches
    correct /= size

    # 计算一个epoch的accuray、recall、precision、AUC
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f}, "
          f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("auc:", total_auc.item())

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()
