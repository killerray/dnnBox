#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch


def train_loop(dataloader, model, loss_fn, optimizer, which_model):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        if which_model == 1:
            X = X.squeeze(1)
        elif which_model == 2:
            X = X.unsqueeze(1)
        else:
            pass
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


def test_loop(dataloader, model, loss_fn, which_model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if which_model == 1:
                X = X.squeeze(1)
            elif which_model == 2:
                X = X.unsqueeze(1)
            else:
                pass
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")
