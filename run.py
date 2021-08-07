#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from models import LeNet
from train_torchmetrics import train_loop, test_loop
# from train_ignite import test_loop, train_loop

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epoches = 500

train_dataset = datasets.MNIST(
    root='./mnist', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./mnist', train=False, transform=transforms.ToTensor(),
    download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def LeNet_mnist_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNet(num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
        print("Done!")
        # Save Models
    torch.save(model.state_dict(), "LeNet.pth")
    print("Saved PyTorch Model State to LeNet.pth")


if __name__ == "__main__":
    LeNet_mnist_train()
