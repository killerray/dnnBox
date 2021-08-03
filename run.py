#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from models import Rnn, AlexNet, LeNet, CNNLSTM, AttBiLSTM
from train import train_loop, test_loop
from dataset.CustomDataset import CustomDataset

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epoches = 5

train_dataset = datasets.MNIST(
    root='./mnist', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./mnist', train=False, transform=transforms.ToTensor(),
    download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def AttBilstm_mnist_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttBiLSTM(10, 28, 128, 2, 0.5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 1)
        test_loop(test_loader, model, loss_fn, 1)
        print("Done!")
    torch.save(model.state_dict(), "AttBilstm.pth")
    print("Saved PyTorch Model State to AttBilstm.pth")


def lstm_mnist_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Rnn(28, 128, 2, 10, False).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 1)
        test_loop(test_loader, model, loss_fn, 1)
        print("Done!")
    torch.save(model.state_dict(), "Lstm.pth")
    print("Saved PyTorch Model State to Lstm.pth")


def alexnet_mnist_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlexNet(num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 0)
        test_loop(test_loader, model, loss_fn, 0)
        print("Done!")
    torch.save(model.state_dict(), "AlexNet.pth")
    print("Saved PyTorch Model State to AlexNet.pth")


def LeNet_mnist_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNet(num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 0)
        test_loop(test_loader, model, loss_fn, 0)
        print("Done!")
        # Save Models
    torch.save(model.state_dict(), "LeNet.pth")
    print("Saved PyTorch Model State to LeNet.pth")


def cnnlstm_mnist_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTM(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 2)
        test_loop(test_loader, model, loss_fn, 2)
        print("Done!")
        # Save Models
    torch.save(model.state_dict(), "cnnlstm.pth")
    print("Saved PyTorch Model State to cnnlstm.pth")


def cnnlstm_custom_train():
    train_custom_dataset = CustomDataset("data1\\train_sample_label.csv",
                                         "data1\\train_sample",
                                         28 * 28 * 5, (5, 1, 28, 28))

    test_custom_dataset = CustomDataset("data1\\test_sample_label.csv",
                                        "data1\\test_sample",
                                        28 * 28 * 5, (5, 1, 28, 28))

    train_custom_loader = DataLoader(train_custom_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
    test_custom_loader = DataLoader(test_custom_dataset, batch_size=batch_size,
                                    shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = CNNLSTM(2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_custom_loader, model, loss_fn, optimizer, 2)
        test_loop(test_custom_loader, model, loss_fn, 2)
        print("Done!")
    # Save Models
    torch.save(model.state_dict(), "cnn_lstm.pth")
    print("Saved PyTorch Model State to cnn_lstm.pth")


if __name__ == "__main__":
    # lstm_mnist_train()
    # alexnet_mnist_train()
    # LeNet_mnist_train()
    # cnnlstm_custom_train()
    # cnnlstm_mnist_train()
    AttBilstm_mnist_train()
