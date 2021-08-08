#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from models import LeNet
from train_torchmetrics import train_loop, test_loop
import optuna
from optuna.trial import TrialState

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epoches = 3

train_dataset = datasets.MNIST(
    root='./mnist', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./mnist', train=False, transform=transforms.ToTensor(),
    download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def LeNet_mnist_train(trial):
    test_accuracy = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNet(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # 优化器集合
    optimizer = trial.suggest_categorical('optimizer',
                                          [torch.optim.SGD,
                                           torch.optim.RMSprop,
                                           torch.optim.Adam])(
        model.parameters(), lr=trial.suggest_loguniform('lr', 1e-3, 1e-2))

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        # 需要将验证集上的accuray作为返回值传出
        test_accuracy = test_loop(test_loader, model, loss_fn)
        print("Done!")
        # Save Models
    torch.save(model.state_dict(), "LeNet.pth")
    print("Saved PyTorch Model State to LeNet.pth")
    return test_accuracy


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=LeNet_mnist_train, n_trials=3)
    pruned_trials = study.get_trials(deepcopy=False,
                                     states=tuple([TrialState.PRUNED]))
    complete_trials = study.get_trials(deepcopy=False,
                                       states=tuple([TrialState.COMPLETE]))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
