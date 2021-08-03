#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


# 定义 单双向LSTM 模型
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, bidirectional):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True,
                            bidirectional=bidirectional)
        if self.bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2, n_class)
        else:
            self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):

        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.classifier(out)
        return out


class Attention(nn.Module):
    def __init__(self, rnn_size: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        # eq.11: r = H
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


class AttBiLSTM(nn.Module):
    def __init__(
            self,
            n_classes: int,
            emb_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float
    ):
        super(AttBiLSTM, self).__init__()

        self.rnn_size = rnn_size

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.attention = Attention(rnn_size)
        self.fc = nn.Linear(rnn_size, n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        rnn_out, _ = self.BiLSTM(x)

        H = rnn_out[:, :, : self.rnn_size] + rnn_out[:, :, self.rnn_size:]

        # attention module
        r, alphas = self.attention(
            H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        h = self.tanh(r)  # (batch_size, rnn_size)

        scores = self.fc(self.dropout(h))  # (batch_size, n_classes)

        return scores


# 定义AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(32 * 5 * 5, 120),
                                        nn.Linear(120, 84),
                                        nn.Linear(84, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.classifier(x)
        return x


class LeNetVariant(nn.Module):
    def __init__(self):
        super(LeNetVariant, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Linear(32 * 5 * 5, 120),
                                        nn.Linear(120, 84))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.classifier(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = LeNetVariant()
        self.lstm = nn.LSTM(input_size=84, hidden_size=128, num_layers=2,
                            batch_first=True)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        cnn_output_list = list()
        for t in range(x_3d.size(1)):
            cnn_output_list.append(self.cnn(x_3d[:, t, :, :, :]))
        x = torch.stack(tuple(cnn_output_list), dim=1)
        out, hidden = self.lstm(x)
        x = out[:, -1, :]
        x = F.relu(x)
        x = self.fc1(x)
        return x
