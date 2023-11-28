#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:50:44 2023

@author: Angel Ayala
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        n_input = state_size[0]
        n_output = action_size[0]
        self.fc1 = nn.Linear(n_input, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 512)
        # self.fc4 = nn.Linear(16, 64)
        self.out = nn.Linear(512, n_output)
        self.leaky_relu = nn.LeakyReLU()
        self.drop = nn.Dropout1d(p=0.05)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = self.leaky_relu(self.fc4(x))
        x = self.drop(x)
        x = self.out(x)
        return x


class QFeaturesNetwork(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.n_features)
        self._h5 = nn.Linear(self.n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        h = torch.relu(self._h1(state))
        h = torch.relu(self._h2(h))
        h = torch.relu(self._h3(h))
        h = torch.relu(self._h4(h.view(-1, 3136)))
        q = self._h5(h)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted
