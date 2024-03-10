#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:50:44 2023

@author: Angel Ayala
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        state_size = input_shape[-1]
        action_size = output_shape[0]
        if len(input_shape) == 2:
            self.conv1 = nn.Conv1d(input_shape[0], 32, kernel_size=state_size)
        else:
            self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 512)
        self.out = nn.Linear(512, action_size)
        self.drop = nn.Dropout1d(p=0.05)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        if hasattr(self, 'conv1'):
            x = self.conv1(x)
            x = x.squeeze(2)
        else:
            x = self.fc1(x)
        x = self.leaky_relu(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return x


class QFeaturesNetwork(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape):
        super().__init__()

        state_size = input_shape[0]
        action_size = output_shape[0]

        self._h1 = nn.Conv2d(state_size, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.n_features)
        self._h5 = nn.Linear(self.n_features, action_size)

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


class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, fc1_units=64, fc2_units=64):
        super(DuelingQNetwork, self).__init__()
        state_size = input_shape[0]
        action_size = output_shape[0]
        self.fc_val = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        val = self.fc_val(state)
        adv = self.fc_adv(state)

        # Dueling Network: combine value and advantage streams
        return val + (adv - adv.mean(dim=1, keepdim=True))
