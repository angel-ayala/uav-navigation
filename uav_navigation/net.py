#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:50:44 2023

@author: Angel Ayala
"""
import torch
import torch.nn as nn


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid], gain)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class MLP(nn.Module):
    """MLP for q-function."""

    def __init__(self, n_input, n_output, hidden_dim, num_layers=2):
        super(MLP, self).__init__()
        self.h_layers = nn.ModuleList([nn.Linear(n_input, hidden_dim)])
        for i in range(num_layers - 1):
            self.h_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.h_layers.append(nn.Linear(hidden_dim, n_output))
        self.num_layers = len(self.h_layers)

    def forward(self, obs):
        h = obs
        for h_layer in self.h_layers:
            h = torch.relu(h_layer(h))

        return h


class Conv1dMLP(MLP):
    def __init__(self, state_shape, out_dim, hidden_dim, num_layers=2):
        super(Conv1dMLP, self).__init__(
            state_shape[-1], out_dim, hidden_dim, num_layers=num_layers)
        if len(state_shape) == 2:
            self.h_layers[0] = nn.Conv1d(state_shape[0], hidden_dim,
                                         kernel_size=state_shape[-1])

    def forward_h(self, obs):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = torch.relu(hidden_layer(h))
            if isinstance(hidden_layer, nn.Conv1d):
                h = h.squeeze(2)
        return h

    def forward(self, obs):
        h = self.forward_h(obs)
        return self.h_layers[-1](h)


class QNetwork(Conv1dMLP):
    def __init__(self, state_shape, action_shape, hidden_dim, num_layers=2):
        super(QNetwork, self).__init__(
            state_shape, action_shape[-1], hidden_dim, num_layers)

        self.drop = nn.Dropout(p=0.5)
        self.apply(weight_init)

    def forward(self, obs):
        z = self.forward_h(obs)
        z = self.drop(z)
        return self.h_layers[-1](z)


class QFeaturesNetwork(nn.Module):
    feature_dim = 512

    def __init__(self, state_shape, action_shape, only_cnn=False):
        super().__init__()

        state_size = state_shape[0]
        action_size = action_shape[0]

        self._h1 = nn.Conv2d(state_size, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.feature_dim)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))

        if not only_cnn:
            self._h5 = nn.Linear(self.feature_dim, action_size)

            nn.init.xavier_uniform_(self._h5.weight,
                                    gain=nn.init.calculate_gain('linear'))
        self.only_cnn = only_cnn

    def forward_conv(self, state):
        h = torch.relu(self._h1(state / 255.))
        h = torch.relu(self._h2(h))
        h = torch.relu(self._h3(h))
        feats = torch.tanh(self._h4(h.view(-1, 3136)))
        return feats

    def forward(self, state, action=None):
        feats = self.forward_conv(state)
        if self.only_cnn:
            return feats
        q = self._h5(feats)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted
