#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:06:16 2024

@author: Angel Ayala
"""

import torch
import torch.nn as nn

from uav_navigation.net import MLP


class Conv1dMLP(MLP):
    def __init__(self, state_shape, action_dim, hidden_dim, num_layers=2):
        super(Conv1dMLP, self).__init__(
            state_shape[-1], action_dim, hidden_dim, num_layers=num_layers)
        if len(state_shape) == 2:
            self.h_layers[0] = nn.Conv1d(state_shape[0], hidden_dim,
                                         kernel_size=state_shape[-1])

    def forward(self, obs):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = torch.relu(hidden_layer(h))
            if isinstance(hidden_layer, nn.Conv1d):
                h = h.squeeze(2)

        return self.h_layers[-1](h)


class Actor(Conv1dMLP):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256,
                 num_layers=2):
        super(Actor, self).__init__(
            state_dim, action_dim, hidden_dim, num_layers=2)

        self.max_action = max_action

    def forward(self, state):
        a = super().forward(state)
        return self.max_action * torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        if len(state_dim) == 2:
            state_action_dim = (state_dim[0], state_dim[1] + action_dim)
        elif len(state_dim) == 1:
            state_action_dim = (state_dim[0] + action_dim, )
        self.q1_network = Conv1dMLP(state_action_dim, 1, hidden_dim,
                                    num_layers=2)
        # Q2 architecture
        self.q2_network = Conv1dMLP(state_action_dim, 1, hidden_dim,
                                    num_layers=2)

    def forward(self, state, action):
        if state.dim() == 3:
            action = action.unsqueeze(1)
            action = action.repeat([1, state.shape[1], 1])
        sa = torch.cat([state, action], state.dim() - 1)
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)

        return q1, q2

    def Q1(self, state, action):
        if state.dim() == 3:
            action = action.unsqueeze(1)
            action = action.repeat([1, state.shape[1], 1])
        sa = torch.cat([state, action], state.dim() - 1)
        return self.q1_network(sa)
