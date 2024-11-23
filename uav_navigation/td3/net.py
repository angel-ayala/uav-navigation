#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:06:16 2024

@author: Angel Ayala
"""

import torch
import torch.nn as nn

from uav_navigation.net import Conv1dMLP
from uav_navigation.net import weight_init
from uav_navigation.srl.net import EncoderWrapper


class Actor(Conv1dMLP):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256,
                 num_layers=2):
        super(Actor, self).__init__(
            state_dim, action_dim, hidden_dim, num_layers=2)

        self.max_action = max_action
        self.apply(weight_init)

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
        self.apply(weight_init)

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


class TD3EncoderWrapper(EncoderWrapper):
    def __init__(self, function, encoder, detach_encoder=True):
        super(TD3EncoderWrapper, self).__init__(function, encoder,
                                                detach_encoder)

    def Q1(self, obs, action):
        z = self.encoder(obs, detach=self.detach)
        q = self.function.Q1(z, action)
        return q
