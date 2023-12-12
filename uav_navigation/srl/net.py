#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:19:08 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
import torch
from torch import nn

from uav_navigation.srl.autoencoder import PixelEncoder


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    # if isinstance(m, nn.Linear):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
        super().__init__()

        self.feature_dim = n_output
        self.num_layers = num_layers
        self.h_layers = nn.ModuleList([nn.Linear(n_input, hidden_dim)])
        for i in range(num_layers - 1):
            self.h_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.h_layers.append(nn.Linear(hidden_dim, n_output))

    def forward(self, obs, detach=False):
        h = self.h_layers[0](obs)
        for i in range(self.num_layers):
            h = torch.relu(self.h_layers[i+1](h))

        if detach:
            h = h.detach()

        return h


class VectorApproximator(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 num_layers=2,
                 hidden_dim=256,
                 feature_dim=50):
        super().__init__()

        n_output = output_shape[0]
        n_input = input_shape[0]
        self.encoder = MLP(n_input, feature_dim, hidden_dim,
                           num_layers=num_layers)
        self.Q = MLP(feature_dim, n_output, hidden_dim,
                     num_layers=num_layers)

    def forward(self, obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        z = self.encoder(obs, detach=detach_encoder)
        q = self.Q(z)

        return q


class PixelApproximator(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 num_layers=2,
                 hidden_dim=256,
                 feature_dim=50,
                 num_filters=32):
        super().__init__()

        n_output = output_shape[0]
        self.encoder = PixelEncoder(input_shape, feature_dim,
                                    num_layers, num_filters)
        self.Q = MLP(self.encoder.feature_dim, n_output, hidden_dim)

    def forward(self, obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        z = self.encoder(obs, detach=detach_encoder)
        q = self.Q(z)

        return q
