#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:19:08 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
from torch import nn

from uav_navigation.srl.autoencoder import PixelEncoder


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
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

    def __init__(self, n_input, n_output, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_output)
        )

    def forward(self, obs, detach=False):
        h = self.trunk(obs)
        if detach:
            h = h.detach()

        return h


class VectorApproximator(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 encoder_feature_dim=50,
                 num_layers=2,
                 num_filters=32,
                 hidden_dim=256,):
        super().__init__()

        n_output = output_shape[0]
        n_input = input_shape[0]
        self.encoder = MLP(n_input, encoder_feature_dim, hidden_dim)
        self.encoder.feature_dim = encoder_feature_dim
        self.Q = MLP(encoder_feature_dim, n_output, hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        z = self.encoder(obs, detach=detach_encoder)
        q = self.Q(z)

        return q


class PixelApproximator(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 encoder_feature_dim=50,
                 num_layers=2,
                 num_filters=32,
                 hidden_dim=256,):
        super().__init__()

        n_output = output_shape[0]
        self.encoder = PixelEncoder(input_shape, encoder_feature_dim,
                                    num_layers, num_filters)
        self.Q = MLP(self.encoder.feature_dim, n_output, hidden_dim)
        self.apply(weight_init)

    def forward(self, obs, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        z = self.encoder(obs, detach=detach_encoder)
        q = self.Q(z)

        return q
