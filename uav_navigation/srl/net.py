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


def slowness_cost(output_batch, lambda_=0.1):
    """
    Compute the slowness cost for a batch of output sequences.

    Parameters:
    - output_batch: A 2D PyTorch tensor representing the batch of output sequences.
                    Each row represents one output sequence.
    - lambda_: A regularization parameter controlling the strength of the penalty.

    Returns:
    - slowness: The average slowness cost over the batch.
    """
    batch_size = output_batch.size(0)
    slowness_total = 0.0

    for i in range(batch_size):
        output_sequence = output_batch[i]
        squared_diffs = torch.square(torch.diff(output_sequence))
        slowness_total += lambda_ * torch.sum(squared_diffs)

    slowness = slowness_total / batch_size
    return slowness


def variability_cost(encoded_batch):
    """
    Compute the variability loss for a batch of encoded representations.

    Parameters:
    - encoded_batch: A 2D PyTorch tensor representing the batch of encoded representations.
                     Each row represents one encoded representation.

    Returns:
    - variability: The variability loss.
    """
    mean_encoded = torch.mean(encoded_batch, dim=0)
    variance = torch.mean(torch.square(encoded_batch - mean_encoded))
    return variance


class MLP(nn.Module):
    """MLP for q-function."""

    def __init__(self, n_input, n_output, hidden_dim, num_layers=2):
        super().__init__()
        self.feature_dim = n_output
        self.num_layers = num_layers
        if type(n_input) != int and len(n_input) == 2:
            first_layer = nn.Conv1d(n_input[0], hidden_dim,
                                    kernel_size=n_input[-1])
        else:
            first_layer = nn.Linear(n_input, hidden_dim)
        self.h_layers = nn.ModuleList([first_layer])
        for i in range(num_layers - 1):
            self.h_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if type(n_output) != int and len(n_output) == 2:
            last_layer = nn.ConvTranspose1d(hidden_dim, n_output[0],
                                            kernel_size=n_output[-1])
        else:
            last_layer = nn.Linear(hidden_dim, n_output)

        self.h_layers.append(last_layer)

    def forward(self, obs, detach=False):
        h = self.h_layers[0](obs)
        if isinstance(self.h_layers[0], nn.Conv1d):
            h = h.squeeze(2)
        for i in range(self.num_layers):
            layer = self.h_layers[i+1]
            if isinstance(layer, nn.ConvTranspose1d):
                h = h.unsqueeze(2)
            h = torch.relu(layer(h))

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
        self.encoder = MLP(input_shape, feature_dim, hidden_dim,
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
