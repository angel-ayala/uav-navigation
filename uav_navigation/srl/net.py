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


OUT_DIM = {2: 39, 4: 35, 6: 31}


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


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def rgb_reconstruction_model(image_shape, latent_dim, num_layers=2,
                             num_filters=32):
    encoder = PixelEncoder(image_shape, latent_dim, num_layers=num_layers,
                           num_filters=num_filters)
    decoder = PixelDecoder(image_shape, latent_dim, num_layers=num_layers,
                           num_filters=num_filters)
    return encoder, decoder


def vector_reconstruction_model(vector_shape, hidden_dim, latent_dim,
                                num_layers=2):
    encoder = MLP(vector_shape[0], latent_dim, hidden_dim, num_layers=num_layers)
    decoder = MLP(latent_dim, vector_shape[0], hidden_dim, num_layers=num_layers)
    return encoder, decoder


def imu2pose_model(imu_shape, pos_shape, hidden_dim, latent_dim,
                   num_layers=2):
    encoder = MLP(imu_shape[0], latent_dim, hidden_dim, num_layers=num_layers)
    decoder1 = MLP(latent_dim, imu_shape[0], hidden_dim, num_layers=num_layers)
    decoder2 = MLP(latent_dim, pos_shape[0], hidden_dim, num_layers=num_layers)
    return encoder, (decoder1, decoder2)

def q_function(latent_dim, action_shape, hidden_dim, num_layers=2):
    return MLP(latent_dim, action_shape[0], hidden_dim, num_layers=num_layers)


class MLP(nn.Module):
    """MLP for q-function."""

    def __init__(self, n_input, n_output, hidden_dim, num_layers=2):
        super().__init__()

        # self.feature_dim = n_output
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


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, state_shape, latent_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(state_shape) == 3

        self.feature_dim = latent_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(state_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class PixelDecoder(nn.Module):
    def __init__(self, state_shape, latent_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            latent_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, state_shape[0], 3, stride=2, output_padding=1
            )
        )

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        obs = self.deconvs[-1](deconv)

        return obs
