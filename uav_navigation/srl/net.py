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
from torch import optim
from adabelief_pytorch import AdaBelief

from uav_navigation.net import Conv1dMLP
from uav_navigation.net import MLP


OUT_DIM = {2: 39, 4: 35, 6: 31}


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def sgd_optimizer(model, learning_rate=1e-5, momentum=0.9, **kwargs):
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                     nesterov=True, **kwargs)


def adabelief_optimizer(model, learning_rate=1e-3):
    return AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16,
                     betas=(0.9, 0.999), weight_decouple=True, rectify=False,
                     print_change_log=False)


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
    encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim, num_layers=num_layers)
    decoder = VectorDecoder(vector_shape, latent_dim, hidden_dim, num_layers=num_layers)
    return encoder, decoder


def imu2pose_model(imu_shape, pos_shape, hidden_dim, latent_dim,
                   num_layers=2):
    encoder = MLP(imu_shape[0], latent_dim, hidden_dim, num_layers=num_layers)
    decoder1 = MLP(latent_dim, imu_shape[0], hidden_dim, num_layers=num_layers)
    decoder2 = MLP(latent_dim, pos_shape[0], hidden_dim, num_layers=num_layers)
    return encoder, (decoder1, decoder2)


class QNetworkWrapper(nn.Module):
    def __init__(self, q_network, encoder_fn):
        super(QNetworkWrapper, self).__init__()
        self.encoder = encoder_fn
        self.q_network = q_network

    def forward(self, obs):
        z = self.encoder(obs, detach=True)
        q = self.q_network(z)
        return q


class VectorEncoder(Conv1dMLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=2):
        super(VectorEncoder, self).__init__(
            state_shape, latent_dim, hidden_dim, num_layers=num_layers)
        self.feature_dim = latent_dim
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, obs, detach=False):
        z = torch.relu(super().forward(obs))
        if detach:
            z.detach()
        out = self.ln(self.fc(z))
        return torch.tanh(out)

    def copy_weights_from(self, source):
        """Tie hidden layers"""
        # only tie hidden layers
        for i in range(self.num_layers):
            tie_weights(src=source.h_layers[i], trg=self.h_layers[i])


class VectorDecoder(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=2):
        super(VectorDecoder, self).__init__(
            latent_dim, state_shape[-1], hidden_dim, num_layers=num_layers)
        if len(state_shape) == 2:
            self.h_layers[-1] = nn.ConvTranspose1d(hidden_dim, state_shape[0],
                                                   kernel_size=state_shape[-1])

    def forward(self, z):
        h = z
        for hidden_layer in self.h_layers[:-1]:
            h = torch.relu(hidden_layer(h))
        last_layer = self.h_layers[-1]
        if isinstance(last_layer, nn.ConvTranspose1d):
            h = h.unsqueeze(2)
        out = last_layer(h)
        return out


class VectorDiffDecoder(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=2):
        super(VectorDiffDecoder, self).__init__(
            latent_dim, state_shape[-1], hidden_dim, num_layers=num_layers)

    def forward(self, z):
        h = z
        for hidden_layer in self.h_layers[:-1]:
            h = torch.relu(hidden_layer(h))
        out = self.h_layers[-1](h)
        return out


class VectorMDPEncoder(VectorEncoder):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=2):
        super(VectorMDPEncoder, self).__init__(
            state_shape, latent_dim, hidden_dim, num_layers)
        self.contrastive = nn.Linear(latent_dim, latent_dim)

    def forward_code(self, obs, detach=False):
        code = self.forward(obs, detach=detach)
        h_fc = self.contrastive(code) + code
        return h_fc


class ChannelAttention(nn.Module):
    def __init__(self, num_filters, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // reduction_ratio, 1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(num_filters // reduction_ratio, num_filters, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze operation
        z = self.avg_pool(x)
        # Excitation operation
        z = self.fc(z)
        # Attention weights
        attention = torch.mul(x, z)
        return attention


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

        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h = h.view(h.size(0), -1)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        h = torch.tanh(h_norm)

        return h

    def copy_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class PixelMDPEncoder(PixelEncoder):
    def __init__(self, state_shape, latent_dim, num_layers=2, num_filters=32):
        super().__init__(state_shape, latent_dim, num_layers, num_filters)
        self.contrastive = nn.Linear(latent_dim, latent_dim)
        self.probabilities = nn.Linear(latent_dim, latent_dim)

    def forward_prob(self, obs, detach=False):
        h = self.forward(obs, detach=detach)
        h_fc = self.probabilities(h)
        return h_fc

    def forward_code(self, obs, detach=False):
        code = self.forward(obs, detach=detach)
        h_fc = self.contrastive(code) + code
        return h_fc


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
            self.deconvs.extend([
                # ChannelAttention(num_filters, reduction_ratio=8),
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            ])
        self.deconvs.extend([
            ChannelAttention(num_filters, reduction_ratio=8),
            nn.ConvTranspose2d(
                num_filters, state_shape[0], 3, stride=2, output_padding=1
            )
        ])
        self.num_layers = len(self.deconvs)

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            if type(self.deconvs[i]) == ChannelAttention:
                deconv_att = self.deconvs[i](deconv)
                deconv = torch.mul(deconv, deconv_att)
            else:
                deconv = torch.relu(self.deconvs[i](deconv))

        obs = self.deconvs[-1](deconv)

        return obs
