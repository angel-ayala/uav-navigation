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
from torch.nn import functional as F
from adabelief_pytorch import AdaBelief

from .loss import circular_difference


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


def q_function(latent_dim, action_shape, hidden_dim, num_layers=2):
    return VectorDecoder(action_shape, latent_dim, hidden_dim, num_layers=num_layers)


class MLP(nn.Module):
    """MLP for q-function."""

    def __init__(self, n_input, n_output, hidden_dim, num_layers=3):
        super().__init__()
        self.h_layers = nn.ModuleList([nn.Linear(n_input, hidden_dim)])
        for i in range(num_layers - 1):
            self.h_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.h_layers.append(nn.Dropout(0.2))
        self.h_layers.append(nn.Linear(hidden_dim, n_output))
        self.num_layers = len(self.h_layers)

    def forward(self, obs, detach=False):
        h = obs
        for i in range(self.num_layers):
            if isinstance(self.h_layers[i], nn.Dropout):
                h = self.h_layers[i](h)
            else:
                h = torch.relu(self.h_layers[i](h))

        if detach:
            h = h.detach()

        return h


class VectorEncoder(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=3):
        super().__init__(state_shape[-1], latent_dim, hidden_dim,
                         num_layers=num_layers-1)
        if len(state_shape) == 2:
            self.h_layers.insert(0, nn.Conv1d(state_shape[0], state_shape[-1],
                                              kernel_size=state_shape[-1]))
        self.feature_dim = latent_dim
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, obs, detach=False):
        h = obs
        for hidden_layer in self.h_layers:
            h = hidden_layer(h)
            if isinstance(hidden_layer, nn.Conv1d):
                h = torch.tanh(h.squeeze(2))
            else:
                h = torch.relu(h)
        h_norm = self.ln(h)
        out = torch.tanh(h_norm)
        if detach:
            out.detach()
        return out


class VectorDecoder(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=3):
        super().__init__(latent_dim, state_shape[-1], hidden_dim,
                         num_layers=num_layers-1)
        if len(state_shape) == 2:
            self.h_layers[-1] = nn.ConvTranspose1d(hidden_dim, state_shape[0],
                                                   kernel_size=state_shape[-1])
        else:
            self.h_layers[-1] = nn.Linear(hidden_dim, state_shape[-1])

    def forward(self, obs, detach=False):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = torch.relu(hidden_layer(h))
        last_layer = self.h_layers[-1]
        if isinstance(last_layer, nn.ConvTranspose1d):
            h = h.unsqueeze(2)
        out = last_layer(h)
        return out


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

    def copy_conv_weights_from(self, source):
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
