#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:50:44 2023

@author: Angel Ayala
"""
import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.nn import functional as F


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


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


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (np.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy.
    taken from https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py"""

    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        hidden_dim1 = hidden_dim
        if isinstance(obs_dim, int):
            f_layer = [nn.Linear(obs_dim, hidden_dim), nn.ReLU(inplace=True)]            
        elif len(obs_dim) == 2:
            f_layer = [nn.LSTM(obs_dim[-1], hidden_dim, num_layers=1,
                               batch_first=True, bidirectional=True)]
            hidden_dim1 = hidden_dim * 2
        elif len(obs_dim) == 1:
            f_layer = [nn.Linear(obs_dim[-1], hidden_dim), nn.ReLU(inplace=True)]            
        else:
            raise NotImplementedError("First layer not implemented.")
        self.f_layer = nn.Sequential(*f_layer)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim),  nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_dim)
        )

        self.apply(weight_init)

    def forward(self, obs):
        if isinstance(self.f_layer[0], nn.LSTM):
            h, _ = self.f_layer(obs)
            h = torch.relu(h)
            h = h[:, -1, :]
        else:
            h = self.f_layer(obs)

        mu, log_std = self.trunk(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist


class QFunction(nn.Module):
    """MLP for action-state function."""
    def __init__(self, obs_dim, action_dim, hidden_dim, preprocess=False):
        super().__init__()

        self.preprocess = preprocess

        if isinstance(obs_dim, int):
            input_dim = obs_dim + action_dim
            f_layer = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        elif len(obs_dim) == 2:
            input_dim = obs_dim[-1] + action_dim
            self.preprocess = nn.Sequential(
                nn.Conv1d(obs_dim[0], obs_dim[-1], kernel_size=obs_dim[-1]),
                nn.Tanh())
            f_layer = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]     
        elif len(obs_dim) == 1:
            input_dim = obs_dim[-1] + action_dim
            f_layer = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]     
        self.f_layer = nn.Sequential(*f_layer)

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        
        if self.preprocess:
            obs = self.preprocess(obs)
            if len(obs.shape) == 3:
                obs = obs.squeeze(-1)

        obs_action = torch.cat([obs, action], dim=1)
        h = self.f_layer(obs_action)
        if isinstance(self.f_layer[0], nn.Conv1d):
            h = h.squeeze(-1)

        return self.trunk(h)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, latent_dim, action_shape, hidden_dim, preprocess=False):
        super().__init__()

        self.Q1 = QFunction(latent_dim, action_shape[0], hidden_dim, preprocess=preprocess)
        self.Q2 = QFunction(latent_dim, action_shape[0], hidden_dim, preprocess=preprocess)

        self.apply(weight_init)

    def forward(self, obs, action):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2
