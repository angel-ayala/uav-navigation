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
from uav_navigation.net import weight_init
from uav_navigation.net import Conv1dMLP


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


class DiagGaussianActor(Conv1dMLP):
    """torch.distributions implementation of an diagonal Gaussian policy.
    taken from https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py"""

    def __init__(self, state_dim, action_dim, hidden_dim, log_std_bounds):
        super(DiagGaussianActor, self).__init__(
            state_dim, 2 * action_dim, hidden_dim, num_layers=2)

        self.log_std_bounds = log_std_bounds
        self.apply(weight_init)

    def forward(self, obs):
        h = self.forward_h(obs)
        mu, log_std = self.h_layers[-1](h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    """MLP for Critic function."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        if len(state_dim) == 2:
            state_action_dim = (state_dim[0], state_dim[1] + action_dim)
        elif len(state_dim) == 1:
            state_action_dim = (state_dim[0] + action_dim, )
        # Q1 architecture
        self.q1_network = Conv1dMLP(state_action_dim, 1, hidden_dim,
                                    num_layers=2)
        # Q2 architecture
        self.q2_network = Conv1dMLP(state_action_dim, 1, hidden_dim,
                                    num_layers=2)
        self.apply(weight_init)

    def forward(self, state, action):
        if state.ndim == 3:
            action = action.unsqueeze(1)
            action = action.repeat([1, state.shape[1], 1])
        sa = torch.cat([state, action], state.ndim - 1)
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)

        return q1, q2
