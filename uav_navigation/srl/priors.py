#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:01:52 2024

@author: Angel Ayala
"""
import torch
# from torch import nn
from torch import optim
from torch.nn import functional as F
# from adabelief_pytorch import AdaBelief

from .net import adabelief_optimizer
from .net import MLP
from .net import VectorDecoder
from .loss import circular_difference


class PriorModel:
    def __init__(self, model, latent_source=['rgb'], obs_target=['vector']):
        self.model = model
        self.name = type(model).__name__
        self.optimizer = None
        self.latent_source = latent_source
        self.obs_target = obs_target
        # self.avg_model = optim.swa_utils.AveragedModel(model)

    # def sgd_optimizer(self, learning_rate, momentum=0.9, **kwargs):
    #     self.optimizer = sgd_optimizer(self.model, learning_rate=learning_rate,
    #                                     momentum=momentum, **kwargs)

    def adabelief_optimizer(self, learning_rate):
        self.optimizer = adabelief_optimizer(self.model,
                                              learning_rate=learning_rate)

    def adam_optimizer(self, learning_rate):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                                              # learning_rate=learning_rate)

    def obs2target(self, observations):
        return self.model.obs2target(observations)

    def compute_loss(self, values_pred, values_true):
        return self.model.compute_loss(values_pred, values_true)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()
        # self.avg_model.update_parameters(self.model)

    def __call__(self, x):
        return self.model(x)


class NorthBelief(VectorDecoder):
    def __init__(self, state_shape, latent_dim, hidden_dim=128,
                 num_layers=3):
        super().__init__((state_shape[0], 1), latent_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers)

    def obs2target(self, obs):
        return obs[:, :, 12].unsqueeze(2)

    def compute_loss(self, orientation, orientation_true):
        assert orientation.shape[-1] == orientation_true.shape[-1]
        orientation = orientation + 2 * torch.pi
        orientation_true = orientation_true + 2 * torch.pi
        loss = circular_difference(orientation, orientation_true)
        # loss = F.mse_loss(orientation, orientation_true)
        # loss = 1 - (F.cosine_similarity(orientation, orientation_true) + 1 ) / 2.
        # loss = loss.abs()
        return loss.mean() * 0.1


class PositionBelief(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim=128,
                 num_layers=3):
        super().__init__(latent_dim, 3, hidden_dim=hidden_dim,
                         num_layers=num_layers)

    def obs2target(self, obs):
        return obs[:, -1, 6:9]

    def compute_loss(self, position, position_true):
        # loss = F.smooth_l1_loss(position, position_true, beta=2.)
        loss = F.huber_loss(position, position_true, delta=2.)
        return loss.mean() * 0.1


class OrientationBelief(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim=128,
                 num_layers=3):
        super().__init__(latent_dim, 2, hidden_dim=hidden_dim,
                         num_layers=num_layers)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)

    def obs2target(self, obs):
        return obs[:, -1, :2]

    def compute_loss(self, inertial, inertial_true):
        inertial = inertial + 2 * torch.pi
        inertial_true = inertial_true + 2 * torch.pi
        # loss = 1 - (F.cosine_similarity(inertial, inertial_true) + 1 ) / 2.
        # loss = F.mse_loss(inertial, inertial_true)
        loss = circular_difference(inertial, inertial_true)
        # loss = loss.abs()
        return loss.mean()


class DistanceBelief(VectorDecoder):
# class DistanceBelief(MLP):
    def __init__(self, latent_dim, hidden_dim=128,
                 num_layers=1):
        super().__init__((3, 3), latent_dim, hidden_dim=hidden_dim,
                         num_layers=num_layers)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)

    def obs2target(self, obs):
        # print('obs', obs.shape)
        # print(obs[:, [-1], [-3, -2, -1]])
        d = obs[:, :, 6:9] - obs[:, :, -3:]
        # d = obs[:, :, -3:]
        return d

    def compute_loss(self, distance, distance_true):
        # inertial = inertial + 2 * torch.pi
        # inertial_true = inertial_true + 2 * torch.pi
        # loss = 1 - (F.cosine_similarity(inertial, inertial_true) + 1 ) / 2.
        assert distance.shape[-1] == distance_true.shape[-1]
        loss = F.mse_loss(distance, distance_true)
        # loss = circular_difference(inertial, inertial_true)
        # loss = loss.abs()
        return loss.mean()


class OdometryBelief(VectorDecoder):
    def __init__(self, state_shape, latent_dim, hidden_dim=128):
        # UAV vector + action
        input_shape = list(state_shape)
        input_shape[-1] = 13
        super().__init__(input_shape, latent_dim=latent_dim,
                         hidden_dim=hidden_dim, num_layers=2)
        self.latent_types = ['rgb']
        self.target_types = ['vector']

    # def forward(self, obs, detach=False):
    #     out = super().forward(obs, detach)
    #     # out = torch.tanh(out)
    #     return out

    def obs2target(self, obs):
        # inertial_diff = obs[:, :6] - obs_t1[:, :6]
        # translational_diff = obs[:, 6:12] - obs_t1[:, 6:12]
        # orientation_diff = obs[:, 12:13] - obs_t1[:, 12:13]
        # return torch.cat((inertial_diff, translational_diff, orientation_diff), dim=1)
        return obs[..., :13]