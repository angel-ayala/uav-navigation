#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:28:24 2024

@author: Angel Ayala
"""

import torch.nn.functional as F
from torch import optim
from .net import preprocess_obs
from .net import rgb_reconstruction_model
from .net import vector_reconstruction_model
from .net import imu2pose_model
from uav_navigation.logger import summary_scalar


class AEModel:

    def __init__(self, ae_type, ae_params):
        self.type = ae_type
        if ae_type == 'rgb':
            self.encoder, self.decoder = rgb_reconstruction_model(
                ae_params['image_shape'],
                ae_params['latent_dim'],
                num_layers=ae_params['num_layers'],
                num_filters=ae_params['num_filters'])
        # Vector observation reconstruction autoencoder model
        if ae_type == 'vector':
            self.encoder, self.decoder = vector_reconstruction_model(
                ae_params['vector_shape'],
                ae_params['hidden_dim'],
                ae_params['latent_dim'],
                num_layers=ae_params['num_layers'])
        if ae_type == 'imu2pose':
            self.encoder, self.decoder = imu2pose_model(
                ae_params['imu_shape'],
                ae_params['pos_shape'],
                ae_params['hidden_dim'],
                ae_params['latent_dim'],
                num_layers=ae_params['num_layers'])

        if type(self.decoder) is not tuple:
            self.decoder = [self.decoder]

    def adam_optimizer(self, encoder_lr, decoders_lr, decoder_weight_decay):
        if type(decoders_lr) is not list:
            decoders_lr = [decoders_lr]
        self.encoder_optim = optim.Adam(self.encoder.parameters(),
                                        lr=encoder_lr,
                                        weight_decay=decoder_weight_decay)
        self.decoder_optim = [optim.Adam(self.decoder[i].parameters(),
                                         lr=decoder_lr,
                                         weight_decay=decoder_weight_decay)
                              for i, decoder_lr in enumerate(decoders_lr)]

    def sgd_optimizer(self, encoder_lr, decoders_lr, decoder_weight_decay):
        if type(decoders_lr) is not list:
            decoders_lr = [decoders_lr]
        self.encoder_optim = optim.SGD(self.encoder.parameters(),
                                       lr=encoder_lr,
                                       momentum=0.9,
                                       weight_decay=decoder_weight_decay)
        self.decoder_optim = [optim.SGD(self.decoder[i].parameters(),
                                        lr=decoder_lr,
                                        momentum=0.9,
                                        weight_decay=decoder_weight_decay)
                              for i, decoder_lr in enumerate(decoders_lr)]

    def to(self, device):
        self.encoder.to(device)
        for dec in self.decoder:
            dec.to(device)

    def __call__(self, observation):
        if self.type == 'imu2pose':
            z = self.encoder(observation[:, :6])
        else:
            z = self.encoder(observation)
        return z

    def reconstruct_obs(self, image):
        h = self.encoder(image)
        rec_obs = self.decoder[0](h)
        return rec_obs, h

    def apply(self, function):
        self.encoder.apply(function)
        for dec in self.decoder:
            dec.apply(function)

    def optimize_reconstruction(self, obs, decoder_latent_lambda):
        rec_obs, h = self.reconstruct_obs(obs)

        if obs.dim() <= 3:
            obs = (obs + 1) / 2.
        # preprocess images to be in [-0.5, 0.5] range
        target_obs = preprocess_obs(obs)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        rloss = rec_loss + decoder_latent_lambda * latent_loss

        summary_scalar('Loss/Decoder', rec_loss.item())
        summary_scalar('Loss/Encoder', latent_loss.item())
        summary_scalar('Loss/Reconstruction', rloss.item())

        # # Compute slowness cost
        # slowness_loss = slowness_cost(h)
        # loss += slowness_loss
        # summary_scalar('Loss/Slowness', slowness_loss.item())
        # # Compute variability cost
        # variability_loss = variability_cost(h)
        # loss += variability_loss
        # summary_scalar('Loss/Variability', variability_loss.item())

        # encoder optimizer
        self.encoder_optim.zero_grad()
        # decoder optimizer
        self.decoder_optim[0].zero_grad()
        rloss.backward()

        self.encoder_optim.step()
        self.decoder_optim[0].step()

        return rloss

    def reconstruct_pose(self, obs_vector):
        h = self.encoder(obs_vector[:, :6])  # attitude (IMU + Gyro) data only
        rec_att = self.decoder[0](h)
        rec_pos = self.decoder[1](h)
        return rec_att, rec_pos, h

    def optimize_pose(self, obs_vector, decoder_latent_lambda):
        rec_att, rec_pos, h = self.reconstruct_pose(obs_vector)

        if obs_vector.dim() == 2:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = preprocess_obs(obs_vector)
            target_att, target_pos = target_obs[:, :6], target_obs[:, 6:12]
        att_loss = F.mse_loss(target_att, rec_att)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # inertial gradients
        r_att_loss = att_loss + decoder_latent_lambda * latent_loss

        # position gradients
        pos_loss = F.mse_loss(target_pos, rec_pos)
        r_pos_loss = pos_loss + decoder_latent_lambda * latent_loss

        # encoder optimizer
        self.encoder_optim.zero_grad()
        # decoder optimizer
        self.decoder_optim[0].zero_grad()
        self.decoder_optim[1].zero_grad()

        r_att_loss.backward(retain_graph=True)
        r_pos_loss.backward()

        self.encoder_optim.step()
        self.decoder_optim[0].step()
        self.decoder_optim[1].step()

        return r_att_loss, r_pos_loss
