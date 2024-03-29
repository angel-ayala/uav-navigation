#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:28:24 2024

@author: Angel Ayala
"""

import torch
import torchvision
from torch.nn import functional as F
from torch import optim
from thop import clever_format
from uav_navigation.utils import profile_model
from uav_navigation.logger import summary_scalar
from uav_navigation.logger import summary_image

from .net import preprocess_obs
from .net import rgb_reconstruction_model
from .net import vector_reconstruction_model
from .net import imu2pose_model
from .net import slowness_cost
from .net import variability_cost
from .net import proportionality_cost
from .net import repeatability_cost
from .net import adabelief_optimizer
from .net import MS_SSIM_Loss, SSIM_Loss
from .net import logarithmic_difference_loss
from .net import BiGRU



def profile_ae_model(ae_model, state_shape, device):
    total_flops, total_params = 0, 0
    flops, params = profile_model(ae_model.encoder, state_shape, device)
    total_flops += flops
    total_params += params
    print('Encoder {}: {} flops, {} params'.format(
        ae_model.type, *clever_format([flops, params], "%.3f")))
    # profile decode stage
    for i, decoder in enumerate(ae_model.decoder):
        flops, params = profile_model(decoder, ae_model.encoder.feature_dim, device)
        total_flops += flops
        total_params += params
        print('Decoder {} ({}): {} flops, {} params'.format(
            ae_model.type, i, *clever_format([flops, params], "%.3f")))
    return total_flops, total_params


class AEModel:
    LOG_FREQ = 50

    def __init__(self, ae_type, ae_params, encoder_only=False):
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

        if encoder_only:
            self.decoder = list()
        elif type(self.decoder) is not tuple:
            self.decoder = [self.decoder]
        self.n_calls = 0
        self.ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True)
        self.ms_ssim_loss = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)

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
                                       weight_decay=decoder_weight_decay,
                                       nesterov=True)
        if len(self.decoder) > 0:
            self.decoder_optim = [optim.SGD(self.decoder[i].parameters(),
                                            lr=decoder_lr,
                                            momentum=0.9,
                                            weight_decay=decoder_weight_decay,
                                            nesterov=True)
                                  for i, decoder_lr in enumerate(decoders_lr)]
        else:
            self.decoder_optim = list()

    def adabelief_optimizer(self, encoder_lr, decoders_lr, decoder_weight_decay):
        if type(decoders_lr) is not list:
            decoders_lr = [decoders_lr]
        self.encoder_optim = adabelief_optimizer(self.encoder,
                                                 learning_rate=encoder_lr)
        if len(self.decoder) > 0:
            self.decoder_optim = [adabelief_optimizer(self.decoder[i],
                                                      learning_rate=decoder_lr)
                                  for i, decoder_lr in enumerate(decoders_lr)]
        else:
            self.decoder_optim = list()

    def to(self, device):
        self.encoder.to(device)
        for dec in self.decoder:
            dec.to(device)

    def create_bim(self, state_shape, hidden_size=10, learning_rate=1e-4):
        self.bim = BiGRU(state_shape, state_shape, hidden_size)
        self.bim_optim = optim.SGD(self.bim.parameters(), lr=learning_rate,
                                   momentum=0.9, nesterov=True)

    def update_bim(self, h_t, h_t1):
        temp_series = torch.cat((h_t, h_t1), dim=1)
        hat_obs_t = self.bim(temp_series)
        bim_loss = F.mse_loss(hat_obs_t, h_t1)
        summary_scalar(f'Loss/{self.type}/BiGRU', bim_loss.item())
        return bim_loss

    def __call__(self, observation, detach=False):
        if self.type == 'imu2pose':
            z = self.encoder(observation[:, :6])
        else:
            z = self.encoder(observation)
        if detach:
            z = z.detach()
        return z

    def reconstruct_obs(self, observation):
        h = self.encoder(observation)
        rec_obs = self.decoder[0](h)
        return rec_obs, h

    def apply(self, function):
        self.encoder.apply(function)
        for dec in self.decoder:
            dec.apply(function)

    # def encoder_slowness(self, h_t, h_t1):
    #     # Compute slowness cost
    #     slowness_loss = slowness_cost(h_t, h_t1)
    #     summary_scalar(f'Loss/{self.type}/Encoder/Slowness', slowness_loss.item())
    #     return slowness_loss

    # def encoder_variability(self, h_t, h_t1):
    #     # Compute slowness cost
    #     variability_loss = variability_cost(h_t, h_t1)
    #     summary_scalar(f'Loss/{self.type}/Encoder/Variability', variability_loss.item())
    #     return variability_loss

    # def encoder_proportionality(self, h_t, h_t1, actions):
    #     # Compute slowness cost
    #     proportionality_loss = proportionality_cost(h_t, h_t1, actions)
    #     summary_scalar(f'Loss/{self.type}/Encoder/Proportionality', proportionality_loss.item())
    #     return proportionality_loss

    # def encoder_repeatability(self, h_t, h_t1, actions):
    #     # Compute slowness cost
    #     repeatability_loss = repeatability_cost(h_t, h_t1, actions)
    #     summary_scalar(f'Loss/{self.type}/Encoder/Repeatibility', repeatability_loss.item())
    #     return repeatability_loss

    def compute_state_priors(self, obs_t, actions, rewards, obs_t1):
        actions_argmax = actions.argmax(-1)
        s_t = self(obs_t)
        s_t1 = self(obs_t1)

        state_diff = s_t1 - s_t
        state_diff_norm = torch.norm(state_diff, p=2., dim=-1)
        similarity = lambda x, y: torch.norm(x - y, p=2., dim=-1)

        slowness_loss = (state_diff_norm ** 2).mean()
        # variability_loss = torch.exp(-similarity(s_t, s_t1)).mean()

        # equal actions
        actions_unique, actions_count = torch.unique(actions_argmax, return_counts=True)
        actions_unique = actions_unique[actions_count > 1]
        causality_loss = 0.
        proportionality_loss = 0.
        repeatability_loss = 0.
        for a in actions_unique:
            actions_mask = actions_argmax == a
            proportionality_loss += (torch.diff(state_diff_norm[actions_mask], dim=0) ** 2.).mean()
            repeatability_loss += torch.exp(-similarity(s_t1[actions_mask], s_t[actions_mask]) ** 2).mean() *\
                (torch.norm(torch.diff(state_diff[actions_mask], dim=0), p=2., dim=-1) ** 2.).mean()
            # print('rewards', rewards[actions_mask].shape)
            # print('rewards', actions_mask.nonzero())
            # print('rewards2', torch.diff(rewards[actions_mask], dim=0).shape)
            # print('rewards2', torch.diff(rewards[actions_mask], dim=0))
            # causality_mask = torch.logical_and(actions_mask, torch.diff(rewards[actions_mask], dim=0) != 0.)
            # if torch.diff(rewards[actions_mask], dim=0) != 0.:
            for i, r_diff in enumerate(torch.diff(rewards[actions_mask], dim=0)):
                amask1 = actions_mask.nonzero().flatten()[i]
                amask2 = actions_mask.nonzero().flatten()[i+1]
                if r_diff != 0:
                    causality_loss += torch.exp(-similarity(s_t1[amask1], s_t1[amask2]) ** 2).mean()
        summary_scalar(f'Loss/{self.type}/Encoder/Slowness', slowness_loss.item())
        # summary_scalar(f'Loss/{self.type}/Encoder/Variability', variability_loss.item())
        summary_scalar(f'Loss/{self.type}/Encoder/Causality', causality_loss.item())
        summary_scalar(f'Loss/{self.type}/Encoder/Proportionality', proportionality_loss.item())
        summary_scalar(f'Loss/{self.type}/Encoder/Repeatibility', repeatability_loss.item())
        # state_priors_loss = slowness_loss + variability_loss + proportionality_loss + repeatability_loss
        state_priors_loss = slowness_loss + causality_loss + proportionality_loss + repeatability_loss
        summary_scalar(f'Loss/{self.type}/Encoder/S+V+P+R', state_priors_loss.item())
        return state_priors_loss

    # def compute_srl_loss(self, obs_t, obs_t1, actions):
    #     actions_argmax = actions.argmax(-1)
    #     h_t = self(obs_t)
    #     h_t1 = self(obs_t1)

    #     srl_loss = self.encoder_slowness(h_t, h_t1)
    #     # srl_loss += self.encoder_variability(h_t, h_t1)
    #     # srl_loss += self.encoder_proportionality(h_t, h_t1, actions_argmax)
    #     # srl_loss += self.encoder_repeatability(h_t, h_t1, actions_argmax)
    #     # enc_loss *= 1e-3
    #     summary_scalar(f'Loss/{self.type}/Encoder/S+V+P+R', srl_loss.item())
    #     return srl_loss

    def compute_reconstruction_loss(self, obs, obs_augm, decoder_latent_lambda):
        rec_obs, h = self.reconstruct_obs(obs_augm)
        if self.n_calls % self.LOG_FREQ == 0:
            rec_seq = rec_obs[-1]
            rec_frames = rec_seq.reshape((3, rec_seq.shape[0] // 3, rec_seq.shape[1], rec_seq.shape[2]))
            img_grid = torchvision.utils.make_grid(rec_frames + 0.5)
            summary_image(f"Agent/reconstruction", img_grid)
            obs_frames = obs[-1].reshape((3, obs[-1].shape[0] // 3, obs[-1].shape[1], obs[-1].shape[2]))
            obs_grid = torchvision.utils.make_grid(obs_frames / 255.)
            summary_image(f"Agent/observation", obs_grid)
            obs_frames = obs_augm[-1].reshape((3, obs[-1].shape[0] // 3, obs[-1].shape[1], obs[-1].shape[2]))
            obs_grid = torchvision.utils.make_grid(obs_frames / 255.)
            summary_image(f"Agent/observation_augm", obs_grid)

        if obs.dim() <= 3:
            obs = (obs + 1) / 2.
        # preprocess images to be in [-0.5, 0.5] range
        target_obs = preprocess_obs(obs)
        rec_obs = rec_obs.reshape(rec_obs.shape[0]*3, 3, rec_obs.shape[-2], rec_obs.shape[-1])
        target_obs = target_obs.reshape(target_obs.shape[0]*3, 3, target_obs.shape[-2], target_obs.shape[-1])
        rec_loss = F.mse_loss(rec_obs, target_obs)
        # rec_loss = F.smooth_l1_loss(rec_obs + 0.5, target_obs + 0.5, beta=0.1)  # good choice
        summary_scalar(f'Loss/{self.type}/ReconstructionLoss', rec_loss.item())
        log_diff_loss = logarithmic_difference_loss(rec_obs + 0.5, target_obs + 0.5, gamma=0.2)
        summary_scalar(f'Loss/{self.type}/LogDiff', log_diff_loss.item())
        rec_loss += log_diff_loss * 1e-6 # 0.000001
        ssim_loss = self.ssim_loss(rec_obs + 0.5, target_obs + 0.5)
        summary_scalar(f'Loss/{self.type}/SSIM', ssim_loss.item())
        rec_loss += ssim_loss * 1e-6  # 0.00001
        
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()
        summary_scalar(f'Loss/{self.type}/EncoderActivationL2', latent_loss.item())

        rloss = rec_loss + decoder_latent_lambda * latent_loss
        summary_scalar(f'Loss/{self.type}/EncoderDecoderLoss', rloss.item())
        self.n_calls += 1
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

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()
        summary_scalar(f'Loss/{self.type}/EncoderActivation', latent_loss.item())

        # inertial gradients
        att_loss = F.mse_loss(target_att, rec_att)
        summary_scalar(f'Loss/{self.type}/AttitudeDecoder', att_loss.item())

        r_att_loss = att_loss + decoder_latent_lambda * latent_loss
        summary_scalar(f'Loss/{self.type}/AttitudeReconstruction', r_att_loss.item())

        # position gradients
        pos_loss = F.mse_loss(target_pos, rec_pos)
        summary_scalar(f'Loss/{self.type}/PositionDecoder', pos_loss.item())

        r_pos_loss = pos_loss + decoder_latent_lambda * latent_loss
        summary_scalar(f'Loss/{self.type}/PositionReconstruction', r_pos_loss.item())

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
