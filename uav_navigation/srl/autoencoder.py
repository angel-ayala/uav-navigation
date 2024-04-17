#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:28:24 2024

@author: Angel Ayala
"""

import torch
from torch.nn import functional as F
from torch import optim
from thop import clever_format
from info_nce import InfoNCE
from uav_navigation.utils import profile_model
from uav_navigation.utils import soft_update_params
from uav_navigation.logger import summary_scalar
from uav_navigation.logger import summary_image

from .net import preprocess_obs
from .net import rgb_reconstruction_model
from .net import PixelDecoder
from .net import PixelMDPEncoder
from .net import vector_reconstruction_model
# from .net import imu2pose_model
# from .net import slowness_cost
# from .net import variability_cost
# from .net import proportionality_cost
# from .net import repeatability_cost
from .net import adabelief_optimizer
from .loss import SSIM_Loss
# from .net import logarithmic_difference_loss
# from .net import BiGRU
from ..logger import log_image_batch


def profile_ae_model(ae_model, state_shape, device):
    total_flops, total_params = 0, 0
    feature_dim = 0
    for i, encoder in enumerate(ae_model.encoder):
        flops, params = profile_model(encoder, state_shape, device)
        total_flops += flops
        total_params += params
        print('Encoder {}({}): {} flops, {} params'.format(
            ae_model.type, i, *clever_format([flops, params], "%.3f")))
        feature_dim += encoder.feature_dim
    # profile decode stage
    for i, decoder in enumerate(ae_model.decoder):
        flops, params = profile_model(decoder, feature_dim, device)
        total_flops += flops
        total_params += params
        print('Decoder {} ({}): {} flops, {} params'.format(
            ae_model.type, i, *clever_format([flops, params], "%.3f")))
    return total_flops, total_params


class AEModel:
    LOG_FREQ = 50

    def __init__(self, ae_type): #, ae_params, encoder_only=False):
        self.type = ae_type

        self.encoder = list()
        self.decoder = list()

        self.n_calls = 0
        self.ssim_loss = SSIM_Loss(data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True)

    def adam_optimizer(self, encoders_lr, decoders_lr, decoder_weight_decay):
        if type(encoders_lr) is not list:
            encoders_lr = [encoders_lr]
        if type(decoders_lr) is not list:
            decoders_lr = [decoders_lr]
        self.encoder_optim = [optim.Adam(self.encoder[i].parameters(),
                                        lr=encoder_lr,
                                        weight_decay=decoder_weight_decay)
                              for i, encoder_lr in enumerate(encoders_lr)]
        self.decoder_optim = [optim.Adam(self.decoder[i].parameters(),
                                         lr=decoder_lr,
                                         weight_decay=decoder_weight_decay)
                              for i, decoder_lr in enumerate(decoders_lr)]

    def sgd_optimizer(self, encoders_lr, decoders_lr, decoder_weight_decay):
        if type(encoders_lr) is not list:
            encoders_lr = [encoders_lr]
        if type(decoders_lr) is not list:
            decoders_lr = [decoders_lr]
        self.encoder_optim = [optim.SGD(self.encoder[i].parameters(),
                                       lr=encoder_lr,
                                       momentum=0.9,
                                       weight_decay=decoder_weight_decay,
                                       nesterov=True)
                              for i, encoder_lr in enumerate(encoders_lr)]
        if len(self.decoder) > 0:
            self.decoder_optim = [optim.SGD(self.decoder[i].parameters(),
                                            lr=decoder_lr,
                                            momentum=0.9,
                                            weight_decay=decoder_weight_decay,
                                            nesterov=True)
                                  for i, decoder_lr in enumerate(decoders_lr)]
        else:
            self.decoder_optim = list()

    def adabelief_optimizer(self, encoders_lr, decoders_lr, decoder_weight_decay):
        if type(encoders_lr) is not list:
            encoders_lr = [encoders_lr]
        if type(decoders_lr) is not list:
            decoders_lr = [decoders_lr]
        self.encoder_optim = [adabelief_optimizer(self.encoder[i],
                                                  learning_rate=encoder_lr)
                              for i, encoder_lr in enumerate(encoders_lr)]
        if len(self.decoder) > 0:
            self.decoder_optim = [adabelief_optimizer(self.decoder[i],
                                                      learning_rate=decoder_lr)
                                  for i, decoder_lr in enumerate(decoders_lr)]
        else:
            self.decoder_optim = list()

    def apply(self, function):
        for enc in self.encoder:
            enc.apply(function)
        for dec in self.decoder:
            dec.apply(function)

    def to(self, device):
        for enc in self.encoder:
            enc.to(device)
        for dec in self.decoder:
            dec.to(device)

    def encoder_optim_zero_grad(self):
        for e in self.encoder_optim:
            e.zero_grad()

    def encoder_optim_step(self):
        for e in self.encoder_optim:
            e.step()

    def decoder_optim_zero_grad(self):
        for d in self.decoder_optim:
            d.zero_grad()

    def decoder_optim_step(self):
        for d in self.decoder_optim:
            d.step()

    def encode_obs(self, observation, encoder_idx=0, detach=False):
        z = self.encoder[encoder_idx](observation)
        if detach:
            z = z.detach()
        return z

    def decode_latent(self, z, decoder_idx=0, detach=False):
        z = self.decoder[decoder_idx](z)
        if detach:
            z = z.detach()
        return z

    def reconstruct_obs(self, observation, ae_idx=0, detach=False):
        h = self.encode_obs(observation, encoder_idx=ae_idx, detach=detach)
        rec_obs = self.decode_latent(h, decoder_idx=ae_idx, detach=detach)
        return rec_obs, h

    def compute_state_priors(self, obs_t, actions, rewards, obs_t1):
        actions_argmax = actions.argmax(-1)
        s_t = self.encode_obs(obs_t)
        s_t1 = self.encode_obs(obs_t1)

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
            for i, r_diff in enumerate(torch.diff(rewards[actions_mask], dim=0)):
                amask1 = actions_mask.nonzero().flatten()[i]
                amask2 = actions_mask.nonzero().flatten()[i+1]
                if r_diff != 0:
                    causality_loss += torch.exp(-similarity(s_t1[amask1], s_t1[amask2]) ** 2).mean()
        summary_scalar(f'Loss/{self.type}/Slowness', slowness_loss.item())
        # summary_scalar(f'Loss/{self.type}/Encoder/Variability', variability_loss.item())
        summary_scalar(f'Loss/{self.type}/Causality', causality_loss.item())
        summary_scalar(f'Loss/{self.type}/Proportionality', proportionality_loss.item())
        summary_scalar(f'Loss/{self.type}/Repeatibility', repeatability_loss.item())
        # state_priors_loss = slowness_loss + variability_loss + proportionality_loss + repeatability_loss
        state_priors_loss = slowness_loss + causality_loss + proportionality_loss + repeatability_loss
        summary_scalar(f'Loss/{self.type}/S+V+P+R', state_priors_loss.item())
        return state_priors_loss        

    def compute_reconstruction_loss(self, obs, obs_augm, decoder_latent_lambda, pixel_obs_log=False):
        # print('AEModel obs', obs.shape, obs.min(), obs.max())
        rec_obs, h = self.reconstruct_obs(obs_augm)
        
        if len(obs.shape) == 3:
            # [-1, 1] -> [-0.5, 0.5]
            true_obs = obs
            output_obs = rec_obs

        if len(obs.shape) == 4:
            # preprocess images to be in [-0.5, 0.5] range
            true_obs = preprocess_obs(obs)
            # de-stack
            obs_shape = obs.shape
            n_stack = obs_shape[1] // 3
            r_shape = (obs_shape[0] * n_stack, 3) + obs_shape[-2:]
            true_obs = true_obs.reshape(r_shape)
            output_obs = rec_obs.reshape(r_shape)

        rec_loss = F.mse_loss(true_obs, output_obs) #* 10
        summary_scalar(f'Loss/Reconstruction/{self.type}/MSE', rec_loss.item())
        # bce_loss = F.binary_cross_entropy(rec_obs + 0.5, target_obs + 0.5)
        # summary_scalar(f'Loss/Reconstruction/{self.type}/BCE', bce_loss.item())
        # rec_loss += bce_loss # * 1e-3
        # rec_loss = F.smooth_l1_loss(rec_obs + 0.5, target_obs + 0.5, beta=0.1)  # good choice
        # summary_scalar(f'Loss/{self.type}/ReconstructionLoss', rec_loss.item())
        # log_diff_loss = logarithmic_difference_loss(rec_obs + 0.5, target_obs + 0.5, gamma=0.2)
        # summary_scalar(f'Loss/{self.type}/LogDiff', log_diff_loss.item())
        # rec_loss += log_diff_loss * 1e-6 # 0.000001
        # ssim_loss = self.ssim_loss(rec_obs + 0.5, target_obs + 0.5)
        # summary_scalar(f'Loss/{self.type}/SSIM', ssim_loss.item())
        # rec_loss += ssim_loss * 1e-6  # 0.00001

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()
        summary_scalar(f'Loss/Encoder/{self.type}/L2', latent_loss.item())

        rloss = rec_loss + decoder_latent_lambda * latent_loss
        # summary_scalar(f'Loss/Reconstruction/{self.type}', rloss.item())

        if pixel_obs_log:
            log_image_batch(obs, "Agent/observation")
            log_image_batch(obs_augm, "Agent/observation_augm")
            log_image_batch(rec_obs + 0.5, "Agent/reconstruction")

        self.n_calls += 1
        return rloss


class RGBModel(AEModel):
    def __init__(self, model_params, encoder_only=False):
        super(RGBModel, self).__init__('RGB')
        rgb_encoder, rgb_decoder = rgb_reconstruction_model(
            model_params['image_shape'],
            model_params['latent_dim'],
            num_layers=model_params['num_layers'],
            num_filters=model_params['num_filters'])
        self.encoder.append(rgb_encoder)
        if not encoder_only:
            self.decoder.append(rgb_decoder)
        # self.avg_encoder = optim.swa_utils.AveragedModel(self.encoder[0])

    def encoder_optim_step(self):
        super().encoder_optim_step()
        # self.avg_encoder.update_parameters(self.encoder[0])

    def compute_loss(self, obs, obs_augm, decoder_latent_lambda):
        return self.compute_reconstruction_loss(obs, obs_augm, decoder_latent_lambda,
                                                pixel_obs_log=self.n_calls % self.LOG_FREQ == 0)


class VectorModel(AEModel):
    def __init__(self, model_params, encoder_only=False):
        super(VectorModel, self).__init__('Vector')
        vector_encoder, vector_decoder = vector_reconstruction_model(
            model_params['vector_shape'],
            model_params['hidden_dim'],
            model_params['latent_dim'],
            num_layers=model_params['num_layers'])
        self.encoder.append(vector_encoder)
        if not encoder_only:
            self.decoder.append(vector_decoder)
        # self.avg_encoder = optim.swa_utils.AveragedModel(self.encoder[0])

    def encoder_optim_step(self):
        super().encoder_optim_step()
        # self.avg_encoder.update_parameters(self.encoder[0])

    def compute_loss(self, obs, obs_augm, decoder_latent_lambda):
        # TODO: process each variable grouply or elementwise?
        return self.compute_reconstruction_loss(obs, obs_augm, decoder_latent_lambda)


class ATCModel(AEModel):
    def __init__(self, model_params, encoder_only=True):
        super(ATCModel, self).__init__('ATC')
        self.encoder.append(PixelMDPEncoder(
            model_params['image_shape'], model_params['latent_dim'],
            num_layers=model_params['num_layers'], num_filters=model_params['num_filters']))
        self.momentum_encoder = PixelMDPEncoder(
            model_params['image_shape'], model_params['latent_dim'],
            num_layers=model_params['num_layers'], num_filters=model_params['num_filters'])
        if not encoder_only:
            self.decoder.append(PixelDecoder(
                model_params['image_shape'], model_params['latent_dim'],
                num_layers=model_params['num_layers'], num_filters=model_params['num_filters']))

        self.momentum_encoder.load_state_dict(self.encoder[0].state_dict())
        self.loss = InfoNCE()
        self.avg_encoder = optim.swa_utils.AveragedModel(self.encoder[0])

    def encoder_optim_step(self):
        super().encoder_optim_step()
        self.avg_encoder.update_parameters(self.encoder[0])

    def update_momentum_encoder(self, tau):
        # Soft update the target network
        soft_update_params(net=self.encoder[0],
                           target_net=self.momentum_encoder,
                           tau=tau)
    def to(self, device):
        super().to(device)
        self.momentum_encoder.to(device)

    def compute_loss(self, obs_augm, obs_t1_augm):
        return self.compute_contrastive_loss(obs_augm, obs_t1_augm)

    def compute_contrastive_loss(self, obs_augm, obs_t1_augm):
        """Compute Augmented Temporal Contrast loss function.

        based on https://arxiv.org/pdf/2009.08319.pdf
        """
        z_t = self.encoder[0].forward_code(obs_augm)  # query
        z_t1 = self.momentum_encoder(obs_t1_augm)  # positive keys
        # TODO: positive keys from positive rewards and viceversa
        nceloss = self.loss(z_t, z_t1)
        summary_scalar(f'Loss/Contrastive/{self.type}/InfoNCE', nceloss.item())
        return nceloss * 1e-8


class ATCRGBModel(ATCModel):
    def __init__(self, model_params):
        super(LossModel, self).__init__(model_params, encoder_only=False)
        self.type = 'ATC-RGB'


class LossModel(ATCModel):
    def __init__(self, model_params, encoder_only=True):
        super(LossModel, self).__init__(model_params, encoder_only=encoder_only)
        self.type = 'ATCLoss'

    def compute_compression_loss(self, obs, obs_t1, temperature=1):
        """Compute compression loss.

        based on https://arxiv.org/pdf/2106.01655.pdf
        """
        # z_t = self.encoder[0](obs)
        # z_t1 = self.momentum_encoder(obs_t1)
        z_t = self.encoder[0].forward_prob(obs)
        z_t1 = self.momentum_encoder.forward_prob(obs_t1)
        # z_t_norm = (z_t + 1.) / 2.
        # z_t1_norm = (z_t1 + 1.) / 2.
        # z_t_norm = F.softmax(z_t, dim=1)
        # z_t1_norm = F.softmax(z_t1, dim=1)
        z_t_norm = torch.clamp(F.softmax(z_t / temperature, dim=-1), 1e-9, 1 - (z_t.shape[0] * 1e-9))
        z_t1_norm = torch.clamp(F.softmax(z_t1 / temperature, dim=-1), 1e-9, 1 - (z_t1.shape[0] * 1e-9))
        # z_t1_norm = z_t1
        # print('z_t', z_t.shape, z_t_norm.max(), z_t_norm.min())
        # print('z_t_norm', z_t_norm.shape, z_t_norm.max(), z_t_norm.min())
        # print('z_t1', z_t1_norm.shape, z_t1_norm.max(), z_t1_norm.min())
        # transition term
        # ce_loss = -torch.sum(torch.mul(z_t_norm, torch.log(z_t1_norm + 1e-8)), dim=0)
        # ce_loss = -torch.sum(z_t_norm * torch.log(z_t1_norm + 1e-8))
        ce_loss = torch.sum(z_t_norm * torch.log(z_t1_norm + 1e-8), dim=1)
        ce_loss = -ce_loss.mean(dim=0)
        # print('ce_loss', ce_loss.shape, ce_loss.max(), ce_loss.min())
        # ce_loss = ce_loss.sum(dim=0)
        summary_scalar(f'Loss/Compression/{self.type}/CE', ce_loss.item())
        # transition entropy term
        avg_prob = z_t_norm.mean(dim=0)
        # print('avg_prob', avg_prob.shape, avg_prob.max(), avg_prob.min())
        # transition_loss = -torch.sum(torch.mul(avg_prob, torch.log(avg_prob)))
        transition_loss = torch.sum(avg_prob * torch.log(avg_prob + 1e-8))
        # print('transition_loss', transition_loss.shape, transition_loss.max(), transition_loss.min())
        summary_scalar(f'Loss/Compression/{self.type}/TCE', transition_loss.item())
        # individual entropy term
        # individual_loss = torch.sum(torch.mul(z_t_norm, torch.log(z_t_norm)), dim=0)
        individual_loss = torch.sum(z_t_norm * torch.log(z_t_norm + 1e-8), dim=1)
        # print('individual_loss', individual_loss.shape, individual_loss.max(), individual_loss.min())
        individual_loss = -individual_loss.mean()
        summary_scalar(f'Loss/Compression/{self.type}/ICE', individual_loss.item())

        compression_loss = ce_loss + 0.4 * transition_loss + 0.1 * individual_loss
        summary_scalar(f'Loss/Compression/{self.type}', compression_loss.item())

        return compression_loss * 1e-3
