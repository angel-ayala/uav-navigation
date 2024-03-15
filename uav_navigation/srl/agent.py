#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:27:24 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
import torch
import numpy as np
from thop import clever_format
from uav_navigation.agent import QFunction
from uav_navigation.agent import DDQNAgent
from uav_navigation.agent import profile_q_approximator
from uav_navigation.utils import profile_model
from uav_navigation.logger import summary_scalar
from .net import weight_init
from .net import sgd_optimizer
from .net import NorthBelief
from .net import PositionBelief
from .net import OrientationBelief
from .autoencoder import AEModel
from .autoencoder import profile_ae_model


def profile_srl_approximator(approximator, state_shape, action_shape):
    total_flops, total_params = 0, 0
    q_feature_dim = 0
    for m in approximator.models:
        if approximator.is_multimodal:
            if m.type == 'rgb':
                flops, params = profile_ae_model(m, state_shape[0], approximator.device)
            if m.type == 'vector':
                flops, params = profile_ae_model(m, state_shape[1], approximator.device)
        else:
            flops, params = profile_ae_model(m, state_shape, approximator.device)
        total_flops += flops
        total_params += params
        q_feature_dim += m.encoder.feature_dim

    # profile q-network
    flops, params = profile_q_approximator(approximator, q_feature_dim,
                                           action_shape)
    total_flops += flops
    total_params += params
    print('Total: {} flops, {} params'.format(
        *clever_format([total_flops, total_params], "%.3f")))
    return total_flops, total_params



class SRLFunction(QFunction):
    def __init__(self, q_app_fn, q_app_params, learning_rate=1e-3,
                 momentum=0.9, tau=0.1, use_cuda=True, is_multimodal=True,
                 decoder_latent_lambda=1e-6):
        super().__init__(q_app_fn, q_app_params, learning_rate, momentum,
                         tau=tau, use_cuda=use_cuda)
        self.models = list()
        self.priors = list()
        self.prior_optims = list()
        self.decoder_latent_lambda = decoder_latent_lambda
        self.is_multimodal = is_multimodal

    def append_autoencoder(self, ae_model,
                           encoder_lr,
                           decoder_lr,
                           decoder_weight_decay):
        ae_model.to(self.device)
        ae_model.apply(weight_init)
        ae_model.sgd_optimizer(encoder_lr, decoder_lr, decoder_weight_decay)
        self.models.append(ae_model)

    def append_prior(self, prior_model, learning_rate=1e-3):
        prior_model.to(self.device)
        prior_model.apply(weight_init)
        self.priors.append(prior_model)
        self.prior_optims.append(sgd_optimizer(prior_model, learning_rate))

    def compute_priors_loss(self, obs, obs_t1, actions):
        obs_2d, obs_1d = self.format_obs(obs)
        obs_2d_t1, obs_1d_t1 = self.format_obs(obs_t1)

        loss_list = list()
        for pmodel in self.priors:
            log_prefix = f"Prior/{type(pmodel).__name__}"
            z = self.compute_z(obs, pmodel.latent_types)
            hat_x = pmodel(z)
            if 'rgb' in pmodel.target_types:
                x_true = pmodel.obs2target(obs_2d)
            elif 'vector' in pmodel.target_types:
                x_true = pmodel.obs2target(obs_1d)

            loss = pmodel.compute_loss(hat_x, x_true)
            summary_scalar(f"{log_prefix}/ReconstructionLoss", loss.item())
            loss_list.append(loss)

        return torch.stack(loss_list)

    def append_models(self, models):
        # ensure empty list
        if hasattr(self, 'models'):
            del self.models
        self.models = list()
        for m, m_params in models.items():
            # RGB observation reconstruction autoencoder model
            ae_model = AEModel(m, m_params)
            self.append_autoencoder(
                ae_model, m_params['encoder_lr'], m_params['decoder_lr'],
                m_params['decoder_weight_decay'])

    def compute_z(self, observations, latent_types=None):
        z_hat = list()
        obs_2d, obs_1d = self.format_obs(observations)

        for m in self.models:
            if latent_types is None or m.type in latent_types:
                if m.type in ['rgb']:
                    z = m(obs_2d.to(self.device))
                if m.type in ['vector', 'imu2pose']:
                    z = m(obs_1d.to(self.device))
                z_hat.append(z)

        z_hat = torch.cat(z_hat, dim=1)
        return z_hat

    def format_obs(self, obs):
        if self.is_multimodal:
            obs_2d = obs[0]
            obs_1d = obs[1]
        else:
            obs_2d = obs
            obs_1d = obs

        if not torch.is_tensor(obs_2d):
            obs_2d = np.array(obs_2d)
            obs_2d = torch.tensor(obs_2d, dtype=torch.float32)
        if not torch.is_tensor(obs_1d):
            obs_1d = np.array(obs_1d)
            obs_1d = torch.tensor(obs_1d, dtype=torch.float32)

        if len(obs_2d.shape) == 3:
            obs_2d = obs_2d.unsqueeze(0)
        if len(obs_1d.shape) == 1:
            obs_1d = obs_1d.unsqueeze(0)
        return obs_2d.to(self.device), obs_1d.to(self.device)

    def compute_q(self, observations, actions=None):
        # Compute Q-values using the inferenced z latent representation
        z_hat = self.compute_z(observations)
        return super().compute_q(z_hat, actions)

    def compute_q_target(self, observations, actions=None):
        # Compute Q-values using the target Q-network
        z_hat = self.compute_z(observations)
        return super().compute_q_target(z_hat, actions)

    def update(self, td_loss):
        for m in self.models:
            m.encoder_optim.zero_grad()
        super().update(td_loss)
        for m in self.models:
            m.encoder_optim.step()

    def update_encoder(self, enc_loss):
        for m in self.models:
            m.encoder_optim.zero_grad()
        enc_loss.backward()
        for m in self.models:
            m.encoder_optim.step()

    def update_reconstruction(self, r_loss):
        for m in self.models:
            for d in m.decoder_optim:
                d.zero_grad()
        self.update_encoder(r_loss)
        for m in self.models:
            for d in m.decoder_optim:
                d.step()

    def update_representation(self, r_loss):
        for optim in self.prior_optims:
            optim.zero_grad()
        self.update_reconstruction(r_loss)
        for optim in self.prior_optims:
            optim.step()

    def save(self, path, ae_models, encoder_only=False):
        q_app_path = str(path) + "_q_function.pth"
        super().save(q_app_path)

        for i, (m, _) in enumerate(ae_models.items()):
            ae_model = self.models[i]
            encoder, decoder = ae_model.encoder, ae_model.decoder
            encoder_opt, decoder_opt = ae_model.encoder_optim, ae_model.decoder_optim

            state_dict = dict(encoder_state_dict=encoder.state_dict(),
                              encoder_optimizer_state_dict=encoder_opt.state_dict())

            if not encoder_only:
                for i, (d, dopt) in enumerate(zip(decoder, decoder_opt)):
                    state_dict.update(
                        {f"decoder_state_dict_{i}": d.state_dict(),
                         f"decoder_optimizer_state_dict_{i}": dopt.state_dict()})

            q_app_path = str(path) + f"_ae_{m}.pth"
            torch.save(state_dict, q_app_path)

    def load(self, path, ae_models, eval_only=False, encoder_only=False):
        q_app_path = str(path) + "_q_function.pth"
        super().load(q_app_path, eval_only=eval_only)

        # ensure empty list
        if hasattr(self, 'models'):
            del self.models
        self.models = list()

        for i, (m, m_params) in enumerate(ae_models.items()):
            ae_model = AEModel(m, m_params, encoder_only=encoder_only)
            self.append_autoencoder(
                ae_model, m_params['encoder_lr'], m_params['decoder_lr'],
                m_params['decoder_weight_decay'])
            encoder, decoder = ae_model.encoder, ae_model.decoder
            encoder_opt, decoder_opt = ae_model.encoder_optim, ae_model.decoder_optim

            q_app_path = str(path) + f"_ae_{m}.pth"
            checkpoint = torch.load(q_app_path, map_location=self.device)

            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if eval_only:
                # Ensure the models are in evaluation mode after loading
                encoder.eval()
            else:
                encoder_opt.load_state_dict(
                    checkpoint['encoder_optimizer_state_dict'])

            if not encoder_only:
                for i, (d, dopt) in enumerate(zip(decoder, decoder_opt)):
                    d.load_state_dict(checkpoint[f"decoder_state_dict_{i}"])
                    if eval_only:
                        # Ensure the models are in evaluation mode after loading
                        d.eval()
                    else:
                        decoder_opt.load_state_dict(
                            checkpoint[f"decoder_optimizer_state_dict_{i}"])


class SRLDDQNAgent(DDQNAgent):
    def __init__(self,
                 state_shape,
                 action_shape,
                 approximator,
                 ae_models,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_steps=500000,
                 memory_buffer=None,
                 srl_loss=False,
                 priors=False):
        super().__init__(
            state_shape,
            action_shape,
            approximator,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_steps=epsilon_steps,
            memory_buffer=memory_buffer)

        self.ae_models = ae_models
        self.init_models()
        self.priors = priors
        self.srl_loss = srl_loss
        if priors:
            self.init_priors()

    def init_models(self):
        self.approximator.append_models(self.ae_models)

    def init_priors(self):
        self.approximator.append_prior(NorthBelief(self.state_shape[1], 50))
        self.approximator.append_prior(OrientationBelief(self.state_shape[1], 50), learning_rate=1e-4)
        self.approximator.append_prior(PositionBelief(self.state_shape[1], 50), learning_rate=1e-4)

    def update_representation(self, obs, obs_t1, actions):
        obs_2d, obs_1d = self.approximator.format_obs(obs)
        obs_2d_t1, obs_1d_t1 = self.approximator.format_obs(obs_t1)

        total_loss = list()
        srl_loss = list()
        for ae_model in self.approximator.models:
            if ae_model.type in ["rgb"]:
                loss = ae_model.compute_reconstruction_loss(obs_2d, self.approximator.decoder_latent_lambda)
                total_loss.append(loss)
                if self.srl_loss:
                    srl_loss.append(ae_model.compute_srl_loss(obs_2d, obs_2d_t1, actions))
            if ae_model.type in ["vector"]:
                loss = ae_model.compute_reconstruction_loss(obs_1d, self.approximator.decoder_latent_lambda)
                total_loss.append(loss)
                if self.srl_loss:
                    srl_loss.append(ae_model.compute_srl_loss(obs_1d, obs_1d_t1, actions))

        tloss = torch.mean(torch.stack(total_loss))
        if self.priors:
            prior_loss = self.approximator.compute_priors_loss(obs, obs_t1, actions)
            tloss += prior_loss.sum() * 0.1
        if self.srl_loss:
            tloss += torch.mean(torch.stack(srl_loss))

        summary_scalar("Loss/Representation/TotalLoss", tloss.item())
        self.approximator.update_representation(tloss)

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.approximator.device)
            td_loss = self.compute_td_loss(sampled_data)
            self.approximator.update(td_loss)
            # update the autoencoder
            obs_data = sampled_data[0][0] if self.is_prioritized else sampled_data[0]
            obs_data_t1 = sampled_data[0][3] if self.is_prioritized else sampled_data[3]
            actions_data = sampled_data[0][1] if self.is_prioritized else sampled_data[1]
            self.update_representation(obs_data, obs_data_t1, actions_data)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, ae_models=self.ae_models)

    def load(self, path, eval_only=True, encoder_only=False):
        self.approximator.load(path, ae_models=self.ae_models,
                               eval_only=eval_only, encoder_only=encoder_only)
