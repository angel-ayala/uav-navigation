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
from uav_navigation.utils import obs2tensor
from .net import weight_init
# from .net import OrientationBelief
from .net import OdometryBelief
from .autoencoder import AEModel
from .autoencoder import PriorModel


def profile_srl_approximator(approximator, state_shape, action_shape):
    total_flops, total_params = 0, 0
    q_feature_dim = 0
    for m in approximator.models:
        # profile encode stage
        flops, params = profile_model(m.encoder, state_shape, approximator.device)
        total_flops += flops
        total_params += params
        print('Encoder {}: {} flops, {} params'.format(
            m.type, *clever_format([flops, params], "%.3f")))
        q_feature_dim += m.encoder.feature_dim
        # profile decode stage
        for i, decoder in enumerate(m.decoder):
            flops, params = profile_model(decoder, m.encoder.feature_dim, approximator.device)
            total_flops += flops
            total_params += params
            print('Decoder {} {}: {} flops, {} params'.format(
                i, m.type, *clever_format([flops, params], "%.3f")))

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
                 memory_buffer=None):
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
        self.prior_models = list()

    def init_priors(self):
        # bmodel = OrientationBelief(50)
        bmodel = OdometryBelief(self.state_shape[1], 50)
        self.prior_models.append(PriorModel(bmodel))
        self.prior_models[-1].sgd_optimizer()

    def init_models(self):
        self.approximator.append_models(self.ae_models)

    def update_representation(self, obs, obs_t1, actions):
        obs_2d, obs_1d = self.approximator.format_obs(obs)
        obs_2d_t1, obs_1d_t1 = self.approximator.format_obs(obs_t1)

        for ae_model in self.approximator.models:
            if ae_model.type in ["rgb"]:
                ae_model.optimize_reconstruction(
                    obs_2d, self.approximator.decoder_latent_lambda)
            if ae_model.type in ["vector"]:
                ae_model.optimize_reconstruction(
                    obs_1d, self.approximator.decoder_latent_lambda)
            if ae_model.type in ["imu2pose"]:
                ae_model.optimize_pose(
                    obs_1d, self.approximator.decoder_latent_lambda)

    def update_encoder(self, obs, obs_t1, actions):
        obs_2d, obs_1d = self.approximator.format_obs(obs)
        obs_2d_t1, obs_1d_t1 = self.approximator.format_obs(obs_t1)
        for ae_model in self.approximator.models:
            if ae_model.type in ["rgb"]:
                ae_model.update_encoder(obs_2d, obs_2d_t1, actions)
            if ae_model.type in ["vector", "imu2pose"]:
                ae_model.update_encoder(obs_1d, obs_1d_t1, actions)


    def update_priors(self, obs, obs_t1, actions):
        for pmodel in self.prior_models:
            pmodel.update(obs, obs_t1, actions, self.approximator)

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
            self.update_encoder(obs_data, obs_data_t1, actions_data)
            self.update_priors(obs_data, obs_data_t1, actions_data)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, ae_models=self.ae_models)

    def load(self, path, eval_only=True, encoder_only=False):
        self.approximator.load(path, ae_models=self.ae_models,
                               eval_only=eval_only, encoder_only=encoder_only)
