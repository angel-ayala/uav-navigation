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
from thop import clever_format
from uav_navigation.agent import QFunction
from uav_navigation.agent import DDQNAgent
from uav_navigation.utils import profile_model
from .net import weight_init
from .autoencoder import AEModel


def profile_agent(agent, state_space_shape, action_space_shape):
    total_flops, total_params = 0, 0
    encoder = agent.encoder
    decoder = agent.decoder
    # profile encode stage
    flops, params = profile_model(encoder, state_space_shape, agent.device)
    total_flops += flops
    total_params += params
    print('Encoder: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    # profile q-network
    flops, params = profile_model(agent.q_network.Q, encoder.feature_dim,
                                  agent.device)
    total_flops += flops
    total_params += params
    print('QFunction: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    # profile decode stage
    flops, params = profile_model(decoder, encoder.feature_dim, agent.device)
    total_flops += flops
    total_params += params
    print('Decoder: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    print('Total: {} flops, {} params'.format(
        *clever_format([total_flops, total_params], "%.3f")))
    return total_flops, total_params



class SRLFunction(QFunction):
    def __init__(self, q_app_fn, q_app_params, learning_rate=1e-3,
                 adam_beta1=0.9, tau=0.005, use_cuda=True,
                 decoder_latent_lambda=1e-6):
        super().__init__(q_app_fn, q_app_params, learning_rate, adam_beta1,
                         tau=tau,
                         use_cuda=use_cuda)
        self.models = list()
        self.decoder_latent_lambda = decoder_latent_lambda

    def update_multimodal(self):
        is_rgb = False
        is_vector = False
        for m in self.models:
            is_rgb = m.type == 'rgb' or is_rgb
            is_vector = m.type == 'vector' or m.type == 'imu2pose' or is_vector

        self.is_multimodal = is_rgb and is_vector

    def append_autoencoder(self, ae_model,
                           encoder_lr,
                           decoder_lr,
                           decoder_weight_decay):
        ae_model.to(self.device)
        ae_model.apply(weight_init)
        ae_model.adam_optimizer(encoder_lr, decoder_lr, decoder_weight_decay)
        self.models.append(ae_model)
        self.update_multimodal()

    def compute_z(self, observations):
        z_hat = list()
        obs_2d = observations

        if self.is_multimodal:
            obs_2d = observations[0]
            observations = observations[1]

        if len(obs_2d.shape) == 3:
            obs_2d = torch.tensor(obs_2d, dtype=torch.float32).unsqueeze(0)

        if len(observations.shape) == 1:
            observations = torch.tensor(observations,
                                        dtype=torch.float32).unsqueeze(0)

        for m in self.models:
            if m.type in ['rgb']:
                z = m(obs_2d.to(self.device))
            if m.type in ['vector', 'imu2pose']:
                z = m(observations.to(self.device))
            z_hat.append(z)

        z_hat = torch.cat(z_hat, dim=1)
        return z_hat

    def compute_q(self, observations, actions=None):
        # Compute Q-values using the inferenced z latent representation
        z_hat = self.compute_z(observations)
        return super().compute_q(z_hat, actions)

    def compute_q_target(self, observations, actions=None):
        # Compute Q-values using the target Q-network
        z_hat = self.compute_z(observations)
        return super().compute_q_target(z_hat, actions)

    def save(self, path, ae_models, encoder_only=False):
        q_app_path = str(path) + "_q_function.pth"
        super().save(q_app_path)

        for i, (m, _) in enumerate(ae_models):
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

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        q_app_path = str(path) + "_q_function.pth"
        super().load(q_app_path, eval_only=eval_only)

        for i, (m, _) in enumerate(ae_models):
            ae_model = self.models[i]

            q_app_path = str(path) + f"_ae_{m}.pth"
            checkpoint = torch.load(q_app_path)

            ae_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if eval_only:
                # Ensure the models are in evaluation mode after loading
                ae_model.encoder.eval()
            else:
                ae_model.encoder_optim.load_state_dict(
                    checkpoint['encoder_optimizer_state_dict'])
                ae_model.encoder_optim.eval()

            if not encoder_only:
                for i, (d, dopt) in enumerate(zip(ae_model.decoder,
                                                  ae_model.decoder_opt)):
                    d.load_state_dict(checkpoint[f"decoder_state_dict_{i}"])

                if eval_only:
                    # Ensure the models are in evaluation mode after loading
                    d.eval()
                else:
                    dopt.load_state_dict(
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

        for m, m_params in ae_models.items():
            # RGB observation reconstruction autoencoder model
            ae_model = AEModel(m, m_params)
            self.approximator.append_autoencoder(
                ae_model, m_params['encoder_lr'], m_params['decoder_lr'],
                m_params['decoder_weight_decay'])

        self.ae_models = ae_models
        ae_types = self.ae_models.keys()
        self.is_multimodal = "rgb" in ae_types and (
            "vector" in ae_types or "imu2pose" in ae_types)

    def update_representation(self, obs):
        obs_2d = obs
        if self.is_multimodal:
            obs_2d = obs[0]
            obs = obs[1]
        for ae_model in self.approximator.models:
            if ae_model.type in ["rgb"]:
                ae_model.optimize_reconstruction(
                    obs_2d, self.approximator.decoder_latent_lambda)
            if ae_model.type in ["vector"]:
                ae_model.optimize_reconstruction(
                    obs, self.approximator.decoder_latent_lambda)
            if ae_model.type in ["imu2pose"]:
                ae_model.optimize_pose(
                    obs, self.approximator.decoder_latent_lambda)

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.approximator.device)
            self.update_approximator(sampled_data)
            # update the autoencoder
            if self.is_prioritized:
                self.update_representation(sampled_data[0][0])
            else:
                self.update_representation(sampled_data[0])

    def save(self, path):
        self.approximator.save(path, ae_models=self.ae_models.items())

    def load(self, path, eval_only=True):
        self.approximator.load(path, ae_models=self.ae_models.items(),
                               eval_only=eval_only)
