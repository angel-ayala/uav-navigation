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
from torch import optim
import torch.nn.functional as F
from thop import clever_format
from uav_navigation.agent import QFunction
from uav_navigation.agent import DDQNAgent
from uav_navigation.utils import profile_model
from .net import weight_init
from .net import preprocess_obs
from .net import rgb_reconstruction_model
from .net import vector_reconstruction_model


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
        self.optimizers = list()
        self.decoder_latent_lambda = decoder_latent_lambda

    def append_autoencoder(self, encoder,
                           encoder_lr,
                           decoder,
                           decoder_lr,
                           decoder_weight_decay):
        encoder.to(self.device)
        decoder.to(self.device)

        encoder.apply(weight_init)
        decoder.apply(weight_init)
        # optimizer for encoder for reconstruction loss
        encoder_optimizer = optim.Adam(encoder.parameters(),
                                       lr=encoder_lr)
        decoder_optimizer = optim.Adam(decoder.parameters(),
                                       lr=decoder_lr,
                                       weight_decay=decoder_weight_decay)
        self.models.append((encoder, decoder))
        self.optimizers.append((encoder_optimizer, decoder_optimizer))

    def compute_q(self, observations, actions=None):
        # Compute Q-values using the Q-network
        state_inference = list()
        for encoder, _ in self.models:
            z = encoder(observations.to(self.device))
            state_inference.append(z)

        state_inference = torch.cat(state_inference)

        return super().compute_q(state_inference, actions)

    def compute_q_target(self, observations, actions=None):
        # Compute Q-values using the target Q-network
        state_inference = list()
        for encoder, _ in self.models:
            z = encoder(observations.to(self.device))
            state_inference.append(z)

        state_inference = torch.cat(state_inference)

        return super().compute_q_target(state_inference, actions)

    def reconstruction_loss(self, obs, target_obs, encoder, decoder):
        h = encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = preprocess_obs(target_obs)
        rec_obs = decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        return loss

    def optimize_reconstruction(self, rloss, optimizer):
        # encoder optimizer
        optimizer[0].zero_grad()
        # decoder optimizer
        optimizer[1].zero_grad()
        rloss.backward()

        optimizer[0].step()
        optimizer[1].step()


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
            if m == 'rgb':
                encoder, decoder = rgb_reconstruction_model(
                    m_params['state_shape'],
                    m_params['latent_dim'],
                    num_layers=m_params['num_layers'],
                    num_filters=m_params['num_filters'])
            # Vector observation reconstruction autoencoder model
            if m == 'vector':
                encoder, decoder = vector_reconstruction_model(
                    m_params['state_shape'],
                    m_params['hidden_dim'],
                    m_params['latent_dim'],
                    num_layers=m_params['num_layers'])
            self.approximator.append_autoencoder(
                encoder, m_params['encoder_lr'],
                decoder, m_params['decoder_lr'],
                m_params['decoder_weight_decay'])

        self.ae_models = ae_models

    def update_reconstruction(self, obs, target_obs):
        for i in range(len(self.ae_models)):
            encoder, decoder = self.approximator.models[i]
            rloss = self.approximator.reconstruction_loss(obs, target_obs,
                                                          encoder, decoder)
            self.approximator.optimize_reconstruction(
                rloss, self.approximator.optimizers[i])

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.approximator.device)
            self.update_approximator(sampled_data)
            # update the autoencoder
            if self.is_prioritized:
                self.update_reconstruction(sampled_data[0][0],
                                           sampled_data[0][0])
            else:
                self.update_reconstruction(sampled_data[0], sampled_data[0])
