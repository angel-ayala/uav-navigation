#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:27:24 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
from torch import optim
import torch.nn.functional as F
from thop import clever_format
from uav_navigation.agent import DDQNAgent
from uav_navigation.utils import profile_model
from .net import weight_init
from .net import MLP
from .net import VectorApproximator
from .net import PixelApproximator
from .autoencoder import PixelDecoder
from .autoencoder import preprocess_obs


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


class AEDDQNAgent(DDQNAgent):
    def __init__(self,
                 state_space_shape,
                 action_space_shape,
                 device,
                 approximator,
                 approximator_lr=1e-3,
                 approximator_beta=0.9,
                 approximator_tau=0.005,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9999,
                 buffer_capacity=2048,
                 latent_dim=256,
                 hidden_dim=1024,
                 num_layers=2,
                 num_filters=32,
                 encoder_lr=1e-3,
                 decoder_lr=1e-3,
                 decoder_latent_lambda=1e-6,
                 decoder_weight_decay=1e-7):

        super(AEDDQNAgent, self).__init__(
            state_space_shape=state_space_shape,
            action_space_shape=action_space_shape,
            device=device,
            approximator=approximator,
            approximator_lr=approximator_lr,
            approximator_beta=approximator_beta,
            approximator_tau=approximator_tau,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_capacity=buffer_capacity)

        # Re initialize Q-networks with proper parameters
        del self.q_network
        del self.target_q_network
        del self.optimizer
        appx_params = dict(
            input_shape=state_space_shape,
            output_shape=action_space_shape,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            feature_dim=latent_dim)
        if approximator == PixelApproximator:
            appx_params['num_filters'] = num_filters

        self.q_network = approximator(**appx_params).to(self.device)
        self.target_q_network = approximator(**appx_params).to(self.device)

        self.encoder = self.q_network.encoder
        self.encoder.apply(weight_init)
        self.encoder.to(self.device)

        # Initialize target network with Q-network parameters
        self._update_target_network()
        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=approximator_lr,
                                    betas=(approximator_beta, 0.999))

        if approximator == VectorApproximator:
            self.decoder = MLP(self.encoder.feature_dim,
                               state_space_shape[0], latent_dim)
        if approximator == PixelApproximator:
            self.decoder = PixelDecoder(state_space_shape,
                                        self.encoder.feature_dim,
                                        num_layers=num_layers,
                                        num_filters=num_filters
                                        ).to(self.device)
        if not hasattr(self, 'decoder'):
            raise ValueError(f"Error, no decoder for {type(approximator)}.")
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder.apply(weight_init)
        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=encoder_lr
        )

        # optimizer for decoder
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_decay
        )

    def update_decoder(self, obs, target_obs):
        h = self.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(self.BATCH_SIZE,
                                              device=self.device)
            self._update_q_network(sampled_data)
            # update the autoencoder
            self.update_decoder(sampled_data[0], sampled_data[0])

        # Anneal exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
