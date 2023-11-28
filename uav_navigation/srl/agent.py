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
from uav_navigation.agent import DDQNAgent
from .net import PixelApproximator
from .net import weight_init
from .autoencoder import PixelDecoder
from .autoencoder import preprocess_obs


class AEDDQNAgent(DDQNAgent):
    def __init__(self,
                 state_space_shape,
                 action_space_shape,
                 device,
                 approximator_lr=1e-3,
                 approximator_beta=0.9,
                 approximator_tau=0.005,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9999,
                 buffer_capacity=2048,
                 encoder_lr=1e-3,
                 decoder_lr=1e-3,
                 decoder_latent_lambda=1e-6,
                 decoder_weight_lambda=1e-7):

        super(AEDDQNAgent, self).__init__(
            state_space_shape=state_space_shape,
            action_space_shape=action_space_shape,
            device=device,
            approximator=PixelApproximator,
            approximator_lr=1e-3,
            approximator_beta=0.9,
            approximator_tau=0.005,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.9999,
            buffer_capacity=2048)

        self.encoder = self.q_network.encoder
        self.decoder = PixelDecoder(state_space_shape,
                                    self.q_network.feature_dim,
                                    self.q_network.num_layers).to(
                                        self.device)
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
            weight_decay=decoder_weight_lambda
        )

    def update_decoder(self, obs, target_obs):
        h = self.q_network(obs)

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

    def update(self, state, action, reward, next_state, done):
        # Store the transition in the replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            self._update_q_network()

        # Anneal exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        self.fit_decoder(state, state)
