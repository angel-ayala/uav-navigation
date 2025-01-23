#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:27:24 2023

@author: Angel Ayala
"""
import torch
import copy

from uav_navigation.srl.agent import SRLAgent
from uav_navigation.srl.agent import SRLFunction
from uav_navigation.srl.autoencoder import latent_l2
from uav_navigation.srl.net import EncoderWrapper
from uav_navigation.logger import summary_scalar

from .agent import SACAgent
from .agent import SACFunction


class SRLSACFunction(SACFunction, SRLFunction):
    def __init__(self, latent_dim, action_shape, obs_space,
                 hidden_dim=256,
                 action_range=[0., 1.],
                 init_temperature=0.1,
                 adjust_temperature=True,
                 alpha_lr=1e-4,
                 alpha_betas=(0.9, 0.999),
                 actor_lr=1e-4,
                 actor_betas=(0.9, 0.999),
                 actor_log_std_bounds=[-5., 2.],
                 actor_update_freq=1,
                 critic_lr=1e-4,
                 critic_betas=(0.9, 0.999),
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 use_cuda=True,
                 is_pixels=False,
                 is_multimodal=False,
                 use_augmentation=True,
                 encoder_tau=0.995,
                 decoder_latent_lambda=1e-6,):
        SACFunction.__init__(self, latent_dim=latent_dim,
                             action_shape=action_shape, obs_space=obs_space,
                             hidden_dim=hidden_dim,
                             action_range=action_range,
                             init_temperature=init_temperature,
                             adjust_temperature=adjust_temperature,
                             alpha_lr=alpha_lr,
                             alpha_betas=alpha_betas,
                             actor_lr=actor_lr,
                             actor_betas=actor_betas,
                             actor_log_std_bounds=actor_log_std_bounds,
                             actor_update_freq=actor_update_freq,
                             critic_lr=critic_lr,
                             critic_betas=critic_betas,
                             critic_tau=critic_tau,
                             critic_target_update_freq=critic_target_update_freq,
                             use_cuda=use_cuda,
                             is_pixels=is_pixels,
                             is_multimodal=is_multimodal,
                             use_augmentation=use_augmentation,
                             preprocess=False)
        SRLFunction.__init__(self, decoder_latent_lambda)
        self.encoder_tau = encoder_tau

    def fuse_encoder2critic(self):
        # Temporal copy
        critic = copy.deepcopy(self.critic)
        encoder = copy.deepcopy(self.models[0].encoder[0])
        # Critic
        self.critic = EncoderWrapper(critic, encoder, detach_encoder=False)
        self.critic_target = copy.deepcopy(self.critic)
        # share autoencoder weights
        self.critic.encoder.copy_weights_from(self.models[0].encoder[0])
        # optimizers
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

    def forward_actor(self, observation):
        return self.actor(self.compute_z(observation).detach())

    def save(self, path, ae_models, encoder_only=False):
        SACFunction.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        SACFunction.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)

    def train_mode(self):
        SACFunction.train_mode(self)
        SRLFunction.train_mode(self)

    def eval_mode(self):
        SACFunction.eval_mode(self)
        SRLFunction.eval_mode(self)


class SRLSACAgent(SACAgent, SRLAgent):
    def __init__(self, action_shape, approximator, ae_models,
                 discount_factor=0.99, memory_buffer=None, batch_size=128,
                 sample=False, reconstruct_freq=1, srl_loss=False, priors=False,
                 encoder_only=False):
        SACAgent.__init__(self, action_shape, approximator, discount_factor,
                          memory_buffer, batch_size, sample)
        SRLAgent.__init__(self, ae_models, reconstruct_freq=reconstruct_freq,
                          srl_loss=srl_loss, priors=priors, encoder_only=encoder_only)
        self.approximator.fuse_encoder2critic()

    def update_critic(self, sampled_data, weight=None):
        critic_loss = self.approximator.compute_critic_loss(
            sampled_data, self.discount_factor, weight=weight)

        if "SPR" not in self.approximator.models[0].type:
            z_l2 = latent_l2(self.approximator.compute_z(sampled_data[0]))
            loss_z = z_l2 * self.approximator.decoder_latent_lambda
            summary_scalar('Loss/Encoder/Critic/L2', z_l2.item())

            self.approximator.update_critic(critic_loss + loss_z)
        else:
            self.approximator.update_critic(critic_loss)

    def update(self, step):
        SACAgent.update(self, step)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, encoder_only=False, eval_only=True):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)
