#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:06:16 2024

@author: Angel Ayala
"""
import torch
import copy

from uav_navigation.td3.agent import TD3Agent
from uav_navigation.td3.agent import TD3Function
from uav_navigation.srl.agent import SRLAgent
from uav_navigation.srl.agent import SRLFunction
from uav_navigation.srl.autoencoder import latent_l2
from uav_navigation.logger import summary_scalar

from .net import TD3EncoderWrapper


class SRLTD3Function(TD3Function, SRLFunction):
    def __init__(self, latent_dim, action_shape, obs_space,
                 hidden_dim=256,
                 action_range=[0., 1.],
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 use_cuda=True,
                 is_pixels=False,
                 is_multimodal=False,
                 use_augmentation=True,
                 decoder_latent_lambda=1e-6):
        TD3Function.__init__(self, latent_dim, action_shape, obs_space,
                             action_range=action_range,
                             hidden_dim=hidden_dim,
                             actor_lr=actor_lr,
                             critic_lr=critic_lr,
                             tau=tau,
                             policy_noise=policy_noise,
                             noise_clip=noise_clip,
                             policy_freq=policy_freq,
                             use_cuda=use_cuda,
                             is_pixels=is_pixels,
                             is_multimodal=is_multimodal,
                             use_augmentation=use_augmentation)
        SRLFunction.__init__(self, decoder_latent_lambda)

    def fuse_encoder2critic(self):
        # Temporal copy
        critic = copy.deepcopy(self.critic)
        # Critic with shared autoencoder weights
        self.critic = TD3EncoderWrapper(critic, self.models[0].encoder[0],
                                        detach_encoder=False)
        self.critic_target = copy.deepcopy(self.critic)
        # optimizers
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

    def forward_actor(self, observation):
        z = self.compute_z(observation, detach=False)
        if 'SPR' in self.models[0].type:
            z = self.critic.encoder.project(z)
        return self.actor(z.detach())

    def forward_critic(self, observation, action):
        z = self.compute_z(observation, detach=False)
        if 'SPR' in self.models[0].type:
            z = self.critic.encoder.project(z)
        return self.critic.function(z, action)

    def forward_critic_target(self, observation, action):
        z = self.critic_target.encoder(observation, detach=False)
        if 'SPR' in self.models[0].type:
            z = self.critic_target.encoder.project(z)
        return self.critic_target.function(z, action)

    def forward_critic_Q1(self, observation, action):
        z = self.critic_target.encoder(observation, detach=False)
        if 'SPR' in self.models[0].type:
            z = self.critic_target.encoder.project(z)
        return self.critic_target.function.Q1(z, action)

    def forward_actor_target(self, observation):
        return self.actor_target(self.compute_z(observation).detach())

    def save(self, path, ae_models, encoder_only=False):
        TD3Function.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        TD3Function.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)

    def train_mode(self):
        TD3Function.train_mode(self)
        SRLFunction.train_mode(self)

    def eval_mode(self):
        TD3Function.eval_mode(self)
        SRLFunction.eval_mode(self)


class SRLTD3Agent(TD3Agent, SRLAgent):
    def __init__(self, action_shape, approximator, ae_models,
                 discount_factor=0.99, memory_buffer=None, batch_size=128,
                 expl_noise=0.1, reconstruct_freq=1, srl_loss=False,
                 priors=False, encoder_only=False):
        TD3Agent.__init__(self, action_shape, approximator, discount_factor,
                          memory_buffer, batch_size, expl_noise)
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
        TD3Agent.update(self, step)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, encoder_only=False, eval_only=True):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)
