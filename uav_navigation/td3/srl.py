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
from uav_navigation.utils import soft_update_params

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
                 encoder_tau=0.995,
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
        self.encoder_tau = encoder_tau

    def init_models(self):
        self.target_encoder = copy.deepcopy(self.models[0].encoder[0])
        # TODO: test with a linear layer between z (latent vector) and actor/critic functions (EncoderInterface)

    def action_inference(self, obs):
        obs_z = self.compute_z(obs).detach()
        return self.actor(obs_z).cpu().data.numpy().flatten()

    def update_critic_target(self):
        super().update_critic_target()
        # Soft update the target network
        soft_update_params(net=self.models[0].encoder[0],
                           target_net=self.target_encoder,
                           tau=self.encoder_tau)

    def compute_critic_loss(self, sampled_data, discount, weight=None):
        obs, action, reward, next_obs, done = sampled_data
        obs_z = self.compute_z(obs, detach=False)
        if 'SPR' in self.models[0].type:
            obs_z = self.models[0].encoder[0].project(obs_z)

        next_obs_z = self.target_encoder(next_obs)
        if 'SPR' in self.models[0].type:
            obs_z = self.target_encoder.project(obs_z)

        return super().compute_critic_loss(
            [obs_z, action, reward, next_obs_z, done], discount, weight)

    def compute_actor_loss(self, obs):
        obs_z = self.compute_z(obs, detach=True)
        return super().compute_actor_loss(obs_z)

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
        self.approximator.init_models()

    def update(self, step):
        TD3Agent.update(self, step)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, encoder_only=False, eval_only=True):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)
