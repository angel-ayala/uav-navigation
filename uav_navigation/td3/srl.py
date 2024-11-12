#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:06:16 2024

@author: Angel Ayala
"""
from uav_navigation.td3.agent import TD3Agent
from uav_navigation.td3.agent import TD3Function
from uav_navigation.srl.agent import SRLAgent
from uav_navigation.srl.agent import SRLFunction
from uav_navigation.srl.autoencoder import latent_l2
from uav_navigation.logger import summary_scalar


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


    def action_inference(self, obs):
        # print('SRL obs', obs.shape)
        state = self.compute_z(obs).detach()
        return super().action_inference(state)

    def compute_critic_loss(self, sampled_data, discount, weight=None):
        # Sample replay buffer
        obs, action, reward, next_obs, done = sampled_data
        # print('SRL obs', obs.shape)
        # print('SRL next_obs', next_obs.shape)
        state = self.compute_z(obs)
        next_state = self.compute_z(next_obs)
        return super().compute_critic_loss([state, action, reward, next_state, done],
                                           discount, weight=weight), state

    def compute_actor_loss(self, obs):
        # detach encoder, so we don't update it with the actor loss
        obs = self.compute_z(obs).detach()
        return super().compute_actor_loss(obs)

    def save(self, path, ae_models, encoder_only=False):
        TD3Function.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        TD3Function.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)


class SRLTD3Agent(TD3Agent, SRLAgent):
    def __init__(self, action_shape, approximator, ae_models,
                 discount_factor=0.99, expl_noise=0.1, memory_buffer=None,
                 batch_size=128, reconstruct_freq=1, srl_loss=False,
                 priors=False, encoder_only=False):
        TD3Agent.__init__(self, action_shape, approximator, discount_factor,
                          expl_noise, memory_buffer, batch_size)
        SRLAgent.__init__(self, ae_models, reconstruct_freq=reconstruct_freq,
                          srl_loss=srl_loss, priors=priors, encoder_only=encoder_only)

    def update_critic(self, sampled_data, weight=None):
        critic_loss, z = self.approximator.compute_critic_loss(
            sampled_data, self.discount_factor, weight=weight)

        z_l2 = latent_l2(z)
        loss_z = 0.1 * z_l2
        summary_scalar('Loss/Encoder/Critic/L2', z_l2.item())

        self.approximator.update_critic(critic_loss + z_l2)

    def update(self, step):
        TD3Agent.update(self, step)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, encoder_only=False, eval_only=True):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)
