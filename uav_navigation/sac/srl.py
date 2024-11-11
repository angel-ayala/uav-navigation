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
import copy
from uav_navigation.srl.agent import SRLAgent
from uav_navigation.srl.agent import SRLFunction
from uav_navigation.srl.autoencoder import latent_l2
from uav_navigation.srl.net import QNetworkWrapper
from uav_navigation.logger import summary_scalar

from .agent import ACFunction
from .agent import SACAgent


class SRLSACFunction(ACFunction, SRLFunction):
    def __init__(self, latent_dim, action_shape, obs_space,       
                 hidden_dim=256,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_min_a=0.,
                 actor_max_a=1.,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,
                 critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 use_cuda=True,
                 is_pixels=False,
                 is_multimodal=False,
                 use_augmentation=True,
                 encoder_tau=0.995,
                 decoder_latent_lambda=1e-6,):
        ACFunction.__init__(self, latent_dim, action_shape, obs_space,
                            hidden_dim,
                            init_temperature,
                            alpha_lr,
                            alpha_beta,
                            actor_lr,
                            actor_beta,
                            actor_min_a,
                            actor_max_a,
                            actor_log_std_min,
                            actor_log_std_max,
                            actor_update_freq,
                            critic_lr,
                            critic_beta,
                            critic_tau,
                            critic_target_update_freq,
                            use_cuda,
                            is_pixels,
                            is_multimodal,
                            use_augmentation)
        SRLFunction.__init__(self, decoder_latent_lambda)
        self.encoder_tau = encoder_tau
    
    def fuse_encoder(self):
        # fuse encoder with Critic function
        critic = copy.deepcopy(self.critic)
        encoder = copy.deepcopy(self.models[0].encoder[0])
        self.critic = QNetworkWrapper(critic, encoder).to(self.device)
        # fuse encoder with Critic target function
        critic_target = copy.deepcopy(self.critic_target)
        self.critic_target = QNetworkWrapper(
            critic_target, encoder).to(self.device)
        # fuse encoder with Actor function
        critic_target = copy.deepcopy(self.critic_target)
        self.critic_target = QNetworkWrapper(
            critic_target, encoder).to(self.device)
        # Initialize target network with same Q-network parameters
        tau_value = copy.deepcopy(self.tau)
        self.tau = 1.
        self.update_critic_target()
        self.tau = tau_value
        # share autoencoder weights
        self.q_network.encoder.copy_weights_from(self.models[0].encoder[0])
        self.actor.encoder.copy_weights_from(self.q_network.encoder)
        # Initialize optimization function
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, amsgrad=True)

        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, amsgrad=True)

    def action_inference(self, obs, sample=False):
        obs = self.compute_z(obs).detach()
        return super().action_inference(obs, sample=sample)

    def compute_critic_loss(self, sampled_data, discount, weight=None):
        obs, action, reward, next_obs, done = sampled_data
        state = self.compute_z(obs)
        next_state = self.compute_z(next_obs)
        return super().compute_critic_loss([state, action, reward, next_state, done], discount, weight), state

    # def update_critic_target(self):
    #     super().update_critic_target()
    #     for m in self.models:
    #         if m.type == 'atc':
    #             m.update_momentum_encoder(self.encoder_tau)

    def update_actor_and_alpha(self, obs, weight=None):
        # detach encoder, so we don't update it with the actor loss
        obs = self.compute_z(obs).detach()
        return super().update_actor_and_alpha(obs, weight)

    def save(self, path, ae_models, encoder_only=False):
        ACFunction.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        ACFunction.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)


class SRLSACAgent(SACAgent, SRLAgent):
    def __init__(self, action_shape, approximator, ae_models,
                 discount_factor=0.99, memory_buffer=None, batch_size=128,
                 reconstruct_freq=1, srl_loss=False, priors=False,
                 encoder_only=False):
        SACAgent.__init__(self, action_shape, approximator,
                          discount_factor, memory_buffer, batch_size)
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
        SACAgent.update(self, step)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, encoder_only=False, eval_only=True):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)
