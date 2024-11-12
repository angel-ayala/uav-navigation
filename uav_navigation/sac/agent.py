#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:49:43 2023

@author: Angel Ayala
Code taken from: https://github.com/denisyarats/pytorch_sac
"""
import numpy as np
import torch
from torch.nn import functional as F

from uav_navigation.agent import GenericAgent
from uav_navigation.agent import GenericFunction
from uav_navigation.utils import soft_update_params
from uav_navigation.logger import summary_scalar

from .net import DiagGaussianActor
from .net import Critic


class SACFunction(GenericFunction):
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
                 preprocess=False):

        super(SACFunction, self).__init__(obs_space, use_cuda, is_pixels,
                                          is_multimodal, use_augmentation)

        # self.preprocess = preproces
        # Actor 
        self.actor_update_freq = actor_update_freq
        self.actor = DiagGaussianActor(latent_dim, action_shape[-1], hidden_dim,
                                       log_std_bounds=actor_log_std_bounds).to(self.device)
        self.action_range = torch.tensor(action_range, device=self.device)

        # Critic
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.critic = Critic(latent_dim, action_shape[-1], hidden_dim
                             ).to(self.device)
        self.critic_target = Critic(latent_dim, action_shape[-1], hidden_dim
                                    ).to(self.device)
        # Initialize target Critic network parameters
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature
        self.log_alpha = torch.tensor(np.log(init_temperature), requires_grad=True).to(self.device)
        # set target entropy to -|A|
        self.target_entropy = -action_shape[-1]
        self.adjust_temperature = adjust_temperature

        # optimizers
        # self.actor_optimizer = torch.optim.Adam(
        #     self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        # self.critic_optimizer = torch.optim.Adam(
        #     self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        # self.log_alpha_optimizer = torch.optim.Adam(
        #     [self.log_alpha], lr=alpha_lr, betas=alpha_betas)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, amsgrad=True)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, amsgrad=True)
        self.log_alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha], lr=alpha_lr, amsgrad=True)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action_inference(self, obs, sample=False):
        # if self.preprocess:
        #     obs = self.preprocess(obs)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return action[0].cpu().detach().numpy()

    def update_critic_target(self):
        # Soft update the target network
        soft_update_params(net=self.critic,
                           target_net=self.critic_target,
                           tau=self.critic_tau)

    def compute_critic_loss(self, sampled_data, discount, weight=None):
        obs, action, reward, next_obs, done = sampled_data

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward.unsqueeze(1) + ((1 - done).unsqueeze(1) * discount * target_V)
            # target_Q = target_Q.detach()


        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        # if weight is not None:
        #     td_error = (F.mse_loss(current_Q1, target_Q, reduction='none') +
        #                 F.mse_loss(current_Q2, target_Q, reduction='none'))
        #     # print('q_loss', q_loss.shape)
        #     # td_error = (current_Q1 - target_Q) + (current_Q2 - target_Q)
        #     print('td_error', td_error.shape)
        #     print('weight', weight.shape)
        #     # q_loss = td_error.pow(2) * weight
        #     q_loss = td_error * weight
        #     print('q_loss', q_loss.shape)
        #     # critic_loss = (weight * q_loss.squeeze()).mean()
        #     # q_loss = td_error[0].pow(2) * weight
        #     critic_loss = 0.5 * q_loss.mean()#.squeeze()
        #     print('critic_loss', critic_loss.shape)
        #     td_error = (td_error / 2.).detach()
        #     print('td_error', td_error.shape)
        # else:
        critic_loss = (F.mse_loss(current_Q1, target_Q) +
                       F.mse_loss(current_Q2, target_Q))
            # td_error = None
        summary_scalar('Loss/Critic', critic_loss.item())

        return critic_loss

    def update_critic(self, critic_loss):
        # Optimize the state-action function
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, weight=None):
        # detach encoder, so we don't update it with the actor loss
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        summary_scalar('Loss/Actor', actor_loss.item())
        summary_scalar('Loss/TargetEntropy', self.target_entropy)
        summary_scalar('Loss/Entropy', -log_prob.mean().item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.adjust_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            summary_scalar('Loss/Alpha', alpha_loss.item())
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def save(self, path):
        ac_app_path = str(path) + "_actor_critic.pth"
        model_chkpt = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict()
            }
        # if self.preprocess:
        #     model_chkpt['preprocess_dict'] = self.preprocess.state_dict()
        torch.save(model_chkpt, ac_app_path)

    def load(self, path, eval_only=True):
        ac_app_path = str(path) + "_actor_critic.pth"
        checkpoint = torch.load(ac_app_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        # if 'preprocess_dict' in checkpoint.keys():
        #     self.preprocess.load_state_dict(checkpoint['preprocess_dict'])

        if not eval_only:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])
        else:
            # Ensure the models are in evaluation mode after loading
            self.actor.eval()
            self.critic.eval()


class SACAgent(GenericAgent):
    def __init__(self,
                 action_shape,
                 approximator,
                 discount_factor=0.99,
                 memory_buffer=None,
                 batch_size=128):
        super(SACAgent, self).__init__(action_shape, approximator,
                                       discount_factor, memory_buffer, batch_size)
        if self.is_prioritized:
            self.memory.update_beta(0)

    def select_action(self, state, sample=True):
        with torch.no_grad():
            state = self.approximator.format_obs(state).unsqueeze(0)
            action = self.approximator.action_inference(state, sample=sample)
        return action

    def update_critic(self, sampled_data, weight=None):
        critic_loss = self.approximator.compute_critic_loss(
            sampled_data, self.discount_factor, weight=weight)

        self.approximator.update_critic(critic_loss)

    def update_sac(self, step):
        sampled_data = self.memory.sample(
                self.batch_size, device=self.approximator.device)
        if self.is_prioritized:
            loss_weights = sampled_data[1]['weights']
            sampled_data = sampled_data[0]
        else:
            # obs, actions, rewards, obs_t1, dones = sampled_data
            loss_weights = None

        # td_errors = self.approximator.update_critic(
        #     self.discount_factor, obs, actions, rewards, obs_t1,
        #     1 - dones, weight=loss_weights)
        self.update_critic(sampled_data)

        if step % self.approximator.actor_update_freq == 0:
            self.approximator.update_actor_and_alpha(sampled_data[0], weight=loss_weights)

        # if self.is_prioritized:
        #     td_errors = td_errors.squeeze().cpu().numpy()
        #     td_errors = np.abs(td_errors) + 1e-6
        #     summary_scalar('Loss/TDError', td_errors.mean())
        #     # https://link.springer.com/article/10.1007/s11370-024-00514-9
        #     self.memory.update_priorities(sampled_data[1]['indexes'],
        #                                   td_errors)

    def update(self, step):
        if self.is_prioritized:
            self.memory.update_beta(step)

        # Update if replay buffer is sufficiently large
        if not self.can_update:
            return False

        self.update_sac(step)

        if step % self.approximator.critic_target_update_freq == 0:
            self.approximator.update_critic_target()
