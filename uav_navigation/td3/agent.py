#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:06:16 2024

@author: Angel Ayala

Code taken from: https://github.com/sfujim/TD3/blob/master/TD3.py
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from uav_navigation.agent import GenericAgent
from uav_navigation.agent import GenericFunction
from uav_navigation.utils import soft_update_params
from uav_navigation.logger import summary_scalar

from .net import Actor
from .net import Critic


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
class TD3Function(GenericFunction):

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
                 use_augmentation=True):
        super(TD3Function, self).__init__(obs_space, use_cuda, is_pixels,
                                          is_multimodal, use_augmentation)
        # Actor
        self.action_range = torch.tensor(action_range, device=self.device)
        self.actor = Actor(latent_dim, action_shape[-1], self.action_range[1],
                           hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic
        self.critic = Critic(latent_dim, action_shape[-1], hidden_dim
                             ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        # self.actor_optimizer = torch.optim.AdamW(
        #     self.actor.parameters(), lr=actor_lr, amsgrad=True)
        # self.critic_optimizer = torch.optim.AdamW(
        #     self.critic.parameters(), lr=critic_lr, amsgrad=True)

        self.tau = tau
        self.policy_noise = torch.tensor(policy_noise, device=self.device)
        self.policy_noise *= self.action_range[1]
        self.noise_clip = torch.tensor(noise_clip, device=self.device)
        self.noise_clip *= self.action_range[1]
        self.policy_freq = policy_freq

    def forward_actor(self, observation):
        return self.actor(observation)

    def forward_actor_target(self, observation):
        return self.actor_target(observation)

    def forward_critic(self, observation, action):
        return self.critic(observation, action)

    def forward_critic_target(self, observation, action):
        return self.critic_target(observation, action)

    def forward_critic_Q1(self, observation, action):
        return self.critic.Q1(observation, action)

    def action_inference(self, obs):
        return self.forward_actor(obs).cpu().data.numpy().flatten()

    def sample_action(self, obs, expl_noise=0.1):
        max_action = self.action_range.cpu().numpy()
        action_noise = np.random.normal(0, max_action[1] * expl_noise)
        action = self.action_inference(obs) + action_noise
        return action.clip(*max_action)

    def update_actor_target(self):
        # Soft update the target network
        soft_update_params(net=self.actor,
                           target_net=self.actor_target,
                           tau=self.tau)

    def update_critic_target(self):
        # Soft update the target network
        soft_update_params(net=self.critic,
                           target_net=self.critic_target,
                           tau=self.tau)

    def compute_critic_loss(self, sampled_data, discount, weight=None):
        # Sample replay buffer
        state, action, reward, next_state, done = sampled_data

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.forward_actor_target(next_state) + noise
                           ).clamp(*self.action_range)

            # Compute the target Q value
            target_Q1, target_Q2 = self.forward_critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.forward_critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        summary_scalar('Loss/Critic', critic_loss.item())
        return critic_loss

    def update_critic(self, critic_loss):
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def compute_actor_loss(self, state):
        # Compute actor losse
        actor_loss = -self.forward_critic_Q1(state, self.forward_actor(state)).mean()
        summary_scalar('Loss/Actor', actor_loss.item())
        return actor_loss

    def update_actor(self, actor_loss):
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        self.update_critic_target()
        self.update_actor_target()

    def save(self, path):
        ac_app_path = str(path) + "_actor_critic.pth"
        model_chkpt = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            }
        # if self.preprocess:
        #     model_chkpt['preprocess_dict'] = self.preprocess.state_dict()
        torch.save(model_chkpt, ac_app_path)

    def load(self, path, eval_only=True):
        ac_app_path = str(path) + "_actor_critic.pth"
        checkpoint = torch.load(ac_app_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        # if 'preprocess_dict' in checkpoint.keys():
        #     self.preprocess.load_state_dict(checkpoint['preprocess_dict'])

        if not eval_only:
            self.actor_target = copy.deepcopy(self.actor)
            self.critic_target = copy.deepcopy(self.critic)
            self.actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])
        else:
            # Ensure the models are in evaluation mode after loading
            self.actor.eval()
            self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


class TD3Agent(GenericAgent):
    def __init__(self,
                 action_shape,
                 approximator,
                 discount_factor=0.99,
                 memory_buffer=None,
                 batch_size=128,
                 expl_noise=0.1):
        super(TD3Agent, self).__init__(action_shape, approximator,
                                       discount_factor, memory_buffer, batch_size)
        self.expl_noise = expl_noise
        self.expl_noise_eval = 0.0
        self._tmp_expl_noise = self.expl_noise
        if self.is_prioritized:
            self.memory.update_beta(0)

    def select_action(self, state):
        with torch.no_grad():
            state = self.approximator.format_obs(state).unsqueeze(0)
            action = self.approximator.sample_action(state, self.expl_noise)
        return action

    def update_critic(self, sampled_data, weight=None):
        critic_loss = self.approximator.compute_critic_loss(
            sampled_data, self.discount_factor, weight=weight)

        self.approximator.update_critic(critic_loss)

    def update_td3(self, step):
        sampled_data = self.memory.sample(
                self.batch_size, device=self.approximator.device)
        loss_weights = None
        # if self.is_prioritized:
        #     obs, actions, rewards, obs_t1, dones = sampled_data[0]
        #     loss_weights = sampled_data[1]['weights']
        # else:
        obs, actions, rewards, obs_t1, dones = sampled_data

        self.update_critic(sampled_data, loss_weights)

        if step % self.approximator.policy_freq == 0:
            actor_loss = self.approximator.compute_actor_loss(obs)
            self.approximator.update_actor(actor_loss)

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

        self.update_td3(step)

        # if step % self.approximator.critic_target_update_freq == 0:
        #     self.approximator.update_critic_target()

    def learn_mode(self):
        # swap exploration noise value for learning
        self.expl_noise = self._tmp_expl_noise
        super().learn_mode()

    def eval_mode(self):
        # swap exploration noise value for evaluation
        self._tmp_expl_noise = self.expl_noise
        self.expl_noise = self.expl_noise_eval
        super().eval_mode()
