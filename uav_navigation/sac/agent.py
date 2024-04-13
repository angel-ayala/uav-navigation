#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:49:43 2023

@author: Angel Ayala
"""
import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.nn import functional as F
from thop import clever_format
from ..agent import GenericAgent
from ..agent import GenericFunction
from ..utils import profile_model
from ..utils import soft_update_params
# from ..utils import obs2tensor
# from ..memory import is_prioritized_memory
from ..logger import summary_scalar
from .net import Actor
from .net import Critic


def profile_actor_critic(approximator, state_shape, action_shape):
    # profile q-network
    actor = approximator.actor
    critic = approximator.critic
    critic_target = approximator.critic_target
    total_flops, total_params = 0, 0
    flops, params = profile_model(actor, state_shape, approximator.device)
    total_flops += flops
    total_params += params
    print('Actor: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    flops, params = profile_model(critic, state_shape, approximator.device, action_shape=action_shape)
    total_flops += flops
    total_params += params
    print('Critic: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    flops, params = profile_model(critic_target, state_shape, approximator.device, action_shape=action_shape)
    total_flops += flops
    total_params += params
    print('Target Critic: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    return total_flops, total_params


class ACFunction(GenericFunction):
    def __init__(self, latent_dim, action_shape,
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
                 use_augmentation=True):

        super(ACFunction, self).__init__(use_cuda, is_pixels, is_multimodal,
                                         use_augmentation)

        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        # Q-networks
        self.actor = Actor(latent_dim, action_shape, hidden_dim,
                           actor_min_a, actor_max_a,
                           actor_log_std_min, actor_log_std_max).to(self.device)

        self.critic = Critic(latent_dim, action_shape, hidden_dim).to(self.device)
        self.critic_target = Critic(latent_dim, action_shape, hidden_dim).to(self.device)

        # Initialize target network with Q-network parameters
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action_inference(self, obs, compute_pi=False, compute_log_pi=False):
        return self.actor(
            obs, compute_pi=compute_pi, compute_log_pi=compute_log_pi
        )
    
    def compute_td_error(self, obs, action, reward, alpha):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(obs)
            current_V = torch.min(*self.critic(obs, policy_action)) - self.alpha.detach() * log_pi
            current_Q = torch.min(*self.critic(obs, action))
        td_error = reward.unsqueeze(1) + alpha * current_V - current_Q
        summary_scalar('Loss/TDError', td_error.mean().item())
        return td_error


    def update_critic_target(self):
        # Soft update the target network
        soft_update_params(net=self.critic,
                           target_net=self.critic_target,
                           tau=self.critic_tau)
        

    def update_critic(self, discount, obs, action, reward, next_obs, not_done, weight=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_pi
            target_Q = reward.unsqueeze(1) + discount * target_V * not_done.unsqueeze(1)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        # print('current_Q1', current_Q1.shape)
        # print('current_Q2', current_Q2.shape)
        if weight is not None:
            q_loss = (F.mse_loss(current_Q1, target_Q, reduction='none') +
                      F.mse_loss(current_Q2, target_Q, reduction='none'))
            critic_loss = (weight * q_loss.squeeze()).mean()
        else:
            critic_loss = (F.mse_loss(current_Q1, target_Q) +
                           F.mse_loss(current_Q2, target_Q))
        summary_scalar('Loss/Critic', critic_loss.item())

        # Optimize the state-action function
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, weight=None):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        summary_scalar('Loss/Entropy', entropy.mean())

        alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
        summary_scalar('Loss/Alpha', alpha_loss.item())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = self.alpha.detach() * log_pi - actor_Q.detach()
        if weight is not None:
            actor_loss = (weight * actor_loss.squeeze()).mean()
        else:
            actor_loss = actor_loss.mean()
        summary_scalar('Loss/Actor', actor_loss.item())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, path):
        ac_app_path = str(path) + "_actor_critic.pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict(),
        }, ac_app_path)

    def load(self, path, eval_only=True):
        ac_app_path = str(path) + "_actor_critic.pth"
        checkpoint = torch.load(ac_app_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
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
                 state_shape,
                 action_shape,
                 approximator,
                 discount_factor=0.99,
                 memory_buffer=None):
        super(SACAgent, self).__init__(state_shape, action_shape, approximator,
                                       discount_factor, memory_buffer)
        if self.is_prioritized:
            self.memory.update_beta(0)

    def select_action(self, state):
        state = self.approximator.format_obs(state)
        with torch.no_grad():
            mu, _, _, _ = self.approximator.action_inference(
                state, compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, state):
        state = self.approximator.format_obs(state)
        with torch.no_grad():
            _, pi, _, _ = self.approximator.action_inference(
                state, compute_pi=True, compute_log_pi=False
            )
        return pi.cpu().data.numpy().flatten()

    def update_sac(self, step):
        sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.approximator.device)
        if self.is_prioritized:
            obs, actions, rewards, obs_t1, dones = sampled_data[0]
            loss_weights = sampled_data[1]['weights']
        else:
            obs, actions, rewards, obs_t1, dones = sampled_data
            loss_weights = None

        self.approximator.update_critic(self.discount_factor, obs, actions,
                                        rewards, obs_t1, 1 - dones, weight=loss_weights)

        if step % self.approximator.actor_update_freq == 0:
            self.approximator.update_actor_and_alpha(obs, weight=loss_weights)
        
        if self.is_prioritized:
            # https://link.springer.com/article/10.1007/s11370-024-00514-9
            td_errors = self.approximator.compute_td_error(
                obs, actions, rewards, self.memory.alpha)
            new_priorities = td_errors.squeeze().detach().cpu().numpy()
            new_priorities = np.abs(new_priorities) + 1e-6
            self.memory.update_priorities(sampled_data[1]['indexes'],
                                          new_priorities)

    def update(self, step):
        if self.is_prioritized:
            self.memory.update_beta(step)

        # Update if replay buffer is sufficiently large
        if not self.can_update:
            return False

        self.update_sac(step)

        if step % self.approximator.critic_target_update_freq == 0:
            self.approximator.update_critic_target()
