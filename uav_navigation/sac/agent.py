#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:49:43 2023

@author: Angel Ayala
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from thop import clever_format
from ..utils import profile_model
from ..utils import soft_update_params
from ..utils import obs2tensor
from ..memory import is_prioritized_memory
from ..logger import summary_scalar
from .net import Actor
from .net import Critic
from .net import VFunction


def profile_actor_critic(approximator, state_shape, action_shape):
    # profile q-network
    actor = approximator.actor
    critic = approximator.critic
    value = approximator.value
    value_target = approximator.value_target
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
    flops, params = profile_model(value, state_shape, approximator.device)
    total_flops += flops
    total_params += params
    print('Value: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    flops, params = profile_model(value_target, state_shape, approximator.device)
    total_flops += flops
    total_params += params
    print('Target Value: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    return total_flops, total_params


class ACFunction:

    def __init__(self, latent_dim, action_shape,
                 hidden_dim=256,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,
                 critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 use_cuda=True):

        self.device = 'cuda'\
            if torch.cuda.is_available() and use_cuda else 'cpu'
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        # Q-networks
        self.actor = Actor(latent_dim, action_shape, hidden_dim,
                           actor_log_std_min, actor_log_std_max).to(self.device)


        self.value = VFunction(latent_dim, hidden_dim).to(self.device)
        self.value_target = VFunction(latent_dim, hidden_dim).to(self.device)

        self.critic = Critic(latent_dim, action_shape, hidden_dim).to(self.device)


        # Initialize target network with Q-network parameters
        # self.update_target_network()

        # tie encoders between value and target
        self.value_target.load_state_dict(self.value.state_dict())

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

        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action_inference(self, obs, compute_pi=False, compute_log_pi=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        return self.actor(
            obs, compute_pi=compute_pi, compute_log_pi=compute_log_pi
        )
    
    def compute_td_error(self, obs, action, reward, beta):
        with torch.no_grad():
            _, _, log_pi, _ = self.actor(obs)
            current_Q = torch.min(*self.critic(obs, action))
            current_V = self.value_target(obs)
        td_error = reward.unsqueeze(1) + beta * current_V - current_Q
        return td_error


    def update_critic_target(self):
        # Soft update the target network
        soft_update_params(net=self.value,
                           target_net=self.value_target,
                           tau=self.critic_tau)
    
    # def compute_target_v(self, state):
    #     with torch.no_grad():
    #         _, policy_action, log_pi, _ = self.actor(state)
    #         target_Q1, target_Q2 = self.critic(state, policy_action)
    #         target_v = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi  
    #     return target_v
        

    def update_critic(self, discount, obs, action, reward, next_obs, not_done, weight=None):
        # Optimize the value-function
        with torch.no_grad():

            _, policy_action, log_pi, _ = self.actor(obs)
            target_Q1, target_Q2 = self.critic(obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi        

        current_V = self.value(obs)
        # target_v = self.compute_target_v(obs)
        value_loss = 0.5 * F.mse_loss(current_V, target_V)
        summary_scalar('Loss/Value', value_loss.item())

        # Optimize the state-action function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # with torch.no_grad():
            # _, policy_action, log_pi, _ = self.actor(next_obs)
            # target_Q1, target_Q2 = self.critic(next_obs, policy_action)
            # target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
        target_V = self.value_target(next_obs).detach()
        target_Q = reward.unsqueeze(1) + discount * target_V * not_done.unsqueeze(1)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs.detach(), action)
        # print('current_Q1', current_Q1.shape)
        # print('current_Q2', current_Q2.shape)
        if weight is not None:
            q_loss = (F.mse_loss(current_Q1, target_Q, reduction='none') +
                      F.mse_loss(current_Q2, target_Q, reduction='none'))
            q_loss = (weight * q_loss).mean()
        else:
            q_loss = (F.mse_loss(current_Q1, target_Q) +
                      F.mse_loss(current_Q2, target_Q))
        critic_loss = 0.5 * q_loss
        # print('critic_loss', critic_loss)
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
            actor_loss = (weight * actor_loss).mean()
        else:
            actor_loss = actor_loss.mean()
        summary_scalar('Loss/Actor', actor_loss.item())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        

    def format_obs(self, obs, is_pixels=True):
        observation = obs2tensor(obs)

        if len(observation.shape) == 3 and is_pixels:
            observation = observation.unsqueeze(0)
        if len(observation.shape) == 1 and not is_pixels:
            observation = observation.unsqueeze(0)

        return observation.to(self.device)

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'value_target_state_dict': self.value_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'log_alpha_optimizer_state_dict': self.log_alpha_optimizer.state_dict(),
        }, path)

    def load(self, path, eval_only=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.value_target.load_state_dict(checkpoint['value_target_state_dict'])
        if not eval_only:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])
        else:
            # Ensure the models are in evaluation mode after loading
            self.actor.eval()
            self.critic.eval()
            self.value.eval()


class SACAgent:
    BATCH_SIZE = 32

    def __init__(self,
                 state_shape,
                 action_shape,
                 approximator,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_steps=500000,
                 memory_buffer=None):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_steps
        self.epsilon = -1
        self.action_shape = action_shape[0]
        # Q-networks
        self.approximator = approximator
        # Replay Buffer
        self.memory = memory_buffer
        if self.is_prioritized:
            self.memory.update_beta(0)

    @property
    def is_prioritized(self):
        return is_prioritized_memory(self.memory)

    def select_action(self, state):
        with torch.no_grad():
            mu, _, _, _ = self.approximator.action_inference(
                state, compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, state):
        with torch.no_grad():
            _, pi, _, _ = self.approximator.action_inference(
                state, compute_pi=True, compute_log_pi=False
            )
        return pi.cpu().data.numpy().flatten()

    def update_sac(self, step):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.approximator.device)
        else:
            return

        if self.is_prioritized:
            obs, actions, rewards, obs_t1, dones = sampled_data[0]
        else:
            obs, actions, rewards, obs_t1, dones = sampled_data


        
        #     self.approximator.update_critic(self.discount_factor, obs, actions,
        #                                     rewards, obs_t1, 1 - dones, weight=new_priorities)
        #     # new_priorities = new_priorities.squeeze().detach().cpu().numpy()
        #     # print('weights', sampled_data[1]['weights'])
        #     # print('indexes', sampled_data[1]['indexes'])
        #     # print('new_priorities', new_priorities)

        #     if step % self.approximator.actor_update_freq == 0:
        #         self.approximator.update_actor_and_alpha(obs, weight=new_priorities)
        # else:
        self.approximator.update_critic(self.discount_factor, obs, actions,
                                        rewards, obs_t1, 1 - dones)
        if step % self.approximator.actor_update_freq == 0:
            self.approximator.update_actor_and_alpha(obs)
        
        if self.is_prioritized:
            # https://link.springer.com/article/10.1007/s11370-024-00514-9
            td_error = self.approximator.compute_td_error(obs, actions, rewards, self.memory.alpha)
            new_priorities = td_error.squeeze().detach()
            self.memory.update_priorities(sampled_data[1]['indexes'],
                                          new_priorities.cpu().numpy())

    def update(self, step):
        # Update the Q-network if replay buffer is sufficiently large
        self.update_sac(step)

        if step % self.approximator.critic_target_update_freq == 0:
            self.approximator.update_critic_target()

        if self.is_prioritized:
            self.memory.update_beta(step)

    def save(self, path):
        self.approximator.save(str(path) + ".pth")

    def load(self, path):
        self.approximator.load(str(path) + ".pth", eval_only=True)
