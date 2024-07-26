#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:49:43 2023

@author: Angel Ayala
"""
import numpy as np
import copy
import torch
from torchvision.transforms import AutoAugment
import torch.nn as nn
import torch.optim as optim
from .utils import soft_update_params
from .utils import format_obs
from .utils import destack
from .memory import is_prioritized_memory
from .logger import summary_scalar
# from .srl.net import adabelief_optimizer


class GenericFunction:
    def __init__(self, obs_space, use_cuda=True, is_pixels=False, is_multimodal=False,
                 use_augmentation=True):
        self.obs_space = obs_space
        self.device = 'cuda'\
            if torch.cuda.is_available() and use_cuda else 'cpu'
        self.is_pixels = is_pixels
        self.is_multimodal = is_multimodal
        self.augment_model = AutoAugment() if use_augmentation else False

    def normalize_obs(self, obs):
        if self.is_multimodal:
            return (obs[0] / torch.tensor(self.obs_space[0].high, device=self.device),
                    obs[1] / torch.tensor(self.obs_space[1].high, device=self.device))
        else:
            return obs / torch.tensor(self.obs_space.high, device=self.device)

    def normalize_image(self, obs):
        if self.is_multimodal:
            return obs[0] / torch.tensor(self.obs_space[0].high, device=self.device)
        else:
            return obs / torch.tensor(self.obs_space.high, device=self.device)

    def normalize_vector(self, obs):
        if self.is_multimodal:
            return obs[1] / torch.tensor(self.obs_space[1].high, device=self.device)
        else:
            return obs / torch.tensor(self.obs_space.high, device=self.device)

    def augment_image(self, obs_2d):
        if self.augment_model:
            # de-stack for correct augmentation
            obs_frames, orig_shape = destack(obs_2d, len_hist=3, is_rgb=True)
            obs_2d = torch.cat([self.augment_model(frame.to(torch.uint8))
                                for frame in obs_frames])
            obs_2d = obs_2d.reshape(orig_shape).to(torch.float32)

        return obs_2d

    def format_multimodal_obs(self, obs, augment=False):
        obs_2d = format_obs(obs[0], is_pixels=True)
        obs_1d = format_obs(obs[1], is_pixels=False)

        if augment:
            obs_2d = self.augment_image(obs_2d)

        return obs_2d.to(self.device), obs_1d.to(self.device)

    def format_unimodal_obs(self, obs, augment=False):
        observation = format_obs(obs, is_pixels=self.is_pixels)

        if self.is_pixels and augment:
            observation = self.augment_image(observation)

        return observation.to(self.device)

    def format_obs(self, obs, augment=False):
        if self.is_multimodal:
            return self.format_multimodal_obs(obs, augment)
        else:
            return self.format_unimodal_obs(obs, augment)


class QFunction(GenericFunction):

    def __init__(self, q_app_fn, q_app_params, obs_space, learning_rate=10e-5,
                 momentum=0.9, tau=0.1, use_cuda=True, is_pixels=False,
                 is_multimodal=False, use_augmentation=True):
        super(QFunction, self).__init__(obs_space, use_cuda,
                                        is_pixels, is_multimodal, use_augmentation)

        # Q-networks
        self.tau = 1.
        self.q_network = q_app_fn(**q_app_params).to(self.device)
        self.target_q_network = q_app_fn(**q_app_params).to(self.device)

        # Initialize target network with same Q-network parameters
        self.update_target_network()
        self.tau = tau
        # optimization function
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        # self.optimizer = optim.SGD(self.q_network.parameters(),
        #                             lr=learning_rate,
        #                             momentum=momentum,
        #                             nesterov=True,
        #                             weight_decay=1e-7)
        self.optimizer = optim.AdamW(self.q_network.parameters(),
                                     lr=learning_rate,
                                     amsgrad=True)
        # self.optimizer = adabelief_optimizer(self.q_network,
        #                                      learning_rate=learning_rate)
        # self.optimizer = optim.Adam(self.q_network.parameters(),
        #                             lr=learning_rate)

    def update_target_network(self):
        # Soft update the target network
        soft_update_params(net=self.q_network,
                           target_net=self.target_q_network,
                           tau=self.tau)

    def compute_q(self, observations, actions=None):
        # Compute Q-values using the Q-network
        q_values = self.q_network(observations)
        if actions is not None:
            actions_argmax = actions.argmax(-1)
            return q_values.gather(
                dim=-1, index=actions_argmax.unsqueeze(-1)).squeeze(-1)
        else:
            return q_values

    def compute_q_target(self, observations, actions=None):
        # Compute Q-values using the target Q-network
        q_values = self.target_q_network(observations)
        if actions is not None:
            actions_argmax = actions.argmax(-1)
            return q_values.gather(
                dim=-1, index=actions_argmax.unsqueeze(-1)).squeeze(-1)
        else:
            return q_values

    def compute_ddqn_target(self, rewards, q_values, next_observations, discount_factor, dones):
        with torch.no_grad():
            next_q_values = self.compute_q(next_observations)
            double_q_values = self.compute_q_target(next_observations,
                                                    next_q_values)

            td_targets = rewards + discount_factor * double_q_values * (
                1 - dones)

        return td_targets

    def update(self, td_loss, retain_graph=False):
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

    def save(self, path):
        q_app_path = str(path) + "_q_function.pth"
        torch.save({
            'q_network_state_dict': copy.deepcopy(self.q_network.state_dict()),
            'target_q_network_state_dict': copy.deepcopy(self.target_q_network.state_dict()),
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
        }, q_app_path)

    def load(self, path, eval_only=True):
        q_app_path = str(path) + "_q_function.pth"
        checkpoint = torch.load(q_app_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(
            checkpoint['target_q_network_state_dict'])
        if not eval_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Ensure the models are in evaluation mode after loading
            self.q_network.eval()
            self.target_q_network.eval()


class GenericAgent:
    def __init__(self,
                 action_shape,
                 approximator,
                 discount_factor=0.99,
                 memory_buffer=None,
                 batch_size=128):
        self.action_shape = action_shape
        self.discount_factor = discount_factor
        # approximation function
        self.approximator = approximator
        # Replay Buffer
        self.memory = memory_buffer
        self.batch_size = batch_size

    @property
    def is_prioritized(self):
        return is_prioritized_memory(self.memory)

    @property
    def can_update(self):
        return len(self.memory) >= self.batch_size

    def save(self, path):
        self.approximator.save(path)

    def load(self, path, eval_only=True):
        self.approximator.load(path, eval_only)


class DDQNAgent(GenericAgent):
    def __init__(self,
                 action_shape,
                 approximator,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_steps=500000,
                 memory_buffer=None,
                 batch_size=128,
                 train_freq=4,
                 target_update_freq=100):
        super(DDQNAgent, self).__init__(
            action_shape, approximator, discount_factor,
            memory_buffer, batch_size)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_steps
        self.update_epsilon(0)
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_shape[-1])  # Explore
        else:
            with torch.no_grad():
                observation = self.approximator.format_obs(state)
                q_values = self.approximator.compute_q(
                    observation).cpu().numpy()
            return np.argmax(q_values)  # Exploit

    def update_epsilon(self, n_step):
        # Anneal exploration rate
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - (self.epsilon_decay * n_step))
        summary_scalar('Agent/Epsilon', self.epsilon)
        if self.is_prioritized:
            self.memory.update_beta(n_step)

    def update(self, step, augment=True):
        self.update_epsilon(step)

        if step % self.train_freq == 0:
            self.update_td(augment)

        if step % self.target_update_freq == 0:
            self.approximator.update_target_network()

    def update_td(self, augment=True):
        if not self.can_update:
            return False
        # Update the Q-network if replay buffer is sufficiently large
        sampled_data = self.memory.sample(
            self.batch_size, device=self.approximator.device)
        if self.is_prioritized:
            states, actions, rewards, next_states, dones = sampled_data[0]
            priorities = sampled_data[1]
        else:
            states, actions, rewards, next_states, dones = sampled_data

        # prepare data
        states = self.approximator.format_obs(states, augment)
        next_states = self.approximator.format_obs(next_states, augment)

        # Compute Q-values using the approximator
        q_values = self.approximator.compute_q(states, actions)
        td_targets = self.approximator.compute_ddqn_target(
            rewards, q_values, next_states, self.discount_factor, dones)

        # Compute the loss and backpropagate
        q_loss = self.approximator.loss_fn(q_values, td_targets)
        if self.is_prioritized:
            td_errors = q_values - td_targets
            # Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            # Update replay buffer priorities
            self.memory.update_priorities(priorities['indexes'],
                                          new_priorities)
            td_loss = torch.mean(q_loss * priorities['weights'])
        else:
            td_loss = torch.mean(q_loss)

        summary_scalar('Loss/TDLoss', td_loss.item())
        self.approximator.update(td_loss)


# TODO: update for PrioritizedReplayBuffer
class DoubleDuelingQAgent(DDQNAgent):
    def __init__(self,
                 state_space_shape,
                 action_space_shape,
                 device,
                 approximator,
                 approximator_lr=1e-3,
                 approximator_beta=0.9,
                 approximator_tau=0.005,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9999,
                 buffer_capacity=2048,
                 latent_dim=256,
                 hidden_dim=1024,
                 num_layers=2,
                 num_filters=32):
        super().__init__(state_space_shape, action_space_shape, device, approximator,
                         approximator_lr, approximator_beta, approximator_tau,
                         discount_factor, epsilon_start, epsilon_end, epsilon_decay,
                         buffer_capacity, latent_dim, hidden_dim, num_layers, num_filters)

        # Q-networks
        self.q_network = approximator(
            state_space_shape, action_space_shape,
            encoder_feature_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_filters=num_filters).to(self.device)
        self.target_q_network = approximator(
            state_space_shape, action_space_shape,
            encoder_feature_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_filters=num_filters).to(self.device)
        # Initialize target network with Q-network parameters
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=approximator_lr,
                                    betas=(approximator_beta, 0.999))

    def _update_q_network(self, sampled_data):
        action_argmax = sampled_data[1].argmax(1)
        # Compute Q-values using the Q-network
        current_q_values = self.q_network(sampled_data[0]).gather(
            dim=1, index=action_argmax.unsqueeze(1))

        # Use the target network for the next Q-values
        next_actions = self.q_network(sampled_data[3]).argmax(dim=1)
        next_q_values = self.target_q_network(sampled_data[3]).gather(
            dim=1, index=next_actions.unsqueeze(1)).squeeze(1).detach()

        target_q_values = sampled_data[2] + self.discount_factor * \
            (1 - sampled_data[4]) * next_q_values

        # Compute the loss and backpropagate
        loss = nn.functional.smooth_l1_loss(
            current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
