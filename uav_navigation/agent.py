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
from thop import clever_format
from .utils import profile_model
from .utils import soft_update_params
from .memory import is_prioritized_memory


def profile_agent(agent, state_space_shape, action_space_shape):
    # profile q-network
    flops, params = profile_model(agent.q_network, state_space_shape,
                                  agent.device)
    print('Q-network: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    flops, params = profile_model(agent.target_q_network, state_space_shape,
                                  agent.device)
    print('Target Q-network: {} flops, {} params'.format(
        *clever_format([flops, params], "%.3f")))
    return flops, params


class QFunction:

    def __init__(self, q_app_fn, q_app_params, learning_rate=1e-3,
                 adam_beta1=0.9, tau=0.005, use_cuda=True):

        self.device = 'cuda'\
            if torch.cuda.is_available() and use_cuda else 'cpu'

        self.tau = tau

        # Q-networks
        self.q_network = q_app_fn(**q_app_params).to(self.device)
        self.target_q_network = q_app_fn(**q_app_params).to(self.device)

        # Initialize target network with Q-network parameters
        self.update_target_network()

        # optimization function
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=learning_rate,
                                    betas=(adam_beta1, 0.999))

    def update_target_network(self):
        # Soft update the target network
        soft_update_params(net=self.q_network,
                           target_net=self.target_q_network,
                           tau=self.tau)

    def compute_q(self, observations, actions=None):
        # Compute Q-values using the Q-network
        if type(observations) is not torch.Tensor:
            obs_tensor = torch.tensor(observations, dtype=torch.float32
                                      ).unsqueeze(0)
        else:
            obs_tensor = observations
        q_values = self.q_network(obs_tensor.to(self.device))
        if actions is not None:
            actions_argmax = actions.argmax(-1)
            return q_values.gather(
                dim=-1, index=actions_argmax.unsqueeze(-1)).squeeze(-1)
        else:
            return q_values

    def compute_q_target(self, observations, actions=None):
        # Compute Q-values using the target Q-network
        if type(observations) is not torch.Tensor:
            obs_tensor = torch.tensor(observations, dtype=torch.float32
                                      ).unsqueeze(0)
        else:
            obs_tensor = observations
        q_values = self.target_q_network(obs_tensor.to(self.device))
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

    def optimize(self, td_loss):
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path, eval_only=False):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if eval_only:
            # Ensure the models are in evaluation mode after loading
            self.q_network.eval()
            self.target_q_network.eval()


class DDQNAgent:
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
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_steps
        self.action_shape = action_shape[0]
        # Q-networks
        self.approximator = approximator
        # Replay Buffer
        self.memory = memory_buffer
        self.update_epsilon(0)

    @property
    def is_prioritized(self):
        return is_prioritized_memory(self.memory)

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_shape)  # Explore
        else:
            with torch.no_grad():
                q_values = self.approximator.compute_q(state).cpu().numpy()
            return np.argmax(q_values)  # Exploit

    def update_epsilon(self, n_step):
        # Anneal exploration rate
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - (self.epsilon_decay * n_step))
        if self.is_prioritized:
            self.memory.update_beta(n_step)

    def update_target(self):
        self.approximator.update_target_network()

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.approximator.device)
            self.update_approximator(sampled_data)

    def update_approximator(self, sampled_data):
        if self.is_prioritized:
            (states, actions, rewards, next_states, dones
             ), priorities = sampled_data
        else:
            states, actions, rewards, next_states, dones = sampled_data

        # Compute Q-values using the approximator
        q_values = self.approximator.compute_q(states, actions)
        td_targets = self.approximator.compute_ddqn_target(
            rewards, q_values, next_states, self.discount_factor, dones)

        # Compute the loss and backpropagate
        losses = self.approximator.loss_fn(q_values, td_targets)
        if self.is_prioritized:
            td_errors = q_values - td_targets
            # Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            # Update replay buffer priorities
            self.memory.update_priorities(priorities['indexes'],
                                          new_priorities)
            loss = torch.mean(losses * priorities['weights'])
        else:
            loss = torch.mean(losses)
        self.approximator.optimize(loss)

    def save(self, path):
        self.approximator.save(str(path) + ".pth")

    def load(self, path):
        self.approximator.load(str(path) + ".pth", eval_only=True)


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
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
