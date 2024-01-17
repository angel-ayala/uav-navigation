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
from .utils import soft_update_params
from .utils import profile_model
from .memory import PrioritizedReplayBuffer


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


class DDQNAgent:
    BATCH_SIZE = 32

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
                 epsilon_steps=500000,
                 memory_buffer=None):
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_steps

        self.action_space_size = action_space_shape[0]
        self.approximator_tau = approximator_tau  # Soft update parameter
        self.device = torch.device(device)
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        # Q-networks
        self.q_network = approximator(
            state_space_shape, action_space_shape).to(self.device)
        self.target_q_network = approximator(
            state_space_shape, action_space_shape).to(self.device)
        # Initialize target network with Q-network parameters
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=approximator_lr,
                                    betas=(approximator_beta, 0.999))

        # Replay Buffer
        self.memory = memory_buffer
        self.update_epsilon(0)

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)  # Explore
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()
            return np.argmax(q_values)  # Exploit

    def update_epsilon(self, n_step):
        # Anneal exploration rate
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - (self.epsilon_decay * n_step))
        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.update_beta(n_step)

    def update_target_network(self):
        # Soft update the target network
        soft_update_params(net=self.q_network,
                           target_net=self.target_q_network,
                           tau=self.approximator_tau)

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(
                self.BATCH_SIZE, device=self.device)
            self._update_q_network(sampled_data)

    def _update_q_network(self, sampled_data):
        if isinstance(self.memory, PrioritizedReplayBuffer):
            (states, actions, rewards, next_states, dones
             ), priorities = sampled_data
        else:
            states, actions, rewards, next_states, dones = sampled_data

        actions_argmax = actions.argmax(-1)
        # Compute Q-values using the Q-network
        current_q_values = self.q_network(states).gather(
            dim=-1, index=actions_argmax.unsqueeze(-1)).squeeze(-1)

        # Use the target network for the next Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            next_actions_argmax = torch.argmax(next_q_values, -1)

            double_q_values = self.target_q_network(next_states).gather(
                dim=-1, index=next_actions_argmax.unsqueeze(-1)).squeeze(-1)

            td_targets = rewards + self.discount_factor * double_q_values * (1 - dones)
            if isinstance(self.memory, PrioritizedReplayBuffer):
                td_errors = current_q_values - td_targets

        # Compute the loss and backpropagate
        losses = self.huber_loss(current_q_values, td_targets)
        if isinstance(self.memory, PrioritizedReplayBuffer):
            # Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
            new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            # Update replay buffer priorities
            self.memory.update_priorities(priorities['indexes'],
                                          new_priorities)
            loss = torch.mean(losses * priorities['weights'])
        else:
            loss = torch.mean(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Ensure the models are in evaluation mode after loading
        self.q_network.eval()
        self.target_q_network.eval()


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
