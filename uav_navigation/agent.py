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


class ReplayBuffer:
    def __init__(self, buffer_size, state_shape, action_shape):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.clear()

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        if type(action) == list:
            self.actions[self.index] = action
        else:
            dims = (self.action_shape[0], self.action_shape[0])
            self.actions[self.index] = np.eye(*dims)[action]
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, device=None):
        indices = np.random.choice(self.size, batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        if device is None:
            return states, actions, rewards, next_states, dones
        else:
            return (
                torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.float32).to(device),
                torch.tensor(rewards, dtype=torch.float32).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device)
            )

    def get_size(self):
        return self.size

    def clear(self):
        state_shape = self.buffer_size, *self.state_shape
        action_shape = self.buffer_size, *self.action_shape
        self.states = np.zeros(state_shape, dtype=np.float32)
        self.actions = np.zeros(action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.next_states = np.zeros(state_shape, dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.index = 0
        self.size = 0

    def __len__(self):
        return self.get_size()


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
                 epsilon_decay=0.9999,
                 buffer_capacity=2048):
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.action_space_size = action_space_shape[0]
        self.approximator_tau = approximator_tau  # Soft update parameter
        self.device = torch.device(device)

        # Q-networks
        self.q_network = approximator(
            state_space_shape, action_space_shape).to(self.device)
        self.target_q_network = approximator(
            state_space_shape, action_space_shape).to(self.device)
        # Initialize target network with Q-network parameters
        self._update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=approximator_lr,
                                    betas=(approximator_beta, 0.999))

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_capacity, state_space_shape,
                                   action_space_shape)

    def _update_target_network(self):
        soft_update_params(net=self.q_network,
                           target_net=self.target_q_network,
                           tau=self.approximator_tau)

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

    def update(self):
        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            sampled_data = self.memory.sample(self.BATCH_SIZE, device=self.device)
            self._update_q_network(sampled_data)

        # Anneal exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def _update_q_network(self, sampled_data):
        action_argmax = sampled_data[1].argmax(1)
        # Compute Q-values using the Q-network
        current_q_values = self.q_network(sampled_data[0]).gather(
            dim=1, index=action_argmax.unsqueeze(1))

        # Use the target network for the next Q-values
        next_q_values = self.target_q_network(sampled_data[3]
                                              ).max(dim=1)[0].detach()
        target_q_values = sampled_data[2] + self.discount_factor * \
            (1 - sampled_data[4]) * next_q_values

        # Compute the loss and backpropagate
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        self._update_target_network()

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
        self._update_target_network()

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

        # Soft update the target network
        self._update_target_network()
