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


class DQNAgent:
    BATCH_SIZE = 32

    def __init__(self, state_space_shape, action_space_shape, approximator,
                 device=None, learning_rate=0.0001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=0.9999,
                 buffer_capacity=2048, tau=0.001):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_space_size = action_space_shape[0]
        self.tau = tau  # Soft update parameter
        self.device = torch.device(device)

        # Q-networks
        self.q_network = approximator(
            state_space_shape, action_space_shape).to(self.device)
        self.target_q_network = approximator(
            state_space_shape, action_space_shape).to(self.device)
        # Initialize target network with Q-network parameters
        self._update_target_network()

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_capacity, state_space_shape,
                                   action_space_shape)

    def _update_target_network(self):
        # Soft update: target_network = tau * Q-network + (1 - tau) * target_network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)  # Explore
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()
            return np.argmax(q_values)  # Exploit

    def update(self, state, action, reward, next_state, done):
        # Store the transition in the replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Update the Q-network if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            self._update_q_network()

        # Anneal exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def _update_q_network(self):
        sampled_data = self.memory.sample(self.BATCH_SIZE, device=self.device)
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
