import torch
import numpy as np


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
