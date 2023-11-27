#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import torch
import numpy as np

import gym
from pathlib import Path
from uav_navigation.agent import DQNAgent
from uav_navigation.net import QNetwork
from uav_navigation.utils import load_json_dict
from uav_navigation.utils import eval_agent
from uav_navigation.utils import PreprocessObservation

from webots_drone.data import StoreStepData

seed_val = 666
torch.manual_seed(seed_val)
np.random.seed(seed_val)


# Define constants
num_tries = 100
episode = 30
LOGS = Path('logs/cartpole_2023-11-26_21-11-41')
agents_path = list(LOGS.glob('**/*.pth'))
agents_path.sort()
agent_path = agents_path[episode + 1]
print('Loading from', agent_path)

# Environment args
environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
env_params = load_json_dict(LOGS / 'args_environment.json')
# Create the environment
env = gym.make(environment_name, **env_params)
env = PreprocessObservation(env)

# Agent params
agent_params = load_json_dict(LOGS / 'args_agent.json')
if 'QNetwork' in agent_params['approximator']:
    agent_params['approximator'] = QNetwork
print('state_space_shape', agent_params['state_space_shape'])
print('action_space_shape', agent_params['action_space_shape'])

agent = DQNAgent(**agent_params)
agent.load(agent_path)

store_callback = StoreStepData(LOGS / f"history_eval_{episode}.csv")

# RL training
training_params = load_json_dict(LOGS / 'args_training.json')
for i in range(num_tries):
    eval_agent(agent, env, training_params['eval_epsilon'],
               step_callback=store_callback)
