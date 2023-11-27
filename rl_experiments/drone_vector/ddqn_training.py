#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import torch
import numpy as np

import gym
import datetime
from pathlib import Path
from uav_navigation.agent import DQNAgent
from uav_navigation.net import QNetwork
from uav_navigation.utils import save_dict_json
from uav_navigation.utils import train_eval_agent
from uav_navigation.utils import PreprocessObservation

from webots_drone.data import StoreStepData


seed_val = 666
torch.manual_seed(seed_val)
np.random.seed(seed_val)


# Environment args
environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
is_pixels = False
env_params = dict(time_limit_seconds=60,  # 1 min
                  max_no_action_seconds=5,  # 5 sec
                  frame_skip=25,  # 200ms
                  goal_threshold=5.,
                  init_altitude=25.,
                  altitude_limits=[11, 75],
                  fire_pos=[-40, 40],
                  fire_dim=[7., 3.5],
                  is_pixels=is_pixels)
# Create the environment
env = gym.make(environment_name, **env_params)
env = PreprocessObservation(env, is_pixels=is_pixels)

# Agent args
agent_params = dict(state_space_shape=env.observation_space.shape,
                    action_space_shape=(env.action_space.n, ),
                    approximator=QNetwork,
                    device='cpu',
                    learning_rate=0.0001,
                    discount_factor=0.99,
                    epsilon_start=0.5,
                    epsilon_end=0.01,
                    epsilon_decay=0.9999,
                    buffer_capacity=2048,
                    tau=0.001)
print(agent_params['state_space_shape'])
print(agent_params['action_space_shape'])

agent = DQNAgent(**agent_params)

# Summary folder
folder_name = './logs/ddqn_' + datetime.datetime.now(
    ).strftime('%Y-%m-%d_%H-%M-%S')
folder_name = Path(folder_name)
folder_name.mkdir(parents=True)

store_callback = StoreStepData(folder_name / 'history.csv')

# RL training
train_eval_params = dict(training_steps=1000000,
                         save_steps=10000,
                         mem_steps=2048,
                         eval_epsilon=0.01,)


save_dict_json(env_params, folder_name / 'args_environment.json')
save_dict_json(agent_params, folder_name / 'args_agent.json')
save_dict_json(train_eval_params, folder_name / 'args_training.json')

train_eval_agent(agent, env, step_callback=store_callback, outpath=folder_name,
                 **train_eval_params)
