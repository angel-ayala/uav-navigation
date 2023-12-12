#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import torch
import numpy as np

import gym
import argparse
from pathlib import Path
from uav_navigation.srl.agent import AEDDQNAgent
from uav_navigation.srl.agent import profile_agent
from uav_navigation.srl.net import VectorApproximator
from uav_navigation.srl.net import PixelApproximator
from uav_navigation.utils import load_json_dict
from uav_navigation.utils import evaluate_agent
from uav_navigation.utils import PreprocessObservation

from webots_drone.data import StoreStepData


def parse_args():
    parser = argparse.ArgumentParser()    # misc
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument(
        '--logs-path',
        default='drone_vector/logs/ddqn-srl_2023-11-28_00-13-33',
        type=str)
    parser.add_argument('--episode', default=-1, type=int)
    parser.add_argument('--render', default=False, action='store_true')

    args = parser.parse_args()
    return args


def run_evaluation(seed_val, log_path, episode):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # Define constants
    log_path = Path(log_path)
    agents_path = list(log_path.glob('**/*.pth'))
    agents_path.sort()

    # Environment args
    environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
    env_params = load_json_dict(log_path / 'args_environment.json')
    # Create the environment
    env = gym.make(environment_name, **env_params)
    env = PreprocessObservation(env)

    # Agent params
    agent_params = load_json_dict(log_path / 'args_agent.json')
    if 'VectorApproximator' in agent_params['approximator']:
        agent_params['approximator'] = VectorApproximator
    if 'PixelApproximator' in agent_params['approximator']:
        agent_params['approximator'] = PixelApproximator
    print('state_space_shape', agent_params['state_space_shape'])
    print('action_space_shape', agent_params['action_space_shape'])

    agent = AEDDQNAgent(**agent_params)
    profile_agent(agent,
                  agent_params['state_space_shape'],
                  agent_params['action_space_shape'])

    training_params = load_json_dict(log_path / 'args_training.json')
    if episode > 0:
        agents_path = [agents_path[episode]]

    for episode, agent_path in enumerate(agents_path):
        print('Loading from', agent_path)
        agent = AEDDQNAgent(**agent_params)
        agent.load(agent_path)
        store_callback = StoreStepData(
            log_path / f"history_eval_{episode:03d}.csv")
        evaluate_agent(agent, env, training_params['eval_epsilon'],
                       step_callback=store_callback)

if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args.seed, args.logs_path, args.episode)