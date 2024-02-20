#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import gym
import argparse
import numpy as np
import torch
from pathlib import Path

from uav_navigation.srl.agent import AEDDQNAgent
from uav_navigation.srl.net import VectorApproximator
from uav_navigation.srl.net import PixelApproximator
from uav_navigation.srl.agent import profile_agent as profile_agent_srl

from uav_navigation.agent import DDQNAgent
from uav_navigation.net import QNetwork
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.agent import profile_agent

from uav_navigation.utils import load_json_dict
from uav_navigation.utils import evaluate_agent
from uav_navigation.utils import ReducedVectorObservation

from webots_drone.data import StoreStepData


def parse_args():
    parser = argparse.ArgumentParser()    # misc
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--logspath', type=str,
                        default='logs/ddqn-srl_2023-11-28_00-13-33',
                        help='Log path with training results.')
    parser.add_argument('--episode', type=int, default=-1,
                        help='Indicate the episode number to execute, set -1 for all of them')
    parser.add_argument('--render', action='store_true',
                        help='Specific if show or not Env.render.')

    args = parser.parse_args()
    return args


def run_evaluation(seed_val, logpath, episode):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # Define constants
    logpath = Path(logpath)
    agents_path = list(logpath.glob('**/*.pth'))
    agents_path.sort()

    # Environment args
    environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
    env_params = load_json_dict(logpath / 'args_environment.json')
    frame_stack = env_params['frame_stack']
    del env_params['frame_stack']
    # Create the environment
    env = gym.make(environment_name, **env_params)
    if not env_params['is_pixels']:
        env = ReducedVectorObservation(env)
    if frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=frame_stack)

    # Agent params
    agent_params = load_json_dict(logpath / 'args_agent.json')
    if 'VectorApproximator' in agent_params['approximator']:
        agent_params['approximator'] = VectorApproximator
        agent_class = AEDDQNAgent
        agent_profiler = profile_agent_srl
    elif 'PixelApproximator' in agent_params['approximator']:
        agent_params['approximator'] = PixelApproximator
        agent_class = AEDDQNAgent
        agent_profiler = profile_agent_srl
    elif 'QFeaturesNetwork' in agent_params['approximator']:
        agent_params['approximator'] = QFeaturesNetwork
        agent_class = DDQNAgent
        agent_profiler = profile_agent
    elif 'QNetwork' in agent_params['approximator']:
        agent_params['approximator'] = QNetwork
        agent_class = DDQNAgent
        agent_profiler = profile_agent
    print('state_space_shape', agent_params['state_space_shape'])
    print('action_space_shape', agent_params['action_space_shape'])

    agent = agent_class(**agent_params)
    agent_profiler(agent,
                   agent_params['state_space_shape'],
                   agent_params['action_space_shape'])

    training_params = load_json_dict(logpath / 'args_training.json')
    for log_ep, agent_path in enumerate(agents_path):
        if episode > 0 and log_ep != episode:
            continue
        print('Loading from', "/".join(str(agent_path).split("/")[-3:]))
        agent = agent_class(**agent_params)
        agent.load(agent_path)
        store_callback = StoreStepData(
            logpath / f"history_eval_{log_ep:03d}.csv")
        evaluate_agent(agent, env, training_params['eval_epsilon'],
                       step_callback=store_callback)


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args.seed, args.logspath, args.episode)
