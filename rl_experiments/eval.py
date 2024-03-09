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
from uav_navigation.stack import ObservationStack
from learn import list_of_float
from learn import list_of_int

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import TargetVectorObservation

def parse_args():
    parser = argparse.ArgumentParser()    # misc
    parser.add_argument('--logspath', type=str,
                        default='logs/ddqn-srl_2023-11-28_00-13-33',
                        help='Log path with training results.')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--episode', type=int, default=-1,
                        help='Indicate the episode number to execute, set -1 for all of them')
    parser.add_argument('--eval-steps', type=int, default=300,  # 1m at 25 frames
                        help='Epsilon value used for evaluation.')
    parser.add_argument("--load-config", action='store_true',
                        help="Whether if force config file's value argument'.")
    parser.add_argument('--render', action='store_true',
                        help='Specific if show or not Env.render.')

    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--time-limit", type=int, default=600,  # 10m
                         help='Max time (seconds) of the mission.')
    arg_env.add_argument("--frame-skip", type=int, default=25,  # 200ms
                         help='Number of simulation steps for a RL step')
    arg_env.add_argument("--goal-threshold", type=float, default=5.,
                         help='Minimum distance from the target.')
    arg_env.add_argument("--init-altitude", type=float, default=25.,
                         help='Minimum height distance to begin the mission.')
    arg_env.add_argument("--altitude-limits", type=list_of_float,
                         default=[11., 75.], help='Vertical flight limits.')
    arg_env.add_argument("--target-pos", type=list_of_int, default=[0, 1, 2, 3],
                         help='Cuadrant number for target position.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[7., 3.5],
                         help="Target's dimension size.")

    args = parser.parse_args()
    return args


def run_evaluation(seed_val, logpath, episode):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # Define constants
    logpath = Path(logpath)
    agents_path = list(logpath.glob('**/*.pth'))
    agents_path.sort()
    episode = episode - 1

    # Environment args
    environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
    env_params = load_json_dict(logpath / 'args_environment.json')
    frame_stack = env_params['frame_stack']
    del env_params['frame_stack']
    add_target = False
    if 'add_target' in env_params.keys():
        add_target = env_params['add_target']
        del env_params['add_target']

    target_pos = args.target_pos

    if not args.load_config:
        env_params['time_limit_seconds'] = args.time_limit
        env_params['frame_skip'] = args.frame_skip
        env_params['goal_threshold'] = args.goal_threshold
        env_params['init_altitude'] = args.init_altitude
        env_params['altitude_limits'] = args.altitude_limits
        env_params['fire_dim'] = args.target_dim

    # Create the environment
    env = gym.make(environment_name, **env_params)
    if not env_params['is_pixels']:
        env = ReducedVectorObservation(env)
    if add_target:
        env = TargetVectorObservation(env)
    if frame_stack > 1:
        env = ObservationStack(env, k=frame_stack)

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
            logpath / f"history_eval_{log_ep+1:03d}.csv")
        for fc in target_pos:
            evaluate_agent(agent, env, training_params['eval_epsilon'],
                           args.eval_steps,
                           fire_cuadrant=fc,
                           step_callback=store_callback)


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args.seed, args.logspath, args.episode)
