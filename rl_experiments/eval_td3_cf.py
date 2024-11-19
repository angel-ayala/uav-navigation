#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import numpy as np
import torch
from pathlib import Path

from uav_navigation.td3.agent import TD3Agent, TD3Function
from uav_navigation.td3.srl import SRLTD3Agent, SRLTD3Function
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.utils import load_json_dict

from learn_cf import instance_env
from learn_cf import wrap_env
from eval_cf import iterate_agents_evaluation
from eval_cf import parse_args


def run_evaluation(seed_val, logpath, episode):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # Define constants
    logpath = Path(logpath)
    agent_paths = list(logpath.glob('**/agent_ep_*_actor*'))
    agent_paths.sort()

    # Environment args
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
    env_params = load_json_dict(logpath / 'args_environment.json')

    target_pos = args.target_pos

    if not args.load_config:
        env_params['time_limit_seconds'] = args.time_limit
        env_params['frame_skip'] = args.frame_skip
        env_params['goal_threshold'] = args.goal_threshold
        env_params['init_altitude'] = args.init_altitude
        env_params['altitude_limits'] = args.altitude_limits
        env_params['target_dim'] = args.target_dim

    # Create the environment
    env, _ = instance_env(env_params, name=environment_name)
    # Observation preprocessing
    env, _ = wrap_env(env, env_params)

    if target_pos is None:
        target_pos = list(range(len(env.quadrants)))

    # Agent params
    agent_params = load_json_dict(logpath / 'args_agent.json')
    is_srl = agent_params['is_srl']
    del agent_params['is_srl']
    approximator_params = agent_params['approximator']
    approximator_params['obs_space'] = env.observation_space
    approximator_params['action_range'] = [env.action_space.low, env.action_space.high]

    if is_srl:
        agent_class = SRLTD3Agent
        policy = SRLTD3Function
    else:
        agent_class = TD3Agent
        policy = TD3Function

    print('state_shape', env_params['state_shape'])
    print('action_shape', agent_params['action_shape'])

    # Instantiate an init evaluation
    agent_params.update(
        dict(approximator=policy(**approximator_params)))

    eval_logpath = logpath / 'eval'
    log_params = {'n_sensors': 4}
    iterate_agents_evaluation(agent_class, agent_params, agent_paths, env,
                              target_pos, args.eval_steps, episode,
                              eval_logpath, log_params, record_video=args.record)


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args.seed, args.logspath, args.episode)
