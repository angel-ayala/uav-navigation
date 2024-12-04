#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import numpy as np
import torch
from pathlib import Path

from uav_navigation.sac.agent import SACAgent, SACFunction
from uav_navigation.sac.srl import SRLSACAgent, SRLSACFunction
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.utils import load_json_dict

from learn_cf import instance_env
from learn_cf import wrap_env
from eval_cf import iterate_agents_evaluation
from eval_cf import parse_args
from eval_cf import args2params


def run_evaluation(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define constants
    logpath = Path(args.logspath)
    agent_paths = list(logpath.glob('**/agent_ep_*_actor*'))
    agent_paths.sort()

    # Environment args
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
    env_params = load_json_dict(logpath / 'args_environment.json')
    env_params = args2params(args, env_params)

    # Create the environment
    env, _ = instance_env(env_params, name=environment_name)
    # Observation preprocessing
    env, _ = wrap_env(env, env_params)

    # Agent params
    agent_params = load_json_dict(logpath / 'args_agent.json')
    is_srl = agent_params['is_srl']
    del agent_params['is_srl']
    approximator_params = agent_params['approximator']
    approximator_params['obs_space'] = env.observation_space
    approximator_params['action_range'] = [env.action_space.low, env.action_space.high]

    if is_srl:
        agent_class = SRLSACAgent
        policy = SRLSACFunction
    else:
        agent_class = SACAgent
        policy = SACFunction

    print('state_shape', env_params['state_shape'])
    print('action_shape', agent_params['action_shape'])

    # Instantiate an init evaluation
    eval_logpath = logpath / 'eval'
    log_params = {'n_sensors': 4}
    iterate_agents_evaluation(agent_paths, agent_class, agent_params, policy,
                              approximator_params, env, args.target_pos,
                              args.eval_steps, args.episode, eval_logpath,
                              log_params, record_video=args.record)


if __name__ == '__main__':
    run_evaluation(parse_args())
