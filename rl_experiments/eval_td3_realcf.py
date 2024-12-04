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
    environment_name = 'webots_drone:webots_drone/RealCrazyflieEnvContinuous-v0'
    env_params = load_json_dict(logpath / 'args_environment.json')
    env_params = args2params(args, env_params)
    env_params['agent_id'] = 8
    env_params['timestep'] = 32

    try:
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
            agent_class = SRLTD3Agent
            policy = SRLTD3Function
        else:
            agent_class = TD3Agent
            policy = TD3Function

        print('state_shape', env_params['state_shape'])
        print('action_shape', agent_params['action_shape'])

        eval_logpath = logpath / 'eval_real'
        log_params = {'n_sensors': 0, 'extra_info': False,
                      'other_cols': ['battery_volts']}
        iterate_agents_evaluation(agent_paths, agent_class, agent_params,
                                  policy, approximator_params, env,
                                  args.target_pos, args.eval_steps, args.episode,
                                  eval_logpath, log_params, record_video=args.record)

    # safety ensurance in case of any error, the drone will land
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)

    except KeyboardInterrupt:
        print('QUITING....')

    finally:
        env.close()


if __name__ == '__main__':
    import traceback
    run_evaluation(parse_args())
