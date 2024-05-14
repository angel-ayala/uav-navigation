#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:33:17 2023

@author: Angel Ayala
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from uav_navigation.sac.agent import SACAgent, ACFunction
from uav_navigation.sac.agent import profile_actor_critic
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.sac.srl import SRLSACAgent, SRLSACFunction
from uav_navigation.sac.srl import profile_srl_approximator
from uav_navigation.utils import load_json_dict
from uav_navigation.utils import evaluate_agent

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import ReducedActionSpace

from learn import list_of_float
from learn import list_of_int
from learn import instance_env
from learn import wrap_env


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
    parser.add_argument('--position-only', action='store_true',
                        help='Specific if show or not Env.render.')

    args = parser.parse_args()
    return args


def action_str2array(action_str):
    numbers = action_str.split()
    if numbers[0] == '[':
        numbers = numbers[1:]
    if numbers[-1] == ']':
        numbers = numbers[:-1]
    daction = " ".join(numbers)
    daction = daction.replace('[', '').replace(']', '')
    # if len(numbers) == 3:
    #     daction += ' 0.'
    action_array = np.fromstring(daction, dtype=np.float32, sep=' ')
    return action_array


def run_evaluation(seed_val, logpath, episode):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    # Define constants
    logpath = Path(logpath)
    agent_paths = [lp.name[:12] for lp in logpath.glob('**/agent_ep_*_actor_critic*')]
    agent_paths.sort()
    episode = episode - 1

    # Environment args
    environment_name = 'webots_drone:webots_drone/DroneEnvContinuous-v0'
    env_params = load_json_dict(logpath / 'args_environment.json')

    target_pos = args.target_pos

    if not args.load_config:
        env_params['time_limit_seconds'] = args.time_limit
        env_params['frame_skip'] = args.frame_skip
        env_params['goal_threshold'] = args.goal_threshold
        env_params['init_altitude'] = args.init_altitude
        env_params['altitude_limits'] = args.altitude_limits
        env_params['fire_dim'] = args.target_dim

    # Create the environment
    env, _ = instance_env(env_params, name=environment_name)
    env = ReducedActionSpace(env)
    # Observation preprocessing
    env, _ = wrap_env(env, env_params)

    # Agent params
    agent_params = load_json_dict(logpath / 'args_agent.json')
    approximator_params = agent_params['approximator']
    approximator_params['actor_max_a'] = action_str2array(approximator_params['actor_max_a'])
    approximator_params['actor_min_a'] = action_str2array(approximator_params['actor_min_a'])

    if agent_params['is_srl']:
        agent_class = SRLSACAgent
        policy = SRLSACFunction
        function_profiler = profile_srl_approximator
        # approximator_params['q_app_fn'] = q_function
    else:
        agent_class = SACAgent
        policy = ACFunction
        function_profiler = profile_actor_critic
        if env_params['is_pixels']:
            approximator_params['preprocess'] = QFeaturesNetwork(
                env_params['state_shape'], env_params['action_shape'], only_cnn=True)
            approximator_params['latent_dim'] = approximator_params['preprocess'].n_features
        else:
            approximator_params['latent_dim'] = env_params['state_shape']
    print('state_shape', env_params['state_shape'])
    print('action_shape', env_params['action_shape'])
    # Profile the approximation function computational costs
    approximation_function = policy(**approximator_params)
    print('====== Full model computational demands ======')
    if agent_params['is_srl']:
        approximation_function.append_autoencoders(agent_params['ae_models'])
    function_profiler(approximation_function, env_params['state_shape'],
                      env_params['action_shape'])
    del approximation_function
    # Profile encoder stage only computational costs
    approximation_function = policy(**approximator_params)
    print('\n====== Reduced inference-only computational demands ======')
    if agent_params['is_srl']:
        approximation_function.load(logpath / agent_paths[0],
                                    ae_models=agent_params['ae_models'],
                                    encoder_only=True)
    function_profiler(approximation_function, env_params['state_shape'],
                      env_params['action_shape'])
    del agent_params['is_srl']
    # Instantiate an init evaluation
    agent_params['approximator'] = policy(**approximator_params)
    agent = agent_class(**agent_params)
    training_params = load_json_dict(logpath / 'args_training.json')
    for log_ep, agent_path in enumerate(agent_paths):
        if episode > 0 and log_ep != episode:
            continue
        print('Loading from', "/".join(str(agent_path).split("/")[-3:]))
        agent = agent_class(**agent_params)
        agent.load(logpath / agent_path)
        store_callback = StoreStepData(
            logpath / f"history_eval_{log_ep+1:03d}.csv")
        store_callback._ep = log_ep
        for fc in target_pos:
            evaluate_agent(agent, env, args.eval_steps, False,
                           fire_cuadrant=fc,
                           step_callback=store_callback)


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args.seed, args.logspath, args.episode)
