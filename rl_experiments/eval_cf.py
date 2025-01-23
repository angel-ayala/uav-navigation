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

from uav_navigation.agent import DDQNAgent, QFunction
from uav_navigation.net import QNetwork, QFeaturesNetwork
from uav_navigation.srl.agent import SRLDDQNAgent, SRLQFunction
from uav_navigation.utils import load_json_dict
from uav_navigation.utils import evaluate_agent
from uav_navigation.logger import summary_create
from uav_navigation.logger import summary_step

from webots_drone.data import StoreStepData
from webots_drone.data import VideoCallback

from learn_cf import list_of_float
from learn_cf import list_of_int
from learn_cf import instance_env
from learn_cf import wrap_env


def list_of_targets(arg):
    if 'random' in arg or 'sample' in arg:
        return arg
    return list_of_int(args)


def parse_args():
    parser = argparse.ArgumentParser()    # misc
    parser.add_argument('--logspath', type=str,
                        default='logs/ddqn-srl_2023-11-28_00-13-33',
                        help='Log path with training results.')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--episode', type=int, default=-1,
                        help='Indicate the episode number to execute, set -1 for all of them')
    parser.add_argument('--eval-steps', type=int, default=60,  # 1m at 25 frames
                        help='Epsilon value used for evaluation.')
    parser.add_argument("--load-config", action='store_true',
                        help="Whether if force config file's value argument'.")
    parser.add_argument('--record', action='store_true',
                        help='Specific if record or not a video simulation.')

    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--time-limit", type=int, default=60,  # 10m
                         help='Max time (seconds) of the mission.')
    arg_env.add_argument("--frame-skip", type=int, default=6,  # 200ms
                         help='Number of simulation steps for a RL step')
    arg_env.add_argument("--goal-threshold", type=float, default=0.25,
                         help='Minimum distance from the target.')
    arg_env.add_argument("--init-altitude", type=float, default=1.0,
                         help='Minimum height distance to begin the mission.')
    arg_env.add_argument("--altitude-limits", type=list_of_float,
                         default=[0.25, 2.], help='Vertical flight limits.')
    arg_env.add_argument("--target-pos", type=list_of_targets, default=None,
                         help='Cuadrant number for target position.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[0.05, 0.02],
                         help="Target's dimension size.")

    args = parser.parse_args()
    return args


def args2params(args, env_params):
    if not args.load_config:
        env_params['time_limit_seconds'] = args.time_limit
        env_params['frame_skip'] = args.frame_skip
        env_params['goal_threshold'] = args.goal_threshold
        env_params['init_altitude'] = args.init_altitude
        env_params['altitude_limits'] = args.altitude_limits
        env_params['target_dim'] = args.target_dim

    return env_params


def args2target(env, arg_tpos):
    target_pos = arg_tpos
    if arg_tpos is None:
        target_pos = list(range(len(env.quadrants)))
    elif 'sample' in arg_tpos:
        target_pos = np.random.choice(range(len(env.quadrants)),
                                      int(target_pos.replace('sample-', '')),
                                      replace=False)
    elif arg_tpos == 'random':
        target_pos = [env.vtarget.get_random_position(env.flight_area)]
    return target_pos


def iterate_agents_evaluation(agent_paths, agent_class, agent_params,
                              approximator, approximator_params, env,
                              target_pos, eval_steps, episode, logpath,
                              log_params, record_video=False):
    summary_create(logpath.parent, logpath.name)
    # Target position for evaluation
    target_pos = args2target(env, target_pos)
    # Video recording callback
    vidcb = VideoCallback(logpath / "videos", env) if record_video else None
    for agent_path in agent_paths:
        agent_name = agent_path.name
        log_ep = int(str(agent_name).replace('agent_ep_', '').split("_")[0]) - 1
        if episode > 0 and log_ep != episode:
            continue
        print('Loading', agent_name[:12])
        # Instantiate an init evaluation
        agent_params.update(
            dict(approximator=approximator(**approximator_params)))
        agent = agent_class(**agent_params)
        agent.load(agent_path.parent / agent_name[:12])
        store_callback = StoreStepData(
            logpath / f"history_{log_ep:03d}.csv", **log_params)
        store_callback._ep = log_ep
        summary_step(log_ep)
        agent.eval_mode()
        for tq in target_pos:
            if vidcb is not None:
                vidcb.start_recording(f"ep{log_ep:03d}_tq{tq:02d}.mp4")
            evaluate_agent(agent, env, eval_steps, target_quadrant=tq,
                           step_callback=store_callback)
            if vidcb is not None:
                vidcb.stop_recording()


def run_evaluation(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define constants
    logpath = Path(args.logspath)
    agent_paths = list(logpath.glob('**/agent_ep_*_q*'))
    agent_paths.sort()

    # Environment args
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvDiscrete-v0'
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
    if is_srl:
        agent_class = SRLDDQNAgent
        q_approximator = SRLQFunction
        approximator_params['q_app_fn'] = QNetwork
    else:
        agent_class = DDQNAgent
        q_approximator = QFunction
        approximator_params['q_app_fn'] = QFeaturesNetwork\
            if env_params['is_pixels'] else QNetwork

    print('state_shape', env_params['state_shape'])
    print('action_shape', agent_params['action_shape'])

    # Instantiate an init evaluation
    eval_logpath = logpath / 'eval'
    log_params = {'n_sensors': 4}
    iterate_agents_evaluation(agent_paths, agent_class, agent_params,
                              q_approximator, approximator_params, env,
                              args.target_pos, args.eval_steps, args.episode,
                              eval_logpath, log_params, record_video=args.record)


if __name__ == '__main__':
    run_evaluation(parse_args())
