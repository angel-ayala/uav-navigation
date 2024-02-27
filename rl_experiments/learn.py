#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:45:58 2023

@author: Angel Ayala
"""

import torch
import numpy as np
import argparse

import gym
import datetime
from pathlib import Path

from uav_navigation.srl.agent import AEDDQNAgent
from uav_navigation.srl.net import VectorApproximator
from uav_navigation.srl.net import PixelApproximator

from uav_navigation.agent import DDQNAgent
from uav_navigation.net import QNetwork
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.memory import ReplayBuffer, PrioritizedReplayBuffer

from uav_navigation.utils import save_dict_json
from uav_navigation.utils import run_agent
from uav_navigation.utils import ReducedVectorObservation
from uav_navigation.stack import ObservationStack


from webots_drone.data import StoreStepData


def list_of_float(arg):
    return list(map(float, arg.split(',')))


def xy_coordinates(arg):
    if arg.lower() == 'random':
        return None
    return list_of_float(arg)


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument("--time-limit", type=int, default=600,  # 10m
                         help='Max time (seconds) of the mission.')
    arg_env.add_argument("--time-no-action", type=int, default=5,
                         help='Max time (seconds) with no movement.')
    arg_env.add_argument("--frame-skip", type=int, default=25,  # 200ms
                         help='Number of simulation steps for a RL step')
    arg_env.add_argument("--frame-stack", type=int, default=1,
                         help='Number of RL step to stack as observation.')
    arg_env.add_argument("--goal-threshold", type=float, default=5.,
                         help='Minimum distance from the target.')
    arg_env.add_argument("--init-altitude", type=float, default=25.,
                         help='Minimum height distance to begin the mission.')
    arg_env.add_argument("--altitude-limits", type=list_of_float,
                         default=[11., 75.], help='Vertical flight limits.')
    arg_env.add_argument("--target-pos", type=xy_coordinates, default=[-40., 40.],
                         help='Initial position of the target.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[7., 3.5],
                         help="Target's dimension size.")
    arg_env.add_argument("--is-pixels", action='store_true',
                         help='Whether if state is image-based or vector-based.')

    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--approximator-lr", type=float, default=10e-4,
                           help='Q approximation function Adam learning rate.'
                           'default as recommended in: '
                           '[Interference and Generalization in Temporal '
                           'Difference Learning]('
                           'https://proceedings.mlr.press/v119/bengio20a.htm)')
    arg_agent.add_argument("--approximator-beta", type=float, default=0.9,
                           help='Q approximation function Adam \beta.')
    arg_agent.add_argument("--approximator-tau", type=float, default=0.1,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--epsilon-start", type=float, default=1.0,
                           help='Initial epsilon value for exploration.')
    arg_agent.add_argument("--epsilon-end", type=float, default=0.01,
                           help='Final epsilon value for exploration.')
    arg_agent.add_argument("--epsilon-steps", type=int, default=90000,  # 5h at 25 frames
                           help='Number of steps to reach minimum value for Epsilon.')
    arg_agent.add_argument("--memory-capacity", type=int, default=65536,  # 2**16
                           help='Maximum number of transitions in the Experience replay buffer.')
    arg_agent.add_argument("--memory-prioritized", action='store_true',
                           help='Whether if memory buffer is Prioritized experiencie replay or not.')
    arg_agent.add_argument("--prioritized-alpha", type=float, default=0.6,
                           help='Alpha prioritization exponent for PER.')
    arg_agent.add_argument("--prioritized-initial-beta", type=float, default=0.4,
                           help='Beta bias for sampling for PER.')
    arg_agent.add_argument("--approximator-momentum", type=float, default=.9,
                           help='Momentum factor factor for the SGD using'
                           'using nesterov')

    arg_srl = parser.add_argument_group(
        'State representation learning variation')
    arg_srl.add_argument("--is-srl", action='store_true',
                         help='Whether if method is SRL-based or not.')
    arg_srl.add_argument("--latent-dim", type=int, default=50,
                         help='Number of features in the latent representation Z.')
    arg_srl.add_argument("--hidden-dim", type=int, default=256,
                         help='Number of units in the hidden layers.')
    arg_srl.add_argument("--num-filters", type=int, default=32,
                         help='Number of filters in the CNN hidden layers.')
    arg_srl.add_argument("--num-layers", type=int, default=2,
                         help='Number of hidden layers.')
    arg_srl.add_argument("--encoder-lr", type=float, default=1e-3,
                         help='Encoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-lr", type=float, default=1e-3,
                         help='Decoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-latent-lambda", type=float, default=1e-6,
                         help='Decoder regularization \lambda value.')
    arg_srl.add_argument("--decoder-weight-decay", type=float, default=1e-7,
                         help='Decoder function Adam weight decay value.')

    arg_training = parser.add_argument_group('Training')
    arg_training.add_argument("--steps", type=int, default=450000,  # 25h at 25 frames
                              help='Number of training steps.')
    arg_training.add_argument('--memory-steps', type=int, default=2048,
                              help='Number of steps for initial population of the Experience replay buffer.')
    arg_training.add_argument("--train-frequency", type=int, default=4,
                              help='Steps interval for Q-network batch training.')
    arg_training.add_argument("--target-update-frequency", type=int, default=1500,  # 5m at 25 frames
                              help='Steps interval for target network update.')
    arg_training.add_argument('--eval-interval', type=int, default=9000,  # 30m at 25 frames
                              help='Steps interval for progress evaluation.')
    arg_training.add_argument('--eval-epsilon', type=float, default=0.01,
                              help='Epsilon value used for evaluation.')
    arg_training.add_argument('--eval-steps', type=int, default=300,  # 1m at 25 frames
                              help='Epsilon value used for evaluation.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--seed', type=int, default=666,
                           help='Seed valu for torch and nummpy.')
    arg_utils.add_argument('--logspath', type=str, default=None,
                           help='Specific output path for training results.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment args
    environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
    env_params = dict(
        time_limit_seconds=args.time_limit,  # 1 min
        max_no_action_seconds=args.time_no_action,  # 5 sec
        frame_skip=args.frame_skip,  # 200ms
        goal_threshold=args.goal_threshold,
        init_altitude=args.init_altitude,
        altitude_limits=args.altitude_limits,
        fire_pos=args.target_pos,
        fire_dim=args.target_dim,
        is_pixels=args.is_pixels)

    # Create the environment
    env = gym.make(environment_name, **env_params)
    if not args.is_pixels:
        env = ReducedVectorObservation(env)

    # Observation preprocessing
    env_params['frame_stack'] = args.frame_stack
    if args.frame_stack > 1:
        env = ObservationStack(env, k=args.frame_stack)

    agent_device = 'cuda'\
        if torch.cuda.is_available() and args.use_cuda else 'cpu'

    if args.is_srl:
        agent_class = AEDDQNAgent
        agent_approximator = PixelApproximator\
            if args.is_pixels else VectorApproximator
    else:
        agent_class = DDQNAgent
        agent_approximator = QFeaturesNetwork if args.is_pixels else QNetwork

    # Agent args
    state_shape = env.observation_space.shape
    agent_params = dict(
        state_space_shape=state_shape,
        action_space_shape=(env.action_space.n, ),
        device=agent_device,
        approximator=agent_approximator,
        approximator_lr=args.approximator_lr,
        approximator_momentum=args.approximator_momentum,
        approximator_tau=args.approximator_tau,
        discount_factor=args.discount_factor,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_steps=args.epsilon_steps,
    )

    # Append SRL parameters
    if args.is_srl:
        agent_params.update(dict(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_filters=args.num_filters,
            num_layers=args.num_layers,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_decay=args.decoder_weight_decay
        ))

    # Memory buffer args
    memory_params = dict(
        buffer_size=args.memory_capacity,
        state_shape=agent_params['state_space_shape'],
        action_shape=agent_params['action_space_shape'],
    )
    memory_class = ReplayBuffer

    if args.memory_prioritized:
        memory_params.update(dict(
            alpha=args.prioritized_alpha,
            beta=args.prioritized_initial_beta,
            beta_steps=args.steps // args.train_frequency
        ))
        memory_class = PrioritizedReplayBuffer

    # RL training
    memory_buffer = memory_class(**memory_params)
    agent_params.update(dict(memory_buffer=memory_buffer))
    agent = agent_class(**agent_params)
    # update params to save info
    memory_params.update(dict(is_prioritized=args.memory_prioritized))
    agent_params.update(dict(memory_buffer=memory_params))

    if args.logspath is None:
        path_prefix = 'drone_pixels' if args.is_pixels else 'drone_vector'
        path_suffix = '-srl' if args.is_srl else ''
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Summary folder
        outfolder = Path(f"logs_{path_prefix}/ddqn{path_suffix}_{timestamp}")
    else:
        outfolder = Path(args.logspath)
    outfolder.mkdir(parents=True)
    print('Saving logs at:', outfolder)

    store_callback = StoreStepData(outfolder / 'history_training.csv',
                                   epsilon=lambda: agent.epsilon)

    run_params = dict(
        training_steps=args.steps,
        mem_steps=args.memory_steps,
        train_frequency=args.train_frequency,
        target_update_steps=args.target_update_frequency,
        eval_interval=args.eval_interval,
        eval_epsilon=args.eval_epsilon,
        eval_steps=args.eval_steps,
        outpath=outfolder)
    run_params_save = run_params.copy()
    run_params_save.update(dict(
        seed=args.seed,
        is_srl=args.is_srl))

    save_dict_json(env_params, outfolder / 'args_environment.json')
    save_dict_json(agent_params, outfolder / 'args_agent.json')
    save_dict_json(run_params_save, outfolder / 'args_training.json')

    run_agent(agent, env, step_callback=store_callback, **run_params)
