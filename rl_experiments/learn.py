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

from uav_navigation.agent import DDQNAgent, QFunction
from uav_navigation.net import QNetwork, QFeaturesNetwork
from uav_navigation.srl.agent import SRLDDQNAgent, SRLQFunction
from uav_navigation.memory import ReplayBuffer, PrioritizedReplayBuffer
from uav_navigation.utils import save_dict_json, run_agent

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import MultiModalObservation
from webots_drone.envs.preprocessor import CustomVectorObservation
from webots_drone.stack import ObservationStack


def list_of_float(arg):
    return list(map(float, arg.split(',')))


def list_of_int(arg):
    return list(map(int, arg.split(',')))


def uav_data_list(arg):
    avlbl_data = ['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors']
    sel_data = list()
    for d in arg.lower().split(','):
        if d in avlbl_data:
            sel_data.append(d)
    return sel_data


def parse_environment_args(parser):
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
    arg_env.add_argument("--target-pos", type=int, default=None,
                         help='Cuadrant number for target position.')
    arg_env.add_argument("--target-dim", type=list_of_float, default=[7., 3.5],
                         help="Target's dimension size.")
    arg_env.add_argument("--zone-steps", type=int, default=0,
                         help='Max number on target area to end the episode with found target.')
    arg_env.add_argument("--is-pixels", action='store_true',
                         help='Whether if reconstruct an image-based observation.')
    arg_env.add_argument("--is-vector", action='store_true',
                         help='Whether if reconstruct a vector-based observation.')
    arg_env.add_argument("--add-target-pos", action='store_true',
                         help='Whether if add the target position to vector state.')
    arg_env.add_argument("--add-target-dist", action='store_true',
                         help='Whether if add the target distance to vector state.')
    arg_env.add_argument("--add-target-dim", action='store_true',
                         help='Whether if add the target dimension to vector state.')
    arg_env.add_argument("--add-action", action='store_true',
                         help='Whether if add the previous action to vector state.')
    arg_env.add_argument("--uav-data", type=uav_data_list,
                         default=['imu', 'gyro', 'gps', 'gps_vel', 'north', 'dist_sensors'],
                         help='Select the UAV sensor data as state, available'
                         ' options are: imu, gyro, gps, gps_vel, north, dist_sensors')
    return arg_env


def parse_agent_args(parser):
    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--approximator-lr", type=float, default=1e-4,
                           help='Q approximation function AdamW learning rate.'
                           'default value is recommended in: '
                           '[Interference and Generalization in Temporal '
                           'Difference Learning]('
                           'https://proceedings.mlr.press/v119/bengio20a.html )')
    arg_agent.add_argument("--approximator-beta", type=float, default=0.9,
                           help='Q approximation function Adam \beta.')
    arg_agent.add_argument("--approximator-tau", type=float, default=0.01,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--approximator-momentum", type=float, default=.9,
                           help='Momentum factor factor for the SGD using'
                           'using nesterov')
    arg_agent.add_argument("--epsilon-start", type=float, default=1.0,
                           help='Initial epsilon value for exploration.')
    arg_agent.add_argument("--epsilon-end", type=float, default=0.01,
                           help='Final epsilon value for exploration.')
    arg_agent.add_argument("--epsilon-steps", type=int, default=90000,  # 5h at 25 frames
                           help='Number of steps to reach minimum value for Epsilon.')
    arg_agent.add_argument('--eval-epsilon', type=float, default=0.01,
                           help='Epsilon value used for evaluation.')
    arg_agent.add_argument("--train-frequency", type=int, default=4,
                           help='Steps interval for Q-network batch training.')
    arg_agent.add_argument("--target-update-frequency", type=int, default=1500,  # 5m at 25 frames
                           help='Steps interval for target network update.')
    return arg_agent


def parse_memory_args(parser):
    arg_mem = parser.add_argument_group('Memory buffer')
    arg_mem.add_argument("--memory-capacity", type=int, default=65536,  # 2**16
                           help='Maximum number of transitions in the Experience replay buffer.')
    arg_mem.add_argument("--memory-prioritized", action='store_true',
                           help='Whether if memory buffer is Prioritized experiencie replay or not.')
    arg_mem.add_argument("--prioritized-alpha", type=float, default=0.6,
                           help='Alpha prioritization exponent for PER.')
    arg_mem.add_argument("--prioritized-initial-beta", type=float, default=0.4,
                           help='Beta bias for sampling for PER.')
    arg_mem.add_argument("--beta-steps", type=float, default=112500,
                           help='Beta bias steps to reach 1.')
    return arg_mem


def parse_srl_args(parser):
    arg_srl = parser.add_argument_group(
        'State representation learning variation')
    arg_srl.add_argument("--is-srl", action='store_true',
                         help='Whether if method is SRL-based or not.')
    arg_srl.add_argument("--latent-dim", type=int, default=50,
                         help='Number of features in the latent representation Z.')
    arg_srl.add_argument("--hidden-dim", type=int, default=512,
                         help='Number of units in the hidden layers.')
    arg_srl.add_argument("--num-filters", type=int, default=32,
                         help='Number of filters in the CNN hidden layers.')
    arg_srl.add_argument("--num-layers", type=int, default=2,
                         help='Number of hidden layers.')
    arg_srl.add_argument("--encoder-lr", type=float, default=1e-3,
                         help='Encoder function Adam learning rate.')
    arg_srl.add_argument("--encoder-tau", type=float, default=0.05,
                         help='Encoder \tau polyak update.')
    arg_srl.add_argument("--decoder-lr", type=float, default=1e-3,
                         help='Decoder function Adam learning rate.')
    arg_srl.add_argument("--decoder-latent-lambda", type=float, default=1e-6,
                         help='Decoder regularization \lambda value.')
    arg_srl.add_argument("--decoder-weight-decay", type=float, default=1e-7,
                         help='Decoder function Adam weight decay value.')
    arg_srl.add_argument("--reconstruct-frequency", type=int, default=1,
                         help='Steps interval for AE batch training.')
    arg_srl.add_argument("--model-rgb", action='store_true',
                         help='Whether if use the RGB reconstruction model.')
    arg_srl.add_argument("--model-vector", action='store_true',
                         help='Whether if use the Vector reconstruction model.')
    arg_srl.add_argument("--model-pose", action='store_true',
                         help='Whether if use the Pose reconstruction model.')
    arg_srl.add_argument("--model-atc", action='store_true',
                         help='Whether if use the Augmented Temporal Contrast model.')
    arg_srl.add_argument("--model-contrastive", action='store_true',
                         help='Whether if use the VectorContrastive model.')
    arg_srl.add_argument("--use-priors", action='store_true',
                         help='Whether if use the Prior models.')
    arg_srl.add_argument("--use-srl-loss", action='store_true',
                         help='Whether if use the SRL loss.')
    arg_srl.add_argument("--encoder-only", action='store_true',
                         help='Whether if use the SRL loss.')
    arg_srl.add_argument("--distance-prior", action='store_true',
                         help='Whether if use the Prior models.')
    arg_srl.add_argument("--north-prior", action='store_true',
                         help='Whether if use the SRL loss.')
    return arg_srl


def parse_training_args(parser):
    arg_training = parser.add_argument_group('Training')
    arg_training.add_argument("--steps", type=int, default=450000,  # 25h at 25 frames
                              help='Number of training steps.')
    arg_training.add_argument('--memory-steps', type=int, default=2048,
                              help='Number of steps for initial population of the Experience replay buffer.')
    arg_training.add_argument("--batch-size", type=int, default=128,
                              help='Minibatch size for training.')
    arg_training.add_argument('--eval-interval', type=int, default=9000,  # 30m at 25 frames
                              help='Steps interval for progress evaluation.')
    arg_training.add_argument('--eval-steps', type=int, default=300,  # 1m at 25 frames
                              help='Number of evaluation steps.')
    return arg_training


def parse_utils_args(parser):
    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--seed', type=int, default=666,
                           help='Seed valu for torch and nummpy.')
    arg_utils.add_argument('--logspath', type=str, default=None,
                           help='Specific output path for training results.')
    return arg_utils


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()

    parse_environment_args(parser)
    parse_agent_args(parser)
    parse_memory_args(parser)
    parse_srl_args(parser)
    parse_training_args(parser)
    parse_utils_args(parser)

    return parser.parse_args()


def instance_env(args, name='webots_drone:webots_drone/DroneEnvDiscrete-v0',
                 seed=666):
    env_params = dict()
    if isinstance(args, dict):
        env_params = args.copy()
        if 'state_shape' in env_params.keys():
            del env_params['state_shape']
        if 'action_shape' in env_params.keys():
            del env_params['action_shape']
        if 'is_vector' in env_params.keys():
            del env_params['is_vector']
        if 'frame_stack' in env_params.keys():
            del env_params['frame_stack']
        if 'target_dist' in env_params.keys():
            del env_params['target_dist']
        if 'target_dim' in env_params.keys():
            del env_params['target_dim']
        if 'action2obs' in env_params.keys():
            del env_params['action2obs']
        if 'uav_data' in env_params.keys():
            del env_params['uav_data']
        if 'is_multimodal' in env_params.keys():
            del env_params['is_multimodal']
        if 'target_pos' in env_params.keys():
            del env_params['target_pos']
    else:
        env_params = dict(
            time_limit_seconds=args.time_limit,
            max_no_action_seconds=args.time_no_action,
            frame_skip=args.frame_skip,
            goal_threshold=args.goal_threshold,
            init_altitude=args.init_altitude,
            altitude_limits=args.altitude_limits,
            fire_pos=args.target_pos,
            fire_dim=args.target_dim,
            is_pixels=args.is_pixels,
            zone_steps=args.zone_steps)

    # Create the environment
    env = gym.make(name, **env_params)
    env.seed(seed)

    if not isinstance(args, dict):
        env_params['frame_stack'] = args.frame_stack
        env_params['is_multimodal'] = args.is_pixels and args.is_vector
        env_params['is_vector'] = args.is_vector
        env_params['target_dist'] = args.add_target_dist
        env_params['target_pos'] = args.add_target_pos
        env_params['target_dim'] = args.add_target_dim
        env_params['action2obs'] = args.add_action
        env_params['uav_data'] = args.uav_data

    env_params['state_shape'] = env.observation_space.shape
    if len(env.action_space.shape) == 0:
        env_params['action_shape'] = (env.action_space.n, )
    else:
        env_params['action_shape'] = env.action_space.shape

    return env, env_params


def wrap_env(env, env_params):
    if env_params['is_multimodal']:
        env = MultiModalObservation(env, frame_stack=env_params['frame_stack'],
                                    add_target=env_params['target_dist'])
    elif not env_params['is_pixels']:
        env = CustomVectorObservation(env, uav_data=env_params['uav_data'],
                                      target_dist=env_params['target_dist'],
                                      target_pos=env_params['target_pos'],
                                      target_dim=env_params['target_dim'],
                                      add_action=env_params['action2obs'])

    if not env_params['is_multimodal'] and env_params['frame_stack'] > 1:
        env = ObservationStack(env, k=env_params['frame_stack'])

    env_params['state_shape'] = env.observation_space.shape
    if len(env.action_space.shape) == 0:
        # discrete action space
        env_params['action_shape'] = (env.action_space.n, )
    else:
        env_params['action_shape'] = env.action_space.shape

    return env, env_params


def args2ae_model(args, env_params):
    ae_models = dict()
    image_shape = env_params['state_shape']
    vector_shape = env_params['state_shape']
    if env_params['is_multimodal']:
        image_shape = image_shape[0]
        vector_shape = vector_shape[1]

    if args.model_rgb:
        assert env_params['is_pixels'], 'RGB model requires is_pixels flag.'
        ae_models['RGB'] = dict(image_shape=image_shape,
                                latent_dim=args.latent_dim,
                                num_layers=args.num_layers,
                                num_filters=args.num_filters,
                                encoder_lr=args.encoder_lr,
                                decoder_lr=args.decoder_lr,
                                decoder_weight_decay=args.decoder_weight_decay)
    if args.model_vector:
        assert env_params['is_vector'], 'Vector model requires is_vector flag.'
        ae_models['Vector'] = dict(vector_shape=vector_shape,
                                   hidden_dim=args.hidden_dim,
                                   latent_dim=args.latent_dim,
                                   num_layers=args.num_layers,
                                   encoder_lr=args.encoder_lr,
                                   decoder_lr=args.decoder_lr,
                                   decoder_weight_decay=args.decoder_weight_decay)
    if args.model_atc:
        assert env_params['is_pixels'], 'ATC model requires is_pixels flag.'
        ae_models['ATC'] = dict(image_shape=image_shape,
                                latent_dim=args.latent_dim,
                                num_layers=args.num_layers,
                                num_filters=args.num_filters,
                                encoder_lr=args.encoder_lr,
                                decoder_lr=args.decoder_lr,
                                decoder_weight_decay=args.decoder_weight_decay)
    if args.model_contrastive:
        assert env_params['is_vector'], 'VectorATC model requires is_vector flag.'
        ae_models['VectorATC'] = dict(vector_shape=vector_shape,
                                      hidden_dim=args.hidden_dim,
                                      latent_dim=args.latent_dim,
                                      num_layers=args.num_layers,
                                      encoder_lr=args.encoder_lr,
                                      decoder_lr=args.decoder_lr,
                                      decoder_weight_decay=args.decoder_weight_decay)
    return ae_models


def args2priors(args, env_params):
    agent_priors = dict()
    # image_shape = env_params['state_shape']
    vector_shape = env_params['state_shape']
    if env_params['is_multimodal']:
        # image_shape = image_shape[0]
        vector_shape = vector_shape[1]

    if args.distance_prior:
        agent_priors['distance'] = dict(latent_dim=args.latent_dim,
                                        hidden_dim=args.hidden_dim,
                                        num_layers=1,
                                        source_obs=['Vector'],
                                        target_obs=['Vector'],
                                        learning_rate=1e-4)
    if args.north_prior:
        agent_priors['north'] = dict(state_shape=vector_shape,
                                     latent_dim=args.latent_dim,
                                     hidden_dim=args.hidden_dim,
                                     num_layers=1,
                                     source_obs=['Vector'],
                                     target_obs=['Vector'],
                                     learning_rate=1e-4)
    return agent_priors


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    environment_name = 'webots_drone:webots_drone/DroneEnvDiscrete-v0'
    env, env_params = instance_env(args, environment_name, seed=args.seed)
    # Observation preprocessing
    env, env_params = wrap_env(env, env_params)

    # Agent args
    agent_params = dict(
        action_shape=env_params['action_shape'],
        discount_factor=args.discount_factor,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_steps=args.epsilon_steps,
        batch_size=args.batch_size,
        train_freq=args.train_frequency,
        target_update_freq=args.target_update_frequency,
    )
    print('state_shape', env_params['state_shape'])
    if args.is_vector or not args.is_pixels:
        print('uav_data', env_params['uav_data'])
    print('action_shape', agent_params['action_shape'])

    # Append SRL models
    approximator_params = dict(
        obs_space=env.observation_space,
        learning_rate=args.approximator_lr,
        momentum=args.approximator_momentum,
        tau=args.approximator_tau,
        use_cuda=args.use_cuda,
        is_pixels=args.is_pixels,
        is_multimodal=env_params['is_multimodal'],
        use_augmentation=True)

    if args.is_srl:
        agent_class = SRLDDQNAgent
        q_approximator = SRLQFunction
        ae_models = args2ae_model(args, env_params)
        approximator_params['q_app_fn'] = QNetwork
        approximator_params['q_app_params'] = dict(
            state_shape=(args.latent_dim * len(ae_models.keys()), ),
            action_shape=agent_params['action_shape'],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers)

        agent_params['ae_models'] = ae_models
        agent_params['reconstruct_freq'] = args.reconstruct_frequency
        agent_params['srl_loss'] = args.use_srl_loss
        agent_params['priors'] = args.use_priors
    else:
        agent_class = DDQNAgent
        q_approximator = QFunction
        if args.is_pixels:
            approximator_params['q_app_fn'] = QFeaturesNetwork
            approximator_params['q_app_params'] = dict(
                state_shape=env_params['state_shape'],
                action_shape=agent_params['action_shape'])
        else:
            approximator_params['q_app_fn'] = QNetwork
            approximator_params['q_app_params'] = dict(
                state_shape=env_params['state_shape'],
                action_shape=agent_params['action_shape'],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers)
    agent_params.update(
        dict(approximator=q_approximator(**approximator_params)))

    # Memory buffer args
    memory_params = dict(
        buffer_size=args.memory_capacity,
        obs_shape=env_params['state_shape'],
        action_shape=agent_params['action_shape'],
        is_multimodal=env_params['is_multimodal']
    )
    memory_class = ReplayBuffer

    if args.memory_prioritized:
        memory_params.update(dict(
            alpha=args.prioritized_alpha,
            beta=args.prioritized_initial_beta,
            beta_steps=args.beta_steps
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
        if env_params['is_multimodal']:
            path_prefix = 'multimodal'
        else:
            path_prefix = 'pixels' if args.is_pixels else 'vector'
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
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        eval_epsilon=args.eval_epsilon,
        outpath=outfolder)
    # update data for log output
    run_params_save = run_params.copy()
    run_params_save.update(dict(
        seed=args.seed,
        use_cuda=args.use_cuda))

    agent_params.update(dict(approximator=approximator_params))
    agent_params_save = agent_params.copy()
    agent_params_save.update(dict(is_srl=args.is_srl))

    save_dict_json(env_params, outfolder / 'args_environment.json')
    save_dict_json(agent_params_save, outfolder / 'args_agent.json')
    save_dict_json(run_params_save, outfolder / 'args_training.json')

    run_agent(agent, env, step_callback=store_callback, **run_params)
