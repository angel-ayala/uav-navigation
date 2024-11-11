#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:53:34 2024

@author: Angel Ayala
"""

import torch
import numpy as np
import argparse

import gym
import datetime
from pathlib import Path

from uav_navigation.td3.agent import TD3Agent, TD3Function
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.td3.srl import SRLTD3Agent, SRLTD3Function
from uav_navigation.memory import ReplayBuffer, PrioritizedReplayBuffer
from uav_navigation.utils import save_dict_json, run_agent

from webots_drone.data import StoreStepData
from webots_drone.envs.preprocessor import ReducedActionSpace

from learn_cf import parse_environment_args
from learn_cf import parse_memory_args
from learn_cf import parse_srl_args
from learn_cf import parse_training_args
from learn_cf import parse_utils_args
from learn_cf import instance_env
from learn_cf import wrap_env
from learn_cf import args2ae_model


def parse_agent_args(parser):
    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--critic-lr", type=float, default=3e-4,
                           help='Critic function Adam learning rate.')
    arg_agent.add_argument("--actor-lr", type=float, default=3e-4,
                           help='Actor function Adams learning rate.')
    arg_agent.add_argument("--tau", type=float, default=0.005,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--policy-freq", type=int, default=2,
                           help='Steps interval for actor batch training.')
    arg_agent.add_argument("--policy-noise", type=list, default=[0.2, 0.2, 0.1, 0.2],
                           help='Policy noise value.')
    arg_agent.add_argument("--noise-clip", type=list, default=[0.5, 0.5, 0.3, 0.5],
                           help='Policy noise clip value.')
    return arg_agent


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


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
    env, env_params = instance_env(args, environment_name, seed=args.seed)
    # Observation preprocessing
    env, env_params = wrap_env(env, env_params)
    env_params['action_shape'] = env.action_space.shape

    # Agent args    
    agent_params = dict(
        action_shape=env_params['action_shape'],
        discount_factor=args.discount_factor,
        batch_size=args.batch_size,
    )
    print('state_shape', env_params['state_shape'])
    if args.is_vector or not args.is_pixels:
        print('uav_data', env_params['uav_data'])
    print('action_shape', agent_params['action_shape'])
    print('max_action', env.action_space.high)

    # Append SRL models
    approximator_params = dict(
        latent_dim=(args.latent_dim,),
        action_shape=env_params['action_shape'],
        obs_space=env.observation_space,
        max_action=env.action_space.high,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,  # 0.005,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        use_cuda=args.use_cuda,
        is_pixels=args.is_pixels,
        is_multimodal=env_params['is_multimodal'],
        use_augmentation=False)

    if args.is_srl:
        agent_class = SRLTD3Agent
        q_approximator = SRLTD3Function
        ae_models = args2ae_model(args, env_params)
        # approximator_params['encoder_tau'] = args.encoder_tau
        approximator_params['latent_dim'] = (args.latent_dim * len(ae_models),)
        agent_params['ae_models'] = ae_models
        agent_params['reconstruct_freq'] = args.reconstruct_frequency
        agent_params['srl_loss'] = args.use_srl_loss
        agent_params['priors'] = args.use_priors
    else:
        agent_class = TD3Agent
        q_approximator = TD3Function
        if args.is_pixels:
            approximator_params['q_app_fn'] = QFeaturesNetwork
            approximator_params['q_app_params'] = dict(
                state_shape=env_params['state_shape'],
                action_shape=agent_params['action_shape'])
        else:
            approximator_params['latent_dim'] = env_params['state_shape']
    agent_params.update(
        dict(approximator=q_approximator(**approximator_params)))

    # Memory buffer args
    memory_params = dict(
        buffer_size=args.memory_capacity,
        obs_shape=env_params['state_shape'],
        action_shape=agent_params['action_shape'],
        is_multimodal=env_params['is_multimodal']
    )
    if env_params['is_multimodal']:
        memory_params['obs_shape'] = env_params['obs_space']
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
        outfolder = Path(f"logs_cf_{path_prefix}/TD3{path_suffix}_{timestamp}")
    else:
        outfolder = Path(args.logspath)
    outfolder.mkdir(parents=True)
    print('Saving logs at:', outfolder)

    store_callback = StoreStepData(outfolder / 'history_training.csv', n_sensors=4)

    run_params = dict(
        training_steps=args.steps,
        mem_steps=args.memory_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        eval_epsilon=None,
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