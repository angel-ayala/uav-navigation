#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 23:45:58 2024

@author: Angel Ayala
"""

import torch
import numpy as np
import argparse

import datetime
from pathlib import Path

from uav_navigation.sac.agent import SACAgent, SACFunction
from uav_navigation.sac.srl import SRLSACAgent, SRLSACFunction
from uav_navigation.net import QFeaturesNetwork
from uav_navigation.memory import ReplayBuffer, PrioritizedReplayBuffer
from uav_navigation.utils import save_dict_json, run_agent

from webots_drone.data import StoreStepData

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
                           help='Actor function Adam learning rate.')
    arg_agent.add_argument("--critic-tau", type=float, default=0.005,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--actor-freq", type=int, default=1,
                           help='Steps interval for actor batch training.')
    arg_agent.add_argument("--critic-target-freq", type=int, default=2,
                           help='Steps interval for target network update.')
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

    # Append SRL models
    approximator_params = dict(
        latent_dim=(args.latent_dim,),
        action_shape=env_params['action_shape'],
        obs_space=env.observation_space,
        hidden_dim=args.hidden_dim,
        init_temperature=0.1,
        action_range=[env.action_space.low, env.action_space.high],
        adjust_temperature=True,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=args.actor_lr,
        actor_betas=(0.9, 0.999),
        actor_log_std_bounds=[-5., 2.],
        actor_update_freq=args.actor_freq,
        critic_lr=args.critic_lr,
        critic_betas=(0.9, 0.999),
        critic_tau=args.critic_tau, # 0.005,
        critic_target_update_freq=args.critic_target_freq,
        use_cuda=args.use_cuda,
        is_pixels=args.is_pixels,
        is_multimodal=env_params['is_multimodal'],
        use_augmentation=False)

    if args.is_srl:
        agent_class = SRLSACAgent
        policy = SRLSACFunction
        ae_models = args2ae_model(args, env_params)
        # approximator_params['encoder_tau'] = args.encoder_tau
        approximator_params['latent_dim'] = (args.latent_dim * len(ae_models),)
        agent_params['ae_models'] = ae_models
        agent_params['encoder_only'] = args.encoder_only
        agent_params['reconstruct_freq'] = args.reconstruct_frequency
        agent_params['srl_loss'] = args.use_srl_loss
    else:
        agent_class = SACAgent
        policy = SACFunction
        if args.is_pixels:
            approximator_params['preprocess'] = QFeaturesNetwork(
                env_params['state_shape'], env_params['action_shape'], only_cnn=True)
            approximator_params['latent_dim'] = approximator_params['preprocess'].feature_dim
        else:
            approximator_params['latent_dim'] = env_params['state_shape']
    agent_params.update(
        dict(approximator=policy(**approximator_params)))

    # Memory buffer args
    memory_params = dict(
        buffer_size=args.memory_capacity,
        obs_shape=env_params['state_shape'],
        action_shape=env_params['action_shape'],
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
        outfolder = Path(f"logs_cf_{path_prefix}/sac{path_suffix}_{timestamp}")
    else:
        outfolder = Path(args.logspath)
    agents_folder = outfolder / 'agents'
    agents_folder.mkdir(parents=True)
    print('Saving logs at:', outfolder)

    store_callback = StoreStepData(outfolder / 'history_training.csv', n_sensors=4)

    run_params = dict(
        training_steps=args.steps,
        mem_steps=args.memory_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        eval_epsilon=None,
        outpath=agents_folder)
    # update data for log output
    run_params_save = run_params.copy()
    run_params_save.update(dict(
        seed=args.seed,
        use_cuda=args.use_cuda))

    agent_params.update(dict(approximator=approximator_params))
    agent_params_save = agent_params.copy()
    agent_params_save.update(dict(is_srl=args.is_srl))

    # environment meta info
    env_params['target_quadrants'] = env.quadrants.tolist()
    env_params['flight_area'] = env.flight_area.tolist()

    save_dict_json(env_params, outfolder / 'args_environment.json')
    save_dict_json(agent_params_save, outfolder / 'args_agent.json')
    save_dict_json(run_params_save, outfolder / 'args_training.json')

    run_agent(agent, env, step_callback=store_callback, **run_params)
