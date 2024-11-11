#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:45:58 2023

@author: Angel Ayala
"""

import torch
import numpy as np
import argparse

import datetime
from pathlib import Path

from uav_navigation.sac.agent import SACAgent, ACFunction
from uav_navigation.sac.srl import SRLSACAgent, SRLSACFunction
from uav_navigation.net import QFeaturesNetwork
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
from learn_cf import args2priors


def parse_agent_args(parser):
    arg_agent = parser.add_argument_group('Agent')
    arg_agent.add_argument("--critic-lr", type=float, default=1e-3,
                           help='Critic function Adam learning rate.')
    arg_agent.add_argument("--critic-beta", type=float, default=0.9,
                           help='Q approximation function Adam \beta.')
    arg_agent.add_argument("--critic-tau", type=float, default=0.995,
                           help='Soft target update \tau.')
    arg_agent.add_argument("--discount-factor", type=float, default=0.99,
                           help='Discount factor \gamma.')
    arg_agent.add_argument("--approximator-momentum", type=float, default=.9,
                           help='Momentum factor factor for the SGD using'
                           'using nesterov')
    arg_agent.add_argument("--actor-freq", type=int, default=1,
                           help='Steps interval for actor batch training.')
    arg_agent.add_argument("--critic-target-freq", type=int, default=2,
                           help='Steps interval for target network update.')
    return arg_agent


def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_env = parse_environment_args(parser)
    arg_agent = parse_agent_args(parser)
    arg_mem = parse_memory_args(parser)
    arg_srl = parse_srl_args(parser)
    arg_training = parse_training_args(parser)
    arg_utils = parse_utils_args(parser)

    return parser.parse_args()


if __name__ == '__main__':    
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    environment_name = 'webots_drone:webots_drone/CrazyflieEnvContinuous-v0'
    env, env_params = instance_env(args, environment_name, seed=args.seed)
    # env = ReducedActionSpace(env)
    # Observation preprocessing
    env, env_params = wrap_env(env, env_params)

    # Agent args
    agent_params = dict(
        action_shape=env_params['action_shape'],
        discount_factor=args.discount_factor,
    )
    print('state_shape', env_params['state_shape'])
    if args.is_vector or not args.is_pixels:
        print('uav_data', env_params['uav_data'])
    print('action_shape', env_params['action_shape'])

    # Append SRL models
    ae_models = dict()
    approximator_params = dict(
        latent_dim=(args.latent_dim, ),
        action_shape=env_params['action_shape'],
        obs_space=env.observation_space,
        hidden_dim=args.hidden_dim,
        init_temperature=0.1,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_min_a=env.action_space.low,
        actor_max_a=env.action_space.high,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=args.actor_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau, # 0.005,
        critic_target_update_freq=args.critic_target_freq,
        use_cuda=args.use_cuda,
        is_pixels=args.is_pixels,
        is_multimodal=env_params['is_multimodal'],
        use_augmentation=True)

    if args.is_srl:
        agent_class = SRLSACAgent
        policy = SRLSACFunction
        ae_models = args2ae_model(args, env_params)
        approximator_params['encoder_tau'] = args.encoder_tau
        approximator_params['latent_dim'] = (args.latent_dim * len(ae_models),)
        agent_params['ae_models'] = ae_models
        agent_params['encoder_only'] = args.encoder_only
        agent_params['reconstruct_freq'] = args.reconstruct_frequency
        agent_params['srl_loss'] = args.use_srl_loss
        agent_params['priors'] = args2priors(args, env_params)
    else:
        agent_class = SACAgent
        policy = ACFunction
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
    memory_class = ReplayBuffer

    if args.memory_prioritized:
        memory_params.update(dict(
            alpha=args.prioritized_alpha,
            beta=args.prioritized_initial_beta,
            beta_steps=args.steps // args.critic_target_freq
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
        outfolder = Path(f"logs_cf_{path_prefix}/SAC{path_suffix}_{timestamp}")
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
        eval_epsilon=False,
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
