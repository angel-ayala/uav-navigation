#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:31:16 2024

@author: Angel Ayala
"""
import argparse
from pathlib import Path
from thop import clever_format

from uav_navigation.agent import DDQNAgent, QFunction
from uav_navigation.net import QNetwork, QFeaturesNetwork
from uav_navigation.srl.agent import SRLDDQNAgent, SRLQFunction
from uav_navigation.utils import load_json_dict
from uav_navigation.utils import profile_model


def parse_args():
    parser = argparse.ArgumentParser()    # misc
    parser.add_argument('--logspath', type=str,
                        default='logs/ddqn-srl_2023-11-28_00-13-33',
                        help='Log path with training results.')

    args = parser.parse_args()
    return args


def run_profile(args):
    # Define constants
    logpath = Path(args.logspath)
    agent_paths = [lp.name[:12] for lp in logpath.glob('**/agent_ep_*_q*')]
    agent_paths.sort()

    # Environment args
    env_params = load_json_dict(logpath / 'args_environment.json')
    # Agent params
    agent_params = load_json_dict(logpath / 'args_agent.json')
    # Training params
    training_params = load_json_dict(logpath / 'args_training.json')

    is_srl = agent_params['is_srl']
    del agent_params['is_srl']
    approximator_params = agent_params['approximator']
    # approximator_params['obs_space'] = env.observation_space
    if is_srl:
        agent_class = SRLDDQNAgent
        q_approximator = SRLQFunction
        approximator_params['q_app_fn'] = QNetwork
    else:
        agent_class = DDQNAgent
        q_approximator = QFunction
        approximator_params['q_app_fn'] = QFeaturesNetwork\
            if env_params['is_pixels'] else QNetwork

    # Instantiate an init computatinal cost profile
    agent_params.update(
        dict(approximator=q_approximator(**approximator_params)))
    agent_path = agent_paths[-1]
    print('Loading from', "/".join(str(agent_path).split("/")[-3:]))
    agent = agent_class(**agent_params)
    agent.load(logpath / agent_path)

    # Architecture
    print("====== Agent's architecture ======")
    if is_srl:
        for i, (ae_type, ae_params) in enumerate(agent_params['ae_models'].items()):
            ae_model = agent.approximator.models[i]
            print(f"--- {ae_type} AutoEncoder ---")
            print(ae_model)
    print("--- Q-function ---")
    print(agent.approximator.q_network)

    # Profiling
    print("====== Agent's computational cost ======")
    device = agent.approximator.device
    q_input_shape = env_params['state_shape']
    q_network = agent.approximator.q_network
    learn_flops, learn_params = 0, 0
    eval_flops, eval_params = 0, 0

    if is_srl:
        latent_dim = 0
        # q_network = q_network.q_network
        for i, (ae_type, ae_params) in enumerate(agent_params['ae_models'].items()):
            ae_model = agent.approximator.models[i]
            print(f"{ae_type} AutoEncoder:")
            enc_flops, enc_params = profile_model(
                ae_model.encoder[0], env_params['state_shape'], device)
            enc_flops_str, enc_params_str = clever_format(
                [enc_flops, enc_params], "%.3f")
            print(f"- Encoder: {enc_flops_str} flops, {enc_params_str} params")
            dec_flops, dec_params = profile_model(
                ae_model.decoder[0], (ae_params['latent_dim'], ), device)
            dec_flops_str, dec_params_str = clever_format(
                [dec_flops, dec_params], "%.3f")
            print(f"- Decoder: {dec_flops_str} flops, {dec_params_str} params")
            eval_flops += enc_flops
            learn_flops += enc_flops + dec_flops
            eval_params += enc_params
            learn_params += enc_params + dec_params
            latent_dim += ae_params['latent_dim']
        q_input_shape = (latent_dim, )
        
    print("Q-values approximation function:")
    q_flops, q_params = profile_model(q_network, q_input_shape, device)
    q_flops_str, q_params_str = clever_format([q_flops, q_params], "%.3f")
    model_name = q_network.__class__.__name__
    print(f"- {model_name}: {q_flops_str} flops, {q_params_str} params")
    learn_flops += q_flops
    eval_flops += q_flops
    learn_params += q_params
    eval_params += q_params
    learn_flops_str, learn_params_str = clever_format(
        [learn_flops, learn_params], "%.3f")
    eval_flops_str, eval_params_str = clever_format([eval_flops, eval_params],
                                                    "%.3f")
    print(f"Learning phase: {learn_flops_str} flops, {learn_params_str} params")
    print(f"Evaluation phase: {eval_flops_str} flops, {eval_params_str} params")


if __name__ == '__main__':
    run_profile(parse_args())
