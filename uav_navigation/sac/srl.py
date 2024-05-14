#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:27:24 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
# import torch
# from torchvision.transforms import AutoAugment
# from torchvision.transforms import AugMix
from thop import clever_format
from .agent import ACFunction
from .agent import SACAgent
from .agent import profile_actor_critic
# from ..logger import summary_scalar
from ..srl.agent import SRLAgent
from ..srl.agent import SRLFunction
# from ..srl.net import weight_init
# from ..srl.priors import PriorModel
# from ..srl.priors import NorthBelief
# from ..srl.priors import PositionBelief
# from ..srl.priors import OrientationBelief
# from ..srl.autoencoder import AEModel
# from ..srl.autoencoder import RGBModel
# from ..srl.autoencoder import ATCModel
from ..srl.autoencoder import profile_ae_model


def profile_srl_approximator(approximator, state_shape, action_shape):
    total_flops, total_params = 0, 0
    q_feature_dim = 0
    for m in approximator.models:
        if approximator.is_multimodal:
            if m.type in ['rgb', 'atc']:
                flops, params = profile_ae_model(m, state_shape[0], approximator.device)
            if m.type == 'vector':
                flops, params = profile_ae_model(m, state_shape[1], approximator.device)
        else:
            flops, params = profile_ae_model(m, state_shape, approximator.device)
        total_flops += flops
        total_params += params
        q_feature_dim += m.encoder[0].feature_dim

    # profile q-network
    flops, params = profile_actor_critic(approximator, q_feature_dim,
                                         action_shape)
    total_flops += flops
    total_params += params
    print('Total: {} flops, {} params'.format(
        *clever_format([total_flops, total_params], "%.3f")))
    return total_flops, total_params



class SRLSACFunction(ACFunction, SRLFunction):
    def __init__(self, latent_dim, action_shape, obs_space,       
                 hidden_dim=256,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_min_a=0.,
                 actor_max_a=1.,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,
                 critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 use_cuda=True,
                 is_pixels=False,
                 is_multimodal=False,
                 use_augmentation=True,
                 encoder_tau=0.995,
                 decoder_latent_lambda=1e-6,):
        ACFunction.__init__(self, latent_dim, action_shape, obs_space,
                            hidden_dim,
                            init_temperature,
                            alpha_lr,
                            alpha_beta,
                            actor_lr,
                            actor_beta,
                            actor_min_a,
                            actor_max_a,
                            actor_log_std_min,
                            actor_log_std_max,
                            actor_update_freq,
                            critic_lr,
                            critic_beta,
                            critic_tau,
                            critic_target_update_freq,
                            use_cuda,
                            is_pixels,
                            is_multimodal,
                            use_augmentation)
        SRLFunction.__init__(self, decoder_latent_lambda)
        self.encoder_tau = encoder_tau

    def action_inference(self, obs, sample=False):
        obs = self.compute_z(obs).detach()
        return super().action_inference(obs, sample=sample)
    
    def update_critic(self, discount, obs, action, reward, next_obs, not_done, weight=None):
        obs = self.compute_z(obs)
        next_obs = self.compute_z(next_obs)
        return super().update_critic(discount, obs, action, reward, next_obs, not_done, weight)

    def update_critic_target(self):
        super().update_critic_target()
        for m in self.models:
            if m.type == 'atc':
                m.update_momentum_encoder(self.encoder_tau)

    def update_actor_and_alpha(self, obs, weight=None):
        # detach encoder, so we don't update it with the actor loss
        obs = self.compute_z(obs).detach()
        return super().update_actor_and_alpha(obs, weight)

    def save(self, path, ae_models, encoder_only=False):
        ACFunction.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        ACFunction.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)


class SRLSACAgent(SACAgent, SRLAgent):
    def __init__(self, action_shape, approximator, ae_models,
                 discount_factor=0.99, memory_buffer=None, batch_size=128,
                 reconstruct_freq=1, srl_loss=False, priors=False,
                 encoder_only=False):
        SACAgent.__init__(self, action_shape, approximator,
                          discount_factor, memory_buffer, batch_size)
        SRLAgent.__init__(self, ae_models, reconstruct_freq=reconstruct_freq,
                          srl_loss=srl_loss, priors=priors, encoder_only=encoder_only)

    def update(self, step):
        SACAgent.update(self, step)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        SRLAgent.save(self, path, encoder_only)

    def load(self, path, eval_only=True, encoder_only=False):
        SRLAgent.load(self, path, eval_only, encoder_only)
