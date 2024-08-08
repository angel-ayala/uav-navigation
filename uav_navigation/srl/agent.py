#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:27:24 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
import torch
import copy
from uav_navigation.agent import QFunction
from uav_navigation.agent import DDQNAgent
from uav_navigation.logger import summary_scalar
from uav_navigation.net import weight_init
from uav_navigation.utils import destack
from .autoencoder import instance_autoencoder
from .autoencoder import latent_l2
from .priors import NorthBelief
from .priors import PositionBelief
from .priors import OrientationBelief
from .priors import DistanceBelief
from .net import QNetworkWrapper


class SRLFunction:
    def __init__(self, decoder_latent_lambda=1e-6):
        self.decoder_latent_lambda = decoder_latent_lambda
        self.models = list()
        self.priors = list()

    def add_autoencoder(self, ae_type, ae_params):
        ae_model, ae_params = instance_autoencoder(ae_type, ae_params)
        ae_model.to(self.device)
        ae_model.apply(weight_init)
        ae_model.adamw_optimizer(ae_params['encoder_lr'],
                                 ae_params['decoder_lr'],
                                 ae_params['decoder_weight_decay'])
        self.models.append(ae_model)

    def compute_z(self, observations, latent_types=None):
        z_hat = list()
        # obs_2d, obs_1d = self.format_obs(observations)
        if self.is_multimodal:
            obs_2d, obs_1d = observations
        elif self.is_pixels:
            obs_2d = observations
        else:
            obs_1d = observations

        for m in self.models:
            if latent_types is None or m.type in latent_types:
                if 'RGB' in m.type:
                    z = m.encode_obs(obs_2d.to(self.device), detach=True)
                if 'Vector' in m.type:
                    z = m.encode_obs(obs_1d.to(self.device), detach=True)
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                z_hat.append(z)
        z_hat = torch.cat(z_hat, dim=1)
        return z_hat

    def compute_ae_loss(self, obs, actions, rewards, obs_t1, dones):
        if self.is_multimodal:
            obs_2d, obs_1d = self.format_obs(obs, augment=False)
            obs_2d_augm, obs_1d_augm = self.format_obs(obs, augment=True)
            obs_2d_t1_augm, obs_1d_t1_augm = self.format_obs(obs_t1, augment=True)
        elif self.is_pixels:
            obs_2d = self.format_obs(obs, augment=False)
            obs_2d_augm = self.format_obs(obs, augment=True)
            obs_2d_t1_augm = self.format_obs(obs_t1, augment=True)
        else:
            obs_1d = self.format_obs(obs, augment=False)
            obs_1d_augm = self.format_obs(obs, augment=True)
            obs_1d_t1_augm = self.format_obs(obs_t1, augment=True)

        total_loss = list()
        for ae_model in self.models:
            if "RGB" in ae_model.type:
                if "ATC" in ae_model.type:
                    loss = ae_model.compute_contrastive_loss(obs_2d_augm, obs_2d_t1_augm)
                    ae_model.update_momentum_encoder(0.01)
                else:
                    loss = ae_model.compute_reconstruction_loss(obs_2d, obs_2d_augm, self.decoder_latent_lambda, pixel_obs_log=True)
                total_loss.append(loss)
            if "Vector" in ae_model.type:
                if "ATC" in ae_model.type:
                    # obs_1d_augm_norm = self.normalize_vector(obs_1d_augm)
                    # obs_1d_t1_augm = self.normalize_vector(obs_1d_t1_augm)
                    loss = ae_model.compute_contrastive_loss(obs_1d_augm, obs_1d_t1_augm, rewards)
                    ae_model.update_momentum_encoder(0.01)
                    total_loss.append(loss)
                # else:
                # compute target distances as reconstruction
                pos_uav = obs_1d[:, :, 6:9]
                pos_target = obs_1d[:, :, -3:]
                dist = pos_uav - pos_target
                obs_1d[:, :, 13:] = dist
                # orientation difference
                orientation = torch.arctan2(dist[:, :, 0], dist[:, :, 1])
                # """Apply UAV sensor offset."""
                orientation -= torch.pi / 2.
                orientation[orientation < -torch.pi] += 2 * torch.pi
                orientation_diff = torch.cos(orientation - obs_1d[:, :, 12])
                obs_1d = self.normalize_vector(obs_1d)
                obs_1d[:, :, 12] = orientation_diff
                loss = ae_model.compute_reconstruction_loss(obs_1d, obs_1d_augm, self.decoder_latent_lambda)
                total_loss.append(loss)
        tloss = torch.sum(torch.stack(total_loss))
        # summary_scalar("Loss/AutoEncoders", tloss.item())
        return tloss

    def update_encoder(self, enc_loss):
        for m in self.models:
            m.encoder_optim_zero_grad()
        enc_loss.backward()
        for m in self.models:
            m.encoder_optim_step()

    def update_reconstruction(self, r_loss):
        for m in self.models:
            m.decoder_optim_zero_grad()
        self.update_encoder(r_loss)
        for m in self.models:
            m.decoder_optim_step()

    def update_representation(self, r_loss):
        for bmodel in self.priors:
            bmodel.optimizer_zero_grad()
        self.update_reconstruction(r_loss)
        for bmodel in self.priors:
            bmodel.optimizer_step()

    def save(self, path, ae_models, encoder_only=False):
        for i, (m, _) in enumerate(ae_models.items()):
            ae_model = self.models[i]
            ae_path = str(path) + f"_ae_{m}.pth"
            self.models[i].save_weights(ae_path, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        # ensure empty list
        if hasattr(self, 'models'):
            del self.models
        self.models = list()

        for i, (m, m_params) in enumerate(ae_models.items()):
            ae_params = m_params.copy()
            ae_params['encoder_only'] = encoder_only
            ae_path = str(path) + f"_ae_{m}.pth"
            self.add_autoencoder(m, ae_params)
            self.models[i].load_weights(ae_path, self.device, encoder_only)

    # def append_prior(self, belief_model, latent_source, obs_target, learning_rate=1e-3):
    #     belief_model.to(self.device)
    #     belief_model.apply(weight_init)
    #     print('belief_model', belief_model)
    #     self.priors.append(PriorModel(belief_model,
    #                                   latent_source=latent_source,
    #                                   obs_target=obs_target))
    #     # self.priors[-1].sgd_optimizer(learning_rate=learning_rate)
    #     self.priors[-1].adam_optimizer(learning_rate=learning_rate)

    # def compute_priors_loss(self, obs, obs_t1, actions):
    #     if self.is_multimodal:
    #         obs_2d, obs_1d = self.format_obs(obs, augment=False)
    #     elif self.is_pixels:
    #         obs_2d = self.format_obs(obs, augment=False)
    #     else:
    #         obs_1d = self.format_obs(obs, augment=False)

    #     loss_list = list()
    #     for pmodel in self.priors:
    #         z = self.compute_z(obs, pmodel.latent_source)
    #         hat_x = pmodel(z)
    #         if 'RGB' in pmodel.obs_target:
    #             x_true = pmodel.obs2target(obs_2d)
    #         elif 'Vector' in pmodel.obs_target:
    #             x_true = pmodel.obs2target(obs_1d)

    #         loss = pmodel.compute_loss(hat_x, x_true)
    #         summary_scalar(f"Loss/Prior/{pmodel.name}", loss.item())
    #         loss_list.append(loss)

    #     ploss = torch.stack(loss_list)
    #     ploss = ploss.sum()
    #     summary_scalar("Loss/Prior", ploss.item())
    #     return ploss


class SRLQFunction(QFunction, SRLFunction):
    def __init__(self, q_app_fn, q_app_params, obs_space, learning_rate=1e-3,
                 momentum=0.9, tau=0.1, use_cuda=True, is_pixels=False,
                 is_multimodal=False, use_augmentation=True,
                 decoder_latent_lambda=1e-6):
        QFunction.__init__(self, q_app_fn, q_app_params, obs_space,
                           learning_rate, momentum, tau=tau, use_cuda=use_cuda,
                           is_pixels=is_pixels, is_multimodal=is_multimodal,
                           use_augmentation=use_augmentation)
        SRLFunction.__init__(self, decoder_latent_lambda)

    def fuse_encoder(self):
        # fuse encoder with Q-network function
        q_network = copy.deepcopy(self.q_network)
        encoder = copy.deepcopy(self.models[0].encoder[0])
        self.q_network = QNetworkWrapper(q_network, encoder).to(self.device)
        # fuse encoder with  Q-network target function
        target_q_network = copy.deepcopy(self.target_q_network)
        self.target_q_network = QNetworkWrapper(
            target_q_network, encoder).to(self.device)
        # Initialize target network with same Q-network parameters
        tau_value = copy.deepcopy(self.tau)
        self.tau = 1.
        self.update_target_network()
        self.tau = tau_value
        # share autoencoder weights
        self.q_network.encoder.copy_weights_from(self.models[0].encoder[0])
        # Initialize optimization function
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(),
                                           lr=learning_rate, amsgrad=True)

    def compute_q(self, observations, actions=None):
        # Compute Q-values using the inferenced z latent representation
        z_hat = self.compute_z(observations)
        return super().compute_q(z_hat, actions)

    def compute_q_target(self, observations, actions=None):
        # Compute Q-values using the target Q-network
        z_hat = self.compute_z(observations)
        return super().compute_q_target(z_hat, actions)

    def update(self, td_loss):
        for m in self.models:
            m.encoder_optim_zero_grad()
        QFunction.update(self, td_loss)
        for m in self.models:
            m.encoder_optim_step()

    def save(self, path, ae_models, encoder_only=False):
        QFunction.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        QFunction.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)


class SRLAgent:
    def __init__(self,
                 ae_models,
                 reconstruct_freq=1,
                 srl_loss=False,
                 priors=False,
                 encoder_only=False):
        self.ae_models = ae_models
        self.reconstruct_freq = reconstruct_freq
        self.srl_loss = srl_loss
        self.priors = priors
        if priors:
            self.init_priors()
        self.init_ae_models(self.ae_models, encoder_only)
        # self.approximator.fuse_encoder()

    def init_ae_models(self, ae_models, encoder_only):
        for m, m_params in ae_models.items():
            ae_params = m_params.copy()
            ae_params['encoder_only'] = encoder_only
            self.approximator.add_autoencoder(m, ae_params)

    def init_priors(self):
        for prior_id, prior_params in self.priors.items():
            if prior_id == 'north':
                pmodel = NorthBelief(prior_params['state_shape'],
                                     prior_params['latent_dim'],
                                     prior_params['hidden_dim'],
                                     prior_params['num_layers'])
            if prior_id == 'orientation':
                pmodel = OrientationBelief(*prior_params)
            if prior_id == 'position':
                pmodel = PositionBelief(*prior_params)
            if prior_id == 'distance':
                pmodel = DistanceBelief(prior_params['latent_dim'],
                                        prior_params['hidden_dim'],
                                        prior_params['num_layers'])
            self.approximator.append_prior(pmodel,
                                           prior_params['source_obs'],
                                           prior_params['target_obs'],
                                           learning_rate=prior_params['learning_rate'])
    def update_reconstruction(self):
        sampled_data = self.memory.random_sample(self.batch_size, device=self.approximator.device)

        if self.is_prioritized:
            obs, actions, rewards, obs_t1, dones = sampled_data[0]
        else:
            obs, actions, rewards, obs_t1, dones = sampled_data

        rec_loss = self.approximator.compute_ae_loss(obs, actions, rewards, obs_t1, dones)
        # TODO: implement state representation learning loss

        if self.priors:
            rec_loss += self.approximator.compute_priors_loss(obs, obs_t1, actions)

        # summary_scalar("Loss/Representation", rec_loss.item())
        self.approximator.update_representation(rec_loss)

    def update(self, step):
        if not self.can_update:
            return False
        if step % self.reconstruct_freq == 0:
            self.update_reconstruction()


class SRLDDQNAgent(DDQNAgent, SRLAgent):
    def __init__(self,
                 action_shape,
                 approximator,
                 ae_models,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_steps=500000,
                 memory_buffer=None,
                 batch_size=128,
                 train_freq=4,
                 target_update_freq=100,
                 reconstruct_freq=1,
                 srl_loss=False,
                 priors=False,
                 encoder_only=False):
        DDQNAgent.__init__(self,
                           action_shape,
                           approximator,
                           discount_factor=discount_factor,
                           epsilon_start=epsilon_start,
                           epsilon_end=epsilon_end,
                           epsilon_steps=epsilon_steps,
                           memory_buffer=memory_buffer,
                           batch_size=batch_size,
                           train_freq=train_freq,
                           target_update_freq=target_update_freq)
        SRLAgent.__init__(self, ae_models,
                          reconstruct_freq=reconstruct_freq,
                          srl_loss=srl_loss,
                          priors=priors,
                          encoder_only=encoder_only)

    def update(self, step):
        DDQNAgent.update(self, step, augment=False)
        SRLAgent.update(self, step)

    def update_td(self, augment=True):
        if not self.can_update:
            return False
        # Update the Q-network if replay buffer is sufficiently large
        sampled_data = self.memory.sample(
            self.batch_size, device=self.approximator.device)
        td_loss = self.compute_td_loss(sampled_data, augment)
        if self.is_prioritized:
            states = sampled_data[0][0]
        else:
            states = sampled_data[0]
        z_l2 = latent_l2(self.approximator.compute_z(states))
        loss_z = 0.1 * z_l2
        summary_scalar(f'Loss/Encoder/QNetwork/L2', z_l2.item())
        self.approximator.update(td_loss + loss_z)

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, encoder_only=False, eval_only=True):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)
