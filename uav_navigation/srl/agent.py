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
from torchvision.transforms import AutoAugment
# from torchvision.transforms import AugMix
from thop import clever_format
from uav_navigation.agent import QFunction
from uav_navigation.agent import DDQNAgent
from uav_navigation.agent import profile_q_approximator
from uav_navigation.logger import summary_scalar
from .net import weight_init
from .net import PriorModel
from .net import NorthBelief
from .net import PositionBelief
from .net import OrientationBelief
from .autoencoder import AEModel
from .autoencoder import RGBModel
from .autoencoder import ATCModel
from .autoencoder import profile_ae_model


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
    flops, params = profile_q_approximator(approximator, q_feature_dim,
                                           action_shape)
    total_flops += flops
    total_params += params
    print('Total: {} flops, {} params'.format(
        *clever_format([total_flops, total_params], "%.3f")))
    return total_flops, total_params



class SRLFunction(QFunction):
    def __init__(self, q_app_fn, q_app_params, learning_rate=1e-3,
                 momentum=0.9, tau=0.1, use_cuda=True, is_multimodal=True,
                 decoder_latent_lambda=1e-6):
        super().__init__(q_app_fn, q_app_params, learning_rate, momentum,
                         tau=tau, use_cuda=use_cuda)
        self.models = list()
        self.priors = list()
        self.decoder_latent_lambda = decoder_latent_lambda
        self.is_multimodal = is_multimodal
        self.augment_model = AutoAugment()
        # self.augment_model = AugMix()

    def append_autoencoder(self, ae_model,
                           encoder_lr,
                           decoder_lr,
                           decoder_weight_decay):
        ae_model.to(self.device)
        ae_model.apply(weight_init)
        ae_model.adabelief_optimizer(encoder_lr, decoder_lr, decoder_weight_decay)
        self.models.append(ae_model)

    def append_prior(self, belief_model, learning_rate=1e-3):
        belief_model.to(self.device)
        belief_model.apply(weight_init)
        self.priors.append(PriorModel(belief_model))
        # self.priors[-1].sgd_optimizer(learning_rate=learning_rate)
        self.priors[-1].adabelief_optimizer(learning_rate=learning_rate)

    def compute_priors_loss(self, obs, obs_t1, actions):
        obs_2d, obs_1d = self.format_obs(obs)
        obs_2d_t1, obs_1d_t1 = self.format_obs(obs_t1)

        loss_list = list()
        for pmodel in self.priors:
            z = self.compute_z(obs, pmodel.latent_source)
            hat_x = pmodel(z)
            if 'rgb' in pmodel.obs_target:
                x_true = pmodel.obs2target(obs_2d)
            elif 'vector' in pmodel.obs_target:
                x_true = pmodel.obs2target(obs_1d)

            loss = pmodel.compute_loss(hat_x, x_true)
            summary_scalar(f"Prior/{pmodel.name}/ReconstructionLoss", loss.item())
            loss_list.append(loss)

        return torch.stack(loss_list)

    def append_models(self, models):
        # ensure empty list
        if hasattr(self, 'models'):
            del self.models
        self.models = list()
        for m, m_params in models.items():
            # RGB observation reconstruction autoencoder model
            if m =='rgb':
                ae_model = RGBModel(m_params)
            if m =='atc':
                ae_model = ATCModel(m_params)
            self.append_autoencoder(
                ae_model, m_params['encoder_lr'], m_params['decoder_lr'],
                m_params['decoder_weight_decay'])

    def format_obs(self, obs, augment=False):
        if self.is_multimodal:
            obs_2d = obs[0]
            obs_1d = obs[1]
        else:
            obs_2d = obs
            obs_1d = obs

        obs_2d = super().format_obs(obs_2d, is_pixels=True)
        obs_1d = super().format_obs(obs_1d, is_pixels=False)

        # augment pixel values
        if augment:
            orig_shape = obs_2d.shape
            obs_frames = obs_2d.reshape((obs_2d.shape[0] * 3, 3, obs_2d.shape[-2], obs_2d.shape[-1]))
            obs_2d = torch.cat([self.augment_model(frame.to(torch.uint8)) for frame in obs_frames])
            obs_2d = obs_2d.reshape(orig_shape).to(torch.float32)

        return obs_2d.to(self.device), obs_1d.to(self.device)

    def compute_z(self, observations, latent_types=None):
        z_hat = list()
        obs_2d, obs_1d = self.format_obs(observations)

        for m in self.models:
            if latent_types is None or m.type in latent_types:
                if m.type in ['rgb', 'atc']:
                    z = m.encode_obs(obs_2d.to(self.device))
                if m.type in ['vector', 'imu2pose']:
                    z = m.encode_obs(obs_1d.to(self.device))
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                z_hat.append(z)
        z_hat = torch.cat(z_hat, dim=1)
        return z_hat

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
            m.encoder_optim[0].zero_grad()
        super().update(td_loss)
        for m in self.models:
            m.encoder_optim[0].step()

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
        q_app_path = str(path) + "_q_function.pth"
        super().save(q_app_path)

        for i, (m, _) in enumerate(ae_models.items()):
            ae_model = self.models[i]
            encoder, decoder = ae_model.encoder, ae_model.decoder
            encoder_opt, decoder_opt = ae_model.encoder_optim, ae_model.decoder_optim

            state_dict = dict()
            for i, (e, eopt) in enumerate(zip(encoder, encoder_opt)):
                state_dict.update(
                    {f"encoder_state_dict_{i}": e.state_dict(),
                     f"encoder_optimizer_state_dict_{i}": eopt.state_dict()})

            if not encoder_only:
                for i, (d, dopt) in enumerate(zip(decoder, decoder_opt)):
                    state_dict.update(
                        {f"decoder_state_dict_{i}": d.state_dict(),
                         f"decoder_optimizer_state_dict_{i}": dopt.state_dict()})

            q_app_path = str(path) + f"_ae_{m}.pth"
            torch.save(state_dict, q_app_path)

    def load(self, path, ae_models, eval_only=False, encoder_only=False):
        q_app_path = str(path) + "_q_function.pth"
        super().load(q_app_path, eval_only=eval_only)

        # ensure empty list
        if hasattr(self, 'models'):
            del self.models
        self.models = list()

        for i, (m, m_params) in enumerate(ae_models.items()):
            if m =='rgb':
                ae_model = RGBModel(m_params, encoder_only=encoder_only)
            if m =='atc':
                ae_model = ATCModel(m_params, encoder_only=encoder_only)
            self.append_autoencoder(
                ae_model, m_params['encoder_lr'], m_params['decoder_lr'],
                m_params['decoder_weight_decay'])
            encoder, decoder = ae_model.encoder, ae_model.decoder
            encoder_opt, decoder_opt = ae_model.encoder_optim, ae_model.decoder_optim

            q_app_path = str(path) + f"_ae_{m}.pth"
            checkpoint = torch.load(q_app_path, map_location=self.device)

            for i, (e, eopt) in enumerate(zip(encoder, encoder_opt)):
                e.load_state_dict(checkpoint[f"encoder_state_dict_{i}"])
                if eval_only:
                    # Ensure the models are in evaluation mode after loading
                    e.eval()
                else:
                    eopt.load_state_dict(
                        checkpoint[f"encoder_optimizer_state_dict_{i}"])

            if not encoder_only:
                for i, (d, dopt) in enumerate(zip(decoder, decoder_opt)):
                    d.load_state_dict(checkpoint[f"decoder_state_dict_{i}"])
                    if eval_only:
                        # Ensure the models are in evaluation mode after loading
                        d.eval()
                    else:
                        dopt.load_state_dict(
                            checkpoint[f"decoder_optimizer_state_dict_{i}"])


class SRLDDQNAgent(DDQNAgent):
    def __init__(self,
                 state_shape,
                 action_shape,
                 approximator,
                 ae_models,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_steps=500000,
                 memory_buffer=None,
                 train_freq=4,
                 target_update_freq=100,
                 reconstruct_freq=1,
                 srl_loss=False,
                 priors=False):
        super().__init__(
            state_shape,
            action_shape,
            approximator,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_steps=epsilon_steps,
            memory_buffer=memory_buffer)

        self.ae_models = ae_models
        self.init_models()
        self.priors = priors
        self.srl_loss = srl_loss
        self.reconstruct_freq = reconstruct_freq
        if priors:
            self.init_priors()

    def init_models(self):
        self.approximator.append_models(self.ae_models)

    def init_priors(self):
        self.approximator.append_prior(NorthBelief(self.state_shape[1], 50))
        self.approximator.append_prior(OrientationBelief(self.state_shape[1], 50), learning_rate=1e-4)
        self.approximator.append_prior(PositionBelief(self.state_shape[1], 50))

    def compute_reconstruction_loss(self, sampled_data):
        obs = sampled_data[0][0] if self.is_prioritized else sampled_data[0]
        actions = sampled_data[0][1] if self.is_prioritized else sampled_data[1]
        rewards = sampled_data[0][2] if self.is_prioritized else sampled_data[2]
        obs_t1 = sampled_data[0][3] if self.is_prioritized else sampled_data[3]

        obs_2d, obs_1d = self.approximator.format_obs(obs, augment=False)
        obs_2d_t1, obs_1d_t1 = self.approximator.format_obs(obs_t1, augment=False)
        obs_2d_augm, _ = self.approximator.format_obs(obs, augment=True)
        obs_2d_t1_augm, _ = self.approximator.format_obs(obs_t1, augment=True)

        total_loss = list()
        srl_loss = list()
        for ae_model in self.approximator.models:
            if ae_model.type in ["atc"]:
                ae_model.update_momentum_encoder(0.01)
                total_loss.append(ae_model.compute_contrastive_loss(obs_2d_augm, obs_2d_t1_augm))
                # total_loss.append(ae_model.compute_compression_loss(obs_2d, obs_2d_t1))
                total_loss.append(ae_model.compute_reconstruction_loss(obs_2d, obs_2d_augm, self.approximator.decoder_latent_lambda))
                # ae_model.update_momentum_encoder(self.approximator.tau)

            if ae_model.type in ["rgb"]:
                total_loss.append(
                    ae_model.compute_loss(
                        obs_2d, obs_2d_augm, self.approximator.decoder_latent_lambda))
                if self.srl_loss:
                    srl_loss.append(ae_model.compute_state_priors(obs_2d, actions, rewards, obs_2d_t1))
            if ae_model.type in ["vector"]:
                loss = ae_model.compute_loss(obs_1d, obs_1d, self.approximator.decoder_latent_lambda)
                total_loss.append(loss)
                if self.srl_loss:
                    srl_loss.append(ae_model.compute_state_priors(obs_1d, actions, rewards, obs_1d_t1))

        tloss = torch.sum(torch.stack(total_loss))
        if self.priors:
            prior_loss = self.approximator.compute_priors_loss(obs, obs_t1, actions)
            tloss += prior_loss.mean() * 0.1
        if self.srl_loss:
            tloss += torch.mean(torch.stack(srl_loss)) * 0.1  # \times lambda

        summary_scalar("Loss/Representation", tloss.item())
        return tloss

    def update_reconstruction(self):
        # Update the autoencoder models if replay buffer is sufficiently large
        if len(self.memory) >= self.BATCH_SIZE:
            # sampled_data = self.memory.random_sample(self.BATCH_SIZE, device=self.approximator.device)
            sampled_data = self.memory.sample(self.BATCH_SIZE, device=self.approximator.device)
            rec_loss = self.compute_reconstruction_loss(sampled_data)
            self.approximator.update_representation(rec_loss)

    def update(self, step):
        super().update(step)
        if step % self.reconstruct_freq == 0:
            self.update_reconstruction()

    def save(self, path, encoder_only=False):
        self.approximator.save(path, ae_models=self.ae_models)

    def load(self, path, eval_only=True, encoder_only=False):
        self.approximator.load(path, ae_models=self.ae_models,
                               eval_only=eval_only, encoder_only=encoder_only)
