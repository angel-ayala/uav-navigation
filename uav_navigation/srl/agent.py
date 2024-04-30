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
# from torchvision.transforms import AutoAugment
# from torchvision.transforms import AugMix
from thop import clever_format
from uav_navigation.agent import QFunction
from uav_navigation.agent import DDQNAgent
from uav_navigation.agent import profile_q_approximator
from uav_navigation.logger import summary_scalar
from .net import weight_init
from .priors import PriorModel
from .priors import NorthBelief
from .priors import PositionBelief
from .priors import OrientationBelief
from .priors import DistanceBelief
from .autoencoder import RGBModel
from .autoencoder import VectorModel
from .autoencoder import ATCModel
from .autoencoder import ATCRGBModel
from .autoencoder import profile_ae_model


def profile_srl_approximator(approximator, state_shape, action_shape):
    total_flops, total_params = 0, 0
    q_feature_dim = 0
    for m in approximator.models:
        if approximator.is_multimodal:
            if any(t in m.type for t in ['RGB', 'ATC']):
                flops, params = profile_ae_model(m, state_shape[0], approximator.device)
            if m.type == 'VECTOR':
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


class SRLFunction:
    def __init__(self, decoder_latent_lambda=1e-6):
        self.decoder_latent_lambda = decoder_latent_lambda
        self.models = list()
        self.priors = list()

    def append_autoencoder(self, ae_model,
                           encoder_lr,
                           decoder_lr,
                           decoder_weight_decay):
        ae_model.to(self.device)
        ae_model.apply(weight_init)
        ae_model.adam_optimizer(encoder_lr, decoder_lr, decoder_weight_decay)
        self.models.append(ae_model)

    def append_autoencoders(self, models, encoder_only=False):
        # ensure empty list
        if hasattr(self, 'models'):
            del self.models

        self.models = list()
        for m, m_params in models.items():
            # RGB observation reconstruction autoencoder model
            if m =='RGB':
                ae_model = RGBModel(m_params, encoder_only=encoder_only)
            if m =='Vector':
                ae_model = VectorModel(m_params, encoder_only=encoder_only)
            if m =='ATC':
                ae_model = ATCModel(m_params, encoder_only=encoder_only)
            if m =='ATC-RGB':
                ae_model = ATCRGBModel(m_params, encoder_only=encoder_only)
            self.append_autoencoder(
                ae_model, m_params['encoder_lr'], m_params['decoder_lr'],
                m_params['decoder_weight_decay'])

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
                if any(t in m.type for t in ['RGB', 'ATC']):
                    z = m.encode_obs(obs_2d.to(self.device))
                if any(t in m.type for t in ['Vector', 'imu2pose']):
                    z = m.encode_obs(obs_1d.to(self.device))
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
            if "ATC" in ae_model.type:
                ae_model.update_momentum_encoder(0.01)
                loss = ae_model.compute_loss(obs_2d_augm, obs_2d_t1_augm)
            if "RGB" in ae_model.type:
                loss = ae_model.compute_loss(obs_2d, obs_2d_augm, self.decoder_latent_lambda)
            if "Vector" in ae_model.type:
                loss = ae_model.compute_loss(obs_1d, obs_1d, self.decoder_latent_lambda)
            total_loss.append(loss)
        tloss = torch.sum(torch.stack(total_loss))
        summary_scalar("Loss/AutoEncoders", tloss.item())
        return tloss

    def append_prior(self, belief_model, latent_source, obs_target, learning_rate=1e-3):
        belief_model.to(self.device)
        belief_model.apply(weight_init)
        print('belief_model', belief_model)
        self.priors.append(PriorModel(belief_model,
                                      latent_source=latent_source,
                                      obs_target=obs_target))
        # self.priors[-1].sgd_optimizer(learning_rate=learning_rate)
        self.priors[-1].adam_optimizer(learning_rate=learning_rate)

    def compute_priors_loss(self, obs, obs_t1, actions):
        if self.is_multimodal:
            obs_2d, obs_1d = self.format_obs(obs, augment=False)
        elif self.is_pixels:
            obs_2d = self.format_obs(obs, augment=False)
        else:
            obs_1d = self.format_obs(obs, augment=False)

        loss_list = list()
        for pmodel in self.priors:
            z = self.compute_z(obs, pmodel.latent_source)
            hat_x = pmodel(z)
            if 'RGB' in pmodel.obs_target:
                x_true = pmodel.obs2target(obs_2d)
            elif 'Vector' in pmodel.obs_target:
                x_true = pmodel.obs2target(obs_1d)

            loss = pmodel.compute_loss(hat_x, x_true)
            summary_scalar(f"Loss/Prior/{pmodel.name}", loss.item())
            loss_list.append(loss)

        ploss = torch.stack(loss_list)
        ploss = ploss.sum()
        summary_scalar("Loss/Prior", ploss.item())
        return ploss

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

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        # ensure empty list
        if hasattr(self, 'models'):
            del self.models
        self.models = list()

        for i, (m, m_params) in enumerate(ae_models.items()):
            if m =='RGB':
                ae_model = RGBModel(m_params, encoder_only=encoder_only)
            if m =='Vector':
                ae_model = VectorModel(m_params, encoder_only=encoder_only)
            if m =='ATC':
                ae_model = ATCModel(m_params, encoder_only=encoder_only)
            if m =='ATC-RGB':
                ae_model = ATCRGBModel(m_params, encoder_only=encoder_only)
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


class SRLAgent:
    def __init__(self,
                 ae_models,
                 reconstruct_freq=1,
                 srl_loss=False,
                 priors=False,
                 encoder_only=False):
        self.ae_models = ae_models
        self.init_models(encoder_only)
        self.reconstruct_freq = reconstruct_freq
        self.srl_loss = srl_loss
        self.priors = priors
        if priors:
            self.init_priors()

    def init_models(self, encoder_only):
        self.approximator.append_autoencoders(self.ae_models,
                                              encoder_only=encoder_only)

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
        # sampled_data = self.memory.random_sample(self.BATCH_SIZE, device=self.approximator.device)
        sampled_data = self.memory.sample(self.batch_size, device=self.approximator.device)

        if self.is_prioritized:
            obs, actions, rewards, obs_t1, dones = sampled_data[0]
        else:
            obs, actions, rewards, obs_t1, dones = sampled_data

        rec_loss = self.approximator.compute_ae_loss(obs, actions, rewards, obs_t1, dones)
        # TODO: implement state representation learning loss

        if self.priors:
            rec_loss += self.approximator.compute_priors_loss(obs, obs_t1, actions)

        summary_scalar("Loss/Representation", rec_loss.item())
        self.approximator.update_representation(rec_loss)

    def update(self, step):
        if not self.can_update:
            return False
        if step % self.reconstruct_freq == 0:
            self.update_reconstruction()

    def save(self, path, encoder_only=False):
        self.approximator.save(path, self.ae_models, encoder_only)

    def load(self, path, eval_only=True, encoder_only=False):
        self.approximator.load(path, self.ae_models, encoder_only, eval_only)


class SRLQFunction(QFunction, SRLFunction):
    def __init__(self, q_app_fn, q_app_params, learning_rate=1e-3,
                 momentum=0.9, tau=0.1, use_cuda=True, is_pixels=False,
                 is_multimodal=False, use_augmentation=True,
                 decoder_latent_lambda=1e-6):
        QFunction.__init__(self, q_app_fn, q_app_params, learning_rate,momentum,
                           tau=tau, use_cuda=use_cuda, is_pixels=is_pixels,
                           is_multimodal=is_multimodal,
                           use_augmentation=use_augmentation)
        SRLFunction.__init__(self, decoder_latent_lambda)

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
        QFunction.update(self, td_loss)
        for m in self.models:
            m.encoder_optim[0].step()

    def save(self, path, ae_models, encoder_only=False):
        QFunction.save(self, path)
        SRLFunction.save(self, path, ae_models, encoder_only)

    def load(self, path, ae_models, encoder_only=False, eval_only=False):
        QFunction.load(self, path, eval_only)
        SRLFunction.load(self, path, ae_models, encoder_only, eval_only)


class SRLDDQNAgent(DDQNAgent, SRLAgent):
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
                 batch_size=128,
                 train_freq=4,
                 target_update_freq=100,
                 reconstruct_freq=1,
                 srl_loss=False,
                 priors=False):
        DDQNAgent.__init__(self,
                           state_shape,
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
                          priors=priors)

    def update(self, step):
        DDQNAgent.update(self, step, augment=False)
        SRLAgent.update(self, step)

    def save(self, path, encoder_only=False):
        SRLAgent.save(self, path, encoder_only)

    def load(self, path, eval_only=True, encoder_only=False):
        SRLAgent.load(self, path, eval_only, encoder_only)
