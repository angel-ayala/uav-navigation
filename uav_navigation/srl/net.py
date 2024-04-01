#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:19:08 2023

@author: Angel Ayala
Based on:
"Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
https://arxiv.org/abs/1910.01741
"""
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from adabelief_pytorch import AdaBelief
from pytorch_msssim import MS_SSIM
from pytorch_msssim import SSIM


OUT_DIM = {2: 39, 4: 35, 6: 31}


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    # if isinstance(m, nn.Linear):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def sgd_optimizer(model, learning_rate=1e-5, momentum=0.9, **kwargs):
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                     nesterov=True, **kwargs)


def adabelief_optimizer(model, learning_rate=1e-3):
    return AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16,
                     betas=(0.9, 0.999), weight_decouple=True, rectify=False)


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def rgb_reconstruction_model(image_shape, latent_dim, num_layers=2,
                             num_filters=32):
    encoder = PixelEncoder(image_shape, latent_dim, num_layers=num_layers,
                           num_filters=num_filters)
    decoder = PixelDecoder(image_shape, latent_dim, num_layers=num_layers,
                           num_filters=num_filters)
    return encoder, decoder


def vector_reconstruction_model(vector_shape, hidden_dim, latent_dim,
                                num_layers=2):
    encoder = VectorEncoder(vector_shape, latent_dim, hidden_dim, num_layers=num_layers)
    decoder = VectorDecoder(vector_shape, latent_dim, hidden_dim, num_layers=num_layers)
    return encoder, decoder


def imu2pose_model(imu_shape, pos_shape, hidden_dim, latent_dim,
                   num_layers=2):
    encoder = MLP(imu_shape[0], latent_dim, hidden_dim, num_layers=num_layers)
    decoder1 = MLP(latent_dim, imu_shape[0], hidden_dim, num_layers=num_layers)
    decoder2 = MLP(latent_dim, pos_shape[0], hidden_dim, num_layers=num_layers)
    return encoder, (decoder1, decoder2)


def q_function(latent_dim, action_shape, hidden_dim, num_layers=2):
    return VectorDecoder(action_shape, latent_dim, hidden_dim, num_layers=num_layers)


def slowness_cost(h_t, h_t1):
    """
    Compute the slowness cost for a batch of encoded representations.

    Parameters:
    - h_t: A 2D PyTorch tensor representing the batch of encoded representations in moment t.
                     Each row represents one encoded representation.
    - h_t1: A 2D PyTorch tensor representing the batch of encoded representations in moment t+1.
                     Each row represents one encoded representation.
    - lambda_: A regularization parameter controlling the strength of the penalty.

    Returns:
    - slowness: The slowness cost.
    """
    # Compute differences between consecutive encoded representations
    differences = h_t1 - h_t

    # Compute squared norm of differences
    squared_norms = torch.norm(differences, p=2, dim=-1) ** 2

    # Compute mean squared norm
    return torch.mean(squared_norms)


def variability_cost(h_t, h_t1):
    """
    Compute the variability loss for a batch of encoded representations.

    Parameters:
    - h_t: A 2D PyTorch tensor representing the batch of encoded representations in moment t.
                     Each row represents one encoded representation.
    - h_t1: A 2D PyTorch tensor representing the batch of encoded representations in moment t+1.
                     Each row represents one encoded representation.

    Returns:
    - variability: The variability loss.
    """
    # Pairwise Euclidean distances between all encoded representations in the batch
    pairwise_distances = torch.cdist(h_t, h_t1, p=2)

    # Exponential of negative distances
    exponential_neg_distances = torch.exp(-pairwise_distances)

    # Exclude self-distances (diagonal elements)
    # mask = torch.eye(h_t.size(0), dtype=torch.bool, device=h_t.device)
    # exponential_neg_distances = exponential_neg_distances.masked_fill(mask, 0)

    # Compute mean of exponential distances
    variability = torch.mean(exponential_neg_distances)

    return variability


def proportionality_cost(h_t, h_t1, actions):
    """
    Compute the proportionality cost for encoded states.

    Parameters:
    - h_t: A 2D PyTorch tensor representing the batch of encoded representations in moment t.
                     Each row represents one encoded representation.
    - h_t1: A 2D PyTorch tensor representing the batch of encoded representations in moment t+1.
                     Each row represents one encoded representation.
    - actions: A 1D PyTorch tensor representing the actions corresponding to each encoded state.

    Returns:
    - proportionality: The proportionality cost.
    """
    # Compute differences in consecutive encoded states
    delta_states = h_t1 - h_t  # Calculate difference directly (avoiding incorrect torch.diff)

    # Filter pairs of consecutive time steps with equal actions
    equal_actions_mask = torch.eq(actions, torch.cat((actions[1:], actions[0].unsqueeze(0))))

    # Filter delta states based on equal actions mask
    delta_states_filtered = delta_states[equal_actions_mask]

    # Compute norm of differences for each pair of consecutive time steps
    norms = torch.norm(delta_states_filtered, dim=1)

    # Compute squared differences of norms
    squared_norm_diffs = torch.square(torch.diff(norms, dim=0))

    # Take the mean
    proportionality = torch.mean(squared_norm_diffs)

    # Replace NaNs with a small constant (e.g., 1e-8)
    proportionality = torch.where(torch.isnan(proportionality), 1e-8, proportionality)

    return proportionality


def repeatability_cost(h_t, h_t1, actions):
    """
    Compute the repeatability cost for encoded states.

    Parameters:
      - h_t: A 2D PyTorch tensor representing the batch of encoded representations in moment t.
            Each row represents one encoded representation.
      - h_t1: A 2D PyTorch tensor representing the batch of encoded representations in moment t+1.
            Each row represents one encoded representation.
      - actions: A 1D PyTorch tensor representing the actions corresponding to each encoded state.

    Returns:
      - repeatability_loss: The repeatability loss.
    """

    # Calculate difference between consecutive encoded states
    delta_states = h_t1 - h_t

    # Filter pairs of consecutive time steps with equal actions
    equal_actions_mask = torch.eq(actions, torch.cat((actions[1:], actions[0].unsqueeze(0))))

    # Filter delta states based on equal actions mask
    delta_states_filtered = delta_states[equal_actions_mask]

    # Calculate squared difference between encoded states (filtered)
    squared_diff = torch.square(delta_states_filtered)
    # Sum the squared differences across dimensions (assuming each state is a vector)
    state_diff_norm = torch.sum(squared_diff, dim=0)

    # Exponentiate the negative squared difference (avoiding overflow with clamp)
    exp_term = torch.exp(-torch.clamp(state_diff_norm, min=0.0))

    # Compute squared differences of consecutive delta states (using valid indices)
    squared_delta_state_diffs = torch.square(delta_states_filtered[1:] - delta_states_filtered[:-1])

    # Combine terms and compute mean over filtered data
    repeatability_loss = torch.mean(exp_term * squared_delta_state_diffs)

    # Replace NaNs with a small constant (e.g., 1e-8)
    repeatability_loss = torch.where(torch.isnan(repeatability_loss), 1e-8, repeatability_loss)

    return repeatability_loss


def angular_loss(orientation, orientation_true):
    """
    Calculates the angular difference loss between predicted and target angles.

    Args:
      orientation: Predicted angles (in radians), PyTorch tensor.
      orientation_true: Ground truth angles (in radians), PyTorch tensor.

    Returns:
      Loss value, a scalar tensor.
    """
    orientation = (orientation + np.pi) % (2 * np.pi) - np.pi
    orientation_true = (orientation_true + np.pi) % (2 * np.pi) - np.pi
    # Calculate difference between angles
    diff = orientation - orientation_true
    # Approximate loss using squared difference
    loss = 1 - ((torch.cos(diff) + 1.) / 2)
    return loss


def circular_difference(predicted_angles, real_angles):
    """
    Calculates the circular difference between predicted and real angles in a batch.

    Args:
      predicted_angles: A PyTorch tensor of shape (batch_size, 3) containing predicted angles in radians.
      real_angles: A PyTorch tensor of shape (batch_size, 3) containing real angles in radians.

    Returns:
      A PyTorch tensor of shape (batch_size) containing the circular difference for each batch element.
    """

    # No conversion needed since angles are already in radians

    # Calculate absolute difference considering rotations
    diff = torch.abs(predicted_angles - real_angles)
    wrapped_diff1 = torch.abs(predicted_angles + torch.tensor(2*torch.pi) - real_angles)
    wrapped_diff2 = torch.abs(predicted_angles - real_angles - torch.tensor(2*torch.pi))

    # Find minimum difference among original and wrapped values
    min_diffs = torch.min(torch.stack([diff, wrapped_diff1, wrapped_diff2]), dim=0)[0]  # Find minimum along first dimension
    circular_diff = torch.min(min_diffs, dim=1).values  # Find minimum along second dimension

    return circular_diff


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 - super(SSIM_Loss, self).forward(img1, img2)


def logarithmic_difference_loss(predicted, ground_truth, gamma=1.0):
    # Ensure non-zero predictions and targets to avoid NaNs in logarithms
    eps = 1e-8
    predicted = torch.clamp(predicted, min=eps, max=float('inf'))
    ground_truth = torch.clamp(ground_truth, min=eps, max=float('inf'))

    # Calculate the loss
    log_diff = torch.abs(torch.log(ground_truth) - torch.log(predicted))
    loss = torch.mean(torch.exp(gamma * log_diff))
    return loss


class MLP(nn.Module):
    """MLP for q-function."""

    def __init__(self, n_input, n_output, hidden_dim, num_layers=3):
        super().__init__()
        self.h_layers = nn.ModuleList([nn.Linear(n_input, hidden_dim)])
        for i in range(num_layers - 1):
            self.h_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.h_layers.append(nn.Dropout(0.2))
        self.h_layers.append(nn.Linear(hidden_dim, n_output))
        self.num_layers = len(self.h_layers)

    def forward(self, obs, detach=False):
        h = obs
        for i in range(self.num_layers):
            if isinstance(self.h_layers[i], nn.Dropout):
                h = self.h_layers[i](h)
            else:
                h = torch.relu(self.h_layers[i](h))

        if detach:
            h = h.detach()

        return h


class BiGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=4, num_layers=1):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, device, detach=False):
        # Combine data for each sample (assuming x is a list of [t, t+1] element pairs)
        combined_data = [torch.cat(pair, dim=0) for pair in x]
        combined_data = torch.stack(combined_data)  # Combine list into tensor
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, combined_data.size(0),
                         self.hidden_size).to(device)
        # Forward propagate GRU
        out, _ = self.gru(combined_data, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class VectorEncoder(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=3):
        super().__init__(state_shape[-1], latent_dim, hidden_dim,
                         num_layers=num_layers-1)
        if len(state_shape) == 2:
            first_layer = nn.Conv1d(state_shape[0], hidden_dim,
                                    kernel_size=state_shape[-1])
        else:
            first_layer = nn.Linear(state_shape[-1], hidden_dim)
        self.feature_dim = latent_dim
        self.h_layers[0] = first_layer  # replace first layer
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, obs, detach=False):
        first_layer = self.h_layers[0]
        h = first_layer(obs)
        if isinstance(first_layer, nn.ConvTranspose1d):
            h = h.squeeze(2)
        for hidden_layer in self.h_layers[1:]:
            h = torch.relu(hidden_layer(h))
        h_norm = self.ln(h)
        out = torch.tanh(h_norm)
        if detach:
            out.detach()
        return out


class VectorDecoder(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim, num_layers=3):
        super().__init__(latent_dim, state_shape[-1], hidden_dim,
                         num_layers=num_layers-1)
        if len(state_shape) == 2:
            last_layer = nn.ConvTranspose1d(hidden_dim, state_shape[0],
                                            kernel_size=state_shape[-1])
        else:
            last_layer = nn.Linear(hidden_dim, state_shape[-1])
        self.h_layers[-1] = last_layer  # replace last layer

    def forward(self, obs, detach=False):
        h = obs
        for hidden_layer in self.h_layers[:-1]:
            h = torch.relu(hidden_layer(h))
        last_layer = self.h_layers[-1]
        if isinstance(last_layer, nn.ConvTranspose1d):
            h = h.unsqueeze(2)
        out = last_layer(h)
        return out


class PriorModel:
    def __init__(self, model, latent_source=['rgb'], obs_target=['vector']):
        self.model = model
        self.name = type(model).__name__
        self.optimizer = None
        self.latent_source = latent_source
        self.obs_target = obs_target
        self.avg_model = optim.swa_utils.AveragedModel(model)

    def sgd_optimizer(self, learning_rate, momentum=0.9, **kwargs):
        self.optimizer = sgd_optimizer(self.model, learning_rate=learning_rate,
                                       momentum=momentum, **kwargs)

    def adabelief_optimizer(self, learning_rate):
        self.optimizer = adabelief_optimizer(self.model,
                                             learning_rate=learning_rate)

    def obs2target(self, observations):
        return self.model.obs2target(observations)

    def compute_loss(self, values_pred, values_true):
        return self.model.compute_loss(values_pred, values_true)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()
        self.avg_model.update_parameters(self.model)

    def __call__(self, x):
        return self.model(x)


class NorthBelief(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim=128,
                 num_layers=3):
        super().__init__(latent_dim, 2, hidden_dim=hidden_dim,
                         num_layers=num_layers)

    def obs2target(self, obs):
        return obs[:, -1, [3, 13]]

    def compute_loss(self, orientation, orientation_true):
        orientation = orientation + 2 * torch.pi
        orientation_true = orientation_true + 2 * torch.pi
        loss = circular_difference(orientation, orientation_true)
        # loss = F.mse_loss(orientation, orientation_true)
        # loss = 1 - (F.cosine_similarity(orientation, orientation_true) + 1 ) / 2.
        # loss = loss.abs()
        return loss.mean() * 0.1


class PositionBelief(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim=128,
                 num_layers=3):
        super().__init__(latent_dim, 3, hidden_dim=hidden_dim,
                         num_layers=num_layers)

    def obs2target(self, obs):
        return obs[:, -1, 6:9]

    def compute_loss(self, position, position_true):
        # loss = F.smooth_l1_loss(position, position_true, beta=2.)
        loss = F.huber_loss(position, position_true, delta=2.)
        return loss.mean() * 0.1


class OrientationBelief(MLP):
    def __init__(self, state_shape, latent_dim, hidden_dim=128,
                 num_layers=3):
        super().__init__(latent_dim, 2, hidden_dim=hidden_dim,
                         num_layers=num_layers)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)

    def obs2target(self, obs):
        return obs[:, -1, :2]

    def compute_loss(self, inertial, inertial_true):
        inertial = inertial + 2 * torch.pi
        inertial_true = inertial_true + 2 * torch.pi
        # loss = 1 - (F.cosine_similarity(inertial, inertial_true) + 1 ) / 2.
        # loss = F.mse_loss(inertial, inertial_true)
        loss = circular_difference(inertial, inertial_true)
        # loss = loss.abs()
        return loss.mean()


class OdometryBelief(VectorDecoder):
    def __init__(self, state_shape, latent_dim, hidden_dim=128):
        # UAV vector + action
        input_shape = list(state_shape)
        input_shape[-1] = 13
        super().__init__(input_shape, latent_dim=latent_dim,
                         hidden_dim=hidden_dim, num_layers=2)
        self.latent_types = ['rgb']
        self.target_types = ['vector']

    # def forward(self, obs, detach=False):
    #     out = super().forward(obs, detach)
    #     # out = torch.tanh(out)
    #     return out

    def obs2target(self, obs):
        # inertial_diff = obs[:, :6] - obs_t1[:, :6]
        # translational_diff = obs[:, 6:12] - obs_t1[:, 6:12]
        # orientation_diff = obs[:, 12:13] - obs_t1[:, 12:13]
        # return torch.cat((inertial_diff, translational_diff, orientation_diff), dim=1)
        return obs[..., :13]


class ChannelAttention(nn.Module):
    def __init__(self, num_filters, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // reduction_ratio, 1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(num_filters // reduction_ratio, num_filters, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze operation
        z = self.avg_pool(x)
        # Excitation operation
        z = self.fc(z)
        # Attention weights
        attention = torch.mul(x, z)
        return attention


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, state_shape, latent_dim, num_layers=2, num_filters=32):
        super().__init__()
        assert len(state_shape) == 3
        self.feature_dim = latent_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList(
            [nn.Conv2d(state_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):

        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        h = h.view(h.size(0), -1)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        h = torch.tanh(h_norm)

        return h

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class PixelMDPEncoder(PixelEncoder):
    def __init__(self, state_shape, latent_dim, num_layers=2, num_filters=32):
        super().__init__(state_shape, latent_dim, num_layers, num_filters)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.code_fc = nn.Linear(num_filters, num_filters)
        self.contrastive = nn.Linear(latent_dim, latent_dim)
        self.probabilities = nn.Linear(latent_dim, latent_dim)

    def forward_prob(self, obs, detach=False):
        h = self.forward(obs, detach=detach)
        h_fc = self.probabilities(h)
        return h_fc

    def forward_code(self, obs, detach=False):
        code = self.forward(obs, detach=detach)
        h_fc = self.contrastive(code) + code
        return h_fc


class PixelDecoder(nn.Module):
    def __init__(self, state_shape, latent_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            latent_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.extend([
                # ChannelAttention(num_filters, reduction_ratio=8),
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            ])
        self.deconvs.extend([
            ChannelAttention(num_filters, reduction_ratio=8),
            nn.ConvTranspose2d(
                num_filters, state_shape[0], 3, stride=2, output_padding=1
            )
        ])
        self.num_layers = len(self.deconvs)

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            if type(self.deconvs[i]) == ChannelAttention:
                deconv_att = self.deconvs[i](deconv)
                deconv = torch.mul(deconv, deconv_att)
            else:
                deconv = torch.relu(self.deconvs[i](deconv))

        obs = self.deconvs[-1](deconv)

        return obs
