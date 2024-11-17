#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:23:24 2024

@author: Angel Ayala
"""
import torch
import numpy as np
# from pytorch_msssim import MS_SSIM
# from pytorch_msssim import SSIM


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


def logarithmic_difference_loss(predicted, ground_truth, gamma=1.0):
    # Ensure non-zero predictions and targets to avoid NaNs in logarithms
    eps = 1e-8
    predicted = torch.clamp(predicted, min=eps, max=float('inf'))
    ground_truth = torch.clamp(ground_truth, min=eps, max=float('inf'))

    # Calculate the loss
    log_diff = torch.abs(torch.log(ground_truth) - torch.log(predicted))
    loss = torch.mean(torch.exp(gamma * log_diff))
    return loss


# class MS_SSIM_Loss(MS_SSIM):
#     def forward(self, img1, img2):
#         return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )


# class SSIM_Loss(SSIM):
#     def forward(self, img1, img2):
#         return 1 - super(SSIM_Loss, self).forward(img1, img2)
