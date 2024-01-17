#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:38:44 2023

@author: Angel Ayala
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from webots_drone.data import ExperimentData
from webots_drone.reward import compute_distance_reward
from webots_drone.reward import compute_orientation_reward
from webots_drone.reward import sum_and_normalize
from webots_drone.utils import min_max_norm
from webots_drone.utils import compute_risk_distance
from webots_drone.data import read_args

import re


def create_area_axis(axis_intervals, is_3d=False):
    # Create a new figure and axis
    fig = plt.figure(figsize=(8, 6))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # Define the vertices of the cube
    vertices = [
        [axis_intervals[0][0], axis_intervals[0][1]],
        [axis_intervals[1][0], axis_intervals[0][1]],
        [axis_intervals[1][0], axis_intervals[1][1]],
        [axis_intervals[0][0], axis_intervals[1][1]]
    ]

    # Define the edges of the cube
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]]
    ]
    if is_3d:
        edges.append([
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]]
        ])

    # Plot the edges
    for edge in edges:
        ax.plot(*zip(*edge), color='b', alpha=0.5)

    # Set labels for each axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if is_3d:
        ax.set_zlabel('Z')

    return ax


def get_flight_area(env_params):
    import webots_drone
    package_path = webots_drone.__path__[0]
    world_path = Path(package_path + '/../worlds/forest_tower.wbt')

    with open(world_path, 'r') as world_dump:
        world_file = world_dump.read()

    area_size_regex = re.search(
        r"DEF FlightArea(.*\n)+  size( \-?\d+\.?\d?){2}\n",
        world_file)
    area_size_str = area_size_regex.group().strip()
    area_size = list(map(float, area_size_str.split(' ')[-2:]))
    # print('area_size', area_size)

    area_size = [fs / 2 for fs in area_size]  # size from center
    flight_area = [[fs * -1 for fs in area_size], area_size]

    # add altitude limits
    flight_area[0].append(env_params['altitude_limits'][0])
    flight_area[1].append(env_params['altitude_limits'][1])

    return flight_area


def plot_fire_zone(area_axis, env_params):
    # add fire and risk zone
    risk_distance = compute_risk_distance(*env_params['fire_dim'])
    goal_distance = risk_distance + env_params['goal_threshold']
    fire_pos = tuple(env_params['fire_pos'])
    fire_spot = plt.Circle(fire_pos, env_params['fire_dim'][1], color='r')
    risk_zone = plt.Circle(fire_pos, risk_distance, color='r', alpha=0.2)
    goal_thr_zone = plt.Circle(fire_pos, goal_distance, color='g', fill=False)
    area_axis.add_patch(fire_spot)
    area_axis.add_patch(risk_zone)
    area_axis.add_patch(goal_thr_zone)
    return area_axis


def plot_trj_reward(captured_positions, captured_orientations, rewards):
    # plt.figure(figsize=(8, 6))
    plt.scatter(captured_positions[:, 0], captured_positions[:, 1], c=rewards,
                cmap='viridis')
    plt.colorbar(label='Reward')
    plt.quiver(captured_positions[:, 0], captured_positions[:, 1],
               np.cos(captured_orientations), np.sin(captured_orientations),
               angles='xy', scale_units='xy', scale=1, color='black')


def compute_total_reward(uav_xy, target_xy, uav_ori, env_params):
    # compute reward components
    orientation_reward = compute_orientation_reward(uav_xy, uav_ori, target_xy)
    distance_reward = compute_distance_reward(
        uav_xy, target_xy, distance_max=50.,
        distance_threshold=compute_risk_distance(*env_params['fire_dim']),
        threshold_offset=env_params['goal_threshold'])
    reward = sum_and_normalize(orientation_reward, distance_reward)
    return reward


def read_args_environment(experiment_path):
    return read_args(experiment_path / 'args_environment.json')


# %% Read Data
exp_path = Path('rl_experiments/logs_drone_vector/ddqn-srl_2023-11-28_00-13-33/history.csv')
exp_path = Path('rl_experiments/logs_drone_pixels/ddqn-srl_2023-11-30_10-58-40/history.csv')
exp_data = ExperimentData(exp_path)
env_params = read_args_environment(exp_path.parent)


# %% Plot learning curve
def get_eval_reward(history_df):
    eval_history = history_df[history_df['phase'] == 'eval']
    reward_values = eval_history.groupby('ep').agg({'reward': 'sum'})
    reward_values = reward_values.reset_index(drop=True)
    reward_values.index += 1
    return reward_values


eval_reward = get_eval_reward(exp_data.history_df)
plt.plot(eval_reward)
plt.show()
np.argmax(eval_reward)

# %% Filter data
trajectory_data = exp_data.get_ep_trajectories(90)
trajectory = trajectory_data[0]
trj_length, init_pos, trj_target = trajectory[:3]
timemarks, rewards, orientations = trajectory[3:6]
states, actions, next_states = trajectory[6:]

print(states.shape, actions.shape, next_states.shape, orientations.shape, rewards.shape)


# %% Plot flight trajectory
def plot_ep_trajectory(episode, exp_data, env_params, limit_area=True):
    # Filter data
    trj_data = exp_data.get_ep_trajectories(episode)[0]
    trj_length = trj_data[0]
    rewards, orientations, states = trj_data[4:7]
    # Plot area and scene elements
    flight_area = get_flight_area(env_params)
    area_axis = create_area_axis(flight_area)
    plot_fire_zone(area_axis, env_params)
    if limit_area:
        area_axis.set_xlim((flight_area[0][0], flight_area[1][0]))
        area_axis.set_ylim((flight_area[0][1], flight_area[1][1]))
    # Plot trajectory
    area_axis.plot(states[:, 0], states[:, 1],
                   label=f"{trj_length} steps reward={rewards.sum():.2f}")
    area_axis.legend(loc="lower left")
    # Plot rewards
    plot_trj_reward(states[:, :2], orientations, rewards)
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title(f"Trajectory reward EP {episode}")
    plt.grid()
    plt.tight_layout()
    plt.show()


plot_ep_trajectory(90, exp_data, env_params, limit_area=False)

# %% Plot computed rewards
comp_rewards = np.zeros_like(rewards)
diff_rewards = np.zeros_like(rewards)

for i, R in enumerate(rewards):
    rtotal = compute_total_reward(states[i, :2], trj_target[:2],
                                  orientations[i], env_params)
    comp_rewards[i] = rtotal
    diff_rewards[i] = rtotal - R
area_axis = create_area_axis(get_flight_area(env_params))
plot_fire_zone(area_axis, env_params)
plot_trj_reward(states[:, :2], orientations, comp_rewards)
