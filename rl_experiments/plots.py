#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:38:44 2023

@author: Angel Ayala
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from webots_drone.data import ExperimentData
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
    world_path = Path(package_path + '/../worlds/forest_tower_200x200_simple.wbt')

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


def plot_fire_zone(area_axis, fire_pos, fire_dim, env_params):
    # add fire and risk zone
    risk_distance = compute_risk_distance(*fire_dim)
    goal_distance = risk_distance + env_params['goal_threshold']
    fire_spot = plt.Circle(fire_pos, fire_dim[1], color='r')
    risk_zone = plt.Circle(fire_pos, risk_distance, color='r', alpha=0.2)
    goal_thr_zone = plt.Circle(fire_pos, goal_distance, color='g', fill=False)
    area_axis.add_patch(fire_spot)
    area_axis.add_patch(risk_zone)
    area_axis.add_patch(goal_thr_zone)
    return area_axis


def plot_scene_elements(area_axis):
    tower = plt.Rectangle((-5.1, -7.7), 10.2, 15.4, color='b', alpha=0.3)
    road_lanes = plt.Rectangle((-5, -50), 10, 42.3, color='black', alpha=0.1)
    area_axis.add_patch(road_lanes)
    area_axis.add_patch(tower)
    return area_axis


def plot_trj_reward(captured_positions, captured_orientations, rewards):
    # plt.figure(figsize=(8, 6))
    plt.scatter(captured_positions[:, 0], captured_positions[:, 1], c=rewards,
                cmap='viridis')
    plt.colorbar(label='Reward')
    plt.quiver(captured_positions[:, 0], captured_positions[:, 1],
               np.cos(captured_orientations), np.sin(captured_orientations),
               angles='xy', scale_units='xy', scale=1, color='black')


def read_exp_data(exp_folder,
                  csv_name='history_training.csv',
                  env_args='args_environment.json',
                  train_args='args_training.json'):
    if not isinstance(exp_folder, Path):
        exp_folder = Path(exp_folder)
    print('Reading experiment data from:', exp_folder)
    exp_data = ExperimentData(exp_folder / csv_name)
    env_params = read_args(exp_folder / env_args)
    train_params = read_args(exp_folder / train_args)
    return exp_data, env_params, train_params


def create_output_path(exp_folder, out_name):
    out_path = exp_folder / out_name
    out_path.mkdir(exist_ok=False)
    return out_path


def get_reward_curve(history_df, phase='eval'):
    eval_history = history_df[history_df['phase'] == phase]
    reward_values = eval_history.groupby('ep').agg({'reward': 'sum'})
    # reward_values = reward_values.reset_index(drop=True)
    # reward_values.index += 1
    return reward_values.to_numpy().flatten()


def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    num_acc = 0
    smoothed = np.zeros_like(scalars)
    for i, next_val in enumerate(scalars):
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - np.power(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed[i] = smoothed_val

    return smoothed


def plot_reward_curve(exp_rewards, exp_rewards_srl, plot_title,
                      ep_length=1, smooth_weight=0., fig_path=None):
    # Ensure new figure
    plt.figure()
    # numbered steps
    steps_number = np.array([ep_length * i
                             for i in range(1, len(exp_rewards) + 1)])
    steps_number_srl = np.array([ep_length * i
                                 for i in range(1, len(exp_rewards_srl) + 1)])
    # Smooth the values
    rewards = smooth(exp_rewards, smooth_weight)
    rewards_srl = smooth(exp_rewards_srl, smooth_weight)
    # Plot the original reward curves as shaded background
    plt.fill_between(steps_number, exp_rewards, alpha=0.3)
    plt.fill_between(steps_number_srl, exp_rewards_srl, alpha=0.3)

    # Plot the smoothed reward curves
    plt.plot(steps_number, rewards,
             label=f"DDQN (mean reward={rewards.mean():.2f})")
    plt.plot(steps_number_srl, rewards_srl,
             label=f"DDQN-SRL (mean reward={rewards_srl.mean():.2f})")
    # Set limits of plot area based on the plotted values
    x_lims = (np.min((steps_number.min(), steps_number_srl.min())) - 1,
              np.max((steps_number.max(), steps_number_srl.max())) + 1)
    min_y = np.min((exp_rewards.min(), exp_rewards_srl.min(),
                    rewards.min(), rewards_srl.min()))
    max_y = np.max((exp_rewards.max(), exp_rewards_srl.max(),
                    rewards.max(), rewards_srl.max()))
    y_lims = (min_y - 100, max_y + 100)
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    # Set tick locator and formatter for x-axis
    plt.gca().xaxis.set_major_locator(
        ticker.MultipleLocator(ep_length*10))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: '{:,.0f}'.format(x / ep_length)))
    # Add a vertical line for min epsilon value
    plt.axvline(x=50*ep_length, color='gray', linestyle='--')
    plt.annotate(r'Min. $\epsilon$', xy=(50*ep_length, y_lims[0]+10),
                 xycoords='data', xytext=(5, 5), textcoords='offset points',
                 arrowprops=dict(
                     arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    # Plot sttings
    plt.xlabel('Learning steps (1e4)')
    plt.ylabel('Accumulative reward')
    plt.title(plot_title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), shadow=True,
               ncol=2)
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path)
    plt.show()


def plot_ep_trajectory(episode, exp_data, env_params, plot_title,
                       limit_area=True, fig_path=None, phase='eval'):
    # Filter data
    try:
        trj_data = exp_data.get_ep_trajectories(episode - 1, phase=phase)[0]
        trj_length = trj_data[0]
        rewards, orientations, states = trj_data[4:7]
        # Plot area and scene elements
        flight_area = get_flight_area(env_params)
        area_axis = create_area_axis(flight_area)
        plot_fire_zone(area_axis, fire_pos=tuple(trj_data[2][:2]),
                       fire_dim=env_params['fire_dim'], env_params=env_params)
        plot_scene_elements(area_axis)
        if limit_area:
            area_axis.set_xlim((flight_area[0][0] - 5, flight_area[1][0] + 5))
            area_axis.set_ylim((flight_area[0][1] - 5, flight_area[1][1] + 5))
        # Plot trajectory
        area_axis.plot(states[:, 0], states[:, 1])
        # Plot rewards
        plot_trj_reward(states[:, :2], orientations, rewards)
        plt.xlabel('Position X (meters)')
        plt.ylabel('Position Y (meters)')
        plot_title += f"\n{trj_length} steps, {rewards.sum():.2f} total reward"
        plt.title(plot_title)
        plt.grid()
        plt.tight_layout()
        if fig_path:
            plt.savefig(fig_path)
        plt.show()
    except IndexError:
        print(f"Error: Episode {episode} trajectory data was not found")


# %% Data source
# First results
# base_path = Path('rl_experiments/results/')
# vector-based DDQN
# exp_path = base_path / 'logs_drone_vector/ddqn_2024-01-31_15-22-09'
# vector-based DDQN-srl
# exp_path_srl = base_path / 'logs_drone_vector/ddqn-srl_2024-02-01_13-01-33'
# pixel-based DDQN
# exp_path = base_path / 'logs_drone_pixels/ddqn_2024-01-31_15-20-06'
# pixel-based DDQN-SRL
# exp_path_srl = base_path / 'logs_drone_pixels/ddqn-srl_2024-01-31_15-17-25/'

# Second results
base_path = Path('rl_experiments/logs_results/')
# vector-based DDQN
exp_path = base_path / 'logs_drone_vector/ddqn_2024-02-09_16-58-55'
# vector-based DDQN-srl
exp_path_srl = base_path / 'logs_drone_vector/ddqn-srl_2024-02-09_16-54-53'
# pixel-based DDQN
# exp_path = base_path / 'logs_drone_pixels/ddqn_2024-02-09_17-00-05'
# pixel-based DDQN-SRL
# exp_path_srl = base_path / 'logs_drone_pixels/ddqn-srl_2024-02-09_17-03-01'

# Third results
base_path = Path('rl_experiments/logs_results2/')
# vector-based DDQN
exp_path = base_path / 'vector/ddqn_2024-02-21_13-44-07'
# vector-based DDQN-srl
exp_path_srl = base_path / 'vector/ddqn-srl_2024-02-21_13-47-03'
# pixel-based DDQN
exp_path = base_path / 'pixels/ddqn_2024-02-21_13-36-28'
# pixel-based DDQN-SRL
exp_path_srl = base_path / 'pixels/ddqn-srl_2024-02-21_13-41-35'

# %% Read data
exp_data, env_params, train_params = read_exp_data(exp_path)
exp_data_srl, env_params_srl, train_params_srl = read_exp_data(exp_path_srl)
out_path = None

# %% Ensure save data
out_path = create_output_path(exp_path.parent, 'assets')


# %% Plot curves
def make_reward_curve(phase, smooth_val):
    exp_rewards = get_reward_curve(exp_data.history_df, phase=phase)
    exp_rewards_srl = get_reward_curve(exp_data_srl.history_df, phase=phase)

    fig_path = out_path / f"reward_curve_{phase}.png" if out_path else None

    plot_title = "Agent's reward curve during "
    plot_title += "evaluation." if phase == 'eval' else "learning."
    plot_title += "\nIn a "
    plot_title += "pixel" if env_params['is_pixels'] else "vector"
    plot_title += "-based "
    plot_title += "random " if env_params['fire_pos'] is None else "fixed "
    plot_title += "fire position "
    if env_params['frame_stack'] > 1:
        plot_title += f"{env_params['frame_stack']} frames-stack "
    plot_title += "environment"

    plot_reward_curve(exp_rewards, exp_rewards_srl, plot_title,
                      ep_length=train_params['eval_interval'],
                      smooth_weight=smooth_val, fig_path=fig_path)


make_reward_curve('eval', smooth_val=0.99)
make_reward_curve('learn', smooth_val=0.99)

# %% Plot trajectories
ddqn_path = create_output_path(out_path, 'ddqn') if out_path else None
ddqn_srl_path = create_output_path(out_path, 'ddqn_srl') if out_path else None
for ep in range(1, 101):
    plot_title = f"DDQN trajectory EP {ep:03d}"
    fig_path = ddqn_path / f"DDQN_trajectory_ep_{ep:03d}.png" if ddqn_path else None
    plot_ep_trajectory(ep, exp_data, env_params, limit_area=True,
                       plot_title=plot_title, fig_path=fig_path)
    plot_title = f"DDQN-SRL trajectory EP {ep:03d}"
    fig_path = ddqn_srl_path / f"DDQN-SRL_trajectory_ep_{ep:03d}.png" if ddqn_srl_path else None
    plot_ep_trajectory(ep, exp_data_srl, env_params, limit_area=True,
                       plot_title=plot_title, fig_path=fig_path)
