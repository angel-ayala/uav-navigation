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


def draw_flight_area(ax, flight_area, is_3d=False):
    # Define the vertices of the cube
    vertices = [
        [flight_area[0][0], flight_area[0][1]],
        [flight_area[1][0], flight_area[0][1]],
        [flight_area[1][0], flight_area[1][1]],
        [flight_area[0][0], flight_area[1][1]]
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


def plot_fire_zone(area_axis, fire_pos, fire_dim, dist_margin):
    # add fire and risk zone
    risk_distance = compute_risk_distance(*fire_dim)
    goal_distance = risk_distance + dist_margin
    fire_spot = plt.Circle(fire_pos, fire_dim[1], color='r')
    risk_zone = plt.Circle(fire_pos, risk_distance, color='r', alpha=0.2)
    goal_thr_zone = plt.Circle(fire_pos, goal_distance, color='g', fill=False)
    area_axis.add_patch(fire_spot)
    area_axis.add_patch(risk_zone)
    area_axis.add_patch(goal_thr_zone)
    return area_axis


def plot_scene_elements(area_axis, dims=(100, 100)):
    tower = plt.Rectangle((-5.1, -7.7), 10.2, 15.4, color='b', alpha=0.3)
    road_width = dims[1] / 2
    road_lanes = plt.Rectangle((-5, -road_width), 10, road_width - 7.7, color='black', alpha=0.1)
    area_axis.add_patch(road_lanes)
    area_axis.add_patch(tower)
    return area_axis


def plot_trj_reward(captured_positions, rewards):
    # plt.figure(figsize=(8, 6))
    plt.scatter(captured_positions[:, 0], captured_positions[:, 1], c=rewards,
                cmap='viridis')
    plt.colorbar(label='Step reward')
    # plt.quiver(captured_positions[:, 0], captured_positions[:, 1],
    #            np.cos(captured_orientations), np.sin(captured_orientations),
    #            color='black')


def create_output_path(exp_folder, out_name, exist_ok=False):
    out_path = exp_folder / out_name
    out_path.mkdir(exist_ok=exist_ok)
    return out_path


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


def plot_reward_curve(exp_rewards, exp_rewards_srl, eps_steps, plot_title,
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
    plt.axvline(x=eps_steps, color='gray', linestyle='--')
    plt.annotate(r'Min. $\epsilon$', xy=(eps_steps, y_lims[0]+10),
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


def plot_trajectory(fig, exp_data, episode, iteration=0,
                    phase='eval', limit_area=True, is_3d=False):
    # Filter data
    trj_data = exp_data.get_ep_trajectories(episode, phase, iteration)[0]
    trj_steps = trj_data['steps']
    rewards = trj_data['rewards']
    states = trj_data['states']
    tquadrant = exp_data.target_pos2quadrant(trj_data['target_pos'])
    # Plot area and scene elements
    flight_area = exp_data.get_flight_area()
    area_size = np.asarray(exp_data.get_flight_area())
    area_size = [np.abs(area_size[:, 1]).sum(), np.abs(area_size[:, 0]).sum()]
    if is_3d:
        area_axis = fig.add_subplot(111, projection='3d')
    else:
        area_axis = fig.add_subplot(111)
    draw_flight_area(area_axis, flight_area, is_3d=is_3d)
    plot_fire_zone(area_axis, fire_pos=tuple(trj_data['target_pos'][:2]),
                   fire_dim=exp_data.env_params['fire_dim'],
                   dist_margin=exp_data.env_params['goal_threshold'])
    plot_scene_elements(area_axis, dims=area_size)
    # Plot trajectory
    area_axis.plot(states[:, 0], states[:, 1])
    # Plot rewards
    plot_trj_reward(states[:, :2], rewards)
    # Plot orientation
    n_dist = list(range(0, len(states), 10))
    plt.quiver(states[n_dist, 0], states[n_dist, 1],
               np.cos(states[n_dist, 12]), np.sin(states[n_dist, 12]),
               color='black')
    
    if limit_area:
        area_axis.set_xlim((flight_area[0][0] - 5, flight_area[1][0] + 5))
        area_axis.set_ylim((flight_area[0][1] - 5, flight_area[1][1] + 5))
    area_axis.set_xlabel(None)
    area_axis.set_ylabel(None)
    plot_title = f"Q{tquadrant} target: {rewards.sum():.2f} total reward in {trj_steps} steps"
    area_axis.set_title(plot_title, fontsize=16)
    area_axis.grid()
    return area_axis


def make_reward_curve(phase, smooth_val):
    exp_rewards = exp_data.get_reward_curve(phase=phase)[:, 0]
    exp_rewards_srl = exp_data_srl.get_reward_curve(phase=phase)[:, 0]
    eps_steps = exp_data.agent_params['epsilon_steps']

    fig_path = out_path / f"reward_curve_{phase}.png" if out_path else None

    plot_title = "Agents' reward curve during "
    plot_title += "evaluation." if phase == 'eval' else "learning."
    plot_title += "\nIn a "
    plot_title += "pixel" if exp_data.env_params['is_pixels'] else "vector"
    plot_title += "-based "
    plot_title += "random " if exp_data.env_params['fire_pos'] is None else "fixed "
    plot_title += "fire position "
    if exp_data.env_params['frame_stack'] > 1:
        plot_title += f"{exp_data.env_params['frame_stack']} frames-stack "
    plot_title += "environment"

    plot_reward_curve(exp_rewards, exp_rewards_srl, eps_steps, plot_title,
                      ep_length=exp_data.train_params['eval_interval'],
                      smooth_weight=smooth_val, fig_path=fig_path)


# %% Best agent data source
# TODO search best_agent
if __name__ == '__main__':
    # First results
    base_path = Path('/home/angel/desarrollo/uav_navigation/rl_experiments/')
    # vector-based DDQN
    # exp_path = base_path / 'logs_vector/ddqn_2024-07-20_21-41-58'
    exp_path = base_path / 'logs_vector_new/ddqn_2024-08-16_09-37-25'
    # exp_path = base_path / 'logs_vector_new/ddqn_2024-08-20_08-53-07'
    # exp_path_srl = base_path / 'logs_vector/ddqn-srl_2024-07-22_14-57-14'
    exp_path_srl = base_path / 'logs_vector_new/ddqn-srl_2024-08-20_08-55-15'
    # exp_path_pid = Path('/home/angel/desarrollo/uav_navigation/pid_experiments/logs/PID_2024-07-05_13-22-36')

    # %% Read data
    exp_data = ExperimentData(exp_path)
    exp_data_srl = ExperimentData(exp_path_srl)
    # exp_data_pid = ExperimentData(exp_path_pid, csv_name='history.csv')
    out_path = base_path / 'logs_vector_new/assets'

    # %% Ensure save data
    # out_path = create_output_path(exp_path.parent, 'assets')
    # exp_df = exp_data.history_df.copy()
    # phase_df = exp_df[exp_df['phase'] == 'eval']
    # phase_df['target_quadrant'].unique()
    # rewards = None
    # target_quadrant = phase_df.groupby(
    #     ['target_pos_x', 'target_pos_y', 'target_pos_z']).agg({'unique'})
    # for i, tpos in enumerate(target_quadrant.index.to_numpy()):
    #     df_query = (f"target_pos_x=={tpos[0]} &" +
    #                 f" target_pos_y=={tpos[1]} &" +
    #                 f" target_pos_z=={tpos[2]}")
    #     quadrant_df = phase_df.query(df_query)
    #     phase_df.loc[quadrant_df.index, "quadrant"] = i

    # %% Plot curves
    make_reward_curve('learn', smooth_val=0.99)
    make_reward_curve('eval', smooth_val=0.99)

    # %% Plot trajectories

    trj_path = create_output_path(out_path, 'trajectories', True) if out_path else None
    eps = [max(0, i - 1) for i in range(0, 76, 15)]
    phase = 'eval'

    for ep in eps:
        print('ep', ep)
        # Create a new figure and axis
        fig = plt.figure(layout='constrained', figsize=(8.5, 8))
        subfigs = fig.subfigures(1, 1, wspace=0.07)
        fig_path = trj_path / f"DDQN_trj_ep_{ep+1:03d}.pdf" if out_path else None

        # for i in range(4):
        plot_trajectory(subfigs, exp_data, ep, phase=phase,
                        iteration=[0, 1, 2, 3], limit_area=True)
        fig.suptitle(f"DDQN EP {ep+1}", fontsize=20)
        if fig_path:
            plt.savefig(fig_path)
        plt.show()

        # Create a new figure and axis
        fig = plt.figure(layout='constrained', figsize=(8.5, 8))
        subfigs = fig.subfigures(1, 1, wspace=0.07)
        fig_path = trj_path / f"DDQN-SRL_trj_ep_{ep+1:03d}.pdf" if out_path else None

        # for i in range(4):
        plot_trajectory(subfigs, exp_data_srl, ep, phase=phase,
                        iteration=[0, 1, 2, 3], limit_area=True)
        fig.suptitle(f"DDQN-SRL EP {ep+1}", fontsize=20)
        if fig_path:
            plt.savefig(fig_path)
        plt.show()


# %% custom plot
def plot_trajectory(fig, exp_data, episode, iteration=0,
                    phase='eval', limit_area=True, is_3d=False):
    # Plot area and scene elements
    flight_area = exp_data.get_flight_area()
    area_size = np.asarray(exp_data.get_flight_area())
    area_size = [np.abs(area_size[:, 1]).sum(), np.abs(area_size[:, 0]).sum()]
    if is_3d:
        area_axis = fig.add_subplot(111, projection='3d')
    else:
        area_axis = fig.add_subplot(111)
    draw_flight_area(area_axis, flight_area, is_3d=is_3d)
    plot_scene_elements(area_axis, dims=area_size)

    # Filter data
    fire_zones = []
    all_rewards = []
    for trj_data in exp_data.get_ep_trajectories(episode, phase, iteration):
        trj_steps = trj_data['steps']
        rewards = trj_data['rewards']
        all_rewards.append(sum(rewards))
        states = trj_data['states']
        tquadrant = exp_data.target_pos2quadrant(trj_data['target_pos'])
        if tquadrant not in fire_zones:
            plot_fire_zone(area_axis, fire_pos=tuple(trj_data['target_pos'][:2]),
                           fire_dim=exp_data.env_params['fire_dim'],
                           dist_margin=exp_data.env_params['goal_threshold'])
            fire_zones.append(tquadrant)
        # Plot trajectory
        area_axis.plot(states[:, 0], states[:, 1])
        # Plot rewards
        area_axis.scatter(states[:, 0], states[:, 1], c=rewards, cmap='viridis')
        # Plot orientation
        n_dist = list(range(0, len(states), 10))
        plt.quiver(states[n_dist, 0], states[n_dist, 1],
                   np.cos(states[n_dist, 12]), np.sin(states[n_dist, 12]),
                   color='black')

    if limit_area:
        area_axis.set_xlim((flight_area[0][0] - 5, flight_area[1][0] + 5))
        area_axis.set_ylim((flight_area[0][1] - 5, flight_area[1][1] + 5))
    area_axis.set_xlabel('Position Y (meters)', fontsize=18)
    area_axis.set_ylabel('Position X (meters)', fontsize=18)
    area_axis.tick_params(labelsize=16)
    plot_title = f"mean accumulative reward: {sum(all_rewards) / len(all_rewards):.2f}"
    # plot_title = "Path to each target location"
    area_axis.set_title(plot_title, fontsize=16)
    area_axis.grid()
    return area_axis
