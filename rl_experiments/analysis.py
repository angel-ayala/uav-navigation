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
from webots_drone.data import EpisodeData
from webots_drone.reward import compute_distance_reward
from webots_drone.reward import compute_orientation_reward
from webots_drone.reward import sum_and_normalize
from webots_drone.utils import min_max_norm

import re


def plot_area(axis_intervals):
    # Create a new figure and axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)  # , projection='3d')

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
        [vertices[3], vertices[0]],
        # [vertices[4], vertices[5]],
        # [vertices[5], vertices[6]],
        # [vertices[6], vertices[7]],
        # [vertices[7], vertices[4]],
        # [vertices[0], vertices[4]],
        # [vertices[1], vertices[5]],
        # [vertices[2], vertices[6]],
        # [vertices[3], vertices[7]]
    ]

    # Plot the edges
    for edge in edges:
        ax.plot(*zip(*edge), color='b', alpha=0.5)

    # Set labels for each axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # Show the plot
    # plt.show()
    return ax

def get_flight_area():
    import webots_drone
    package_path = webots_drone.__path__
    world_path = Path(package_path +'/worlds/forest_fire.wbt')

    with open(world_path, 'r') as world_dump:
        world_file = world_dump.read()

    area_size_regex = re.search(
        "DEF FlightArea(.*\n)+  size( \-?\d+\.?\d?){2}\n",
        world_file)
    area_size_str = area_size_regex.group().strip()
    area_size = list(map(float, area_size_str.split(' ')[-2:]))
    # print('area_size', area_size)

    area_size = [fs / 2 for fs in area_size]  # size from center
    flight_area = [[fs * -1 for fs in area_size], area_size]

    return flight_area
    
def plot_reward_heatmap(captured_positions, captured_orientations, rewards):
    # plt.figure(figsize=(8, 6))
    plt.scatter(captured_positions[:, 0], captured_positions[:, 1], c=rewards, cmap='viridis')
    plt.colorbar(label='Reward')
    plt.quiver(captured_positions[:, 0], captured_positions[:, 1], 
               np.cos(captured_orientations), np.sin(captured_orientations), 
               angles='xy', scale_units='xy', scale=1, color='black')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title('Reward Heatmap considering Captured Position (X, Y) and Orientation')
    plt.show()
    
def plot_orientation(positions, headings_rad):
    # Extract x and y coordinates from the position data
    x = positions[:, 0]
    y = positions[:, 1]

    # Convert headings to radians
    # headings_rad = np.deg2rad(headings)

    # Calculate vector components
    dx = np.cos(headings_rad)
    dy = np.sin(headings_rad)

    plt.figure(figsize=(8, 6))
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)  # Plotting the orientation vectors
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Orientation Vectors')
    plt.grid()
    plt.show()
    
def compute_total_reward(uav_xy, target_xy,uav_ori):
    # uav_xy = info['position'][:2]
    # target_xy = self.sim.get_target_pos()[:2]
    # # orientation values from [-1, 1] to [0, 2 * pi]
    # uav_ori = (info['north_deg'] + 1.) * np.pi
    # compute reward components
    orientation_reward = compute_orientation_reward(uav_xy, uav_ori,
                                                    target_xy)
    distance_reward = compute_distance_reward(
        uav_xy, target_xy)
    reward = sum_and_normalize(orientation_reward, distance_reward)
    return reward

# %% Read Data
experiment_path = Path('drone_vector/logs/test_2023-11-25_18-49-04/history.csv')
ep_data = EpisodeData(experiment_path)

trajectory_data = ep_data.get_ep_trajectories(2, [0])
trajectory = trajectory_data[0]
trj_length, init_pos, trj_target = trajectory[:3]
timemarks, rewards, orientations = trajectory[3:6]
states, actions, next_states = trajectory[6:]

print(states.shape, actions.shape, next_states.shape, orientations.shape, rewards.shape)

# %% Plot flight area and others scene elements
flight_area = get_flight_area()
# add altitude limits
flight_area[0].append(11.)
flight_area[1].append(75.)

area_axis = plot_area(flight_area)
area_axis.plot(*trj_target[:2], 'ro')

# Plot trajectory
area_axis.plot(states[:, 0], states[:, 1],
               label=f"reward={rewards.sum():.2f}")
area_axis.legend(loc="lower left")
plt.title(f"EP {trj_length} steps")


# Plot rewards
# plot_reward_heatmap_custom(x_coordinates, y_coordinates, rewards)
plot_reward_heatmap(states[:, :2], orientations, rewards)
plt.tight_layout()
# plt.show()

#%% Plot computed rewards
comp_rewards = np.zeros_like(rewards)
diff_rewards = np.zeros_like(rewards)

for i, R in enumerate(rewards):
    rtotal = compute_total_reward(states[i, :2], trj_target[:2], orientations[i])
    comp_rewards[i] = rtotal
    diff_rewards[i] = rtotal - R

plot_reward_heatmap(states[:, :2], orientations, comp_rewards)

# %% PLot
plot_orientation( states[:, :2], orientations)

# %% Plot splines
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Create a cubic spline interpolation
# for action_offset in [2]:  # range(2):
# states_filt = states[action_offset*10:10 + action_offset*10]
states_filt = states.copy()
npoints = states_filt[:, 0].shape[0]
csx = CubicSpline(range(npoints), states_filt[:, 0])
csy = CubicSpline(range(npoints), states_filt[:, 1])

# Generate more points for plotting the spline curve
offset_points = 5
interp = np.linspace(0, npoints + offset_points, 100)
x_interp = csx(interp)
y_interp = csy(interp)

# Plot the original points and the spline curve
plt.figure(figsize=(5, 3))
plt.scatter(states_filt[:, 0], states_filt[:, 1], label='Original Points')
plt.plot(x_interp[-offset_points*2:], y_interp[-offset_points*2:], label='Cubic Spline', color='red')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title(f"Cubic Spline Interpolation")
plt.legend()
plt.grid(True)
plt.show()
    
# %%

def angular_to_body_velocities(phi, theta, psi, p, q, r):
    # Calculate derivatives of orientation angles in the body frame
    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * (np.sin(phi) / np.cos(theta)) + r * (np.cos(phi) / np.cos(theta))
    
    # Calculate body frame velocities using derivatives of orientation angles
    dot_x_b = d_phi + np.sin(phi) * np.tan(theta) * d_theta + np.cos(phi) * np.tan(theta) * d_psi
    dot_y_b = np.cos(phi) * d_theta - np.sin(phi) * d_psi
    dot_z_b = np.sin(phi) / np.cos(theta) * d_theta + np.cos(phi) / np.cos(theta) * d_psi
    
    return dot_x_b, dot_y_b, dot_z_b

def body_to_global_velocities(phi, theta, psi, dot_x_b, dot_y_b, dot_z_b):
    # Conversion equations
    dot_x = (dot_x_b * np.cos(theta) * np.cos(psi) + 
             dot_y_b * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) + 
             dot_z_b * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)))
    
    dot_y = (dot_x_b * np.cos(theta) * np.sin(psi) + 
             dot_y_b * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) + 
             dot_z_b * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)))
    
    dot_z = (-dot_x_b * np.sin(theta) + 
             dot_y_b * np.sin(phi) * np.cos(theta) + 
             dot_z_b * np.cos(phi) * np.cos(theta))
    
    return dot_x, dot_y, dot_z

x, y, z = states[:, 0], states[:, 1], states[:, 2]
phi, theta, psi = states[:, 3], states[:, 4], states[:, 5]
dot_x, dot_y, dot_z = states[:, 6], states[:, 7], states[:, 8]
p, q, r = states[:, 9], states[:, 10], states[:, 11]


dot_x_b, dot_y_b, dot_z_b = angular_to_body_velocities(phi, theta, psi, p, q, r)
hat_dot_x, hat_dot_y, hat_dot_z = body_to_global_velocities(phi, theta, psi, dot_x_b, dot_y_b, dot_z_b)

# %% 

# Function to update position based on angular velocities
def update_position(position, orientation, angular_velocities, dt):
    # Simulate integration by Euler's method
    orientation += angular_velocities * dt  # Update orientation
    # Update position based on orientation (here, assuming constant velocities)
    position += np.sin(orientation) * dt  # Just an example calculation
    
    return position, orientation

state_t = states[1]
state_t1 = states[2]
position = state_t[:3]
orientation = state_t[3:6]
angular_vel = state_t[-3:]
dt = 1
print(state_t)
print('pos', position)
print('pos', orientation)
new_pos, new_ori = update_position(position, orientation, angular_vel, dt)
print('new_pos', new_pos)
print('new_ori', new_ori)
print(state_t1)
