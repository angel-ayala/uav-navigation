#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:38:44 2023

@author: Angel Ayala
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# from geokernels.distance import geodist
from webots_drone.data import ExperimentData
import threading


out_path = Path('/home/angel/desarrollo/uav_navigation/rl_experiments/logs_vector_new/assets')
# out_path = None
if out_path is not None and not out_path.exists():
    out_path.mkdir()


def append_list2dict(dict_elm, key, value):
    if key in dict_elm.keys():
        dict_elm[key].append(value)
    else:
        dict_elm[key] = [value]


# %% RL Data reading and preprocessing
base_path = Path('/home/angel/desarrollo/uav_navigation/rl_experiments/logs_vector_new')
exp_paths = list(base_path.iterdir())
exp_paths.sort(reverse=True)

phase = 'eval'

exp_summ = None
rewards = dict()
rewards_step = dict()
nav_metrics = dict()
epsilon = dict()
mem_beta = dict()
rewards_tpos = {}

threads = []


for exp_path in exp_paths:
    if 'random' in str(exp_path) or 'assets' in str(exp_path) or 'ddqn_2024-07-22_14-54-03' in str(exp_path):
        continue
    exp_data = ExperimentData(exp_path)
    exp_info = exp_data.get_info()
    if exp_summ is None:
        exp_summ = pd.DataFrame([exp_info])
    else:
        exp_summ = pd.concat((exp_summ, pd.DataFrame([exp_info])))

    dict_key = 'DDQN-SRL' if exp_data.agent_params['is_srl'] else 'DDQN'
    try:
        # append_list2dict(rewards, dict_key, exp_data.get_reward_curve(phase)[:, 0])
        x = threading.Thread(target=append_list2dict,
                             args=(rewards, dict_key, exp_data.get_reward_curve(phase)[:, 0]))
        threads.append(x)
        x.start()
        # append_list2dict(rewards_step, dict_key, exp_data.get_reward_curve(phase)[:, 1])
        # append_list2dict(epsilon, dict_key, exp_data.get_epsilon_curve())
        x = threading.Thread(target=append_list2dict,
                             args=(epsilon, dict_key, exp_data.get_epsilon_curve()))
        threads.append(x)
        x.start()
        # append_list2dict(mem_beta, dict_key, exp_data.get_mem_beta_curve())
        ep_metrics = pd.DataFrame(exp_data.get_nav_metrics(phase))
        # append_list2dict(nav_metrics, dict_key, ep_metrics)
        x = threading.Thread(target=append_list2dict,
                             args=(nav_metrics, dict_key, ep_metrics))
        threads.append(x)
        x.start()
        # append_list2dict(rewards_tpos, dict_key, 
        #     exp_data.get_reward_curve(phase, by_quadrant=True)[:, :, 0])
        x = threading.Thread(target=append_list2dict,
                             args=(rewards_tpos, dict_key, exp_data.get_reward_curve(phase, by_quadrant=True)[:, :, 0]))
        threads.append(x)
        x.start()

        for thrd in threads:
            thrd.join()
        # append_list2dict(rewards_tpos, dict_key, exp_data.get_reward_by_quadrant(phase)[:, 1])
    except AssertionError:
        print(exp_path, "does not have", phase, "data")
        continue


# %% PID Data reading and preprocessing
# pid_path = Path('/home/angel/desarrollo/uav_navigation/pid_experiments/logs/'
#                 'PID_2024-07-08_22-36-27')
# pid_data = ExperimentData(pid_path, csv_name='history.csv')

# pid_rewards = pid_data.get_reward_curve('eval')
# append_list2dict(rewards, 'PID', pid_rewards.repeat(len(nav_metrics[dict_key][-1])))

# pid_metrics = pd.DataFrame(pid_data.get_nav_metrics('eval'))
# pid_metrics = pid_metrics.loc[
#     pid_metrics.index.repeat(len(nav_metrics[dict_key][-1]))
#     ].reset_index(drop=True)
# append_list2dict(nav_metrics, 'PID', pid_metrics)


# %% Ensure save data
# out_name = 'pixels_assets' if env_params['is_pixels'] else 'vector_assets'
# out_path = create_output_path(exp_path.parent, out_name + 'first')

# %% Overall results

print(exp_summ)


# %% Navigation metrics
plots = [  # metric_id, y_label, plt_title, is_percent
    ('SR', 'SR', 'Success rate comparison', True),
    ('SPL', 'SPL', 'Success path length comparison', True),
    ('SSPL', 'SSPL', 'Soft success path length comparison', True),
    ('DTS', 'DTS (meters)', 'Distance to success comparison', False)
]


def plot_nav_metrics(metrics_data, metric_id, y_label, plt_title, is_percent,
                     fig_path=None):
    metrics_keys = list(metrics_data.keys())
    # metrics_keys.sort(reverse=True)
    # plot data
    for alg in metrics_keys:
        metrics = metrics_data[alg]
        metric_value = np.asarray([metric[metric_id].to_numpy()
                                   for metric in metrics])
        mean_values = metric_value.mean(axis=0)
        std_values = metric_value.std(axis=0)
        max_values = mean_values + std_values  # metric_value.max(axis=0)
        min_values = mean_values - std_values  # metric_value.min(axis=0)
        mean_lbl = mean_values.mean()
        if is_percent:
            mean_lbl *= 100
        v_label = f"{alg} (mean={mean_lbl:.2f}"
        if is_percent:
            v_label += "%)"
        else:
            v_label += ")"
        plt.plot(mean_values, label=v_label)
        plt.fill_between(range(min_values.shape[-1]),
                         min_values, max_values, alpha=0.3)

    # Plot settings
    plt.xlabel('Episodes')
    plt.ylabel(y_label)
    if is_percent:
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, _: '{:,.0f}%'.format(x * 100)))
        plt.ylim((-0.01, 1.01))
    plt.title(plt_title)
    plt.grid()
    # plt.legend()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), shadow=True,
               ncol=2)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(out_fig)
    plt.show()


for p_config in plots:
    out_fig = out_path / f"nav_metric_{p_config[0]}.pdf" if out_path is not None else out_path
    plot_nav_metrics(nav_metrics, *p_config, fig_path=out_fig)
    
# %% Plot reward
def plot_reward_curve(rewards_data, plot_title, fig_path=None):
    # Ensure new figure
    plt.figure()
    
    reward_keys = list(rewards_data.keys())
    # reward_keys.sort(reverse=True)
    # plot data
    for alg in reward_keys:
        rewards = np.asarray([arr for arr in rewards_data[alg] if arr.shape[-1] > 0])
        mean_values = rewards.mean(axis=0)
        std_values = rewards.std(axis=0)
        max_values = mean_values + std_values  # rewards.max(axis=0)
        min_values = mean_values - std_values  # rewards.min(axis=0)
        mean_lbl = mean_values.mean()
        v_label = f"{alg} (mean={mean_lbl:.2f})"
        plt.plot(range(1, len(mean_values) + 1), mean_values, label=v_label)
        plt.fill_between(range(1, min_values.shape[-1] + 1),
                         min_values, max_values, alpha=0.3)
    # Add a vertical line for min epsilon value
    eps_steps = 36
    plt.axvline(x=eps_steps, color='gray', linestyle='--')
    plt.annotate(r'Min. $\epsilon$', xy=(eps_steps, min_values.min()),
                 xycoords='data', xytext=(5, 5), textcoords='offset points',
                 arrowprops=dict(
                     arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    # Plot settings
    plt.xlabel('Episodes')
    plt.ylabel('Accumulative reward')
    plt.title(plot_title)
    plt.grid()
    plt.xlim((0.5, len(mean_values) + 0.5))
    # plt.legend(loc='lower right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), shadow=True,
               ncol=2)
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path)
    plt.show()

plot_reward_curve(rewards, f"Reward curve during {phase}",
                  fig_path=out_path / f"rewards_{phase}.pdf" if out_path is not None else out_path)
# plot_reward_curve(rewards_step, f"Step reward curve during {phase}",
#                   fig_path=out_path / f"rewards_step_{phase}.pdf" if out_path is not None else out_path)


# %% Plot Reward by target quadrant
keys = list(rewards_tpos.keys())
elms = len(rewards_tpos[keys[0]][0])
for i in range(elms):
    tq_reward = {}
    for k in keys:
        # Select quadrant related rewards
        tq_reward[k + f"_q{i}"] = np.asarray(rewards_tpos[k])[:, i]
        # tq_reward[k + f"_q{i}"] = np.asarray([arr for arr in rewards_tpos[k] if arr.shape[-1] > 0])
    plot_reward_curve(tq_reward, f"Step reward curve during {phase} for TargetQuadrant{i}",
                      fig_path=out_path / f"rewards_step_{phase}_{i}.pdf" if out_path is not None else out_path)

# %% Using rliable
# Aggregate metrics with 95% Stratified Bootstrap CIs
# IQM, Optimality Gap, Median, Mean
# https://github.com/google-research/rliable

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


algorithms = ['DDQN', 'DDQN-SRL']
# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices, each of which is of size `(num_runs x num_games)`.
# ddqn_r = exp_data.get_reward_curve('learn')
# ddqn_srl_r = exp_data_srl.get_reward_curve('learn') #'eval')
# ddqn_r = np.asarray(rewards['DDQN'])
# ddqn_srl_r = np.asarray(rewards['DDQN-SRL'])

# max_val = np.min((ddqn_r.max(), ddqn_srl_r.max()))
# max_val = pid_data.get_reward_curve()[0] / 4

ddqn_r = np.asarray(rewards_tpos['DDQN']).sum(axis=-1)
ddqn_srl_r = np.asarray(rewards_tpos['DDQN-SRL']).sum(axis=-1)

max_val = ddqn_r.max()
ddqn_r /= max_val
ddqn_srl_r /= max_val

atari_200m_normalized_score_dict = {'DDQN': ddqn_r,
                                    'DDQN-SRL': ddqn_srl_r}
aggregate_func = lambda x: np.array([
   metrics.aggregate_median(x),
  metrics.aggregate_iqm(x),
   metrics.aggregate_mean(x),
  metrics.aggregate_optimality_gap(x)])
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  atari_200m_normalized_score_dict, aggregate_func, reps=50000)
fig, axes = plot_utils.plot_interval_estimates(
  aggregate_scores, aggregate_score_cis,
   metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
  # metric_names=['IQM', 'Optimality Gap'],
  algorithms=algorithms, xlabel='Max-DDQN Normalized Score',
  xlabel_y_coordinate=-1.,
   row_height=1.2)
  # row_height=2)
plt.tight_layout()
# plt.legend()
# out_fig = out_path / f"rliable_agg_metrics.pdf" if out_path is not None else out_path
out_fig = out_path / f"rliable_agg_metrics_less.pdf" if out_path is not None else out_path
if out_fig is not None:
    plt.savefig(out_fig)
plt.show()

#%% Probability of Improvement
# ddqn_r = exp_data.get_reward_curve('learn')
# ddqn_srl_r = exp_data_srl.get_reward_curve('learn')

# # max_val = np.max((ddqn_r.max(), ddqn_srl_r.max()))
# ddqn_r /= max_val
# ddqn_srl_r /= max_val

# procgen_algorithm_pairs = {'DDQN,DDQN-SRL': (ddqn_r[np.newaxis, ], ddqn_srl_r[np.newaxis, ])}
procgen_algorithm_pairs = {'DDQN-SRL,DDQN': (ddqn_srl_r, ddqn_r)}
average_probabilities, average_prob_cis = rly.get_interval_estimates(
  procgen_algorithm_pairs, metrics.probability_of_improvement, reps=2000)
plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
plt.tight_layout()
out_fig = out_path / f"rliable_prob_improvement.pdf" if out_path is not None else out_path
if out_fig is not None:
    plt.savefig(out_fig)
plt.show()

#%% Performance profile
import seaborn as sns
# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices, each of which is of size `(num_runs x num_games)`.
algorithms = ['DDQN', 'DDQN-SRL']
# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices, each of which is of size `(num_runs x num_games)`.
# ddqn_r = exp_data.get_reward_curve('learn')
# ddqn_srl_r = exp_data_srl.get_reward_curve('learn')
# max_val = np.max((ddqn_r.max(), ddqn_srl_r.max()))
# ddqn_r /= max_val
# ddqn_srl_r /= max_val

atari_200m_normalized_score_dict = {'DDQN': ddqn_r,
                                    'DDQN-SRL': ddqn_srl_r}

# Human normalized score thresholds
atari_200m_thresholds = np.linspace(0.0, 1.25, 100)
score_distributions, score_distributions_cis = rly.create_performance_profile(
    atari_200m_normalized_score_dict, atari_200m_thresholds)
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
plot_utils.plot_performance_profiles(
  score_distributions, atari_200m_thresholds,
  performance_profile_cis=score_distributions_cis,
  colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
  xlabel=r'Max-DDQN Normalized Score $(\tau)$',
  ax=ax)
plt.tight_layout()
plt.legend()
out_fig = out_path / f"rliable_performance_profile.pdf" if out_path is not None else out_path
if out_fig is not None:
    plt.savefig(out_fig)
plt.show()

# %% Sample efficiency curve

algorithms = ['DDQN', 'DDQN-SRL']
# Load ALE scores as a dictionary mapping algorithms to their human normalized
# score matrices across all 200 million frames, each of which is of size
# `(num_runs x num_games x 200)` where scores are recorded every million frame.

ddqn_r = np.asarray(rewards_tpos['DDQN'])
ddqn_srl_r = np.asarray(rewards_tpos['DDQN-SRL'])

# ddqn_srl_r = exp_data_srl.get_reward_by_quadrant()

# max_val = np.max((ddqn_r.max(), ddqn_srl_r.max()))
max_val = ddqn_r.max()
ddqn_r /= max_val
ddqn_srl_r /= max_val

ale_all_frames_scores_dict = {'DDQN': ddqn_r,
                              'DDQN-SRL': ddqn_srl_r}
# frames = np.array([1, 10, 25, 50, 75, 100, 125, 150, 175, 200]) - 1
# frames = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) - 1
# frames = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) - 1
frames = np.array([max(1, x) for x in range(0, 76, 5)]) - 1
# frames = np.array([1, 5]) -1 #, 10, 15, 20, 25, 30, 35, 40, 45, 50]) - 1
ale_frames_scores_dict = {algorithm: score[:, :, frames] for algorithm, score
                          in ale_all_frames_scores_dict.items()}
iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                               for frame in range(scores.shape[-1])])
iqm_scores, iqm_cis = rly.get_interval_estimates(
  ale_frames_scores_dict, iqm, reps=50000)
plot_utils.plot_sample_efficiency_curve(
    frames+1, iqm_scores, iqm_cis, algorithms=algorithms,
    xlabel='Number of Episodes',
    ylabel='IQM Max-DDQN Normalized Score')
plt.tight_layout()
plt.legend()
out_fig = out_path / f"rliable_sample_efficiency.pdf" if out_path is not None else out_path
if out_fig is not None:
    plt.savefig(out_fig)
plt.show()

