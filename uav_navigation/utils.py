#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:54:52 2023

@author: Angel Ayala
"""
import json
import gym
import numpy as np
import time
import sys
import torch
from thop import profile
from tqdm import tqdm

from .logger import summary
from .logger import summary_create
from .logger import summary_step


def profile_model(model, input_shape, device):
    """Profiling developed models.

    based on https://github.com/angel-ayala/kutralnet/blob/master/utils/profiling.py"""
    x = torch.randn(input_shape).unsqueeze(0).to(device)
    flops, params = profile(model, verbose=False,
                            inputs=(x, ),)
    return flops, params


def save_dict_json(dict2save, json_path):
    proc_dic = dict2save.copy()
    dict_json = json.dumps(proc_dic,
                           indent=4,
                           default=lambda o: str(o))
    with open(json_path, 'w') as jfile:
        jfile.write(dict_json)
    return dict_json


def load_json_dict(json_path):
    json_dict = dict()
    with open(json_path, 'r') as jfile:
        json_dict = json.load(jfile)
    return json_dict


def soft_update_params(net, target_net, tau):
    # Soft update: target_network = tau * network + (1 - tau) * target_network
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def do_step(agent, env, state, callback=None, must_remember=True):
    # Choose action using the agent's policy
    action = agent.select_action(state)

    # Take the chosen action
    next_state, reward, done, trunc, info = env.step(action)
    if callback:
        callback((state, action, reward, next_state, done, done), info)

    # Update the agent based on the observed transition
    ended = done or trunc
    if must_remember:
        # Store the transition in the replay buffer if must
        agent.memory.add(state, action, reward, next_state, ended)

    return action, reward, next_state, ended


def obs2tensor(observations):
    if type(observations) is not torch.Tensor:
        obs_tensor = torch.tensor(np.array(observations),
                                  dtype=torch.float32).unsqueeze(0)
    else:
        obs_tensor = observations
    return obs_tensor


def run_agent(agent, env, training_steps, mem_steps, train_frequency,
              target_update_steps, eval_interval, eval_epsilon, eval_steps,
              outpath, step_callback=None):
    summary_create(outpath / 'logs')
    ended = True
    total_reward = 0
    total_episodes = 1
    total_iterations = 0
    ep_reward = 0
    ep_steps = 0
    timemark = time.time()

    if mem_steps:
        membar = tqdm(range(mem_steps), desc='Memory init', leave=False)
        for step in membar:
            if ended:
                state, info = env.reset()
                if step_callback:
                    step_callback.set_init_state(state, info)
            action, reward, next_state, ended = do_step(
                agent, env, state, step_callback, must_remember=True)
            state = next_state
        elapsed_time = time.time() - timemark
        membar.clear()
        del membar
        print(f"Memory fill at {elapsed_time:.3f} seconds")
        ended = True

    tbar = tqdm(range(eval_interval), desc=f"Episode {total_episodes:03d}",
                leave=False, unit='step',
                bar_format='{desc}: {n:04d}|{bar}|[{rate_fmt}]')
    for step in range(training_steps):
        summary_step(step)
        if ended:
            total_iterations += 1
            ep_reward = 0
            ep_steps = 0
            state, info = env.reset()
            if step_callback:
                step_callback.set_init_state(state, info)
                step_callback.set_learning()

        action, reward, next_state, ended = do_step(
            agent, env, state, step_callback, must_remember=True)

        if step % train_frequency == 0:
            agent.update()

        if step % target_update_steps == 0:
            agent.update_target()

        ep_reward += reward
        state = next_state
        ep_steps += 1
        summary().add_scalar('Learning/StepReward', reward, step)

        agent.update_epsilon(step)
        tbar.update(1)
        # after training steps, began evaluation
        if (step + 1) % eval_interval == 0:
            elapsed_time = time.time() - timemark
            tbar.clear()
            print(f"Episode {total_episodes:03d}\n- Learning: {elapsed_time:.3f} seconds\tR: {ep_reward:.4f}\tS: {ep_steps}")
            agent.save(outpath / f"agent_ep_{total_episodes:03d}.pth")
            if eval_steps > 0:
                for fc in range(4):
                    e_reward, e_steps, e_time = evaluate_agent(
                        agent, env, eval_epsilon, eval_steps, fire_cuadrant=fc,
                        step_callback=step_callback)
                    summary().add_scalar(f"Evaluation/EpRewardC{fc}", e_reward, total_episodes)
                    summary().add_scalar(f"Evaluation/EpNumberStepsC{fc}", e_steps, total_episodes)
            total_episodes += 1
            tbar.reset()
            tbar.set_description(f"Episode {total_episodes:03d}")
            ended = True
            if step_callback:
                step_callback.new_episode()
            timemark = time.time()

        if ended:
            total_reward += ep_reward
            summary().add_scalar('Learning/EpReward', ep_reward, total_iterations)
            summary().add_scalar('Learning/EpNumberSteps', ep_steps, total_iterations)
            summary().flush()

    summary().close()
    return total_reward, total_episodes


def evaluate_agent(agent, env, eval_epsilon, eval_steps, fire_cuadrant=2, step_callback=None):
    timemark = time.time()
    state, info = env.reset(fire_cuadrant=fire_cuadrant)
    ep_reward = 0
    ep_steps = 0
    end = False
    curr_epsilon = agent.epsilon
    agent.epsilon = eval_epsilon

    if step_callback:
        step_callback.set_init_state(state, info)
        step_callback.set_eval()

    while not end:
        action, reward, next_state, end = do_step(
            agent, env, state, step_callback, must_remember=False)
        state = next_state
        ep_steps += 1
        ep_reward += reward
        sys.stdout.write(f"\rR: {ep_reward:.4f}\tS: {ep_steps}")
        sys.stdout.flush()
        if ep_steps == eval_steps:
            end = True

    elapsed_time = time.time() - timemark
    sys.stdout.flush()
    sys.stdout.write(f"\r- Evaluation: {elapsed_time:.3f} seconds\tR: {ep_reward:.4f}\tS: {ep_steps}\n")
    sys.stdout.flush()
    agent.epsilon = curr_epsilon
    return ep_reward, ep_steps, elapsed_time


class ReducedVectorObservation(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_shape = (13, )
        self.obs_type = np.float32
        self.observation_space = gym.spaces.Box(low=float('-inf'),
                                                high=float('inf'),
                                                shape=self.obs_shape,
                                                dtype=self.obs_type)
    def observation(self, obs):
        return obs[:13]

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        return self.observation(obs), rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        return self.observation(obs), info
