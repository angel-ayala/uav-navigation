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

from webots_drone.utils import min_max_norm


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


def run_agent(agent, env, training_steps, mem_steps, train_frequency,
              target_update_steps, eval_interval, eval_epsilon, outpath,
              step_callback=None):
    ended = True
    total_reward = 0
    total_episodes = 0
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

    tbar = tqdm(range(eval_interval), desc='Episode 0', leave=False,
                unit='step', bar_format='{desc}{n:04d}|{bar}|[{rate_fmt}]')
    for step in range(training_steps):
        # after training steps, began evaluation
        if step % eval_interval == 0:
            elapsed_time = time.time() - timemark
            tbar.clear()
            if total_episodes == 0:
                print(f"Memory fill at {elapsed_time:.3f} seconds")
            else:
                print(f"Episode {total_episodes:03d}\n- Learning: {elapsed_time:.3f} seconds\tR: {ep_reward:.4f}\tS: {ep_steps}")
            agent.save(outpath / f"agent_ep_{total_episodes:03d}.pth")
            eval_reward, eval_steps, eval_time = evaluate_agent(
                agent, env, eval_epsilon, step_callback)
            total_episodes += 1
            total_reward += ep_reward
            tbar.reset()
            tbar.set_description(f"Episode {total_episodes:03d}")
            ended = True
            if step_callback:
                step_callback.new_episode()
            timemark = time.time()

        if ended:
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
            agent.update_target_network()

        ep_reward += reward
        state = next_state
        ep_steps += 1

        agent.update_epsilon(step)
        tbar.update(1)

    return total_reward, total_episodes


def evaluate_agent(agent, env, eval_epsilon, step_callback=None):
    state, info = env.reset()
    ep_reward = 0
    ep_steps = 0
    end = False
    curr_epsilon = agent.epsilon
    agent.epsilon = eval_epsilon

    if step_callback:
        step_callback.set_init_state(state, info)
        step_callback.set_eval()

    timemark = time.time()
    while not end:
        action, reward, next_state, end = do_step(
            agent, env, state, step_callback, must_remember=False)
        state = next_state
        ep_steps += 1
        ep_reward += reward
        sys.stdout.write(f"\rR: {ep_reward:.4f}\tS: {ep_steps}")
        sys.stdout.flush()

    elapsed_time = time.time() - timemark
    sys.stdout.flush()
    sys.stdout.write(f"\r- Evaluation: {elapsed_time:.3f} seconds\tR: {ep_reward:.4f}\tS: {ep_steps}\n")
    sys.stdout.flush()
    agent.epsilon = curr_epsilon
    return ep_reward, ep_steps, elapsed_time


class PreprocessObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, is_pixels=True):
        super().__init__(env)
        if is_pixels:
            self.preprocess_fn = self.preprocess_pixels
        else:
            self.preprocess_fn = self.preprocess_vector

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        return self.preprocess_fn(obs), rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        return self.preprocess_fn(obs), info

    def preprocess_pixels(self, obs):
        return obs.astype(np.float32) / 255.

    def preprocess_vector(self, obs):
        # Normalize angular values
        obs[3] = min_max_norm(obs[3], a=-1, b=1, minx=-np.pi, maxx=np.pi)
        obs[4] = min_max_norm(obs[4], a=-1, b=1, minx=-np.pi/2, maxx=np.pi/2)
        obs[5] = min_max_norm(obs[5], a=-1, b=1, minx=-np.pi, maxx=np.pi)
        return obs
