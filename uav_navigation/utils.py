#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:54:52 2023

@author: Angel Ayala
"""
import json
import gym
import copy
import numpy as np
from tqdm import tqdm


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


def do_step(agent, env, state, callback=None, must_update=False):
    # Choose action using the agent's policy
    action = agent.select_action(state)

    # Take the chosen action
    next_state, reward, done, trunc, info = env.step(action)
    if callback:
        callback((state, action, reward, next_state, done, done), info)

    # Update the agent based on the observed transition
    ended = done or trunc
    if must_update:
        agent.update(state, action, reward, next_state, ended)

    return action, reward, next_state, ended


def eval_agent(agent, env, eval_epsilon, step_callback=None):
    state, info = env.reset()
    ep_reward = 0
    ep_steps = 0
    end = False
    curr_epsilon = copy.copy(agent.epsilon)
    agent.epsilon = copy.copy(eval_epsilon)

    if step_callback:
        step_callback.set_init_state(state, info)
        step_callback.set_eval()

    while not end:
        action, reward, next_state, end = do_step(
            agent, env, state, step_callback, must_update=False)
        state = next_state
        ep_steps += 1
        ep_reward += reward
    print("Evaluation ends:",
          f"\tlength: {ep_steps} steps\tr: {ep_reward}\te: {agent.epsilon}")
    agent.epsilon = copy.copy(curr_epsilon)
    return ep_reward, ep_steps


def fill_memory(agent, env, num_steps, step_callback=None):
    state, info = env.reset()
    curr_epsilon = copy.copy(agent.epsilon)
    if step_callback:
        step_callback.set_init_state(state, info)
    print('Initializing memory...')
    for step in tqdm(range(num_steps)):
        action, reward, next_state, ended = do_step(
            agent, env, state, step_callback, must_update=False)
        agent.memory.add(state, action, reward, next_state, ended)
        state = next_state
        if ended:
            state, info = env.reset()
            if step_callback:
                step_callback.set_init_state(state, info)
    agent.epsilon = copy.copy(curr_epsilon)


def train_agent(agent, env, num_steps, step_callback=None):
    state, info = env.reset()
    total_reward = 0
    total_episodes = 0
    ep_reward = 0
    ep_steps = 0
    if step_callback:
        step_callback.set_init_state(state, info)
        step_callback.set_learning()

    for step in range(num_steps):
        action, reward, next_state, ended = do_step(
            agent, env, state, step_callback, must_update=True)

        ep_reward += reward
        state = next_state
        ep_steps += 1

        if ended:
            total_episodes += 1
            total_reward += ep_reward
            print(
                f"Iterarion {total_episodes},\ts:{ep_steps}\tr: {ep_reward}",
                'e:', agent.epsilon)
            ep_reward = 0
            # ep_steps = 0
            state, info = env.reset()
            if step_callback:
                step_callback.set_init_state(state, info)

    return total_reward, total_episodes


def train_eval_agent(agent, env, training_steps, save_steps, mem_steps,
                     eval_epsilon, outpath, step_callback=None):
    max_ep_steps = training_steps // save_steps
    fill_memory(agent, env, mem_steps, step_callback)
    for e in range(max_ep_steps):
        print('Starting ep', e)
        accum_reward, n_episodes = train_agent(agent, env, save_steps,
                                               step_callback)
        agent.save(outpath / f"agent_ep_{e:03d}.pth")
        eval_reward, eval_steps = eval_agent(agent, env, eval_epsilon,
                                             step_callback)


class PreprocessObservation(gym.core.Wrapper):
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
        print(obs)
        return obs
