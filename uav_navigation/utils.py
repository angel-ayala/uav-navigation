#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:54:52 2023

@author: Angel Ayala
"""
import json
import numpy as np
import time
import sys
import torch
from thop import profile
from tqdm import tqdm

from .logger import summary
from .logger import summary_create
from .logger import summary_step
from .logger import summary_scalar


def profile_model(model, input_shape, device, action_shape=None):
    """Profiling developed models.

    based on https://github.com/angel-ayala/kutralnet/blob/master/utils/profiling.py"""
    x = torch.randn(input_shape).unsqueeze(0).to(device)
    if action_shape:
        y = torch.randn(action_shape).unsqueeze(0).to(device)
        flops, params = profile(model, verbose=False,
                                inputs=(x, y),)
    else:
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


def do_step(agent, env, state, callback=None, must_remember=True, random_step=False):
    # Choose action using the agent's policy
    if random_step:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)

    # Take the chosen action
    next_state, reward, done, trunc, info = env.step(action)
    if callback:
        callback((state, action, reward, next_state, done, trunc), info)

    # Update the agent based on the observed transition
    if must_remember:
        # Store the transition in the replay buffer if must
        agent.memory.add(state, action, reward, next_state, done)
    ended = done or trunc

    return action, reward, next_state, ended


def obs2tensor(observations):
    if torch.is_tensor(observations):
        return observations
    else:
        return torch.tensor(np.array(observations), dtype=torch.float32)


def format_obs(observation, is_pixels=False):
    observation = obs2tensor(observation)

    if len(observation.shape) == 3 and is_pixels:
        observation = observation.unsqueeze(0)

    return observation


def run_agent(agent, env, training_steps, mem_steps, eval_interval,
              eval_steps, eval_epsilon, outpath, step_callback=None):
    summary_create(outpath.parent)
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
                agent, env, state, step_callback, must_remember=True,
                random_step=True)
            state = next_state
        elapsed_time = time.time() - timemark
        membar.clear()
        del membar
        print(f"Memory fill at {elapsed_time:.3f} seconds")
        ended = True

    tbar = tqdm(range(eval_interval), desc=f"Episode {total_episodes:03d}",
                leave=False, unit='step',
                bar_format='{desc}: {n:04d}|{bar}|[{rate_fmt}]')

    agent.learn_mode()
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

        agent.update(step)

        ep_reward += reward
        state = next_state
        ep_steps += 1
        summary().add_scalar('Learning/StepReward', reward, step)

        tbar.update(1)
        # after training steps, began evaluation
        if (step + 1) % eval_interval == 0:
            elapsed_time = time.time() - timemark
            tbar.clear()
            print(f"Episode {total_episodes:03d}\n- Learning: {elapsed_time:.3f} seconds\tR: {ep_reward:.4f}\tS: {ep_steps}")
            agent.save(outpath / f"agent_ep_{total_episodes:03d}")
            if eval_steps > 0:
                summary_step(total_episodes)
                agent.eval_mode()
                for tq in range(len(env.quadrants)):
                    evaluate_agent(agent, env, eval_steps, target_quadrant=tq,
                                   step_callback=step_callback)
                summary().flush()
                agent.learn_mode()
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


def evaluate_agent(agent, env, eval_steps, target_quadrant=2, step_callback=None):
    timemark = time.time()
    state, info = env.reset(target_pos=target_quadrant)
    ep_reward = 0
    ep_steps = 0
    end = False

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
    sys.stdout.write(f"\r- Evaluation: {elapsed_time:.3f} seconds\t"
                     f"R: {ep_reward:.4f}\tS: {ep_steps}\n")
    sys.stdout.flush()

    if isinstance(target_quadrant, (int, np.integer)):
        summary_scalar(f"Evaluation/EpRewardQ{target_quadrant:02d}", ep_reward)
        summary_scalar(f"Evaluation/EpNumberStepsQ{target_quadrant:02d}", ep_steps)
    else:
        summary_scalar("Evaluation/EpRewardRandomPos", ep_reward)
        summary_scalar("Evaluation/EpNumberStepsRandomPos", ep_steps)

    return ep_reward, ep_steps, elapsed_time


def destack(obs_stack, len_hist=3, is_rgb=False):
    orig_shape = obs_stack.shape
    if is_rgb:
        n_stack = (orig_shape[1] // len_hist) * orig_shape[0]
        obs_destack = obs_stack.reshape((n_stack, 3) + orig_shape[-2:])
    else:
        obs_destack = obs_stack.reshape(
            (orig_shape[0] * len_hist, orig_shape[-1]))
    return obs_destack, orig_shape
