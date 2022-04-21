import os
from git import Actor
import numpy as np
import torch
import torch.nn as nn
import gym
from gym.wrappers import Monitor

def clip_inputs(observation, goal,cliprange ):
    obs, goal = np.clip(observation, -cliprange, cliprange), \
        np.clip(goal, -cliprange, cliprange)
    return obs, goal

def normalise_and_clip(obs, goal,obs_mean, goal_mean, obs_std, goal_std,cliprange):
    obs_norm = (obs-obs_mean) / obs_std
    goal_norm = (goal-goal_mean)/goal_std
    obs_norm = np.clip(obs_norm, -cliprange, cliprange)
    goal_norm = np.clip(goal_norm, -cliprange, cliprange)
    return obs_norm,goal_norm

def concat_inputs(obs, goal):
    inputs = np.concatenate([obs, goal])
    return torch.tensor(inputs, dtype=torch.float32)

def test_agent(args, env,agent):
    path = 'per_colab/actor'
    # env = Monitor(env, './video', force=True)
    obs_mean, obs_std, goal_mean, goal_std, state_dict = torch.load(path, map_location=lambda storage, loc: storage) 
    
    # create actor network and load state dict
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    success_rate = []
    for ep in range(args.eval_episodes):

        observation = env.reset()
        obs = observation['observation']
        goal = observation['desired_goal']
        ep_success = []
        for t in range(env.maxtimestamps):
            env.render()
            obs, goal = clip_inputs(obs, goal, args.clip_observation)
            obs_norm, goal_norm = normalise_and_clip(obs, goal, obs_mean, goal_mean, obs_std, goal_std, args.cliprange)
            inputs = concat_inputs(obs_norm, goal_norm)
            with torch.no_grad():
                pi = agent.actor(inputs)
            actions = pi.cpu().detach().numpy().squeeze()
            next_observation, reward, done, info = env.step(actions)
            obs = next_observation['observation']
            ep_success.append(info['is_success'])
        ep_mean_success = np.mean(ep_success)
        success_rate.append(ep_mean_success)
        print(f'epoch: {ep} -> succes rate: {ep_mean_success}')
    print(f'overall success rate: {np.mean(success_rate)}')





