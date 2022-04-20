import os
import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F
import gym
from herwithddpg import HERDDPG
import time
from mpi4py import MPI
import datetime
import matplotlib.pyplot as plt

def train_agent(args, env, agent=HERDDPG):
    success_rate = []
    #np.zeros(args.episodes)
    
    eval_every = 10
    
    std_success_rate = []
    #np.zeros(args.episodes)
    for epoch in range(args.episodes):
        for cycle in range(args.cycles):
            cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions = [],[],[],[] # collects statistics for each cycle
            for _ in range(args.rollouts):
                ep_obs, ep_achieved_goal, ep_goal, ep_actions = [],[],[],[] # collects statistics for each rollout within a cycle
                episode_reward = 0
                observation = env.reset()

                obs = observation['observation']
                achieved_goal = observation['achieved_goal']
                goal = observation['desired_goal']
                # collect samples for 'env.maxtimestamps' timestamps
                for t in range(env.maxtimestamps):
                    with torch.no_grad():
                        # obs_norm = agent.obs_norm.normalize(obs)
                        # goal_norm = agent.goal_norm.normalize(goal)
                        # input_tensor = agent.concat_inputs(obs_norm, goal_norm)
                        # print(f'input_tensor: {input_tensor}')
                        input_tensor = agent._preproc_inputs(obs, goal)
                        pi = agent.actor(input_tensor)
                        #action = agent.actor(input_tensor)
                        action = agent.choose_action_wnoise(pi,args.noise_prob,args.random_prob,1)
                    # print(f'action: {action}')
                    # collect statistics about next observation, achieved goals and desired goals..
                    next_observation, reward, done, _ = env.step(action)
                    obs_next = next_observation['observation']
                    achieved_goal_next = next_observation['achieved_goal']

                    ep_obs.append(obs.copy())
                    ep_achieved_goal.append(achieved_goal.copy())
                    ep_goal.append(goal.copy())
                    ep_actions.append(action.copy())

                    obs = obs_next
                    achieved_goal = achieved_goal_next
                
                ep_obs.append(obs.copy())
                ep_achieved_goal.append(achieved_goal.copy())
                cycle_obs.append(ep_obs)
                cycle_achieved_goal.append(ep_achieved_goal)
                cycle_actions.append(ep_actions)
                cycle_goal.append(ep_goal)
            cycle_obs = np.array(cycle_obs)
            cycle_achieved_goal = np.array(cycle_achieved_goal)
            cycle_goal = np.array(cycle_goal)
            cycle_actions = np.array(cycle_actions)
            episode_reward += agent.rewards.mean()
            agent.remember([cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions])
            # apply HER
            # # store new goals from her buffer into replay buffer
            agent.normalise_her_samples([cycle_obs, cycle_achieved_goal,cycle_goal, cycle_actions])
            
            #perform ddpg optimisation for args.optimsteps=40 steps
            for _ in range(args.optimsteps):
                # perform ddpg optimization...
                agent.learn()
            agent.update_network_params()      
            print(f'cycle: {cycle}, epoch: {epoch}')
        print(f' epoch: {epoch} -> reward: {episode_reward}')

        
        # if MPI.COMM_WORLD.Get_rank() == 0:
            
        if epoch%10==0:
            success_rate_epoch, std_success_rate_epoch = _eval_agent(env,args, agent)
            print(f'epoch: {epoch} -> success rate: {success_rate_epoch}')
            print(f'epoch: {epoch} -> standard dev: {std_success_rate_epoch}')
            if MPI.COMM_WORLD.Get_rank() ==0:
                
                success_rate.append(success_rate_epoch)
                #[epoch] = success_rate_epoch
                std_success_rate.append(std_success_rate_epoch)
                #[epoch] = std_success_rate_epoch
                if epoch>0:
                  plt.figure()
                  episodes = [i*10 for i in range(len(success_rate))]
                  print(f'episodes: {episodes}')
                  print(f'success_rate: {success_rate}')

                  print(f'std_success_rate: {std_success_rate}')
                  plt.plot(episodes, success_rate, label="HER+DDPG")
                  plt.fill_between(episodes, np.array(success_rate) - np.array(std_success_rate), np.array(success_rate) + np.array(std_success_rate), alpha=0.4)
                  plt.legend()
                  plt.title(f'Success rate vs episodes')
                  plt.xlabel("Episode")
                  plt.ylabel("Success Rate")
                  plt.savefig(f"plots/{epoch}.jpg", dpi=200, bbox_inches='tight')
                agent.save_models()

def _eval_agent(env, args, agent):
    total_success_rate = []
    for _ in range(10):
        per_success_rate = []
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        for _ in range(env.maxtimestamps):
            with torch.no_grad():
                input_tensor = agent._preproc_inputs(obs, g)
                pi = agent.actor(input_tensor)
                # convert the actions
                actions = pi.detach().cpu().numpy().squeeze()
            observation_new, _, _, info = env.step(actions)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    local_success_rate_std = np.std(total_success_rate[:, -1])

    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    global_success_rate_std = MPI.COMM_WORLD.allreduce(local_success_rate_std, op=MPI.SUM)

    return global_success_rate / MPI.COMM_WORLD.Get_size(),global_success_rate_std/  MPI.COMM_WORLD.Get_size()  
