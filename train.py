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
    
    eval_every = 20
    priority_sum = 0
    for epoch in range(args.episodes):
        for cycle in range(args.cycles):
            cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions,cycle_priority = [],[],[],[],[] # collects statistics for each cycle
            for _ in range(args.rollouts):
                ep_obs, ep_achieved_goal, ep_goal, ep_actions,ep_priority = [],[],[],[],[] # collects statistics for each rollout within a cycle
                episode_reward = 0
                observation = env.reset()

                obs = observation['observation']
                achieved_goal = observation['achieved_goal']
                goal = observation['desired_goal']
                # collect samples for 'env.maxtimestamps' timestamps
                for t in range(env.maxtimestamps):
                    with torch.no_grad():
                        input_tensor = agent.prepare_inputs(obs, goal)
                        pi = agent.actor(input_tensor)
                        action = agent.choose_action_wnoise(pi,args.noise_prob,args.random_prob,1)
                    # collect statistics about next observation, achieved goals and desired goals..
                    next_observation, reward, done, _ = env.step(action)
                    obs_next = next_observation['observation']
                    achieved_goal_next = next_observation['achieved_goal']

                    if args.per:
                        with torch.no_grad():
                            obs1, goal1 = agent.preprocess_inputs(obs), agent.preprocess_inputs(goal)
                            input_tensor = agent.concat_inputs(obs1, goal1)
                            action_tensor = torch.tensor(action, dtype=torch.float32)
                            action_tensor = action_tensor.view(1,-1)
                            # current q value:
                            q_curr = agent.critic(input_tensor, action_tensor)
                            q_curr = q_curr.detach().cpu().numpy().squeeze()

                            # get next observation and goal
                            obs_next1, goal_next1 = agent.preprocess_inputs(obs_next), agent.preprocess_inputs(goal)
                            input_tensor_next = agent.concat_inputs(obs_next1, goal_next1)

                            # get next action
                            pi_next = agent.actor(input_tensor_next)
                            action_next = agent.choose_action_wnoise(pi_next,args.noise_prob,args.random_prob,1)
                            action_next_tensor = torch.tensor(action_next, dtype=torch.float32)
                            action_next_tensor = action_next_tensor.reshape(1,-1)
                            # get next q values
                            q_next = agent.target_critic(input_tensor_next, action_next_tensor)
                            q_next = q_next.detach().cpu().numpy().squeeze()

                            # find the td error for q value
                            td_error = np.abs(q_next - q_curr)
                            priority = (td_error + args.epsilon) ** args.alpha
                            priority_sum +=priority
                            priority_prob = priority / priority_sum
                            ep_priority.append(priority_prob)


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
                if args.per:
                    cycle_priority.append(ep_priority)
            cycle_obs = np.array(cycle_obs)
            cycle_achieved_goal = np.array(cycle_achieved_goal)
            cycle_goal = np.array(cycle_goal)
            cycle_actions = np.array(cycle_actions)

            episode_reward += agent.rewards.mean()
            if args.per:
                cycle_priority = np.array(cycle_priority)
                agent.remember([cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions,cycle_priority])
                # apply HER
                # # store new goals from her buffer into replay buffer
                agent.normalise_her_samples([cycle_obs, cycle_achieved_goal,cycle_goal, cycle_actions,cycle_priority])
            else:
                agent.remember([cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions])
                # apply HER
                # # store new goals from her buffer into replay buffer
                agent.normalise_her_samples([cycle_obs, cycle_achieved_goal,cycle_goal, cycle_actions])
            
            #perform ddpg optimisation for args.optimsteps=40 steps
            for _ in range(args.optimsteps):
                # perform ddpg optimization...
                agent.learn()
            agent.update_network_params()      

        
       
        if epoch%eval_every==0:
            success_rate_epoch = evaluate_agent(env,args, agent)
            if MPI.COMM_WORLD.Get_rank() ==0:
                success_rate.append(success_rate_epoch)
                if epoch>0:
                  plt.figure()
                  episodes = [i*eval_every for i in range(len(success_rate))]
                  print(f'episodes: {episodes}')
                  print(f'success_rate: {success_rate}')
                  plt.plot(episodes, success_rate, label="HER+DDPG")
                  plt.legend()
                  plt.title(f'Success rate vs episodes')
                  plt.xlabel("Episode")
                  plt.ylabel("Success Rate")
                  plt.savefig(f"plots/{epoch}.jpg", dpi=200, bbox_inches='tight')
                agent.save_models()

def evaluate_agent(env, args, agent):
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
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size()
