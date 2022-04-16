import os
import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F
import gym
from herwithddpg import HERDDPG
import time
from mpi4py import MPI

def train_agent(args, env, agent=HERDDPG):
    for epoch in range(args.episodes):
        
        for _ in range(args.cycles):
            # reset state, goal data 
            ep_obs, ep_achieved_goal, ep_goal, ep_actions = [],[],[],[]
            episode_reward = 0
            observation = env.reset()

            obs = observation['observation']
            achieved_goal = observation['achieved_goal']
            goal = observation['desired_goal']
            # collect samples for 'args.policy_episodes' episodes
            for ep in range(args.policy_episodes):
                # collect samples
                #agent = HERDDPG() ##TODO: comment before running..

                input_tensor = agent.concat_inputs(obs, goal)
                # print(f'input_tensor: {input_tensor}')
                action = agent.choose_action_wnoise(input_tensor, 1)
                # print(f'action: {action}')
                # collect statistics about next observation, achieved goals and desired goals..
                next_observation, reward, done, _ = env.step(action)
                obs_next = next_observation['observation']
                achieved_goal_next = next_observation['achieved_goal']
                # store in replay buffer
                agent.remember(obs, obs_next, action, reward, goal, done)
                ep_obs.append(obs.copy())
                ep_achieved_goal.append(achieved_goal.copy())
                ep_goal.append(goal.copy())
                ep_actions.append(action.copy())
                obs = obs_next
                achieved_goal = achieved_goal_next

            ep_obs.append(obs.copy())
            ep_achieved_goal.append(achieved_goal.copy())

            # convert episode data into array for easier computation
            ep_obs = np.array(ep_obs)
            ep_achieved_goal = np.array(ep_achieved_goal)
            ep_goal = np.array(ep_goal)
            ep_actions = np.array(ep_actions)
            episode_reward += agent.rewards.mean()
            
            # apply HER
            
            # # store new goals from her buffer into replay buffer
            # for i in range(len_transitions):
            agent.normalise_her_samples([ep_obs, ep_achieved_goal,ep_goal, ep_actions])

            n = len(agent.hindsight_buffer['reward'])
            for i in range(n):
                    temp_done = 1 if (i == n-1) else 0
                    agent.remember(agent.hindsight_buffer['state'][i],
                                       agent.hindsight_buffer['next_state'][i],
                                       agent.hindsight_buffer['actions'][i],
                                       agent.hindsight_buffer['reward'][i],
                                       agent.hindsight_buffer['goal'][i],
                                       temp_done
                                       )
            for _ in range(args.optimsteps):
                # perform ddpg optimization...
                agent.learn()
        
        # print losses
        print("Critic loss : ",agent.critic_loss )
        print("Actor loss : " , agent.actor_loss)
        print("Achieved goal : " , next_observation['achieved_goal'])
        print("Desired goal : " , next_observation['desired_goal'])
        print("[*] Number of episodes : {} Reward : {}".format(epoch, episode_reward))
        print("[*] End of epoch ",epoch)

        if epoch%5==0:
            success_rate = _eval_agent(args, agent)
            print(f'success rate: {success_rate}')
        if epoch%20==0:
            agent.actor.save_model()
            agent.target_actor.save_model()
            agent.critic.save_model()
            agent.target_critic.save_model()
        

def _eval_agent(args,agent):
    total_success_rate = []
    per_success_rate = []
    env = agent.testenv
    observation = env.reset()
    obs = observation['observation']
    g = observation['desired_goal']
    for _ in range(50):
        with torch.no_grad():
            input_tensor = agent.concat_inputs(obs, g)
            pi = agent.choose_action(input_tensor,1)
            print(f'actions: {pi}')
            # convert the actions
            actions = pi
        observation_new, _, _, info = env.step(actions)
        obs = observation_new['observation']
        g = observation_new['desired_goal']
        per_success_rate.append(info['is_success'])
    print(f'per_success_rate: {per_success_rate}')
    total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size()