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
    success_rate = np.zeros(args.episodes)
    for epoch in range(args.episodes):
        
        for cycle in range(args.cycles):
            # reset state, goal data 
            cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions = [],[],[],[]
            for _ in range(args.rollouts):
                ep_obs, ep_achieved_goal, ep_goal, ep_actions = [],[],[],[]
                episode_reward = 0
                observation = env.reset()

                obs = observation['observation']
                achieved_goal = observation['achieved_goal']
                goal = observation['desired_goal']
                # collect samples for 'args.policy_episodes' episodes
                for ep in range(env.maxtimestamps):
                    # collect samples
                    #agent = HERDDPG() ##TODO: comment before running..
                    with torch.no_grad():
                        input_tensor = agent.concat_inputs(obs, goal)
                        # print(f'input_tensor: {input_tensor}')
                        action = agent.actor(input_tensor)
                        action = agent.choose_action_wnoise(action,args.noise, 1)
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
                print(f'cycle: {cycle}')
                print(f'actions: {ep_actions}')
            # convert episode data into array for easier computation
            cycle_obs = np.array(cycle_obs)
            cycle_achieved_goal = np.array(cycle_achieved_goal)
            cycle_goal = np.array(cycle_goal)
            cycle_actions = np.array(cycle_actions)
            episode_reward += agent.rewards.mean()
            # store in replay buffer
            print(f'for cycle: {cycle}')
            print(f'cycle_obs: {cycle_obs.shape}')
            print(f'cycle_achieved_goal: {cycle_achieved_goal.shape}')
            print(f'cycle_goal: {cycle_goal.shape}')
            print(f'cycle_actions:{cycle_actions.shape} ')
            agent.remember([cycle_obs, cycle_achieved_goal, cycle_goal, cycle_actions])
            print(f'agent remembered the transition')
            # apply HER
            
            # # store new goals from her buffer into replay buffer
            # for i in range(len_transitions):
            agent.normalise_her_samples([cycle_obs, cycle_achieved_goal,cycle_goal, cycle_actions])
            print(f'normalised the samples')
            # n = len(agent.hindsight_buffer['reward'])
            # for i in range(n):
            #         temp_done = 1 if (i == n-1) else 0
            #         agent.remember(agent.hindsight_buffer['state'][i],
            #                            agent.hindsight_buffer['next_state'][i],
            #                            agent.hindsight_buffer['actions'][i],
            #                            agent.hindsight_buffer['reward'][i],
            #                            agent.hindsight_buffer['goal'][i],
            #                            temp_done
            #                            )
            print(f'starting model optimisation..')
            for _ in range(args.optimsteps):
                # perform ddpg optimization...
                agent.learn()
            print(f'model optimised..')
            agent.update_network_params()      
            print("in epoch..",epoch)
        # print losses
        print("Critic loss : ",agent.critic_loss )
        print("Actor loss : " , agent.actor_loss)
        print("Achieved goal : " , next_observation['achieved_goal'])
        print("Desired goal : " , next_observation['desired_goal'])
        print("[*] Number of episodes : {} Reward : {}".format(epoch, episode_reward))
        print("[*] End of epoch ",epoch)

        #if epoch%5==0:
        print(f'eval agent started for epoch... {epoch}')
        success_rate_epoch = _eval_agent(env,args, agent)
        success_rate[epoch] = np.mean(success_rate_epoch)
        print(f'success rate: {np.mean(success_rate_epoch)}')
        if epoch%20==0:
            agent.actor.save_model()
            agent.target_actor.save_model()
            agent.critic.save_model()
            agent.target_critic.save_model()
        
        with open('successlog.log') as f:
            f.write('success_rate[epoch]:'+success_rate[epoch])
# do the evaluation
def _eval_agent(env, args, agent):
    total_success_rate = []
    for _ in range(args.rollouts):
        per_success_rate = []
        observation = env.reset()
        obs = observation['observation']
        g = observation['desired_goal']
        for _ in range(env.maxtimestamps):
            with torch.no_grad():
                input_tensor = agent.concat_inputs(obs, g)
                action = agent.actor(input_tensor)
                pi =agent.choose_action_wnoise(action,args.noise, 1)
                # convert the actions
                actions = pi
            observation_new, _, _, info = env.step(actions)
            obs = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
        total_success_rate.append(per_success_rate)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size()