import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI # reference: https://mpi4py.readthedocs.io/en/stable/. 
from replaybuffer import ReplayBuffer
from actor_critic import Actor, Critic
from ounoise import OUNoise
from normaliser import normalizer
from her import HER

class HERDDPG(object):
    def __init__(self, lr_actor, lr_critic, tau, env, envname, gamma, buffer_size,fc1_dims, fc2_dims, fc3_dims,cliprange,  batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.replay_memory = ReplayBuffer(max_size=buffer_size, nS = env.nS, nA = env.nA, nG = env.nG)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.batch_size = batch_size
        self.actor_inputdims = env.nS+env.nG
        self.critic_input_dims = env.nS+env.nG+env.nA
        self.cliprange = cliprange
        self.envname = envname

        # define actor and critic networks
        # actor network
        self.actor = Actor(self.lr_actor, self.actor_inputdims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'actor')
        # critic network
        self.critic = Critic(self.lr_actor, self.critic_input_dims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'critic')

        # define taget networks
        # actor network
        self.target_actor = Actor(self.lr_actor, self.actor_inputdims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'actor')
        # critic network
        self.target_critic = Critic(self.lr_actor, self.critic_input_dims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'critic')
        
        # adding OU noise to make the policy noisy and generate noisy actions
        self.ounoise = OUNoise(mu=np.zeros(env.nA))

        # normalise observation and goal

        self.obs_norm = normalizer(size = env.nS, default_clip_range=self.cliprange)
        self.goal_norm = normalizer(size = env.nG, default_clip_range=self.cliprange)

        # create a test environment to avoid breaking things in train environment
        self.testenv = gym.make(envname)
        self.her = HER(self.env.calculate_reward)

         # move the model on to a device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def choose_action_wnoise(self, observation, clip_val = 1):
        '''
        Adds Noise to actions and clips the values of actions 
        in between a specific range defined by clip_val

        Parameters:
        -----------
        observation: int 
            Observed state
        
        clip_val: int
            Upper and lower bounds to clip value of action
            Default: action is clipped in [-1, 1] range

        Returns:
        -------
        mu_prime: int
           Noisy action clipped with clip_val
        '''
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.ounoise(), dtype=torch.float).to(self.actor.device)

        self.actor.train()
        mu_prime = np.clip(mu_prime, -clip_val, clip_val) # clipping to keep actions in a valid range after adding noise
        return mu_prime.cpu().detach().numpy()
    
    def concat_inputs(self, observation, goal):
        '''
        Concats state and goal to create pairing of form s||g
        Normalizes state and goal to standard normal and then concats them

        Parameters:
        -----------
        observation: int 
            Observed state
        
        goal: int
            Goal state

        Returns:
        -------
        obs_goal_pair: torch.tensor
           tensor containing observed state and goal state concatenated
        '''

        obs_norm = self.obs_norm.normalize(observation)
        goal_norm = self.goal_norm.normalize(goal)

        obs_goal_norm = np.concatenate([obs_norm, goal_norm],axis=1)
        obs_goal_norm = torch.tensor(obs_goal_norm, dtype=torch.float32)

        obs_goal_norm.to(self.device)
        return obs_goal_norm
    
    def generate_hindsight(self, transitions):
        '''
        Generates hindsight buffer by sampling new goals from a set of goals
        and then setting new goals as the achieved goal for state, action pair
        and updates next goals and rewards for the transitions

        Parameters:
        -----------
        transitions: tuple 
            Tuple containing state, achieved goal, actual goal and actions
        
        Returns:
        -------
        buffer_experience: Dict
           Buffer containing hindsight from experience replay
        '''

        obs, ag, g, actions = transitions
        obs_ = obs[1:,:] # next observations
        ag_ = ag[1:, :] # next goals
        
        # HER buffer
        buffer_her = {
            'obs': obs,
            'obs_':obs_,
            'g':g,
            'ag':ag,
            'ag_':ag_,
            'actions':actions
        }
        buffer_experience = self.her.get_hindsight(buffer = buffer_her)
        return buffer_experience

     

       
        
