import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI # reference: https://mpi4py.readthedocs.io/en/stable/. 
from replaybuffer import ReplayBuffer
from actor_critic import Actor, Critic
from ounoise import OUNoise

class HERDDPG(object):
    def __init__(self, lr_actor, lr_critic, tau, env, gamma, buffer_size,fc1_dims, fc2_dims, fc3_dims, batch_size=64):
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
        
        # adding OU noise to make the policy noisy
        self.ounoise = OUNoise(mu=np.zeros(env.nA))

