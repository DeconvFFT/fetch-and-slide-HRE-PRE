import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI # reference: https://mpi4py.readthedocs.io/en/stable/. 
from replaybuffer import ReplayBuffer
from actor_critic import Actor, Critic
from normaliser import normalizer
from her import HER
from mpiutils import *

class HERDDPG(object):
    def __init__(self, lr_actor, lr_critic, tau, env, envname, gamma, buffer_size,fc1_dims, fc2_dims, fc3_dims,cliprange, clip_observation,future, batch_size, per):
        '''
        Parameters:
        -----------
        lr_actor: float32
            Learning rate for actor network

        lr_critic: float32
            Learning rate for critic network

        tau: float32
            Polyak average to create weighted average of actor/critic models with target actor/critic models

        env: gym.env
            OpenAi gym environment under consideration

        envname: str
            Name of the environment under consideration

        gamma: float32
            Discount factor

        buffer_size: int
            Size of the replay buffer

        fc1_dims: int
            Dimensions of first fully connected layer in actor/critic network

        fc2_dims: int
            Dimensions of second fully connected layer in actor/critic network

        fc3_dims: int
            Dimensions of third fully connected layer in actor/critic network

        cliprange: int
            Clipping value for normalised observations/goals

        clip_observation: int
            Clipping value for observations/goals

        future: int
            How many future trajectories to consider

        batch_size: int
            Minibatch size for training

        per: bool
            Whether to use Prioritized Experience Replay or not

        Returns:
        --------
        None
        '''
        self.gamma = gamma
        self.tau = tau
        #ReplayBuffer(max_size=buffer_size, nS = env.nS, nA = env.nA, nG = env.nG)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.batch_size = batch_size
        self.actor_inputdims = env.nS+env.nG
        self.critic_input_dims = env.nS+env.nG+env.nA
        self.cliprange = cliprange
        self.clip_observation = clip_observation
        self.envname = envname
        self.env = env
        self.her_buffer = None
        self.critic_loss = None
        self.actor_loss = None
        self.per = per
        self.rewards = np.zeros(1)

        # define actor and critic networks
        # actor network
        self.actor = Actor(self.lr_actor, self.actor_inputdims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'actor')
        # critic network
        self.critic = Critic(self.lr_critic, self.critic_input_dims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'critic')

        # synchronize networks across cpus
        sync_networks(self.actor)
        sync_networks(self.critic)

        # define taget networks
        # actor network
        self.target_actor = Actor(self.lr_actor, self.actor_inputdims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'target_actor')
        # critic network
        self.target_critic = Critic(self.lr_critic, self.critic_input_dims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'target_critic')
        
        # load the weights into the target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # normalise observation and goal
        self.obs_norm = normalizer(size = env.nS, default_clip_range=self.cliprange)
        self.goal_norm = normalizer(size = env.nG, default_clip_range=self.cliprange)

        # create her instance
        self.her = HER(future,self.env.compute_reward, self.per)
        # replay buffer
        self.replay_memory = ReplayBuffer(max_size=buffer_size,nS = env.nS, nA=env.nA, nG=env.nG, timestamps =env.maxtimestamps, sampler = self.her.sample_transitions, per=self.per)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def choose_action_wnoise(self, action, noise_prob,random_prob, clip_val = 1):
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
       
        mu = action.cpu().numpy().squeeze()
        mu_prime = mu + noise_prob*np.random.randn(*mu.shape)
        mu_prime = np.clip(mu_prime, -clip_val, clip_val) # clipping to keep actions in a valid range after adding noise
        mu_random = np.random.uniform(low=-1, high=1,
                                           size=self.env.nA)
        mu_prime += np.random.binomial(1, random_prob, 1)[0] * (mu_random - mu_prime)
        return mu_prime
    
    
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
        if observation.shape[0]==self.env.nS:
            inputs = np.concatenate([observation, goal])
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        else:
            inputs = np.concatenate([observation, goal], axis = 1)
            inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs.to(self.device)
        return inputs
     
    def generate_hindsight_buffer(self, transitions):
        '''
        Generates hindsight buffer by sampling new goals from a set of goals
        and then setting new goals as the achieved goal for state, action pair
        and updates next goals and rewards for the transitions

        Parameters:
        -----------
        transitions: list() 
            List containing state, achieved goal, actual goal and actions
        
        Returns:
        -------
        buffer_experience: dict()
           Buffer containing hindsight from experience replay
        '''

        s, ag, g, actions = transitions
        s_ = s[:,1:,:] # next observations
        ag_ = ag[:,1:, :] # next goals
        n_transitions = actions.shape[1] # length is equal to T
                    
        # HER buffer containing observation, achieved goal, desired goal, actions, next state and next achieved goal
        buffer_her = {
            'observation': s,
            'achieved_goal':ag,
            'goal':g,
            'actions':actions,
            'next_state':s_,
            'achieved_goal_next':ag_
        }
        buffer_experience = self.her.sample_transitions(buffer_batch=buffer_her,batchsize=n_transitions)
        return buffer_experience


    def normalise_her_samples(self, transitions):
        '''
        Normalises observation and goals in hindsight buffer and recomputes
        mean and standard deviation for observations and goals using normaliser

        Parameters:
        -----------
       transitions: list() 
            Dictionary containing list of observations, next observations, actual goals, 
            achieved goals, next achieved goals and a list of actions
        
        Returns:
        -------
        None
        '''
        self.hindsight_buffer = self.generate_hindsight_buffer(transitions)
        state, goal,  = self.hindsight_buffer['observation'], self.hindsight_buffer['goal']
        state, goal = self.preprocess_inputs(state, self.clip_observation),  self.preprocess_inputs(goal, self.clip_observation) # clip values of state and goal
        self.hindsight_buffer['observation'], self.hindsight_buffer['goal'] =  state, goal # update state, goal in buffer to reflect clipped values
        
        # update normaliser and recompute stats
        self.obs_norm.update_params(self.hindsight_buffer['observation'])
        self.goal_norm.update_params(self.hindsight_buffer['goal'])
        self.obs_norm.recompute_stats()
        self.goal_norm.recompute_stats()

    
    def preprocess_inputs(self, observation,cliprange = np.inf):
        '''
        Clips an input in desired range given by cliprange

        Parameters:
        -----------
        observation: np.float32
            Any float value 
        
        Returns:
        -------
        clipped_input: float32
            Input clipped in a range given by cliprange
        '''
        clipped_input = np.clip(observation, -cliprange,cliprange)
        return clipped_input
    
    def update_network_params(self,tau=None):
        # implemented from ddpg paper: https://arxiv.org/pdf/1509.02971.pdf
        '''
        Moves target network based on tau. 
        Takes a weighted average of original models and target models to update the target models
        With a weight tau, it takes value from actor/critic and with weight (1-tau),
        it takes value from target actor/critic model

        Parameters:
        -----------
        tau: np.float32
            weight of actor/critic model for target actor/critic model 
        
        Returns:
        -------
        None
        '''
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((tau) * param.data + (1-tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau)* target_param.data)
        

    def remember(self, transitions):
        '''
        Stores transitions into replay memory
        '''
        self.replay_memory.store_transitions(transitions)
    

    def learn(self):
        '''
        Samples trajectories from the replay buffer, genrates actions and q values based on that actions.
        Generates target actions and target q values from target networks and then calculates error
        between actual and target q values as the loss function. Backpropagates actor and critic loss and 
        takes an optimiser step for both actor and critic models.

        Parameters:
        -----------
        None
        
        Returns:
        -------
        None
        '''
        transitions = self.replay_memory.sample_buffer(self.batch_size)

        # get observation, next observation, goal and next goal
        state, goal, next_state = transitions['observation'], transitions['goal'], transitions['observation_next']
        transitions['observation'], transitions['goal'] = self.preprocess_inputs(state, self.clip_observation), self.preprocess_inputs(goal,self.clip_observation) # preprocess state, goal
        transitions['observation_next'], transitions['goal_next'] = self.preprocess_inputs(next_state,self.clip_observation), self.preprocess_inputs(goal,self.clip_observation)
        
        # normalise state and goal
        state_norm = self.obs_norm.normalize(transitions['observation'])
        goal_norm = self.goal_norm.normalize(transitions['goal'])
        next_state_norm =  self.obs_norm.normalize(transitions['observation_next'])
        next_goal_norm = self.goal_norm.normalize(transitions['goal_next'])

        # combine state and action as per "st||g and  st+1||g; || = concatenaton" and convert to tensors
        obs_goal = np.concatenate([state_norm, goal_norm], axis=1)

        obsnext_goal = np.concatenate([next_state_norm, next_goal_norm], axis=1)
        obs_goal = torch.tensor(obs_goal, dtype=torch.float32)
        obsnext_goal = torch.tensor(obsnext_goal, dtype=torch.float32)
        # convert actions, reward into tensors
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        rewards_tensor = torch.tensor(transitions['reward'], dtype=torch.float32)
        self.rewards = rewards_tensor
        # move tensors to device
        obs_goal.to(self.device)
        obsnext_goal.to(self.device)
        actions_tensor.to(self.device)
        rewards_tensor.to(self.device)

        with torch.no_grad():
            target_actions = self.target_actor(obsnext_goal)
            target_q = self.target_critic(obsnext_goal, target_actions)
            target_q = target_q.detach()
            target_q = rewards_tensor + self.gamma * target_q 
            target_q = target_q.detach()
            # clipping targets as per experiments section of original paper: https://arxiv.org/pdf/1707.01495.pdf page 14
            # clipping value is between 1/(1-gamma) and 0
            target_clip_val = 1 / (1-self.gamma)
            # torch.clip and torch.clamp are similar. 
            target_q = torch.clamp(target_q, -target_clip_val, 0)

        actual_q = self.critic(obs_goal, actions_tensor)

        # calculate mse loss
        critic_mse_loss = (target_q - actual_q).pow(2).mean()
        self.critic_loss = critic_mse_loss.item()
        mu_b = self.actor(obs_goal)
        actor_loss = -self.critic(obs_goal, mu_b).mean() # actor loss is -(expected loss from critic)
        actor_loss += (mu_b).pow(2).mean()
        
        self.actor.optimiser.zero_grad()
        self.actor_loss = actor_loss.item()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor.optimiser.step()

        self.critic.optimiser.zero_grad() 
        critic_mse_loss.backward()
        sync_grads(self.critic)
        self.critic.optimiser.step()

    def prepare_inputs(self, obs, g):
        '''
        Parameters:
        ----------
        obs: list()
            25 d list containing states

        goal: list()
            3 d list containing goals

        Returns:
        --------
        inputs: torch.tensor
            torch tensor containing observation and goals concatenated
        '''
        obs_norm = self.obs_norm.normalize(obs)
        g_norm = self.goal_norm.normalize(g)
        # concatenate the stuffs
        inputs = self.concat_inputs(obs_norm, g_norm)
        inputs.to(self.device)
        return inputs
    
    def save_models(self):
        self.actor.save_model(self.obs_norm.mean, self.obs_norm.std, self.goal_norm.mean, self.goal_norm.std)
        self.critic.save_model(self.obs_norm.mean, self.obs_norm.std, self.goal_norm.mean, self.goal_norm.std)
        self.target_actor.save_model(self.obs_norm.mean, self.obs_norm.std, self.goal_norm.mean, self.goal_norm.std)
        self.target_critic.save_model(self.obs_norm.mean, self.obs_norm.std, self.goal_norm.mean, self.goal_norm.std)
    
    def load_models(self):
        obs_mean, obs_std, goal_mean, goal_std = self.actor.load_model()
        return obs_mean, obs_std, goal_mean, goal_std




    







