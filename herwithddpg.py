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
from mpiutils import *

class HERDDPG(object):
    def __init__(self, lr_actor, lr_critic, tau, env, envname, gamma, buffer_size,fc1_dims, fc2_dims, fc3_dims,cliprange, clip_observation,future, batch_size=128):
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
        self.rewards = np.zeros(1)
        # define actor and critic networks
        # actor network
        self.actor = Actor(self.lr_actor, self.actor_inputdims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'actor')
        # critic network
        self.critic = Critic(self.lr_critic, self.critic_input_dims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'critic')

        sync_networks(self.actor)
        sync_networks(self.critic)

        # define taget networks
        # actor network
        self.target_actor = Actor(self.lr_actor, self.actor_inputdims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'target_actor')
        # critic network
        self.target_critic = Critic(self.lr_critic, self.critic_input_dims,self.fc1_dims, self.fc2_dims, self.fc3_dims, env.nA, 'target_critic')
        
        # adding OU noise to make the policy noisy and generate noisy actions
        self.ounoise = OUNoise(env.nA,123)

        # normalise observation and goal

        self.obs_norm = normalizer(size = env.nS, default_clip_range=self.cliprange)
        self.goal_norm = normalizer(size = env.nG, default_clip_range=self.cliprange)

        # create a test environment to avoid breaking things in train environment
        self.testenv = gym.make(envname)

        self.her = HER(future, self.env.compute_reward)
        self.replay_memory = ReplayBuffer(max_size=buffer_size,nS = env.nS, nA=env.nA, nG=env.nG, timestamps =env.maxtimestamps, sampler = self.her.sample_transitions)

        # move the model on to a device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.update_network_params(tau=1)

    def choose_action_wnoise(self, action, noise, clip_val = 1):
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
        #self.actor.eval()
        #observation = observation.reshape(1,-1)
        #observation = observation.to(self.actor.device)
        #with torch.no_grad():
        mu = action.cpu().numpy().squeeze()
       # mu = self.actor(observation)#.to(self.actor.device)

        # mu_prime = mu + torch.tensor(self.ounoise(), dtype=torch.float).to(self.actor.device)
        #self.actor.train()
        # mu_prime1 =  torch.flatten(mu_prime).detach().cpu().numpy()
        mu_prime = mu + noise*self.ounoise()
        mu_prime = np.clip(mu_prime, -clip_val, clip_val) # clipping to keep actions in a valid range after adding noise

         # random actions...
        random_actions = np.random.uniform(low=-1, high=1,
                                           size=self.env.nA)
        # choose if use the random actions
        mu_prime += np.random.binomial(1, 0.3, 1)[0] * (random_actions - mu_prime)
        return mu_prime
    
    def choose_action(self,observation, clip_val):
        self.actor.eval()
        observation = observation.reshape(1,-1)
        observation = observation.to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        self.actor.train()
        mu1 =  torch.flatten(mu).detach().cpu().numpy()
        return np.clip(mu1, -clip_val, clip_val)
    
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

        # obs_norm = self.obs_norm.normalize(observation)
        # goal_norm = self.goal_norm.normalize(goal)

        # if observation.shape[0] == 25:
        #     inputs = np.concatenate([observation, goal])
        # else:
        inputs = np.concatenate([observation, goal], axis = 1)
        inputs = torch.tensor(inputs, dtype=torch.float32)

        inputs.to(self.device)
        return inputs
     # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.obs_norm.normalize(obs)
        g_norm = self.goal_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs.to(self.device)
        return inputs

    def generate_hindsight_buffer(self, transitions):
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

        s, ag, g, actions = transitions
        s_ = s[:,1:,:] # next observations
        ag_ = ag[:,1:, :] # next goals
        n_transitions = actions.shape[1] # length is equal to T
                    
        # HER buffer
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
        hindsight buffer: dict() 
            Dictionary containing observations, next observations, actual goals, 
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
        
        # get actor and critic parameters
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        # get target actor and target critic parameters
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # create new state dicts from named params 

        ## actor and critic state dict
        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)

        ## target actor and target critic state dict
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)
        
        # do a weighted update of state dicts 
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau) * target_actor_state_dict[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1-tau) * target_critic_state_dict[name].clone()
        
        # load the updated state dicts into target models
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)


    def remember(self, transitions):
        '''
        Stores transitions into replay memory
        '''
        self.replay_memory.store_transitions(transitions)
    

    def learn(self):
        '''
        Learns from stored transitions inside 

        Parameters:
        -----------
        tau: np.float32
            weight of actor/critic model for target actor/critic model 
        
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
        obs_goal= self.concat_inputs(state_norm, goal_norm)
        obsnext_goal = self.concat_inputs(next_state_norm, next_goal_norm)
        # convert actions, reward into tensors
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        rewards_tensor = torch.tensor(transitions['reward'], dtype=torch.float32)
        self.rewards = rewards_tensor
        # move tensors to device
        obs_goal.to(self.device)
        obsnext_goal.to(self.device)
        actions_tensor.to(self.device)
        rewards_tensor.to(self.device)

        # sending target networks and critic network into eval mode
        # self.target_actor.eval()
        # self.target_critic.eval()
        # self.critic.eval()
        with torch.no_grad():
            target_actions = self.target_actor.forward(obsnext_goal)
            target_q = self.target_critic.forward(obsnext_goal, target_actions)
            target_q = target_q.detach()
            target_q = rewards_tensor + self.gamma * target_q 
            target_q = target_q.detach()
            # clipping targets as per experiments section of original paper: https://arxiv.org/pdf/1707.01495.pdf page 14
            # clipping value is between 1/(1-gamma) and 0
            target_clip_val = 1 / (1-self.gamma)
            # torch.clip and torch.clamp are similar. 
            target_q = torch.clamp(target_q, target_clip_val, 0)
            # actual_q = self.critic.forward(obsnext_goal, actions_tensor)

        # # get terminal flag
        # terminals = torch.tensor(transitions['terminals'])

        #target_q = rewards_tensor + self.gamma *target_q* terminals
        # target = []
        # for j in range(self.batch_size):
        #     target.append(rewards_tensor[j] + self.gamma*target_q[j]*(1-int(terminals[j])))
        # target = torch.tensor(target).to(self.critic.device)

        #target_q = target.view(self.batch_size, 1)

        

        # sending critic back to training
        #self.critic.train()

        
        
    
        # calculating actor model loss
        # setting critic back in eval as we need to calculate loss from the actor model  
        # by taking action generated from a behavioural policy (target policy + OU noise)
        #self.critic.eval()
        actual_q = self.critic(obs_goal, actions_tensor)

        # calculate mse loss
        critic_mse_loss = F.mse_loss(target_q, actual_q)
        self.critic_loss = critic_mse_loss.item()
        actions_real = self.actor(obs_goal)

        actor_loss = -self.critic(obs_goal, actions_real).mean()
        actor_loss += (actions_real).pow(2).mean()
        
        self.actor.optimiser.zero_grad()
        # mu_b = self.actor.forward(obs_goal)
        #self.actor.train()
        # actor_loss = -self.critic.forward(obs_goal, mu_b)
        # actor_loss = torch.mean(actor_loss) # actor loss is -(expected loss from critic)
        self.actor_loss = actor_loss.item()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor.optimiser.step()

        self.critic.optimiser.zero_grad() 
        critic_mse_loss.backward()
        sync_grads(self.critic)
        self.critic.optimiser.step()

        # update network params

    
    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
        self.target_actor.save_model()
        self.target_critic.save_model()
    
    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
        self.target_actor.load_model()
        self.target_critic.load_model()




    







