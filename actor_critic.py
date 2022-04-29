import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from mpi4py import MPI
# define actor class as per HER paper: https://arxiv.org/pdf/1707.01495.pdf
# 3 hidden layers, 64 hidden units in each layers
# ReLu activation for hidden layers, tanh activation for actor output
# rescale tanh output to [-5cm, 5cm] range
# add square of preactivations to actor's cost function
# input dims of size -> number of states + goal 
class Actor(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,fc3_dims, nA,name,chkpt_dir='models'):
        '''
        Parameters:
        ----------
        lr: float
            Learning rate for actor model

        input_dims: int 
            Input dimension for actor network
            shape of state space + shape of goal

        fc1_dims: int
            Number of hidden units for fc1 layer * number of actions in action space

        fc2_dims: int
            Number of hidden units for fc2 layer * number of actions in action space
        
        fc3_dims: int
            Number of hidden units for fc3 layer * number of actions in action space

        nA: int
            Number of actions in action space
        
        name: string
            Name of the model
        
        chkpt_dir: string
            Directory to save the model in
        
        Returns:
        -------
        None

        '''
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.lr = lr
        self.nA = nA
        self.chkpt_file = os.path.join(chkpt_dir,name)

        # define network architecture
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) #hidden layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # hidden layer 2
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims) # hidden layer 3

        self.mu = nn.Linear(self.fc3_dims, self.nA) # output layer-> noisy version of original policy

        # define optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), lr = self.lr)

        # move the model on to a device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        '''
        Parameters:
        ----------
        state: int
            Current state, goal pair

        Returns:
        -------
        output: int
            Distance from the goal state
        '''
        output = self.fc1(state)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        output = F.relu(output)
        output = torch.tanh(self.mu(output))
        return output
    
    def save_model(self, obs_mean, obs_std, goal_mean, goal_std):
        print(f'Saving actor model at checkpoint')
        if MPI.COMM_WORLD.Get_rank() == 0:
            torch.save([obs_mean, obs_std, goal_mean, goal_std,self.state_dict()], self.chkpt_file)
    
    def load_model(self):
        print(f'Loading actor model from checkpoint')
        if MPI.COMM_WORLD.Get_rank() == 0:
            obs_mean, obs_std, goal_mean, goal_std, state_dict = torch.load(self.chkpt_file)
            self.load_state_dict(state_dict)
            return obs_mean, obs_std, goal_mean, goal_std


class Critic(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims,nA, name, chkpt_dir = 'models'):
        '''
        Parameters:
        ----------
        lr: float
            Learning rate for critic model

        input_dims: int 
            Input dimension for critic network
            shape of state space + shape of goal + shape of action space

        fc1_dims: int
            Number of hidden units for fc1 layer * number of actions in action space

        fc2_dims: int
            Number of hidden units for fc2 layer * number of actions in action space
        
        fc3_dims: int
            Number of hidden units for fc3 layer * number of actions in action space

        nA: int
            Number of actions in action space
        
        name: string
            Name of the model
        
        chkpt_dir: string
            Directory to save the model in
        
        Returns:
        -------
        None

        '''
        super(Critic, self).__init__()
        self.input_dims = input_dims # nS+ nG + nA
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.lr = lr
        self.nA = nA
        self.chkpt_file = os.path.join(chkpt_dir,name)

        # define network architecture
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) #hidden layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # hidden layer 2
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims) # hidden layer 3
        self.Q = nn.Linear(self.fc3_dims, 1) # output -> Q value based on state and action(from target policy from actor model)
        
        # define optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), lr = self.lr)
        # move the model on to a device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # define forward class
    def forward(self, state, action):
        '''
        Parameters:
        ----------
        state: int
            Current state, goal pair
        
        action: float32
            Current action value

        Returns:
        -------
        q_value: float32
            Q value corresponding to (s,a,g) pair
        '''
        # create state, action value pair
        state_action_value = torch.cat([state, action], dim=1) 
        state_action_value = self.fc1(state_action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc2(state_action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc3(state_action_value)
        state_action_value = F.relu(state_action_value)
        q_value = self.Q(state_action_value)
        return q_value
    
    def save_model(self, obs_mean, obs_std, goal_mean, goal_std):
        '''
        Parameters:
        ----------
        obs_mean: float32
            Mean of observations

        obs_std: float32
            Standard Deviation of observations

        goal_mean: float32
            Mean of goals

        goal_std: float32
            Standard deviaiton of goals

        Returns:
        -------
        None
        '''
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f'Saving actor model at checkpoint')
            torch.save([obs_mean, obs_std, goal_mean, goal_std,self.state_dict()], self.chkpt_file)
    
    def load_model(self):
        '''
        Parameters:
        -----------
        None

        Returns:
        --------
        obs_mean: float32
            Mean of observations

        obs_std: float32
            Standard Deviation of observations

        goal_mean: float32
            Mean of goals

        goal_std: float32
            Standard deviaiton of goals
        '''
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f'Loading actor model from checkpoint')
            obs_mean, obs_std, goal_mean, goal_std, state_dict = torch.load(self.chkpt_file)
            self.load_state_dict(state_dict)
            return obs_mean, obs_std, goal_mean, goal_std