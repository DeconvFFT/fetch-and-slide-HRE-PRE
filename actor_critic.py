import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

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
        self.chkpt_file = os.join(chkpt_dir,name)

        # define network architecture
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) #hidden layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # hidden layer 2
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims) # hidden layer 3

        self.mu = nn.Linear(self.fc2_dims, self.nA) # output layer-> noisy version of original policy

        # add layer normalisation (batch norm here) as per ddpg documentation from openAI: https://spinningup.openai.com/en/latest/algorithms/ddpg.html 
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        self.bn3 = nn.BatchNorm1d(self.fc3_dims)

        # initialise network weights and biases
        fan1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -fan1, fan1)
        torch.nn.init.uniform_(self.fc1.bias.data, -fan1, fan1)

        fan2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -fan2, fan2)
        torch.nn.init.uniform_(self.fc2.bias.data, -fan2, fan2)

        fan3 = 1/np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc3.weight.data, -fan3, fan3)
        torch.nn.init.uniform_(self.fc3.bias.data, -fan3, fan3)


        fan4 = 0.003 # fixed value as per DDPG paper: https://arxiv.org/pdf/1509.02971.pdf
        torch.nn.init.uniform_(self.mu.weight.data, -fan4, fan4)
        torch.nn.init.uniform_(self.mu.bias.data, -fan4, fan4)

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
        output = self.bn1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = self.bn2(output)
        output = F.relu(output)
        output = self.fc3(output)
        output = self.bn3(output)
        output = F.relu(output)
        output = torch.tanh(self.mu(output))
        return output
    
    def save_model(self):
        print(f'Saving actor model at checkpoint')
        torch.save(self.state_dict(), self.chkpt_file)
    
    def load_model(self):
        print(f'Loading actor model from checkpoint')
        self.load_state_dict(torch.load(self.chkpt_file))


class Critic(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims,nA, name, chkpt_dir = 'models'):
        super(Critic, self).__init__()
        self.input_dims = input_dims # nS+ nG + nA
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.lr = lr
        self.nA = nA
        self.chkpt_file = os.join(chkpt_dir,name)

        # define network architecture
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) #hidden layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # hidden layer 2
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        #self.action_value = nn.Linear(self.nA, self.fc2_dims) # hidden layer 3 -> contains state, action value pair
        self.Q = nn.Linear(self.fc3_dims, 1)

        # add layer normalisation (batch norm here) as per ddpg documentation from openAI: https://spinningup.openai.com/en/latest/algorithms/ddpg.html 
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        # initialise network weights and biases
        fan1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -fan1, fan1)
        torch.nn.init.uniform_(self.fc1.bias.data, -fan1, fan1)

        fan2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -fan2, fan2)
        torch.nn.init.uniform_(self.fc2.bias.data, -fan2, fan2)

        fan3 = 1/np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc3.weight.data, -fan3, fan3)
        torch.nn.init.uniform_(self.fc3.bias.data, -fan3, fan3)
        

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
        # state = s || g ; || -> concatenation
        state_action_value = torch.cat([state, action], dim=1) 
        state_action_value = self.fc1(state_action_value)
        state_action_value = self.bn1(state_action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc2(state_action_value)
        state_action_value = self.bn2(state_action_value)
        #action_value = F.relu(self.action_value(action))
        q_value = self.Q(state_action_value)
        return q_value
    
    def save_model(self):
        print(f'Saving actor model at checkpoint')
        torch.save(self.state_dict(), self.chkpt_file)
    
    def load_model(self):
        print(f'Loading actor model from checkpoint')
        self.load_state_dict(torch.load(self.chkpt_file))