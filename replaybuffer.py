from sys import maxsize
import threading
import numpy as np

# replay buffer class
class ReplayBuffer(object):
    def __init__(self, max_size, nS, nA, nG, timestamps, sampler, per):
        '''
        Sets replay buffer parameters and generates a standard or
        prioritized buffer with default parameters

        Parameters:
        -----------
        maxsize: int
            Maximum size of the replay buffer

        nS: int
            Size of environment's state space

        nA: int
            Size of environment's action space

        nG: int
            Size of environment's goal space

        timestamps: int
            Number of timestamps per episode

        sampler: func
            Sampling function for replay buffer. It is the same as sampling
            function for HER class

        per: bool
            Whether to use Stochastic Prioritization
            over replay buffer trajectories or not

        Returns:
        --------
        None
        '''
        self.maxsize = max_size
        self.horizon = timestamps
        self.size = self.maxsize // self.horizon
        self.counter = 0 # tracks size of buffer as we fill it
        self.sampler = sampler # function which samples data from buffer. Same as HER buffer
        self.nS = nS
        self.nG = nG
        self.nA = nA
        self.per = per
        if self.per:
             # create a buffer
            self.buffer = {
                'observation': np.zeros([self.size, self.horizon+1, self.nS]), # till T+1 because we need to fetch S[T] from mapping
                'achieved_goal': np.zeros([self.size, self.horizon+1, self.nG]), # till T+1 because we need to fetch G[T] from mapping
                'goal': np.zeros([self.size, self.horizon, self.nG]),
                'actions':np.zeros([self.size, self.horizon, self.nA]),
                'priority':np.empty([self.size, self.horizon])
            }   
        else:
            # create a buffer
            self.buffer = {
                'observation': np.zeros([self.size, self.horizon+1, self.nS]), # till T+1 because we need to fetch S[T] from mapping
                'achieved_goal': np.zeros([self.size, self.horizon+1, self.nG]), # till T+1 because we need to fetch G[T] from mapping
                'goal': np.zeros([self.size, self.horizon, self.nG]),
                'actions':np.zeros([self.size, self.horizon, self.nA])
            }
        self.lock = threading.Lock() # acquire thread lock
    
    def store_transitions(self, transition):
        '''
        Stores transitions at an index in the replay buffer

        Parameters:
        -----------
        transitons: dict
            list containing observation, achieved goal, desired goal,
            action, and/or priorities (stochastic based on TD error)

        Returns:
        --------
        None
        '''
        if self.per: 
            observation, achieved_goal, goal, actions, priorities = transition
        else:
            observation, achieved_goal, goal, actions = transition
        batch_size = observation.shape[0]
        with self.lock:
            if self.counter+batch_size <= self.size:
                index = np.arange(self.counter, self.counter+batch_size)
            elif self.counter < self.size:
                overflow = batch_size - (self.size - self.counter)
                idx_a = np.arange(self.counter, self.size)
                idx_b = np.random.randint(0, self.counter, overflow)
                index = np.concatenate([idx_a, idx_b])
            else:
                index = np.random.randint(0, self.size, batch_size)
            self.counter = min(self.size, self.counter+batch_size)
            self.buffer['observation'][index] = observation
            self.buffer['achieved_goal'][index] = achieved_goal
            self.buffer['goal'][index] = goal
            self.buffer['actions'][index] = actions
            if self.per:
                self.buffer['priority'][index] = priorities

    def sample_buffer(self, batch_size):
        '''
        Samples trajectories from the replay buffer
        
        Parameters:
        -----------
        batchsize: int
            size of minibatch to genrate trajectories

        Returns:
        --------
        transitions: dict
            trajectories with hindsight experience applied 
        '''
        tmp_buffer = {}
        with self.lock:
            for key in self.buffer.keys():
                tmp_buffer[key] = self.buffer[key][:self.counter]
            tmp_buffer['observation_next'] = tmp_buffer['observation'][:,1:,:] # from t+1 to T(assuming t=0)
            tmp_buffer['achieved_goal_next'] = tmp_buffer['achieved_goal'][:,1:,:] # from t+1 to T(assuming t=0)
        transitions = self.sampler(tmp_buffer, batch_size)
        return transitions