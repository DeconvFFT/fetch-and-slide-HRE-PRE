from sys import maxsize
import threading
import numpy as np

# replay buffer class
class ReplayBuffer(object):
    def __init__(self, max_size, nS, nA, nG, timestamps, sampler, per):

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
        #index = self.counter % self.size  # added a wrap around condition incase the buffer runs out of space
        if self.per: 
            observation, achieved_goal, goal, actions, priorities = transition
        else:
            observation, achieved_goal, goal, actions = transition
        batch_size = observation.shape[0]
        with self.lock:
            
            index = self._get_storage_idx(inc=batch_size)

            self.buffer['observation'][index] = observation
            self.buffer['achieved_goal'][index] = achieved_goal
            self.buffer['goal'][index] = goal
            self.buffer['actions'][index] = actions
            if self.per:
                print(f'actions.shape: {actions.shape}')
                self.buffer['priority'][index] = priorities

    def sample_buffer(self, batch_size):
        tmp_buffer = {}
        with self.lock:
            for key in self.buffer.keys():
                tmp_buffer[key] = self.buffer[key][:self.counter]
            tmp_buffer['observation_next'] = tmp_buffer['observation'][:,1:,:] # from t+1 to T(assuming t=0)
            tmp_buffer['achieved_goal_next'] = tmp_buffer['achieved_goal'][:,1:,:] # from t+1 to T(assuming t=0)
        transitions = self.sampler(tmp_buffer, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None): ## TODO: remove this and replace with own logic
        inc = inc or 1
        if self.counter+inc <= self.size:
            idx = np.arange(self.counter, self.counter+inc)
        elif self.counter < self.size:
            overflow = inc - (self.size - self.counter)
            idx_a = np.arange(self.counter, self.size)
            idx_b = np.random.randint(0, self.counter, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.counter = min(self.size, self.counter+inc)
        if inc == 1:
            idx = idx[0]
        return idx
    ##TODO: add function documentation for replay buffer class