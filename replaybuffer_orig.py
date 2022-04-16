from sys import maxsize
import numpy as np

# replay buffer class
class ReplayBuffer(object):
    def __init__(self, max_size, nS, nA, nG):

        self.maxsize = max_size
        self.size = 0
        self.index = 0
        self.state_memory = np.zeros((self.size,nS), dtype=np.float32)
        self.new_state_memory = np.zeros((self.size,nS), dtype=np.float32)
        self.action_memory = np.zeros((self.size, nA), dtype=np.float32)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.goal_memory = np.zeros((self.size, nG))
        self.terminal_memory = np.zeros(self.size, dtype=np.float32)
    
    def store_transitions(self, state, action, reward, state_, goal, done):
        self.state_memory[self.index] = state
        self.new_state_memory[self.index] = state_
        self.reward_memory[self.index] = reward
        self.action_memory[self.index] = action
        self.terminal_memory[self.index] = done
        self.goal_memory[self.index] = goal
        self.index = (self.index+1) % self.size # added a wrap around condition incase the buffer runs out of space

        self.size = min(self.counter+1, self.maxsize)

    def sample_buffer(self, batch_size):
        
        batch = np.random.choice(self.size, batch_size)

        state = self.state_memory[batch]
        new_state = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        goals = self.goal_memory[batch]
        transitions = {
            'state': state,
            'next_state':new_state,
            'actions': actions,
            'reward': rewards,
            'goal':goals,
            'terminals': terminals,

        }
        return transitions

    ##TODO: add function documentation for replay buffer class