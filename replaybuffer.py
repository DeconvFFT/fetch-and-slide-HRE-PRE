from sys import maxsize
import numpy as np

# replay buffer class
class ReplayBuffer(object):
    def __init__(self, max_size, nS, nA, nG):

        self.size = max_size
        self.counter = 0
        self.state_memory = np.zeros((self.size,nS), dtype=np.float32)
        self.new_state_memory = np.zeros((self.size,nS), dtype=np.float32)
        self.action_memory = np.zeros((self.size, nA), dtype=np.float32)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.goal_memory = np.zeros((self.size, nG))
        self.terminal_memory = np.zeros(self.size, dtype=np.float32)
    
    def store_transitions(self, state, action, reward, state_, goal, done):
        index = self.counter % self.size # added a wrap around condition incase the buffer runs out of space
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1-int(done)
        self.goal_memory[index] = goal
        self.counter+=1

    def sample_buffer(self, batch_size):
        max_memory = min(self.counter, self.size)
        batch = np.random.choice(max_memory, batch_size)

        state = self.state_memory[batch]
        new_state = self.new_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        goal = self.goal_memory[batch]
        return state, action, reward, new_state, goal, terminal