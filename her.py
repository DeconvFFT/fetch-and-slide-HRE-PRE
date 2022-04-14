# stores transition from her and samples results from her
import numpy as np

class HER(object):
    def __init__(self, k, reward_func = None):
        '''
        Parameters:
        ----------
        k: int 
            Number of timesteps to lookahead in the future
        
        reward_func: function()
            Function to get rewards

        Returns:
        -------
        None
        '''
        self.future_prob = 1 - (1./(1+k))
        self.reward_func = reward_func
    
    def sample_transitions(self, buffer_batch, batchsize):
        T = buffer_batch['actions'].shape[0]
        
        # create samples from 0 to T-1
        trajectories = np.random.randint(T-1, size=batchsize)
        # create transitions from trajectories
        transitions = {k:buffer_batch[k][trajectories].copy() for k in buffer_batch.keys()}
        # get indices for hindsight experience buffer
        random_batch = np.random.uniform(size=batchsize)
        indices = np.where(random_batch<self.future_prob)
        future_offset = random_batch*(T-trajectories)
        future_offset = future_offset.astype(int)

        future_trajectories = (trajectories+1+future_offset)[indices]

        # set the current reached state after trajectory as a new goal
        future_goal = buffer_batch['g_'][future_trajectories]
        transitions['g'][indices] = future_goal

        # store new rewards from reward function based on the updated goals
        transitions['r'][indices] = np.expand_dims(self.reward_func(transitions['s_'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshpe(batchsize, *transitions[k].shape[1:]) for k in transitions.keys()}
    
    def set_hindsight(self, buffer):
        traj_len = len(buffer['actions'])
        new_goal = buffer['s_'][-1] ## S[T] is the new goal from trajectory 0...T

        her_buffer = buffer.copy()
        # need to collect rewards and goals
        her_buffer['r'] = np.zeros(traj_len)
        for i in range(traj_len):
            her_buffer['g'][i] = new_goal ## S[T] is the new goal for each state in the trajectory
            if i == traj_len-1 : ## end of episodes
                 her_buffer['r'][i] = 0
            else:
                her_buffer['r'][i] = -1
            her_buffer['g_'] = her_buffer['g'][1:, :] # setting next goals
        
        return her_buffer