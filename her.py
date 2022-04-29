# stores transition from her and samples results from her
import numpy as np

class HER(object):
    def __init__(self, k, reward_func = None, per= False):
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
        self.per = per
    
    def sample_transitions(self, buffer_batch, batchsize):
        '''
        Samples transitions from replay buffer(buffer_batch). If
        per is enabled, samples will be selected based on 
        stochastic prioritization using TD error

        Parameters:
        ----------
        buffer_batch: dict 
            A dictionary containing memory buffer from standard experience replay
        
        batchsize: int
            Size of batch to be processed

        Returns:
        -------
        transitions: dict()
            Trajectory samples from replay buffer
        '''
        T = buffer_batch['actions'].shape[1]
        buffer_size = buffer_batch['actions'].shape[0]
        episode_idxs = np.random.randint(0, buffer_size, batchsize) #TODO: change this
        # create samples from 0 to T-1

        trajectories = np.random.randint(T, size=batchsize)
        if self.per:
            priorities = buffer_batch['priority']
            priotity_sum = np.sum(priorities[0])
            prior = [p/priotity_sum for p in priorities[0]]
            trajectories = np.random.choice(T, size=batchsize, p=prior)

        # create transitions from trajectories
        transitions = {k:buffer_batch[k][episode_idxs,trajectories].copy() for k in buffer_batch.keys()}
        # get indices for hindsight experience buffer
        indices = np.where(np.random.uniform(size=batchsize)<self.future_prob)
        future_offset = np.random.uniform(size=batchsize)*(T-trajectories)
        future_offset = future_offset.astype(int)

        future_trajectories = (trajectories+1+future_offset)[indices]
        # set the current reached state after trajectory as a new goal
        future_goal = buffer_batch['achieved_goal'][episode_idxs[indices],future_trajectories]
        transitions['goal'][indices] = future_goal

        # store new rewards from reward function based on the updated goals
        transitions['reward'] = np.expand_dims(self.reward_func(transitions['achieved_goal_next'], transitions['goal'], None), 1)
        transitions = {k: transitions[k].reshape(batchsize, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions
