import numpy as np
import threading
from mpi4py import MPI


# Code for normaliser adapted from OpenAI baselines normaiser: https://github.com/openai/baselines/blob/master/baselines/her/normalizer.py

class normalizer(object):
    def __init__(self, size, default_clip_range = np.inf, eps = 1e-2):
        '''
        A normaliser that shapes observations into a standard normal distribution with zero mean and unit variance

        Parameters:
        -----------
        size: int 
            Length of observation to be normaized
        
        eps: float
            Constant to avoid underflow issues
        
        default_clip_range: float
            Clipping range to avoid overflow and underflow issues with 
            state space and output of tanh activation function
            Default value: np.inf

        Returns:
        -------
        None
        '''

        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # collect local sum and counts
        self.local_sum = np.zeros(self.size, dtype=np.float32)
        self.local_sumsq = np.zeros(self.size, dtype=np.float32)
        self.local_count = np.zeros(1, dtype=np.float32)

        # collect global sum and counts
        self.global_sum = np.zeros(self.size, dtype=np.float32)
        self.global_sumsq = np.zeros(self.size, dtype=np.float32)
        self.global_count = np.zeros(1, dtype=np.float32)

        # collect mean and std 
        self.mean = np.zeros(self.size, dtype=np.float32)
        self.std = np.zeros(self.size, dtype=np.float32)

        # acquire thread lock
        self.lock = threading.Lock()
    
    def update_params(self, obs):
        '''
        Updates local parameters (sums and count) based on observation

        Parameters:
        -----------
        obs: np.array() 
            Observation from episode
    
        Returns:
        -------
        None
        '''
        obs = obs.reshape(-1, self.size)
        if self.lock:
            self.local_sum += obs.sum(axis=0)
            self.local_sumsq += (np.square(obs)).sum(axis=0)
            self.local_count += obs.shape[0]
    
    def _mpi_average(self, x):
        '''
        Averages data in x across cpus

        Parameters:
        -----------
        x: data to be averaged 

        Returns:
        -------
        buffer: np.array
            Array containing data averaged across cpu workers
        '''

        buffer = np.zeros_like(x)
        ## reduces sum of elements in x to buffer 
        MPI.COMM_WORLD.Allreduce(x, buffer, op=MPI.SUM)
        buffer /= MPI.COMM_WORLD.Get_size()
        return buffer
    
    def sync_params(self, local_sum, local_sumsq, local_count):
        '''
        Syncs sums and count across cpus as averages

        Parameters:
        -----------
        local_sum: np.array
            Array containing local sum for each cpu for the array of observations
        
        local_sumsq: np.array
            Array containing local sum of squares for each cpu for the array of observations
        
        local_count: np.array
            Array containing local count for each cpu for the array of observations

        Returns:
        -------
        local_sum: np.array
            Array containing average sum across cpus for the array of observations
        
        local_sumsq: np.array
            Array containing average sum of squares across cpus for the array of observations
        
        local_count: np.array
            Array containing average count across cpus for the array of observations
        '''

        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)

        return local_sum, local_sumsq, local_count
    
    def recompute_stats(self):
        '''
        Recomputes mean and standard deviation after collecting values across cpus

        Parameters:
        -----------
        None

        Returns:
        -------
        None
        '''

        with self.lock(): # compute stats when we acquire the thread lock
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            local_count = self.local_count.copy()

            # reset local stats 
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
            self.local_count[...] = 0
        
        # synchronise stats from all cpus and set global sums and count
        synced_sum, synced_sumsq, synced_count = self.sync_params(local_sum, local_sumsq, local_count)
        self.global_sum += synced_sum
        self.global_sumsq += synced_sumsq
        self.global_count += synced_count

        # calculate mean and standard deviation
        self.mean = self.global_sum / self.global_count
        self.std =  np.sqrt(np.maximum(np.square(self.eps), (self.global_sumsq / self.global_count) - np.square(self.global_sum / self.global_count)))

    
    def normalize(self, obs, clip_val = None):
        '''
        Converts observation to a standar normal observation and clips the values 
        in a range from [-clip_val to clip_val] to avoid over/under flow issues

        Parameters:
        -----------
        obs: int
            Observed state
        
        clip_val: int
            Value to clip the observation after converting it to standard normal
            Default: np.inf

        Returns:
        -------
        clipped_obs: int
            Standard normal observation with values clipped in the 
            range specified by clip_val
        '''
        if clip_val is None:
            clip_val = self.default_clip_range
        obs_stdnorm = (obs - self.mean)/(self.std)
        clipped_obs =  np.clip(obs_stdnorm, -clip_val, clip_val)
        return clipped_obs



