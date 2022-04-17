import numpy as np
import random
import copy
class OUNoise(object): 
    def __init__(self,size,seed,mu=0., theta=0.15, sigma=0.2):
        self.theta = theta
        self.mu = mu*np.ones(size)
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def __call__(self):
        # x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        #     self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        # self.x_prev = x
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        #return x
    
    def reset(self):
        #self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.state = copy.copy(self.mu)