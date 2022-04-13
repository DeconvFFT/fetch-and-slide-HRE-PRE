import numpy as np
import argparse
from train import *
from test import *

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000, help='Number of epidoes')
    parser.add_argument('--maxsteps', type=int, default=500, help='Maximum number of time stamps per episode')
    parser.add_argument('--buffersize', type=int, default=1e6, help='Size of replay buffer')
    parser.add_argument('--mode', type=str, default='train', help='Train or eval mode')
    parser.add_argument('--episodes', type=int, default=50, help='Number of evaluation epidoes')
    parser.add_argument('--future', type=int, default=4, help='How many future episodes to consider')
    parser.add_argument('--algorithm',type=str, default='ddpg', help='Algorithm to use for training the agent')
    parser.add_argument('--actor_lr', type=float, default=0.0001, help='Learning rate for actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='Learning rate for critic')
    parser.add_argument('--clip_range', type=int, default=5, help='Clipping range for inputs')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount Factor')

    args = parser.parse_args()

    if args.mode =='train':
        train_agent(args)
    else:
        test_agent(args)

