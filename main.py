import numpy as np
import argparse
from train import *
from test import *
from herwithddpg import HERDDPG
import random
def create_environment(envname):
    '''
    Creates openai gym environments and adds some properties
    like number of actions, number of states, number of 
    goals and maximum timestamps per episode to the environment.
    
    Parameters:
    -----------
    envname: str
        name of environment
    
    Returns:
    --------
    env: gym.env
        OpenAi gym environment
    '''
    env = gym.make(envname)
    env.nA = env.action_space.shape[0]
    env.nG = env.observation_space['achieved_goal'].shape[0]
    env.nS = env.observation_space['observation'].shape[0]
    env.maxtimestamps = env._max_episode_steps
    return env

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # default arguments set as per OpenAI baselines: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/her/experiment/config.py#L182
    parser.add_argument('--episodes', type=int, default=200, help='Number of epidoes')
    parser.add_argument('--cycles', type=int, default=50, help='Number of cycles/episode')
    parser.add_argument('--optimsteps', type=int, default=40, help='Number of optimisation steps in each cycle')
    parser.add_argument('--buffersize', type=int, default=int(1e6), help='Size of replay buffer')
    parser.add_argument('--mode', type=str, default='train', help='Train or eval mode')
    parser.add_argument('--np_seed', type=int, default=123, help='random seed for random modules in numpy and random')
    parser.add_argument('--env_seed', type=int, default=123, help='random seed for environment')
    parser.add_argument('--torch_seed', type=int, default=123, help='random seed for torch')

    parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation epidoes')
    parser.add_argument('--future_episodes', type=int, default=4, help='How many future episodes to consider')
    parser.add_argument('--actor_lr', type=float, default=0.001, help='Learning rate for actor')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='Learning rate for critic')
    parser.add_argument('--cliprange', type=int, default=5, help='Clipping range for normalised inputs')
    parser.add_argument('--clip_observation', type=int, default=200, help='Clipping range for inputs')

    parser.add_argument('--gamma', type=float, default=0.98, help='Discount Factor')
    parser.add_argument('--tau', type=float, default=0.05, help='Proportion of actual model to be used in target model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--fc1_dims', type=int, default=256, help='Dimensions for fc1')
    parser.add_argument('--fc2_dims', type=int, default=256, help='Dimensions for fc2')
    parser.add_argument('--fc3_dims', type=int, default=256, help='Dimensions for fc3')
    parser.add_argument('--envname', type=str, default='FetchSlide-v1', help='Name of the environment')
    parser.add_argument('--rollouts', type=int, default=2, help='Number of rollouts')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='Probabilityfor OU noise')
    parser.add_argument('--random_prob', type=float, default=0.3, help='Probability for selecting random actions')
    parser.add_argument('--per', type=bool, default=False, help='Use PER or Not')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Small constant to allow visiting transitions even when their TD error is zero')
    parser.add_argument('--alpha', type=float, default=0.5, help='How much priortization to use')


    # parse arguments
    args = parser.parse_args()

    # create environment
    envname = args.envname
    env = create_environment(envname)
    
    # set random, environment and torch seeds
    env.seed(args.env_seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.np_seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.np_seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.torch_seed + MPI.COMM_WORLD.Get_rank())

    # create agent
    agent = HERDDPG(args.actor_lr, args.critic_lr, args.tau,env, envname, args.gamma, args.buffersize, args.fc1_dims, args.fc2_dims, args.fc3_dims, args.cliprange,args.clip_observation,args.future_episodes,args.batch_size, args.per)
    
    # set environment variables for mpi
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # train/test agent
    if args.mode =='train':
        train_agent(args,env, agent)
    else:
        test_agent(args, env, agent)

