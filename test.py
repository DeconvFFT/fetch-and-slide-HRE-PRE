import os
import numpy as np
import torch
import torch.nn as nn
import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring import video_recorder

def clip_inputs(observation, goal,cliprange ):
    '''
    Clips inputs into range (-cliprange, cliprange)
    Parameters:
    ----------
    observation: list()
        25d list containing information about current state

    goal: list()
        3d list containing goal position of the robot arm

    cliprange: int
        observation and goal will be clipped in the range (-cliprange, cliprange)

    Returns:
    -------
    obs: list()
        observation clipped using cliprange
    goal: list()
        goal clipped using cliprange
    '''
    obs, goal = np.clip(observation, -cliprange, cliprange), \
        np.clip(goal, -cliprange, cliprange)
    return obs, goal

def normalise_and_clip(obs, goal,obs_mean, goal_mean, obs_std, goal_std,cliprange):
    '''
    Normalises observations using means and standard deviaiton and then
    clips them into a range (-cliprange, cliprange).

    Parameters:
    ----------
    obs: list()
        25d list containing information about current state
    
    goal: list()
        3d list containing goal position of the robot arm

    obs_mean: float32
        Mean of observations

    goal_mean: float32
        Mean of goals

    obs_std: float32
        Standard Deviation of observations

    goal_std: float32
        Standard Deviation of goals

    cliprange: int
        range to clip observation and goal

    Returns:
    -------
    obs_norm: list()
        25d list with normalised observations clipped with cliprange

    goal_norm: list()
        3d list with normalised goals clipped with cliprange
    '''
    obs_norm = (obs-obs_mean) / obs_std
    goal_norm = (goal-goal_mean)/goal_std
    obs_norm = np.clip(obs_norm, -cliprange, cliprange)
    goal_norm = np.clip(goal_norm, -cliprange, cliprange)
    return obs_norm,goal_norm

def concat_inputs(obs, goal):
    '''
    Concatenates observation and goal 
    and returns resulting tensor

    Parameters:
    ----------
    obs: list()
        25d list with state information

    goal: list()
        3d list with goal information

    Returns:
    -------
    inputs: torch.tensor
        torch tensor with observation and goal concatenated
    '''
    inputs = np.concatenate([obs, goal])
    return torch.tensor(inputs, dtype=torch.float32)

def test_agent(args, env,agent):
    '''
    Parameters:
    ----------
    args: cmd arguments
        arguments passed from CLI
    
    env: gym.env
        OpenAi gym environment
    
    agent: HERDDPG 
        HERDDPG
        
    Returns:
    -------
    None
    '''
    path = 'models/per_colab/different_seed/actor_per_140'
    video_path = './video/'+path.split('models/')[1]
    print(video_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    vid = video_recorder.VideoRecorder(env,path=video_path+"/vid.mp4")

    obs_mean, obs_std, goal_mean, goal_std, state_dict = torch.load(path, map_location=lambda storage, loc: storage) 
    
    # create actor network and load state dict
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    success_rate = []
    for ep in range(args.eval_episodes):

        observation = env.reset()
        obs = observation['observation']
        goal = observation['desired_goal']
        ep_success = []
        for t in range(env.maxtimestamps):
            vid.capture_frame()
            obs, goal = clip_inputs(obs, goal, args.clip_observation)
            obs_norm, goal_norm = normalise_and_clip(obs, goal, obs_mean, goal_mean, obs_std, goal_std, args.cliprange)
            inputs = concat_inputs(obs_norm, goal_norm)
            with torch.no_grad():
                pi = agent.actor(inputs)
            actions = pi.cpu().detach().numpy().squeeze()
            next_observation, reward, done, info = env.step(actions)
            obs = next_observation['observation']
            ep_success.append(info['is_success'])
        ep_mean_success = np.mean(ep_success)
        success_rate.append(ep_mean_success)
        print(f'epoch: {ep} -> succes rate: {ep_mean_success}')
    print(f'overall success rate: {np.mean(success_rate)}')





