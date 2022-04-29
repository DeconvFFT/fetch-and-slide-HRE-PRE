from mpi4py import MPI
import numpy as np
import torch

# gradient sync code inspired from : https://github.com/openai/baselines/tree/master/baselines/her
def sync_networks(network):
    '''
    Broadcasts parameters from actor and critic networks across cpus and sets parameters 
    for actor and critic networks across cpus
    Parameters:
    ----------
    network: torch.nn
        Actor/Critic network for which we want to synchronize parameters
    Returns:
    --------
    None
    '''
    flat_params = get_flat_prarms_or_grad(network, mode='params')
    MPI.COMM_WORLD.Bcast(flat_params, root=0)
    set_flat_params_or_grad(network, flat_params, mode='params')

def sync_grads(network):
    '''
    Synchronizes gradients across all cpus after gradient backpropagation occurs 
    on actor and critic networks
    Parameters:
    ----------
    network: torch.nn
        Actor/Critic network for which we want to synchronize gradients
    Returns:
    --------
    None
    '''
    flat_grads = get_flat_prarms_or_grad(network, mode='grads')
    global_grads = np.zeros_like(flat_grads)
    MPI.COMM_WORLD.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    set_flat_params_or_grad(network, global_grads, mode='grads')

def get_flat_prarms_or_grad(network, mode='params'):
    '''
    Extracts gradients or parameters from the network
    Parameters:
    ----------
    network: torch.nn
        Actor/Critic network for which we want to get parameters or gradients
    Returns:
    --------
    data: list()
        List of parameters or gradients from the network
    '''
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])

def set_flat_params_or_grad(network, flat_params, mode='params'):
    '''
    Sets the parameters or gradients or the networks
    Parameters:
    ----------
    network: torch.nn
        Actor/Critic network for which we want to set parameters or gradients
    Returns:
    --------
    None
    '''
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()