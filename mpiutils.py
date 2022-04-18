from mpi4py import MPI
import numpy as np
import torch

def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    flat_params = get_flat_prarms_or_grad(network, mode='params')
    MPI.COMM_WORLD.Bcast(flat_params, root=0)
    set_flat_params_or_grad(network, flat_params, mode='params')

def sync_grads(network):
    flat_grads = get_flat_prarms_or_grad(network, mode='grads')
    global_grads = np.zeros_like(flat_grads)
    MPI.COMM_WORLD.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    set_flat_params_or_grad(network, global_grads, mode='grads')

def get_flat_prarms_or_grad(network, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])

def set_flat_params_or_grad(network, flat_params, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()