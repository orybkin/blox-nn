import numpy as np
import torch
from blox.core import optional
from blox.tensor import ndim
from blox.tensor.core import *


def add_n_dims(generalized_tensor, n, dim=-1):
    """ Adds n new dimensions of size 1 to the end of the tensor or array """
    for i in range(n):
        generalized_tensor = ndim.unsqueeze(generalized_tensor, dim)
    return generalized_tensor


def broadcast_final(t1, t2):
    """ Adds trailing dimensions to t1 to match t2 """
    return add_n_dims(t1, len(t2.shape) - len(t1.shape))


def broadcast_initial(t1, t2):
    """ Adds leading dimensions to t1 to match t2 """
    return add_n_dims(t1, len(t2.shape) - len(t1.shape), dim=0)


def get_dim_inds(generalized_tensor):
    """ Returns a tuple 0..length, where length is the number of dimensions of the tensors"""
    return tuple(range(len(generalized_tensor.shape)))


def batchwise_index(generalized_tensor, index, dim=1):
    """ Indexes the tensor with the _index_ along dimension dim.
    Works for numpy arrays and torch tensors
    
    :param generalized_tensor:
    :param index: must be a tensor of shape [batch_size]
    :return tensor t2 such that t2[i] = tensor[i,index[i]]
    """
    
    bs = generalized_tensor.shape[0]
    return ndim.swapaxes(generalized_tensor, 1, dim)[np.arange(bs), index]


def batchwise_assign(tensor, index, value):
    """ Assigns the _tensor_ elements at the _index_ the _value_. The indexing is along dimension 1

    :param tensor:
    :param index: must be a tensor of shape [batch_size]
    :return tensor t2 where that t2[i, index[i]] = value
    """
    bs = tensor.shape[0]
    tensor[np.arange(bs), index] = value


@optional()
def remove_spatial(tensor):
    if len(tensor.shape) == 4 or len(tensor.shape) == 5:
        return tensor.mean(dim=[-1, -2])
    elif len(tensor.shape) == 2:
        return tensor
    else:
        raise ValueError("Are you sure you want to do this? Got tensor shape {}".format(tensor.shape))


def concat_inputs(*inp):
    """ Concatenates tensors together. Used if the tensors need to be passed to a neural network as input. """
    max_n_dims = np.max([len(tensor.shape) for tensor in inp])
    inp = torch.cat([add_n_dims(tensor, max_n_dims - len(tensor.shape)) for tensor in inp], dim=1)
    return inp