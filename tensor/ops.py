import numpy as np
import torch
import torch.nn.functional as F
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
    if len(index.shape) == 1:
        bs = generalized_tensor.shape[0]
        return ndim.swapaxes(generalized_tensor, 1, dim)[np.arange(bs), index]
    else:
        # If index is two-dimensional, retrieve multiple values per batch element
        bs, queries = index.shape
        ar_array = np.arange(bs)[:, None].repeat(queries, 1)
        
        out = ndim.swapaxes(generalized_tensor, 1, dim)[ar_array.reshape(-1), index.reshape(-1)]
        return out.reshape([bs, queries] + list(out.shape[1:]))


def batchwise_assign(tensor, index, value):
    """ Assigns the _tensor_ elements at the _index_ the _value_. The indexing is along dimension 1

    :param tensor:
    :param index: must be a tensor of shape [batch_size]
    :return tensor t2 where that t2[i, index[i]] = value
    """
    bs = tensor.shape[0]
    tensor[np.arange(bs), index] = value


def one_hot(a, length):
  return np.eye(length)[a.reshape(-1)]


@optional()
def remove_spatial(tensor):
    if len(tensor.shape) == 4 or len(tensor.shape) == 5:
        return tensor.mean(dim=[-1, -2])
    elif len(tensor.shape) == 2:
        return tensor
    else:
        raise ValueError("Are you sure you want to do this? Got tensor shape {}".format(tensor.shape))


def concat_inputs(*inp, dim=1):
    """ Concatenates tensors together. Used if the tensors need to be passed to a neural network as input. """
    inp = list(filter(lambda ten: ten is not None, inp))
    max_n_dims = np.max([len(tensor.shape) for tensor in inp])
    inp = torch.cat([add_n_dims(tensor, max_n_dims - len(tensor.shape)) for tensor in inp], dim=dim)
    return inp


def pad_to(tensor, size, dim=-1, mode='back', value=0):
    # TODO: padding to the back somehow pads to the beginning
    kwargs = dict()
    
    padding = size - tensor.shape[dim]
    if mode == 'front':
        kwargs['pad_front'] = padding
    elif mode == 'back':
        kwargs['pad_back'] = padding
    elif mode == 'equal':
        kwargs['pad_front'] = int(padding / 2 + 0.5)
        kwargs['pad_back'] = int(padding / 2)
    
    return pad(tensor, **kwargs, dim=dim, value=value)


def pad(generalized_tensor, pad_front=0, pad_back=0, dim=-1, value=0):
    """ Pads a tensor at the specified dimension"""
    l = len(generalized_tensor.shape)
    if dim < 0:
        dim = l + dim
    
    if isinstance(generalized_tensor, torch.Tensor):
        # pad takes dimensions in reversed order for some reason
        size = (l - dim - 1) * 2 * [0] + [pad_front, pad_back] + (dim) * 2 * [0]
        return F.pad(generalized_tensor, size, value=value)
    elif isinstance(generalized_tensor, np.ndarray):
        size = (dim) * 2 * [0] + [pad_front, pad_back] + (l - dim - 1) * 2 * [0]
        size = list(zip(size[::2], size[1::2]))
        return np.pad(generalized_tensor, size, mode='constant', constant_values=value)
    
    
def pad_sequences(list, dim=1, value=0):
    """ Pads a list of tensors to have equal length at the dimension `dim` """
    length = np.max([seq.shape[dim] for seq in list])
    list = [pad_to(seq, length, dim, value=value) for seq in list]
    return list
