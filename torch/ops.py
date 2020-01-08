from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from blox.tensor import ndim
from blox.tensor.ops import *
from blox.torch.core import *

def apply_linear(layer, val, dim):
    """Applies a liner layer to the specified dimension."""
    assert isinstance(layer, nn.Linear)     # can only apply linear layers
    return layer(val.transpose(dim, -1)).transpose(dim, -1)


def make_one_hot(index, length):
    """ Converts indices to one-hot values"""
    oh = index.new_zeros([index.shape[0], length])
    batchwise_assign(oh, index, 1)
    return oh


@ndim.numpied
def make_one_hot_image(index, height, width):
    """ Converts 2-d indices to one-hot images"""

    # Mask for the one values
    mask_h = add_n_dims(torch.arange(height), len(index.shape) - 1, dim=0)[..., :, None]
    mask_w = add_n_dims(torch.arange(width), len(index.shape) - 1, dim=0)[..., None, :]
    mask = (mask_h == index[..., 0, None, None]) & (mask_w == index[..., 1, None, None])
    
    image = index.new_zeros(index.shape[:-1] + (height, width))
    image[mask] = 1
    # Add channels
    image = image[..., None, :, :].repeat_interleave(3, -3)
    
    return image


def first_nonzero(tensor, dim=-1):
    """ Returns the first nonzero element of the tensor """

    mask = (tensor != 0).float()
    weight = tensor2indices(tensor, dim).float().flip(dim)
    
    return (mask * weight).argmax(dim)
    
    
def tensor2indices(tensor, dim):
    """ Returns a tensor with the shape of 'tensor' but values substituted with index along 'dim' """
    l = len(tensor.shape)
    if dim < 0:
        dim = l + dim
    
    indices = torch.arange(tensor.shape[dim], device=tensor.device)
    indices = add_n_dims(add_n_dims(indices, n=dim, dim=0), n=l - dim - 1)
    indices = indices.expand(tensor.shape)
    
    return indices


def batch_cdist(x1, x2, reduction='sum'):
    """ Compute batchwise L2 norm matrix: for each of n vectors in x1, compute L2 norm between it
    and each of m vectors in x2 and outputs the corresponding matrix.
    This is a memory-efficient version using quadratic expansion, adapted from a suggestion somewhere online
    (look for pytorch github issue comments on cdist). It does not require duplicating the arrays, and only scales
    with max(n,m) as opposed to n x m memory size.
    
    :param x1: a tensor of shape batchsize x n x dim
    :param x2: a tensor of shape batchsize x m x dim
    :return: a tensor of distances batchsize x n x m
    """
    x1 = x1.flatten(start_dim=2)
    x2 = x2.flatten(start_dim=2)
    
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

    # the einsum is broken, and probably will also be slower
    # torch.einsum('einlhw, eitlhw->nt', torch.stack([x, torch.ones_like(x)]), torch.stack([torch.ones_like(y), y]))
    res = torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)

    # Zero out negative values
    res.clamp_min_(0)
    if reduction == 'mean':
        res = res / x1.shape[2]
    elif reduction == 'sum':
        pass
    else:
        raise NotImplementedError
    return res


def cdist(x1, x2, reduction='sum'):
    return batch_cdist(x1[None], x2[None], reduction=reduction)[0]


def list2ten(list, device=None, dtype=None):
    ten = torch.from_numpy(np.asarray(list))
    if device is not None:
        ten = ten.to(device)
    if dtype is not None:
        ten = torch.Tensor.type(ten, dtype)
    
    return ten


def dim2list(tensor, dim):
    return [t.squeeze(dim) for t in torch.split(tensor, 1, dim)]


def mask_out(tensor, start_ind, end_ind, value, dim=1):
    """ Set the elements before start_ind and after end_ind (both inclusive) to the value. """
    if dim != 1:
        raise NotImplementedError
    
    batch_size, time = list(tensor.shape)[:2]
    # (oleg) This creates the indices every time, but doesn't seem to affect the speed a lot.
    inds = torch.arange(time, device=tensor.device, dtype=start_ind.dtype).expand(batch_size, -1)
    mask = (inds >= end_ind[:, None]) | (inds <= start_ind[:, None])
    tensor[mask] = value
    return tensor, mask


def log_sum_exp(tensor, dim=-1):
    """ Safe log-sum-exp operation """
    return torch.logsumexp(tensor, dim)
    #
    # max = tensor.max(dim)[0]
    # if torch.isinf(max).any():
    #     if not torch.isinf(max).all():
    #         raise NotImplementedError('perhaps I should implement masking for mixed cases')
    #     return tensor.exp().sum(dim).log()
    #
    # return (tensor - max[..., None]).exp().sum(dim).log() + max


def combine_dim(x, dim_begin, dim_end=None):
    if dim_end is None: dim_end = len(x.shape)
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.reshape(combined_shape)


def like(func, tensor):
    return partial(func, device=tensor.device, dtype=tensor.dtype)


def slice_tensor(tensor, start, step, dim):
    """ I couldn't find this in pytorch... """
    if dim == 0:
        return tensor[start::step]
    elif dim == 1:
        return tensor[:, start::step]
    else:
        raise NotImplementedError
    

def pad_to(tensor, size, dim=-1, mode='back'):
    # TODO: padding to the back somehow pads to the beginning
    kwargs = dict()
    
    padding = size - tensor.shape[dim]
    if mode == 'front':
        kwargs['pad_front'] = padding
    else:
        kwargs['pad_back'] = padding
        
    return pad(tensor, **kwargs, dim=dim)


def pad(generalized_tensor, pad_front=0, pad_back=0, dim=-1):
    """ Pads a tensor at the specified dimension"""
    l = len(generalized_tensor.shape)
    if dim < 0:
        dim = l + dim
  
    if isinstance(generalized_tensor, torch.Tensor):
        # pad takes dimensions in reversed order for some reason
        size = (l - dim - 1) * 2 * [0] + [pad_front, pad_back] + (dim) * 2 * [0]
        return F.pad(generalized_tensor, size)
    elif isinstance(generalized_tensor, np.ndarray):
        size = (dim) * 2 * [0] + [pad_front, pad_back] + (l - dim - 1) * 2 * [0]
        size = list(zip(size[::2], size[1::2]))
        return np.pad(generalized_tensor, size, mode='constant')