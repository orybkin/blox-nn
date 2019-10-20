from functools import partial

import torch.nn as nn
from blox.tensor.ops import *


def apply_linear(layer, val, dim):
    """Applies a liner layer to the specified dimension."""
    assert isinstance(layer, nn.Linear)     # can only apply linear layers
    return layer(val.transpose(dim, -1)).transpose(dim, -1)


def make_one_hot(index, length):
    """ Converts indices to one-hot values"""
    oh = index.new_zeros([index.shape[0], length])
    batchwise_assign(oh, index, 1)
    return oh


def batch_cdist(x1, x2, reduction='sum'):
    """ Compute batchwise L2 matrix using quadratic expansion. For each of n vectors in x1, compute L2 norm between it
    and each of m vectors in x2 and outputs the corresponding matrix.
    Adapted from a suggestion somewhere online (look for pytorch github issue comments on cdist).
    
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


def ten2ar(tensor):
    return tensor.detach().cpu().numpy()


def ar2ten(array, device):
    return torch.from_numpy(array).to(device)


def list2ten(list, device=None, dtype=None):
    ten = torch.from_numpy(np.asarray(list))
    if device is not None:
        ten = ten.to(device)
    if dtype is not None:
        ten = torch.Tensor.type(ten, dtype)
    
    return ten


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


def combine_dim(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


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