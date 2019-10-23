import numpy as np
import torch
from blox.tensor import ndim
from blox.basic_types import map_dict, listdict2dictlist
from blox.core import optional


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


def batch_apply(tensors, fn, separate_arguments=False, unshape_inputs=False, ):
    """ Applies the fn to the tensors while treating two first dimensions of tensors as batch.

    :param tensors: can be a single tensor, tuple or a list.
    :param fn: _fn_ can return a single tensor, tuple or a list
    :param separate_arguments: if true, the highest-level list will be fed into the function as a
    list of arguments
    :param unshape_inputs: if true, reshapes the inputs back to original (in case they have references to classes)"""
    
    reference_tensor, success = find_tensor(tensors, min_dim=2)
    if not success:
        raise ValueError("couldn't find a reference tensor")
    
    batch, time = reference_tensor.shape[:2]
    reshape_to = make_recursive(lambda tensor: tensor.view([batch * time] + list(tensor.shape[2:])))
    reshape_from = make_recursive(lambda tensor: tensor.view([batch, time] + list(tensor.shape[1:])))
    
    input_reshaped = reshape_to(tensors)
    if separate_arguments:
        if isinstance(input_reshaped, dict):
            output = fn(**input_reshaped)
        else:
            output = fn(*input_reshaped)
    else:
        output = fn(input_reshaped)
    output_reshaped_back = reshape_from(output)
    if unshape_inputs: reshape_from(input_reshaped)
    return output_reshaped_back


def find_tensor(tensors, min_dim=None):
    """ Finds a single tensor in the structure """

    if isinstance(tensors, list) or isinstance(tensors, tuple):
        tensors_items = list(tensors)
    elif isinstance(tensors, dict):
        tensors_items = list(tensors.items())
    else:
        # Base case, if not dict or iterable
        success = isinstance(tensors, torch.Tensor)
        if min_dim is not None: success = success and len(tensors.shape) >= min_dim
        return tensors, success

    # If dict or iterable, iterate and find a tensor
    for tensors_item in tensors_items:
        tensors_result, success = find_tensor(tensors_item)
        if success:
            return tensors_result, success

    return None, False


def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """
    
    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))
    
    return recursive_map


def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)


rmap = map_recursive
rmap_list = map_recursive_list
recursively = make_recursive


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