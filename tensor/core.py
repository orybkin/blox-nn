from functools import partial

from blox.basic_types import map_dict, listdict2dictlist
import torch


def batch_apply(tensors, fn, separate_arguments=False, unshape_inputs=False):
    """ Applies the fn to the tensors while treating two first dimensions of tensors as batch.

    :param tensors: can be a single tensor, tuple or a list.
    :param fn: _fn_ can return a single tensor, tuple or a list
    :param separate_arguments: if true, the highest-level list will be fed into the function as a
    list of arguments
    :param unshape_inputs: if true, reshapes the inputs back to original (in case they have references to classes)"""
    
    reference_tensor = find_tensor(tensors, min_dim=2)
    if not isinstance(reference_tensor, torch.Tensor):
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


def find(inp, success_fn):
    """ Finds an element for which the success_fn responds true """

    def rec_find(structure):
        if isinstance(structure, list) or isinstance(structure, tuple):
            items = list(structure)
        elif isinstance(structure, dict):
            items = list(structure.items())
        else:
            # Base case, if not dict or iterable
            success = success_fn(structure)
            return structure, success
    
        # If dict or iterable, iterate and find a tensor
        for item in items:
            result, success = rec_find(item)
            if success:
                return result, success

        return None, False
    
    return rec_find(inp)[0]


def find_tensor(structure, min_dim=None):
    """ Finds a single tensor in the structure """
    def success_fn(x):
        success = isinstance(x, torch.Tensor)
        if min_dim is not None:
            success = success and len(x.shape) >= min_dim
            
        return success
    
    return find(structure, success_fn)


find_element = lambda structure: find(structure, success_fn=lambda x: True)


def make_recursive(fn, *argv, target_class=torch.Tensor, strict=False, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """
    
    def recursive_map(tensors):
        if isinstance(tensors, target_class):
            return fn(tensors, *argv, **kwargs)
        elif tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        else:
            try:
                assert not strict
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


def map_recursive(fn, tensors, **kwargs):
    return make_recursive(fn, **kwargs)(tensors)


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)


rmap = map_recursive
rmap_list = map_recursive_list
recursively = make_recursive