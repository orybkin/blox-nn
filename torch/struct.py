from blox import AttrDict, rmap, rmap_list
from blox.torch import porch

import torch
import os


class Struct(AttrDict):
    """ A struct is the base class for object-oriented programming in pytorch.
    It is a nested dictionary containing tensors. It defines operations to be executed on all of the
    tensors simultaneously. """
    # TODO should this inherit from AttrDict?
    # TODO: make a way for such classes to include a custom init function (and still work with rmap)
    # TODO a simplest way of doing that seems to be to support a to_dict and from_dict function in rmap.
    
    # TODO: make this unpicklable (currently impossible because of inheritance from dict overriden __setitem__
    
    # TODO what is the proper way to query the dict by name (as a variable)? Having an internal dict variable seems good
    
    # TODO use porch?
    def detach(self):
        return porch.detach(self)
    
    def __getitem__(self, *args, **kwargs):
        return rmap(lambda x: x.__getitem__(*args, **kwargs), self)
    
    def __setitem__(self, indices, values):
        return rmap_list(lambda x, y: x.__setitem__(indices, y), [self, values])
    
    def clone(self):
        return porch.clone(self)

    def contiguous(self):
        return rmap(lambda x: x.contiguous(), self)

    @property
    def dict(self):
        """
        A property that can be used to access attributes by square brackets. E.g.:
        
        s = Struct(a=torch.zeros(10))
        s.dict['a'] == torch.zeros(10)
        """
        return DictAccess(self)

    # def __getattr__(self, item):
    #     return getattr(porch, getattr(torch, item))(self)


class DictAccess():
    # A simple access class that forwards item calls to struct's attributes
    def __init__(self, struct):
        super().__setattr__('struct', struct)

    def __getitem__(self, item):
        return dict.__getitem__(self.struct, item)

    def __setitem__(self, key, value):
        return dict.__setitem__(self.struct, key, value)

def save(struct, path):
    # TODO remove this once you can save structures with torch.save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_inputs = rmap(AttrDict, struct, target_class=Struct, only_target=True)
    torch.save(save_inputs, path)
