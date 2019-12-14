from blox import AttrDict, rmap, rmap_list
from blox.torch import porch


class Struct(AttrDict):
    """ A struct is the base class for object-oriented programming in pytorch.
    It is a nested dictionary containing tensors. It defines operations to be executed on all of the
    tensors simultaneously. """
    # TODO should this inherit from AttrDict?
    # TODO: make a way for such classes to include a custom init function (and still work with rmap)
    # TODO a simplest way of doing that seems to be to support a to_dict and from_dict function in rmap.
    
    # TODO use porch?
    def detach(self):
        return porch.detach(self)
    
    def __getitem__(self, *args, **kwargs):
        return rmap(lambda x: x.__getitem__(*args, **kwargs), self)
    
    def __setitem__(self, indices, values):
        return rmap_list(lambda x, y: x.__setitem__(indices, y), [self, values])
    
    def clone(self):
        return porch.clone(self)
