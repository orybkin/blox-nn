import torch
from blox.tensor.ops import recursively


class Porch:
    """ An adapter for torch methods that makes all functions recursive.
    To be used as: porch.zeros_like(tensor)
    
    """
    # TODO come up with a better name
    def __getattr__(self, item):
        fn = getattr(torch, item)
        return recursively(fn)
    
    
porch = Porch()