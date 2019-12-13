import torch
import torch.nn as nn
from contextlib import contextmanager

from blox.torch.dist import ProbabilisticModel


class BaseModel(nn.Module):
    def call_children(self, fn, cls):
        def conditional_fn(module):
            if isinstance(module, cls):
                getattr(module, fn).__call__()
        
        self.apply(conditional_fn)
        
    def apply_to(self, fn, cls):
        def conditional_fn(module):
            if isinstance(module, cls):
                fn(module)

        self.apply(conditional_fn)
        
    @contextmanager
    def prior_mode(self):
        """ Changes the mode to sampling from the prior. To be used like: with model.val_mode(): ...<do something>..."""
        self.call_children('switch_to_prior', ProbabilisticModel)
        yield
        self.call_children('switch_to_inference', ProbabilisticModel)
