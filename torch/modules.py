import numpy as np
import torch
import torch.nn as nn
from blox.tensor.ops import concat_inputs
from blox import AttrDict, batch_apply
from blox.basic_types import map_dict


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        return AttrDict()
    
    def loss(self, *args, **kwargs):
        return {}


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class AttrDictPredictor(nn.ModuleDict):
    """ Holds a dictionary of modules and applies them to return an output dictionary """
    def forward(self, *args, **kwargs):
        output = AttrDict()
        for key in self:
            output[key] = self[key](*args, **kwargs)
        return output


class GetIntermediatesSequential(nn.Sequential):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, input):
        """Computes forward pass through the network outputting all intermediate activations with final output."""
        skips = []
        for i, module in enumerate(self._modules.values()):
            input = module(input)
            
            if i % self.stride == 0:
                skips.append(input)
            else:
                skips.append(None)
        return input, skips[:-1]


class SkipInputSequential(nn.Sequential):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        
    def forward(self, input, skips):
        """Computes forward pass through the network and concatenates input skips."""
        skips = [None] + skips[::-1]   # start applying skips after the first decoding layer
        for i, module in enumerate(self._modules.values()):
            if i < len(skips) and skips[i] is not None:
                input = torch.cat((input, skips[i]), dim=1)
                
            input = module(input)
        return input


class ConcatSequential(nn.Sequential):
    """ A sequential net that accepts multiple arguments and concatenates them along dimension 1
    The class also broadcasts the tensors to fill last dimensions.
    """
    def __init__(self, *args, detached=False):
        super().__init__(*args)
        self.detached = detached
    
    def forward(self, *inp):
        inp = concat_inputs(*inp)
        if self.detached:
            inp = inp.detach()
        return super().forward(inp)


class Batched(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, *args):
        return batch_apply(args, self.net, separate_arguments=True)


class Updater(nn.Module):
    """ A class for non-optimization updates. An Updater defines a 'step' which is called every training step.
    
    This is implemented as a module so that all updaters that are class fields are easily accessible.
    """
    def __init__(self):
        self.it = 0
        super().__init__()
    
    def step(self):
        self.it += 1


class ExponentialDecayUpdater(Updater):
    def __init__(self, parameter, n_iter, update_freq=10, min_limit=-np.inf):
        """
        Decays the parameter such that every n_iter the parameter is reduced by 10.
        
        :param parameter:
        :param n_iter:
        :param update_freq:
        """
        super().__init__()
        
        assert parameter.numel() == 1
        assert not parameter.requires_grad
        self.parameter = parameter
        self.update_freq = update_freq
        self.min_limit = min_limit
        
        self.decay = self.determine_decay(n_iter, update_freq)
        
    def determine_decay(self, n_iter, update_freq):
        n_updates = n_iter / update_freq
        decay = 0.1 ** (1 / n_updates)
        
        return decay
    
    def step(self):
        if self.it % self.update_freq == 0 and self.parameter.data[0] * self.decay > self.min_limit:
            self.parameter.data[0] = self.parameter.data[0] * self.decay
        super().step()


class LinearUpdater(Updater):
    def __init__(self, parameter, n_iter, target, update_freq=10, name=None):
        """
        Linearly interpolates the parameter between the current and target value during n_iter iterations

        """
        super().__init__()
        
        assert parameter.numel() == 1
        assert not parameter.requires_grad
        self.parameter = parameter
        self.update_freq = update_freq
        self.n_iter = n_iter
        self.target = target
        self.name = name
    
    @property
    def upd(self):
        n_updates = (self.n_iter - self.it) / self.update_freq
        upd = (self.target - self.parameter.data[0]) / n_updates
        
        return upd

    def step(self):
        if self.it % self.update_freq == 0 and self.it < self.n_iter:
            self.parameter.data[0] = self.parameter.data[0] + self.upd
        super().step()

    def log_outputs_stateful(self, step, log_images, phase, logger):
        if self.name:
            logger.log_scalar(self.parameter, self.name, step, phase)


class ConstantUpdater(Updater):
    def __init__(self, parameter, n_iter, name=None):
        """
        Keeps the parameter constant for n_iter
        """
        super().__init__()
        
        assert parameter.numel() == 1
        # assert not parameter.requires_grad
        self.parameter = parameter
        self.n_iter = n_iter
        self.name = name
        self.val = parameter.data
        
    def step(self):
        # TODO this should depend on the global step
        if self.it < self.n_iter:
            self.parameter.data = self.val.to(self.parameter.device)
        
        super().step()
    
    def log_outputs_stateful(self, step, log_images, phase, logger):
        if self.name:
            logger.log_scalar(self.parameter, self.name, step, phase)
            

def num_parameters(model, level=0):
    """  Returns the number of parameters used in a module.
    
    Known bug: if some of the submodules are repeated, their parameters will be double counted
    :param model:
    :param level: if level==1, returns a dictionary of submodule names and corresponding parameter counts
    :return:
    """
    
    if level == 0 or len(model.named_children()) == 0:
        return sum([p.numel() for p in model.parameters()])
    else:
        return map_dict(lambda x: num_parameters(x, level - 1), dict(model.named_children()))

