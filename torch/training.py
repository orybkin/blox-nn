import numpy as np
import torch
from torch.nn.parallel._functions import Gather
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch.utils.data as torchd
from functools import partial


class LossSpikeHook:
    def __init__(self, loss_name):
        self.loss_name = loss_name
    
    def run(self, inputs, output, losses, epoch):
        if self.loss_name in losses.keys():
            pass
        
        
class NanGradHook:
    def __init__(self, trainer):
        self.trainer = trainer
        
    def run(self, inputs, output, losses, epoch):
        # TODO fix the scope here
        self.trainer.nan_grads_hook(inputs, output, losses, epoch)


class NoneGradHook:
    def __init__(self, trainer):
        self.trainer = trainer
    
    def run(self, inputs, output, losses, epoch):
        none_list = [n for n, p in filter(lambda x: x[1] is None, self.trainer.model.named_parameters())]
        if none_list: print(none_list)
        

class RepeatedDataLoader(DataLoader):
    """ A data loader that returns an iterator cycling through data n times """
    def __init__(self, *args, n_repeat=1, **kwargs):
        super().__init__(*args, **kwargs)
        if n_repeat != 1:
            self.batch_sampler = RepeatedSampler(self.batch_sampler, n_repeat)
            
            
class RepeatedSampler(Sampler):
    """ A sampler that repeats the data n times """
    
    def __init__(self, sampler, n_repeat):
        super().__init__(sampler)
        
        self._sampler = sampler
        self.n_repeat = n_repeat
        
    def __iter__(self):
        for i in range(self.n_repeat):
            for elem in self._sampler:
                yield elem

    def __len__(self):
        return len(self._sampler) * self.n_repeat


class DataLoader(torchd.DataLoader):
    def __init__(self, *args, **kwargs):
        # Inititalize the workers with different random seeds. If this is not done, the numpy random seed will be the
        # same for all workers, which can cause less randomness than one hoped for
        
        # The random seed of the global process controls the workers' seeds: the whole process is reproducible
        
        kwargs['worker_init_fn'] = lambda x: np.random.seed(np.random.randint(65536) + x)
        super().__init__(*args, **kwargs)
        
# Alternative implementation:
# DataLoader = partial(torchd.DataLoader, worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))


class DataParallelWrapper(torch.nn.DataParallel):
    """Wraps a pytorch Module for multi-GPU usage but gives access to original model attributes"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def gather(self, outputs, output_device):
        """Overrides the standard gather function to handle custom classes that implement a 'reduce' function."""
        return gather(outputs, output_device, dim=self.dim)


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        try:
            return type(out)(map(gather_map, zip(*outputs)))
        except:
            return type(out).reduce(*outputs)

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class MiniTrainer:
    def __init__(self, model=None, closure=None, parameters=None):
        """
        
        :param model: Either model or parameters have to be specified
        :param closure:
        :param parameters: Either model or parameters have to be specified
        """
        if parameters is None:
            parameters = model.parameters()
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=0.001)
        self.closure = closure
    
    def step(self, i):
        self.optimizer.zero_grad()
        loss = self.closure(i)
        loss.backward()
        if i % 1 == 0: print(loss)
        self.optimizer.step()
    
    def train(self, time):
        [self.step(i) for i in range(time)]


def get_clipped_optimizer(*args, optimizer_type=None, **kwargs):
    assert optimizer_type is not None   # need to set optimizer type!

    class ClipGradOptimizer(optimizer_type):
        def __init__(self, *args, gradient_clip=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.gradient_clip = gradient_clip

        def step(self, *args, **kwargs):
            if self.gradient_clip is not None:
                params = np.concatenate([group['params'] for group in self.param_groups])
                torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)

            super().step(*args, **kwargs)

    return ClipGradOptimizer(*args, **kwargs)
