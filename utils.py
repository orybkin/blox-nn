import functools
import heapq
import time
from contextlib import contextmanager

from blox.tensor.core import map_recursive, map_recursive_list
from blox import AttrDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]


class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)


@contextmanager
def dummy_context():
    yield


@contextmanager
def timing(text, name=None, interval=10):
    start = time.time()
    yield
    elapsed = time.time() - start
    
    if name:
        if not hasattr(timing, name):
            setattr(timing, name, AverageMeter())
        meter = getattr(timing, name)
        meter.update(elapsed)
        if meter.count % interval == 0:
            print("{} {}".format(text, meter.avg))
        return
            
    print("{} {}".format(text, elapsed))
    
    
class timed:
    """ A function decorator that prints the elapsed time """
    
    def __init__(self, text):
        """ Decorator parameters """
        self.text = text
    
    def __call__(self, func):
        """ Wrapping """
        
        def wrapper(*args, **kwargs):
            with timing(self.text):
                result = func(*args, **kwargs)
            return result
        
        return wrapper


def lazy_property(function):
    """ Caches the property such that the code creating it is only executed once.
    Adapted from Dani Hafner (https://danijar.com/structuring-your-tensorflow-models/) """
    # TODO can I just use lru_cache?
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class HasParameters:
    def __init__(self, **kwargs):
        self.build_params(kwargs)
        
    def build_params(self, inputs):
        # If params undefined define params
        try:
            self.params
        except AttributeError:
            self.params = self.get_default_params()
            self.params.update(inputs)
    
    # TODO allow to access parameters by self.<param>


class ParamDict(AttrDict):
    def overwrite(self, new_params):
        for param in new_params:
            print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self
