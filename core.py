import inspect

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


class optional:
    """ A function decorator that returns the first argument to the function if yes=False
     I chose a class-based decorator since I find the syntax less confusing. """

    def __init__(self, n=0):
        """ Decorator parameters """
        self.n = n

    def __call__(self, func):
        """ Wrapping """
    
        def wrapper(*args, yes=True, **kwargs):
            if yes:
                return func(*args, **kwargs)
        
            n = self.n
            if inspect.ismethod(func):
                n += 1
            return args[n]
    
        return wrapper