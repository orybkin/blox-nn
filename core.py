import inspect
import collections


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return dict.__getitem__(self, attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def rename(self, old, new):
        self[new] = self.pop(old)
        
    def get(self, key, default=None):
        if key not in self:
            return default
        return self.__getitem__(key)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d

    @property
    def safe(self):
        """
        A property that can be used to safely access the dictionary.
        When a key is not found, the access getter will return None.

        To be used as:
        AttrDict().safe.keypoint == None
        """
        return DictAccess(self)


class AttrDefaultDict(collections.defaultdict):
    __setattr__ = collections.defaultdict.__setitem__
    
    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return collections.defaultdict.__getitem__(self, attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)
    
    def rename(self, old, new):
        self[new] = self.pop(old)
    
    def get(self, key, default=None):
        if key not in self:
            return default
        return self.__getitem__(key)
    
    def __getstate__(self):
        return self
    
    def __setstate__(self, d):
        self = d
    
    @property
    def safe(self):
        """
        A property that can be used to safely access the dictionary.
        When a key is not found, the access getter will return None.

        To be used as:
        AttrDict().safe.keypoint == None
        """
        return DictAccess(self)
    

class DictAccess():
    # A simple access class that returns None if a key is not found
    def __init__(self, parent):
        super().__setattr__('parent', parent)
    
    def __getattr__(self, item):
        try:
            return dict.__getitem__(self.parent, item)
        except KeyError:
            return None
    

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
