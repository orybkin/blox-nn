import random
import re
from functools import reduce
import collections

import numpy as np
from blox import AttrDict


def dict_concat(d1, d2):
    if not set(d1.keys()) == set(d2.keys()):
        raise ValueError("Dict keys are not equal. got {} vs {}.".format(d1.keys(), d2.keys()))
    for key in d1:
        d1[key] = np.concatenate((d1[key], d2[key]))


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def filter_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(filter(fn, d.items()))


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    # Note dict.__getitem__ is necessary for subclasses of dict that override getitem
    return type(LD[0])({k: [dict.__getitem__(d, k) for d in LD] for k in keys})


def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL,t)) for t in zip(*DL.values())]


def subdict(dict, keys, strict=True):
    if not strict:
        keys = dict.keys() & keys
    return AttrDict((k, dict[k]) for k in keys)


def maybe_retrieve(d, key):
    if hasattr(d, key):
        return d[key]
    else:
        return None


def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None


def float_regex():
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    return rx


def rand_split_list(list, frac=0.5, seed=None):
    rng = random.Random()
    if seed is not None: rng.seed(seed)
    rng.shuffle(list)
    split = int(frac * len(list))
    return list[:split], list[split:]


def dot2nesteddict(dot_map_dict):
    """
    Convert something like
    ```
    {
        'one.two.three.four': 4,
        'one.six.seven.eight': None,
        'five.nine.ten': 10,
        'five.zero': 'foo',
    }
    ```
    into its corresponding nested dict.
    
    Taken from rlkit by Vitchyr Pong
    http://stackoverflow.com/questions/16547643/convert-a-list-of-delimited-strings-to-a-tree-nested-dict-using-python
    :param dot_map_dict:
    :return:
    """
    tree = {}

    for key, item in dot_map_dict.items():
        split_keys = key.split('.')
        if len(split_keys) == 1:
            if key in tree:
                raise ValueError("Duplicate key: {}".format(key))
            tree[key] = item
        else:
            t = tree
            for sub_key in split_keys[:-1]:
                t = t.setdefault(sub_key, {})
            last_key = split_keys[-1]
            if not isinstance(t, dict):
                raise TypeError(
                    "Key inside dot map must point to dictionary: {}".format(
                        key
                    )
                )
            if last_key in t:
                raise ValueError("Duplicate key: {}".format(last_key))
            t[last_key] = item
    return tree


def nested2dotdict(d, parent_key=''):
    """
    Convert a recursive dictionary into a flat, dot-map dictionary.

    Taken from rlkit by Vitchyr Pong
    :param d: e.g. {'a': {'b': 2, 'c': 3}}
    :param parent_key: Used for recursion
    :return: e.g. {'a.b': 2, 'a.c': 3}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + "." + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(nested2dotdict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
