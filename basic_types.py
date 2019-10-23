import random
import re
from functools import reduce

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


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = reduce(lambda x,y: x & y, (map(lambda d: d.keys(), LD)))
    return type(LD[0])({k: [dic[k] for dic in LD] for k in keys})


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