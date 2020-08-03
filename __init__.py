from blox.core import *
from blox.tensor.core import batch_apply, rmap, rmap_list, recursively

import blox.tensor

try:
    import torch
    import blox.torch
    from blox.torch.struct import save
except:
    pass