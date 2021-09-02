import copy

from torch import nn


def is_pair(img_size):
    return img_size if isinstance(img_size, tuple) else (img_size, img_size)


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])