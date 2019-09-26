import typing as t

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'get_activation',
    'mlp_fn'
]


def get_activation(name, *args, **kwargs):
    """Get activation function by name"""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(*args, **kwargs)
    if name == 'elu':
        return nn.ELU(*args, **kwargs)
    if name == 'selu':
        return nn.SELU(*args, **kwargs)
    raise ValueError('Activation not implemented')


def mlp_fn(
    x: torch.Tensor,
    cond: torch.Tensor,
    mlp_func: t.Callable,
    chunk_sizes: t.List[int] = [1, 2]
) -> t.Tuple[torch.Tensor]:
    x = mlp_func(torch.cat(x, cond), dim=-1)
    if chunk_sizes is None:
        chunk_sizes = x.size(1) // 2
    mu, var = torch.split(x, chunk_sizes, -1)
    var = F.softplus(var) / math.log(2)
    return mu, var
