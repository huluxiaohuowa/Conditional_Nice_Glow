import json
import typing as t

import torch
from torch import nn

import distrib_zoo as dz
from func import *


class SimpleLinear(nn.Module):
    def __init__(
        self,
        num_in_feats: int,
        num_out_feats: int,
        bias: bool=False,
        activation: str='elu'
    ):
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_in_feats)
        self.fc = nn.Linear(num_in_feats, num_out_feats)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor):
        # x = self.bn(x)
        x = self.activation(
            self.fc(x)
        )
        return x


class Nice(dz.Flow):
    """The nice block"""
    def __init__(
        self,
        affine_F: t.Callable,
        affine_G: t.Callable,
        chunk_sizes: t.List[int] = [1, 2]
    ):
        """The NICE block

        Args:
            affine_F ([t.Callable]): the F affine function
            affine_G ([t.Callable]): the G affine function
            chunk_sizes (t.List[int]):
                chunk size per section of the splited tensor
                Defaults to [1, 2]
        """
        super(Nice, self).__init__()
        self._affine_F = affine_F
        self._affine_G = affine_G
        self.chunk_sizes = chunk_sizes
        self.num_blocks = num_blocks

    def flow(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """The forward flow of NICE

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x_1, x_2 = torch.split(x, self.chunk_sizes, dim=-1)
        mean_F, var_F = self._affine_F(x_2, *args, **kwargs)
        x_1 = mean_F + x_1 * var_F.sqrt()

        mean_G, var_G = self._affine_G(x_1, *args, **kargs)
        x_2 = mean_G + x_2 * var_G.sqrt()

        x = torch.cat((x_1, x_2), dim=-1)
        return x

    def inverse(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        x_1, x_2 = torch.split(x, self.chunk_sizes, dim=-1)
        mean_G, var_G = self._affine_G(x_1, *args, **kwargs)
        x_2 = (x_2 - mean_G) / var_G.sqrt()

        mean_F, var_F = self._affine_F(x_2, *args, **kwargs)
        x_1 = (x_1 - mean_F) / var_F.sqrt()

        x = torch.cat((x1, x2), dim=-1)
        return x

    def likelihood(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        x_1, x_2 = torch.split(x, self.chunk_sizes, dim=-1)
        mean_G, var_G = self._affine_G(x_1, *args, **kwargs)
        x_2 = (x_2 - mean_G) / var_G.sqrt()

        mean_F, var_F = self._affine_F(x_2, *args, **kwargs)
        x_1 = (x_1 - mean_F) / var_F.sqrt()

        x = torch.cat((x1, x2), dim=-1)
        ll = torch.sum(torch.log(torch.cat(var_F, var_G), dim=-1), dim=-1)
        return x, ll


class GlowBlock(dz.Flow):
    """The glow block for starting coordinates prediction"""
    def __init__(
        self,
        num_features: int,
        num_cond_features: int,
        num_blocks: int,
        chunk_sizes: t.List[int]=[1, 2]
    ):
        super().__init__()
        self.num_features = num_features
        self.num_cond_features = num_cond_features
        self.num_blocks = num_blocks
        self.chunk_sizes = chunk_sizes

        bn_list = []
        linear_list = []
        nice_list = []
        for _ in range(self._num_blocks):
            bn_list.append(dz.BatchNormFlow(num_features))
            linear_list.append(dz.InvLinear(num_features))
            affine_F = SimpleLinear(
                chunk_sizes[1],
                chunk_sizes[0]
            )
            affine_G = SimpleLinear(
                chunk_sizes[0],
                chunk_sizes[1]
            )
            nice_list.append(Nice(
                mlp_fn(affine_F, in, out, chunk_sizes),
                mlp_fn(affine_G, in, out, chunk_sizes),
                chunk_sizes
            ))
        

    def flow(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        sdf

    def inverse(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ):
        pass


# class GlowNet(dz.Distribution):
#     """The Glow network"""
#     def __init__(
#         self,
#         num_blocks: int,
        
#     ):