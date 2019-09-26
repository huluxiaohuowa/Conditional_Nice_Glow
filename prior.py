"""Implementation of the glow-based prior"""
import math
import typing as t

import torch
from torch import nn
from torch.nn import functional as F

import distrib_zoo as dz
import model


__all__ = ['NICEWeave', 'Prior']


class NICEWeave(dz.Distribution):
    """Implementation of the glow-based flow using weavenet"""
    def __init__(self,
                 num_features: int,
                 num_cond_features: int,
                 num_blocks: int,
                 hidden_size: int,
                 activation: str = 'elu'):
        """
        The constructor

        Args:
            num_features (int):
                The number of features for random variable attached to each node
            num_cond_features (int):
                The number of conditional features for each node
            num_blocks (int):
                The number of NICE blocks to use
            hidden_size (int):
                The hidden size used in each NICE
            activation (str, optional):
                The activation used. Defaults to 'elu'.
        """
        # Calling parent constructor
        super(NICEWeave, self).__init__()

        # Saving parameters
        self._num_features = num_features
        self._num_blocks = num_blocks
        self._hidden_size = hidden_size
        self._activation = activation
        self._num_cond_features = num_cond_features

        # Making submodules
        bn_list = []
        linear_list = []
        nice_in_list, nice_mid_list, nice_out_list = [], [], []
        block_list = []
        for _ in range(self._num_blocks):
            bn = dz.BatchNormFlow(self._num_features)
            linear = dz.InvLinear(self._num_features, bias=False)
            nice_in = model.WeaveLayer(self._num_features // 2 +
                                       self._num_cond_features,
                                       self._hidden_size,
                                       self._activation)
            nice_mid = nn.Sequential(nn.BatchNorm1d(self._hidden_size +
                                                    self._num_cond_features),
                                     nn.Linear(self._hidden_size +
                                               self._num_cond_features,
                                               self._hidden_size,
                                               bias=False))
            nice_out = model.ZeroWeaveLayer(self._hidden_size +
                                            self._num_cond_features,
                                            self._num_features,
                                            self._activation)

            # pylint: disable=cell-var-from-loop
            def _affine_fn(_x, _x_cond, _adj):
                _x = nice_in(torch.cat((_x, _x_cond), dim=-1), _adj)
                _x = nice_mid(torch.cat((_x, _x_cond), dim=-1))
                _x = nice_out(torch.cat((_x, _x_cond), dim=-1), _adj)
                _mu, _var = torch.split(_x, self._num_features // 2, dim=-1)
                _var = F.softplus(_var) / math.log(2)
                return _mu, _var
            block = dz.NICE(_affine_fn)

            bn_list.append(bn)
            linear_list.append(linear)
            nice_in_list.append(nice_in)
            nice_mid_list.append(nice_mid)
            nice_out_list.append(nice_out)
            block_list.append(block)

        self.bn_list = nn.ModuleList(bn_list)
        self.linear_list = nn.ModuleList(linear_list)
        self.nice_in_list = nn.ModuleList(nice_in_list)
        self.nice_mid_list = nn.ModuleList(nice_mid_list)
        self.nice_out_list = nn.ModuleList(nice_out_list)
        self.block_list = nn.ModuleList(block_list)

        self.gaussian = dz.Gaussian(self._num_features)

    def sample(self,
               x_cond: torch.Tensor,
               adj: t.Union[torch.sparse.FloatTensor,
                            torch.cuda.sparse.FloatTensor]
               ) -> torch.Tensor:
        """
        Sample from the distribution

        Args:
            x_cond (torch.Tensor):
                The conditional features attached to each node
            adj (t.Union[torch.sparse.FloatTensor,
                         torch.cuda.sparse.FloatTensor]):
                The adjacency matrix of the graph, represented as a
                sparse tensor

        Returns:
            torch.Tensor:
                The sampled results
        """
        # Get device information
        device = x_cond.device
        # Sample from gaussian
        x = self.gaussian.sample(x_cond.size(0), device=device)
        # Perform transformation
        bn: dz.BatchNormFlow
        linear: dz.InvLinear
        nice: dz.NICE
        for bn, linear, nice in reversed(list(zip(self.bn_list,
                                                  self.linear_list,
                                                  self.block_list))):
            x = nice.flow(x, x_cond, adj)
            x = linear.flow(x)
            x = bn.flow(x)
        return x

    def likelihood(self,
                   x: torch.Tensor,
                   x_cond: torch.Tensor,
                   adj: t.Union[torch.sparse.FloatTensor,
                                torch.cuda.sparse.FloatTensor]
                   ) -> torch.Tensor:
        """
        Get the likelihood of x

        Args:
            x (torch.Tensor):
                The sample for which likelihood values are calculated
            x_cond (torch.Tensor):
                The conditional values
            adj (t.Union[torch.sparse.FloatTensor,
                         torch.cuda.sparse.FloatTensor]):
                The adjacency matrix for the graph

        Returns:
            torch.Tensor:
                The likelihood values
        """
        ll = 0.0
        # Perform transformation
        bn: dz.BatchNormFlow
        linear: dz.InvLinear
        nice: dz.NICE
        for bn, linear, nice in zip(self.bn_list,
                                    self.linear_list,
                                    self.block_list):
            x, ll_bn = bn.likelihood(x)
            x, ll_linear = linear.likelihood(x)
            x, ll_nice = nice.likelihood(x, x_cond, adj)
            ll_block = ll_bn + ll_linear + ll_nice
            ll = ll + ll_block
        ll_gaussian = self.gaussian.likelihood(x)
        ll = ll + ll_gaussian
        return ll


class Prior(dz.Distribution):
    """The prior network (DenseNet + Glow)"""
    def __init__(self,
                 num_node_types: int,
                 num_embeddings: int,
                 casual_hidden_sizes: t.Iterable,
                 num_dense_bottlenec_feat: int,
                 num_k: int,
                 num_dense_layers: int,
                 num_dense_out_features: int,
                 latent_size: int,
                 num_nice_blocks: int,
                 nice_hidden_size: int,
                 activation: str = 'elu'):
        """
        The constructor

        Args:
            num_node_types (int):
                The number of node types
            num_embeddings (int):
                The number of input features to densenet
            casual_hidden_sizes (t.Iterable):
                The number of hidden features for casual layers
            num_dense_bottlenec_feat (int):
                The number of bottlenec features for densenet
            num_k (int):
                The growth rate of densenet
            num_dense_layers (int):
                The number of layers in densenet
            num_dense_out_features (int):
                The number of output features for densenet
            latent_size (int):
                The number of latent features
            num_nice_blocks (int):
                The number of nice blocks
            nice_hidden_size (int):
                The size of hidden layers in each nice block
            activation (str, optional):
                The type of activation used. Defaults to 'elu'.
        """
        # Calling parent constructor
        super(Prior, self).__init__()

        # Building submodules
        self.embedding = nn.Embedding(num_node_types, num_embeddings)
        self.densenet = model.DenseNet(num_feat=num_embeddings,
                                       casual_hidden_sizes=casual_hidden_sizes,
                                       num_botnec_feat=num_dense_bottlenec_feat,
                                       num_k_feat=num_k,
                                       num_dense_layers=num_dense_layers,
                                       num_out_feat=num_dense_out_features,
                                       activation=activation)
        self.nice = NICEWeave(num_features=latent_size,
                              num_cond_features=num_dense_out_features,
                              num_blocks=num_nice_blocks,
                              hidden_size=nice_hidden_size,
                              activation=activation)

    def sample(self,
               x: torch.Tensor,
               adj: t.Union[torch.sparse.FloatTensor,
                            torch.cuda.sparse.FloatTensor]
               ) -> torch.Tensor:
        """
        Perform sampling

        Args:
            x (torch.Tensor):
                The type of each node
            adj (t.Union[torch.sparse.FloatTensor,
                         torch.cuda.sparse.FloatTensor]):
                The adjacency matrix

        Returns:
            torch.Tensor:
                The result of the sampling
        """
        # Embed node types
        x = self.embedding(x)
        # Dense net
        x_cond = self.densenet(x, adj)
        # Sampling
        z = self.nice.sample(x_cond, adj)
        return z

    def likelihood(self,
                   z: torch.Tensor,
                   x: torch.Tensor,
                   adj: t.Union[torch.sparse.FloatTensor,
                                torch.cuda.sparse.FloatTensor]
                   ) -> torch.Tensor:
        """
        Calculate the likelihood of z

        Args:
            z (torch.Tensor):
                The latent variable for which the likelihood need to be calculated
            x (torch.Tensor):
                The type of each node
            adj (t.Union[torch.sparse.FloatTensor,
                         torch.cuda.sparse.FloatTensor]):
                The adjacency matrix

        Returns:
            torch.Tensor:
                The calculated likelihood
        """
        # Embed node types
        x = self.embedding(x)
        # Dense net
        x_cond = self.densenet(x, adj)
        # Get likelihood
        ll = self.nice.likelihood(z, x_cond, adj)
        return ll
