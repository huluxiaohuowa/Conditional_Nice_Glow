#!/usr/bin/python3.7
"""Implementing the script for model training"""
import json
import math
import os
import sys
import typing as t
import time
import gc

from adabound import AdaBound
import torch
from torch import nn
import torch_scatter

from data_utils import ComLoader
from model import GraphInf
from prior import Prior
import model
scaffold_inference_network_architecture = model


def _get_loader(db_train_loc: str,
                db_test_loc: str,
                num_workers: int,
                batch_size: int,
                batch_size_test: int
                ) -> t.Tuple[ComLoader, ComLoader]:
    """
    Get dataloader

    Args:
        db_train_loc (str): The location of training dataset
        db_test_loc (str): The location of test dataset
        num_workers (int): The number of worker used for loading training set
        batch_size (int): The size of mini-batch for training set
        batch_size_test (int): The size of mini-batch for test set

    Returns:
        t: The loader for training and test set
    """
    train_loader = ComLoader(original_scaffolds_file=db_train_loc,
                             num_workers=num_workers,
                             batch_size=batch_size)
    test_loader = ComLoader(original_scaffolds_file=db_test_loc,
                            num_workers=1,
                            batch_size=batch_size_test)
    return train_loader, test_loader


def _load_vae(ckpt_loc: str) -> GraphInf:
    """
    Load VAE model from checkpoint

    Args:
        ckpt_loc (str):
            The location of the checkpoint

    Returns:
        GraphInf:
            The model loaded from file
    """
    with open(os.path.join(ckpt_loc, 'model_config.json')) as f:
        model_config = json.load(f)
    model_new = GraphInf(**model_config)
    model_loc = os.path.join(ckpt_loc, 'vae.ckpt')
    model_new.load_state_dict(torch.load(model_loc))
    return model_new


def _next(iterator: t.Iterator,
          iterable: t.Iterable
          ) -> t.Tuple[t.Any, t.Iterator]:
    try:
        record = next(iterator)
    except StopIteration:
        iterator = iter(iterable)
        record = next(iterator)
    return record, iterator


def _loss(mini_batch: t.Tuple,
          model_prior: Prior,
          model_vae: GraphInf,
          num_post_samples: int,
          do_backprop: bool) -> torch.Tensor:
    """
    Get the loss from the mini_batch

    Args:
        mini_batch (t.Tuple): The mini-batch input
        model_prior (Prior): The prior network
        model_vae (GraphInf): The VAE model
        num_post_samples (int): The number of samples from the posterior
        do_backprop (bool): Whether to perform backpropagation

    Returns:
        torch.Tensor: The calculated loss
    """
    device = next(model_prior.parameters()).device
    (_,
     nums_nodes,
     nums_edges,
     seg_ids,
     bond_info_all,
     nodes_o,
     nodes_c) = mini_batch

    num_total_nodes = sum(nums_nodes)
    num_total_edges = sum(nums_edges)

    values = torch.ones(num_total_edges)

    s_adj = torch.sparse_coo_tensor(
        bond_info_all.T,
        values,
        torch.Size([num_total_nodes, num_total_nodes])
    ).to(device)

    node_features = torch.from_numpy(nodes_o).to(device)
    node_features_csk = torch.from_numpy(nodes_c).to(device)
    seg_ids = torch.from_numpy(seg_ids).long().to(device)

    with torch.no_grad():
        mu, var = model_vae.inference_net(node_features,
                                          node_features_csk,
                                          s_adj)
        entropy = 0.5 * torch.log(2 * math.pi * var * math.e)
        entropy = entropy.sum(-1)
        entropy = torch_scatter.scatter_add(entropy, seg_ids, dim=0)
        entropy = entropy.mean()

    loss_list = []
    for _ in range(num_post_samples):
        eps_i = torch.zeros_like(mu).normal_()
        z_i = mu + var.sqrt() * eps_i
        ll_i = model_prior.likelihood(z_i, node_features_csk, s_adj)
        ll_i = torch_scatter.scatter_add(ll_i, seg_ids, dim=0)
        print(ll_i.shape)
        loss_i = - ll_i.mean()
        loss_i = loss_i - entropy
        loss_i = loss_i / num_post_samples
        if do_backprop:
            loss_i.backward()
        loss_list.append(loss_i.item())
    loss = sum(loss_list)
    return loss


def _train_step(optim: AdaBound,
                train_iter: t.Iterator,
                train_loader: ComLoader,
                model_prior: Prior,
                model_vae: GraphInf,
                clip_grad: float,
                num_post_samples: int
                ) -> torch.Tensor:
    """
    Perform one-step training

    Args:
        optim (AdaBound): The optimizer
        train_iter (t.Iterator): The iterator for training
        train_loader (ComLoader): The Loader for training
        mini_batch (t.Tuple): The mini-batch input
        model_prior (Prior): The prior network
        model_vae (GraphInf): The VAE model
        clip_grad (float): Cliping gradient

    Returns:
        torch.Tensor: The calculated loss
    """
    model_prior.train()
    optim.zero_grad()
    record, train_iter = _next(train_iter, train_loader)
    loss = _loss(record, model_prior, model_vae, num_post_samples, True)
    # Clip gradient
    torch.nn.utils.clip_grad_value_(model_prior.parameters(), clip_grad)
    optim.step()
    return loss


def _test_step(model_prior: Prior,
               model_vae: GraphInf,
               test_iter: t.Iterator,
               test_loader: ComLoader,
               num_post_samples: int
               ) -> torch.Tensor:
    """
    Performing one-step test

    Args:
        model_prior (Prior): The prior network
        model_vae (GraphInf): The VAE model
        test_iter (t.Iterator): The iterator for testing
        test_loader (ComLoader): The Loader for testing

    Returns:
        torch.Tensor: The calculated loss
    """
    model_prior.eval()
    record, test_iter = _next(test_iter, test_loader)
    with torch.no_grad():
        loss = _loss(record, model_prior, model_vae, num_post_samples, False)
    return loss


def _save(model_prior: Prior,
          ckpt_loc: str,
          optim: AdaBound):
    """
    Save checkpoint

    Args:
        model_prior (Prior): The prior network
        ckpt_loc (str): Checkpoint location
        optim (AdaBound): The optimizer
    """
    torch.save(model_prior.state_dict(),
               os.path.join(ckpt_loc, 'mdl.ckpt'))
    torch.save(optim.state_dict(),
               os.path.join(ckpt_loc, 'optimizer.ckpt'))


def _engine(ckpt_loc: str = 'ckpt/prior',
            db_train_loc: str = 'data-center/train.smi',
            db_test_loc: str = 'data-center/test.smi',
            num_workers: int = 1,
            batch_size: int = 128,
            batch_size_test: int = 256,
            device_id: int = 0,
            num_embeddings: int = 8,
            casual_hidden_sizes: t.Iterable = (16, 32),
            num_dense_bottlenec_feat: int = 48,
            num_k: int = 12,
            num_dense_layers: int = 10,
            num_dense_out_features: int = 64,
            num_nice_blocks: int = 24,
            nice_hidden_size: int = 128,
            activation: str = 'elu',
            lr: float = 1e-3,
            final_lr: float = 0.1,
            clip_grad: float = 3.0,
            num_iterations: int = int(1e4),
            summary_steps: int = 200,
            num_post_samples: int = 10,
            ):
    """Script to start model training

    Args:
        ckpt_loc (str, optional):
            Checkpoint location. Defaults to 'ckpt/prior'.
        db_train_loc (str, optional):
            The location of training dataset.
            Defaults to 'data-center/train.smi'.
        db_test_loc (str, optional):
            The location of test dataset. Defaults to 'data-center/test.smi'.
        num_workers (int, optional):
            The number of worker used for loading training set. Defaults to 1.
        batch_size (int, optional):
            The size of mini-batch for training set. Defaults to 128.
        batch_size_test (int, optional):
            The size of mini-batch for test set. Defaults to 256.
        device_id (int, optional):
            Which device should the model be put to. Defaults to 0.
        num_embeddings (int, optional):
            The embedding size for nodes. Defaults to 8.
        casual_hidden_sizes (t.Iterable, optional):
            The hidden sizes for casual layers. Defaults to (16, 32).
        num_dense_bottlenec_feat (int, optional):
            The size of bottlenec layer for densenet. Defaults to 48.
        num_k (int, optional):
            The growth rate. Defaults to 12.
        num_dense_layers (int, optional):
            The number of layers in densenet. Defaults to 10.
        num_dense_out_features (int, optional):
            The number of output features for densenet. Defaults to 64.
        num_nice_blocks (int, optional):
            The number of nice blocks. Defaults to 10.
        nice_hidden_size (int, optional):
            The size of hidden layers in each nice block. Defaults to 128.
        activation (str, optional):
            The type of activation used. Defaults to 'elu'.. Defaults to 'elu'.
        lr (float, optional):
            Initial learning rate for AdaBound. Defaults to 1e-3.
        final_lr (float, optional):
            The final learning rate AdaBound should converge to.
            Defaults to 0.1.
        clip_grad (float, optional):
            The scale of gradient clipping. Defaults to 3.0.
        num_iterations (int, optional):
            How many iterations should the training be performed.
            Defaults to 1e4.
        summary_steps (int, optional):
            Create summary for each `summary_steps` steps. Defaults to 200.
    """
    device = torch.device(f'cuda:{device_id}')
    # Get training and test set loaders
    train_loader, test_loader = \
        _get_loader(db_train_loc,
                    db_test_loc,
                    num_workers,
                    batch_size,
                    batch_size_test)
    train_iter, test_iter = iter(train_loader), iter(test_loader)
    # Load VAE
    model_vae = _load_vae(ckpt_loc)
    # Move model to GPU
    model_vae = model_vae.to(device).eval()
    # Disable gradient
    p: nn.Parameter
    for p in model_vae.parameters():
        p.requires_grad_(False)

    # Initialize model and optimizer
    model_prior = Prior(model_vae.num_c_feat,
                        num_embeddings,
                        casual_hidden_sizes,
                        num_dense_bottlenec_feat,
                        num_k,
                        num_dense_layers,
                        num_dense_out_features,
                        model_vae.num_z_feat,
                        num_nice_blocks,
                        nice_hidden_size,
                        activation)
    model_prior.to(device)
    # optim = AdaBound(model_prior.parameters(),
    #                  lr=lr,
    #                  final_lr=final_lr)
    optim = torch.optim.SGD(model_prior.parameters(), lr=final_lr * 0.1)

    with open(os.path.join(ckpt_loc, 'log.out'), 'w') as f:
        f.write('Global step\t'
                'Time\t'
                'Training loss\t'
                'Test loss\n')
        t0 = time.time()
        for step_id in range(num_iterations):
            train_loss = _train_step(optim,
                               train_iter,
                               train_loader,
                               model_prior,
                               model_vae,
                               clip_grad,
                               num_post_samples)
            print(train_loss)
            if step_id % summary_steps == 0:
                test_loss = _test_step(model_prior,
                                  model_vae,
                                  test_iter,
                                  test_loader,
                                  num_post_samples)
                f.write(f'{step_id}\t'
                        f'{float(time.time() - t0) / 60}\t'
                        f'{train_loss}\t'
                        f'{test_loss}\n')
                f.flush()
                _save(model_prior, ckpt_loc, optim)
        test_loss = _test_step(model_prior,
                          model_vae,
                          test_iter,
                          test_loader,
                          num_post_samples)
        f.write(f'{step_id}\t'
                f'{float(time.time() - t0) / 60}\t'
                f'{train_loss}\t'
                f'{test_loss}\n')
        f.flush()
        _save(model_prior, ckpt_loc, optim)


def main(ckpt_loc):
    """Program entrypoint"""
    with open(os.path.join(ckpt_loc, 'config.json')) as f:
        config = json.load(f)
    config['ckpt_loc'] = ckpt_loc
    _engine(**config)


if __name__ == '__main__':
    main(sys.argv[1])
