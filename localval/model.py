#!/usr/bin/env python3
# Copyright (c) 2024 Sven Lämmle
#
# This source code is licensed under the BSD 3 license found in the
# LICENSE file in the root directory of this source tree.

"""
Created on 13-06-2023
@author: Sven Lämmle

Gaussian Process Model
"""
from typing import Optional

import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.priors import GammaPrior, HalfCauchyPrior
from torch import Tensor


def _get_kernel(
    ard_num_dims: Optional[int] = None,
    batch_shape: Optional[torch.Size] = None,
    scale_kernel: bool = True,
):
    k1 = RBFKernel(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=HalfCauchyPrior(2.0),
    )
    k2 = MaternKernel(
        ard_num_dims=ard_num_dims,
        nu=1 / 2,
        batch_shape=batch_shape,
        lengthscale_prior=HalfCauchyPrior(2.0),
    )
    k3 = MaternKernel(
        ard_num_dims=ard_num_dims,
        nu=3 / 2,
        batch_shape=batch_shape,
        lengthscale_prior=HalfCauchyPrior(2.0),
    )
    k4 = MaternKernel(
        ard_num_dims=ard_num_dims,
        nu=5 / 2,
        batch_shape=batch_shape,
        lengthscale_prior=HalfCauchyPrior(2.0),
    )
    k5 = RQKernel(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=HalfCauchyPrior(2.0),
    )

    K = k1 + k2 + k3 + k4 + k5

    if scale_kernel:
        return ScaleKernel(
            base_kernel=K,
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
    return K


class FullSingleTaskGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: Optional[Kernel] = None,
        *args,
        **kwargs
    ):
        if covar_module is None:
            covar_module = _get_kernel(ard_num_dims=train_X.shape[-1])
        super().__init__(
            train_X=train_X, train_Y=train_Y, covar_module=covar_module, *args, **kwargs
        )
