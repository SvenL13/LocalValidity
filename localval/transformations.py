#!/usr/bin/env python3
# Copyright (c) 2024 Sven Lämmle
#
# This source code is licensed under the BSD 3 license found in the
# LICENSE file in the root directory of this source tree.

"""
Created on 13-06-2023
@author: Sven Lämmle

Outcome Transformation
"""
import math
from typing import Optional, Tuple

import torch
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import (
    GPyTorchPosterior,
    Posterior,
    TorchPosterior,
    TransformedPosterior,
)
from torch import Tensor

from .folded_normal import FoldedNormal

BOUNDS_THRESHOLD = 1e-7


def norm_to_folded_mean(mu: Tensor, var: Tensor) -> Tensor:
    """Compute mean of a folded-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vector of the Normal distribution.

    Returns:
        The `batch_shape x n` mean vector of the folded-Normal distribution.
    """
    sigma = var.sqrt()
    mu_folded = (
        sigma * math.sqrt(2 / math.pi) * torch.exp((-(mu**2)) / (2 * sigma**2))
    )
    mu_folded = mu_folded + mu * torch.erf(mu / torch.sqrt(2 * sigma**2))
    return mu_folded


def norm_to_folded_var(mu: Tensor, var: Tensor) -> Tensor:
    """Compute variance of a folded-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vector of the Normal distribution.

    Returns:
        The `batch_shape x n` variance vector of the folded-Normal distribution.
    """
    mu_folded = norm_to_folded_mean(mu, var)
    return mu**2 + var - mu_folded**2


class ShiftedFoldedTransform(OutcomeTransform):
    """
    Apply only the inverse transform Z = t - |X|, where t is a scalar.
    Useful for learning the limit state.
    """

    def __init__(self, tolerance_level: float):
        super().__init__()
        self.register_buffer("tolerance", torch.as_tensor(tolerance_level))

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return Y, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if Yvar is not None:
            raise NotImplementedError
        Y_tf = self.tolerance - torch.abs(Y)
        return Y_tf, Yvar

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        def shifted_mean(mean: Tensor, var: Tensor) -> Tensor:
            return self.tolerance - norm_to_folded_mean(mean, var)

        if type(posterior) is GPyTorchPosterior and posterior._is_mt:
            raise ValueError("Multitask is not supported")

        if type(posterior) is not GPyTorchPosterior:
            # fall back to TransformedPosterior
            return TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: self.untransform(s)[0],
                mean_transform=shifted_mean,
                variance_transform=norm_to_folded_var,
            )
        posterior: GPyTorchPosterior
        folded_norm = FoldedNormal(
            loc=posterior.mean,
            scale=posterior.variance.sqrt(),
            shift=self.tolerance,
            flip=True,
        )
        return TorchPosterior(distribution=folded_norm)
