#!/usr/bin/env python3
# Copyright (c) 2024 Sven Lämmle
#
# This source code is licensed under the BSD 3 license found in the
# LICENSE file in the root directory of this source tree.

"""
Created on 13-06-2023
@author: Sven Lämmle

Acquisition functions for local validation

.. [Ech2011akmcs]
    B. Echard, N.Gayton, M. Lemaire. AK-MCS: An active learning reliability method
    combining Kriging and Monte Carlo Simulation, Structural Safety 33, 2011,
    DOI: 10.1016/j.strusafe.2011.01.002.
.. [Läm2024MCP]
    S. Lämmle, C. Bogoclu, R. Voßhall, A. Haselhoff, D. Roos - Quantifying Local Model
    Validity using Active Learning, UAI 2024.
"""

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.posteriors import TorchPosterior
from botorch.utils import t_batch_mode_transform
from torch import Tensor

from .folded_normal import FoldedNormal

__all__ = ["MCProb", "NegUFun"]


def misclassification_probability(folded_normal: FoldedNormal, omega: Tensor) -> Tensor:
    """
    Evaluate the misclassification probability of the limit state represented as a folded
    Normal distribution.
    """
    if omega.nelement() != 1:
        raise ValueError("Omega should be scalar.")
    if not isinstance(folded_normal, FoldedNormal):
        raise TypeError("`folded_normal` should be instance of `FoldedNormal`.")
    if folded_normal.flip is False:
        raise NotImplementedError(
            f"Only implemented for flipped FoldedNormal. But "
            f"got flip={folded_normal.flip}."
        )

    idx_valid = folded_normal.loc.abs() <= folded_normal.shift
    prob = folded_normal.cdf(-omega)
    prob[~idx_valid] = 1 - folded_normal.cdf(omega)[~idx_valid]  # false negative
    fc_prob = prob.squeeze(-2).squeeze(-1)
    return fc_prob


class MCProb(AnalyticAcquisitionFunction):
    r"""Single-outcome Misclassification Probability (analytic).

    Compute the Misclassification Probability [Läm2024MCP] over the limit-state, using the
    analytic formula for a folded Normal posterior distribution. This relies on the
    posterior at single test point being a folded Gaussian, as given by the
    `ShiftedFoldedTransform`. Only supports the case of `q=1`. The model must be
    single-outcome.

    Example:
        >>> tolerance = 0.5
        >>> model = SingleTaskGP(train_X,
        >>>                      train_Y,
        >>>                      outcome_transform = ShiftedFoldedTransform(tolerance),)
        >>> mc_prob = MCProb(model)
        >>> mcp = mc_prob(test_X)
    """

    def __init__(self, model, omega: float = 0.0):
        """Single-outcome Misclassification Probability (analytic).

        Parameters
        ----------
        model: A fitted single-outcome model with folded Normal posterior (e.g., as given
            with `ShiftedFoldedTransform` applied to the outcome).
        omega: A scalar value representing the exploration-exploitation trade-off, i.e.,
            it gives the misclassification probability around the limit state with a small
            slack variable ω < ξ. `omega` > 0 will increase exploration.
        """
        super().__init__(model=model)
        if omega < 0.0:
            raise ValueError(
                f"Expected `omega` to be greater or equal to zero. Given omega: {omega}."
            )
        self.register_buffer("omega", torch.as_tensor(omega))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        self.to(device=X.device)

        # speed up calculation if we can eliminate q-dimension
        if (len(X.shape) == 3) and (X.shape[-2] == 1):
            X = X.squeeze(-2)

        posterior = self.model.posterior(
            X=X,
            posterior_transform=self.posterior_transform,
            observation_noise=False,
        )

        if not isinstance(posterior, TorchPosterior):
            raise TypeError(
                "Expected posterior distribution to be a `TorchPosterior`, "
                "which should be a flipped `FoldedNormal` distribution. One "
                "reason could be, that `ShiftedFoldedTransform` is not "
                "applied to the model output."
            )
        return misclassification_probability(posterior.distribution, omega=self.omega)


class NegUFun(AnalyticAcquisitionFunction):
    r"""Single-outcome Negative U-function (analytic).

    Calculate the negative U-function [Ech2011akmcs]. In the setting of validation with
    two limit states the Aq behaves different as originally intended, see [Läm2024MCP].

    The acquisition function is defined as

    .. math::
        \phi(x)=-\frac{\mu(x)}{\sigma(x)},

    where :math:`\mu(x)` and :math:`\sigma(x)` are the predictive mean and variance of the
    model, respectively.
    """

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        # speed up calculation if we can eliminate q-dimension
        if (len(X.shape) == 3) and (X.shape[-2] == 1):
            X = X.squeeze(-2)
        mean, sigma = self._mean_and_sigma(X)
        neg_u = torch.div(torch.abs(mean), sigma).mul(-1.0)
        return neg_u
