#!/usr/bin/env python3
# Copyright (c) 2024 Sven Lämmle
#
# This source code is licensed under the BSD 3 license found in the
# LICENSE file in the root directory of this source tree.

"""
Created on 13-06-2023
@author: Sven Lämmle

Folded Normal Distribution
"""
import math

import numpy as np
import scipy.special as sc
import torch
from numpy import vectorize
from scipy.optimize import brentq
from torch.distributions import ExponentialFamily, constraints
from torch.distributions.constraints import _GreaterThanEq, _LessThan
from torch.distributions.utils import broadcast_all


def _ppf_to_solve(x: np.ndarray, q: float, loc: float, scale: float) -> np.ndarray:
    sqrt_ = scale * math.sqrt(2)
    return 0.5 * (sc.erf((x - loc) / sqrt_) + sc.erf((x + loc) / sqrt_)) - q


def _ppf_single(q, *args):
    xtol = 1e-14
    factor = 10.0
    left, right = 0.0, np.inf

    if np.isinf(right):
        right = max(factor, left)
        while _ppf_to_solve(right, q, *args) < 0.0:
            left, right = right, right * factor
        # right is now such that cdf(right) >= q
    return brentq(_ppf_to_solve, left, right, args=(q,) + args, xtol=xtol)


class _LessThanEq(_LessThan):
    """
    Constrain to a real half line `[-inf, upper_bound]`.
    """

    def check(self, value):
        return self.upper_bound >= value


class FoldedNormal(ExponentialFamily):
    r"""
    Creates a folded normal distribution Z, parameterized by
    :attr:`loc` and :attr:`scale`.

    Z = |X|, where X ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): mean of the normal distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the normal distribution
            (often referred to as sigma)
        shift (float or Tensor): shift the location of the distribution, ie Y = shift + Z
        flip (bool): flip the distribution, ie Y = -Z
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "shift": constraints.real,
    }
    # default folded normal support is only positive, but since we can flip distribution
    # the support is changing
    support = constraints.real

    def _get_support(self):
        if self.flip:
            return -torch.inf, self.shift
        return self.shift, torch.inf

    def _support_mask(self, x):
        a, b = self._get_support()
        return (a <= x) & (x <= b)

    def mean_(self):
        sigma = self.scale
        mu_folded = (
            sigma
            * math.sqrt(2 / math.pi)
            * torch.exp((-(self.loc**2)) / (2 * sigma**2))
        )
        mu_folded = mu_folded + self.loc * torch.erf(
            self.loc / torch.sqrt(2 * sigma**2)
        )
        return mu_folded

    @property
    def mean(self):
        mu_folded = self.mean_()
        if self.flip:
            mu_folded = mu_folded.mul(-1)
        return mu_folded + self.shift

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        return self.loc**2 + self.scale**2 - self.mean_() ** 2

    def __init__(
        self,
        loc,
        scale,
        shift=0.0,
        flip: bool = False,
        cache_quantiles: bool = True,
        validate_args=None,
    ):
        self.flip = bool(flip)
        if self.flip and isinstance(shift, (float, int)):
            self.support = _LessThanEq(shift)
        elif isinstance(shift, (float, int)):
            self.support = _GreaterThanEq(shift)

        self.loc, self.scale, self.shift = broadcast_all(loc, scale, shift)
        if isinstance(loc, (int, float, bool)) and isinstance(
            scale, (int, float, bool)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

        # cache quantile calculation to speed up things
        self.cache_quantiles = bool(cache_quantiles)
        self._q_cache = {}

    def clean_cache(self):
        self._q_cache = {}

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        lw, up = self._get_support()
        lw_supp = value < lw
        up_supp = value > up

        value = value - self.shift
        if self.flip:
            value = value.mul(-1)

        sqrt_two = math.sqrt(2)
        cdf_ = 0.5 * (
            torch.erf((value - self.loc) * self.scale.reciprocal() / sqrt_two)
            + torch.erf((value + self.loc) * self.scale.reciprocal() / sqrt_two)
        )
        if self.flip:
            cdf_ = 1 - cdf_

        # enforce [0, 1]
        cdf_[lw_supp] = 0.0
        cdf_[up_supp] = 1.0
        return cdf_

    def icdf(self, value: torch.Tensor) -> torch.Tensor:

        if value.nelement() == 1 and float(value.round(decimals=4)) in self._q_cache:
            return self._q_cache[float(value.round(decimals=4))]

        batch_shape = self.batch_shape

        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Input should be torch.Tensor. Given {type(value)}")

        if (value <= 0).any() or (value >= 1).any():
            raise ValueError("Outside of range, q should be within (0, 1)")

        value_ = value
        if value_.nelement() == 1:
            value_ = value_.expand(batch_shape)

        loc = self.loc
        scale = self.scale
        if batch_shape.numel() == 1 and value_.nelement() > 1:
            loc = loc.expand(value_.shape)
            scale = scale.expand(value_.shape)
            batch_shape = loc.shape

        if value_.shape != batch_shape:
            raise ValueError(f"Batch_shapes should match. Expected: {batch_shape}.")

        if self.flip:  # flip quantile values
            value_ = 1 - value_
        output = self._icdf(
            q=value_.flatten(),
            loc=loc.flatten(),
            scale=scale.flatten(),
        )
        if self.flip:
            output = output.mul(-1)
        output = output.reshape(batch_shape) + self.shift

        if self.cache_quantiles and value.nelement() == 1:
            self._q_cache[float(value.round(decimals=4))] = output
        return output

    def _icdf(self, q, loc, scale):
        q, loc, scale = q.detach().numpy(), loc.detach().numpy(), scale.detach().numpy()
        ppfvec = vectorize(_ppf_single, otypes="d")
        output = ppfvec(*(q, loc, scale))
        return torch.from_numpy(output)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            samples = torch.normal(
                self.loc.expand(shape), self.scale.expand(shape)
            ).abs()
        if self.flip:
            samples = samples.mul(-1)
        samples_ = samples + self.shift
        return samples_

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        cond = self._support_mask(value)
        value = value - self.shift
        if self.flip:
            value = value.mul(-1)
        norm = self.scale * math.sqrt(2 * math.pi)
        normal_1 = torch.exp(-0.5 * (value - self.loc) ** 2 / (self.scale**2))
        normal_2 = torch.exp(-0.5 * (value + self.loc) ** 2 / (self.scale**2))

        pdf_ = normal_1 / norm + normal_2 / norm
        pdf_[~cond] = 0.0
        return pdf_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.prob(value).log()
