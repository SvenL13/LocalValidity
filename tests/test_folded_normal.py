from typing import Callable

import torch
from botorch.utils.testing import BotorchTestCase
from localval.folded_normal import FoldedNormal
from scipy.stats import foldnorm


def wrapper_np_torch(x: torch.Tensor, fn: Callable, *param) -> torch.Tensor:
    res = fn(x.numpy(), *param)
    return torch.from_numpy(res)


class TestFoldedNormal(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp()
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)

    def test_folded_normal(self):

        loc = torch.randn(5, 5, requires_grad=True)
        scale = torch.randn(5, 5).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(FoldedNormal(loc, scale).sample().size(), (5, 5))
        self.assertEqual(FoldedNormal(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(FoldedNormal(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(FoldedNormal(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(FoldedNormal(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(FoldedNormal(-0.7, 50.0).sample((1,)).size(), (1,))

        params = (
            [3, 2, 0, False],
            [3, 2, 1, False],
            [3, 2, 0, True],
            [3, 2, 1, True],
            [3, 2, -1, True],
            [-3, 2, -1, True],
        )

        for param in params:
            q = torch.tensor([0.001, 0.5, 0.999], dtype=torch.double)
            fn = FoldedNormal(*param)
            vals = fn.icdf(q)
            self.assertAllClose(q, fn.cdf(vals))

    def test_simulation(self):

        rtol = 1e-2
        atol = 1e-3
        loc = torch.randn(2, 2, dtype=torch.double)
        scale = torch.randn(2, 2, dtype=torch.double).abs()

        for param in [
            (0, False),
            (0, True),
            (1, False),
            (0, True),
            (-1, True),
            (-1, False),
        ]:
            fn = FoldedNormal(loc, scale, shift=param[0], flip=param[1])
            samples = fn.sample(torch.Size((500_000,)))

            self.assertAllClose(samples.mean(dim=0), fn.mean, rtol=rtol, atol=atol)
            self.assertAllClose(samples.std(dim=0), fn.stddev, rtol=rtol, atol=atol)
            self.assertAllClose(samples.var(dim=0), fn.variance, rtol=rtol, atol=atol)

    def test_scipy(self):
        params = [
            (0, 1, 0),
            (-4, 2, 0),
            (3, 2, 0),
            (0, 1, 1),
            (-4, 2, -1),
            (3, 2, 3),
        ]
        x = torch.linspace(0.01, 15, 200, dtype=torch.double)
        q = torch.tensor([0.1, 0.5, 0.99], dtype=torch.double)
        for p in params:
            x_ = x + p[2]
            c = abs(p[0]) / p[1]
            fn_scipy = foldnorm(c=c, loc=p[2], scale=p[1])
            fn_torch = FoldedNormal(loc=p[0], scale=p[1], shift=p[2])

            self.assertAllClose(fn_torch.prob(x_), wrapper_np_torch(x_, fn_scipy.pdf))
            self.assertAllClose(fn_torch.cdf(x_), wrapper_np_torch(x_, fn_scipy.cdf))
            self.assertAllClose(fn_torch.icdf(q), wrapper_np_torch(q, fn_scipy.ppf))

    def test_normal_shape_scalar_params(self):
        fnormal = FoldedNormal(0, 1)
        self.assertEqual(fnormal._batch_shape, torch.Size())
        self.assertEqual(fnormal._event_shape, torch.Size())
        self.assertEqual(fnormal.sample().size(), torch.Size())
        self.assertEqual(fnormal.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            fnormal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            fnormal.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_normal_shape_tensor_params(self):
        fnormal = FoldedNormal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(fnormal._batch_shape, torch.Size((2,)))
        self.assertEqual(fnormal._event_shape, torch.Size(()))
        self.assertEqual(fnormal.sample().size(), torch.Size((2,)))
        self.assertEqual(fnormal.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            fnormal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, fnormal.log_prob, self.tensor_sample_2)
        self.assertEqual(fnormal.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))
