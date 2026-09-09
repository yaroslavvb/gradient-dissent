"""Tiny deterministic CPU checks for post-ReLU probes; no model training."""
import json
import math
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .mechanisms import run_mechanisms, multilinear_weights, survival_probabilities
except ImportError:
    from mechanisms import run_mechanisms, multilinear_weights, survival_probabilities


class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.widths = (2, 2, 2, 2, 2, 2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2, dtype=torch.float64) for _ in range(6)])
        self.residual = True
        self.input_scale = 1.
        self.output_relu = False
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.zero_()
                layer.bias.zero_()
            self.layers[0].weight.copy_(torch.eye(2))
            self.layers[0].bias.fill_(1.)
            for layer in list(self.layers)[1:-1]:
                layer.bias.copy_(torch.tensor([.2, 0.]))
            self.layers[-1].weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))


def forward(model, x, gains):
    h = F.relu(model.layers[0](x*model.input_scale))
    for layer, gain in zip(list(model.layers)[1:-1], gains):
        h = F.relu(h[..., :layer.out_features]+gain*layer(h))
    logits = model.layers[-1](h)
    return F.relu(logits) if model.output_relu else logits


def corners(model, x):
    with torch.no_grad():
        return torch.stack([forward(model, x, [float(bool(mask & (1 << j))) for j in range(4)])
                            for mask in range(16)])


class MechanismChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.model = TinyResidual()
        self.x = torch.tensor([[.2, .4], [.5, .3], [.1, .2]], dtype=torch.float64)
        self.y = torch.tensor([0, 1, 0])

    def run_probe(self, recipe="residual", epoch=100):
        return run_mechanisms(self.model, self.x, self.y, corners(self.model, self.x), epoch, recipe, forward)

    def test_gradient_is_per_example_and_direction_is_actual_local_delta(self):
        result = self.run_probe()
        logits = forward(self.model, self.x, [1.]*4)
        probs = logits.softmax(-1).detach().numpy()
        bias = self.model.layers[1].bias[0].item()
        expected = np.where(self.y.numpy() == 0, 2*bias*probs[:, 1], -2*bias*probs[:, 0])
        for block in result["blocks"]:
            np.testing.assert_allclose(block["per_example"]["gradient_dot_delta"], expected, atol=1e-13)
            self.assertEqual(block["full_correct_examples"]["n"], 2)
            self.assertEqual(block["all_examples"]["n"], 3)
            for values in block["finite_difference"]["deletion"]["centered_logit_directional_response_l2"].values():
                np.testing.assert_allclose(values, math.sqrt(2)*bias, atol=2e-11)

    def test_relu_deletion_is_not_negative_affine_increment(self):
        with torch.no_grad():
            self.model.layers[1].bias.copy_(torch.tensor([-2., 0.]))
        result = self.run_probe()
        actual_delta_norm = result["blocks"][0]["per_example"]["delta_norm"]
        np.testing.assert_allclose(actual_delta_norm, (self.x[:, 0]+1).numpy(), atol=1e-13)
        self.assertFalse(np.allclose(actual_delta_norm, 2.))

    def test_affine_logits_interpolate_exactly_but_corner_ce_does_not(self):
        result = self.run_probe()
        interior = [row for row in result["fractional_gate_interpolation"] if row["alpha"] == .5]
        for row in interior:
            self.assertLess(row["summary"]["centered_logit_multilinear_error_rms"]["rms"], 1e-12)
            self.assertLess(row["summary"]["centered_logit_degree1_error_rms"]["rms"], 1e-12)
            self.assertGreater(row["summary"]["ce_minus_corner_ce_interpolant"]["mean_absolute"], 1e-5)
            self.assertLess(row["summary"]["ce_minus_ce_of_interpolated_logits"]["rms"], 1e-12)

    def test_survival_mean_logit_bias_and_ce_jensen_gap_are_separate(self):
        result = self.run_probe("sd_constant", 50)
        self.assertEqual(len(result["survival_mean_bias"]), 4)
        for row in result["survival_mean_bias"]:
            self.assertLess(row["summary"]["centered_logit_mean_bias_rms"]["rms"], 1e-12)
            self.assertGreater(row["summary"]["ce_jensen_gap"]["mean"], 0.)
        final = self.run_probe("sd_annealed", 100)
        for row in final["survival_mean_bias"]:
            self.assertEqual(row["survival_probability"], 1.)
            self.assertEqual(row["summary"]["centered_logit_mean_bias_rms"]["rms"], 0.)
            self.assertEqual(row["summary"]["ce_jensen_gap"]["rms"], 0.)

    def test_private_rng_no_parameter_or_grad_mutation_and_outer_inference_mode(self):
        self.model.train()
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = [p.detach().clone() for p in self.model.parameters()]
        rng = torch.random.get_rng_state().clone()
        with torch.inference_mode():
            x = self.x.clone()
            y = self.y.clone()
            table = corners(self.model, x)
            result = run_mechanisms(self.model, x, y, table, 100, "residual", forward)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for previous, parameter in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(previous, parameter))
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))
        json.dumps(result, allow_nan=False)

    def test_zero_vectors_have_explicit_null_angles(self):
        with torch.no_grad():
            for layer in self.model.layers:
                layer.weight.zero_()
                layer.bias.zero_()
        result = self.run_probe()
        for row in result["blocks"]:
            self.assertEqual(row["per_example"]["angle_radians"], [None]*3)
            self.assertEqual(row["per_example"]["relative_delta_norm"], [0.]*3)
            self.assertIsNone(row["all_examples"]["first_order_r2_vs_zero_change"])

    def test_recipe_exclusion_and_survival_indexing(self):
        self.assertEqual(self.run_probe("plain")["status"], "excluded")
        np.testing.assert_allclose(survival_probabilities("sd_constant", 1), [.9, .8, .7, .6])
        np.testing.assert_allclose(survival_probabilities("sd_annealed", 1), [.8, .6, .4, .2])
        self.assertEqual(survival_probabilities("sd_annealed", 100), [1.]*4)
        with self.assertRaises(ValueError):
            survival_probabilities("sd_constant", 0)

    def test_corner_order_and_extrapolation_weights(self):
        for mask in range(16):
            gains = [float(bool(mask & (1 << j))) for j in range(4)]
            weights = multilinear_weights(gains)
            expected = np.zeros(16)
            expected[mask] = 1
            np.testing.assert_array_equal(weights, expected)
        self.assertAlmostEqual(float(multilinear_weights([2., 1., 1., 1.]).sum()), 1.)


if __name__ == "__main__":
    unittest.main()
