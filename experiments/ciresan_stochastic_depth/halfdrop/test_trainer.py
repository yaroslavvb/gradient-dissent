"""Meaningful CPU qualification; no datasets, cloud calls or paid devices."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from halfdrop.trainer import (probabilities, validate_spec, epoch_randomness,
                             parameter_hash, prediction_arrays, evaluate)
from model_data import CiresanMLP
from optimization.stochastic_graph import forward_with_mask, active_parameter_indices, update_active_sgd
from telemetry.metrics import probe_statistics


class HalfDropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        self.x = torch.rand(9, 8)*255; self.y = torch.arange(9)%10

    def model(self):
        return CiresanMLP(recipe='sd_constant', pmax=0., widths=(8, 12, 10, 8, 6, 4, 10))

    def spec(self, **changes):
        return {'run_id': 'halfdrop-m02-s201-v1', 'stage': 'evaluate', 'seed': 201,
                'drop_mask': 2, 'epochs': 100, **changes}

    def test_all16_probability_vectors_and_validation(self):
        for mask in range(16):
            p = probabilities(mask)
            self.assertEqual(sum(v == .5 for v in p), mask.bit_count())
            self.assertEqual(p, [.5 if mask & (1 << j) else 0. for j in range(4)])
            self.assertEqual(validate_spec(self.spec(drop_mask=mask))['pmax'], 0)
        for mask in (-1, 16, True, .5):
            with self.assertRaises(ValueError): probabilities(mask)
        for change in ({'lr': .03}, {'pmax': .5}, {'precision': 'bf16'}, {'seed': 101},
                       {'drop_probabilities': [0, 0, 0, 0]}, {'epochs': 99},
                       {'telemetry_probe_n': 256}, {'telemetry': False}, {'stage': 'tune'}):
            with self.assertRaises(ValueError): validate_spec(self.spec(**change))
        self.assertEqual(validate_spec(self.spec(stage='pilot', epochs=2))['epochs'], 2)

    def test_fixed_draws_and_order_for_all_subsets_across_epochs(self):
        reference = None
        for mask in range(16):
            order_rng = torch.Generator().manual_seed(100201)
            mask_rng = torch.Generator().manual_seed(200201)
            rows = []
            for _ in range(3):
                order, draws, active = epoch_randomness(137, 8, probabilities(mask), order_rng, mask_rng)
                self.assertEqual(draws.shape, (17, 4))
                for j in range(4):
                    expected = draws[:, j] >= .5 if mask & (1 << j) else torch.ones(17, dtype=torch.bool)
                    self.assertTrue(torch.equal(active[:, j], expected))
                rows.append((order, draws))
            if reference is None: reference = rows
            else:
                for (order, draws), (old_order, old_draws) in zip(rows, reference):
                    self.assertTrue(torch.equal(order, old_order))
                    self.assertTrue(torch.equal(draws, old_draws))

    def test_identical_initialization_and_dense_subset0_operator(self):
        hashes = []
        for _ in range(16):
            torch.manual_seed(201); model = self.model()
            hashes.append(parameter_hash(model))
        self.assertEqual(len(set(hashes)), 1)
        torch.manual_seed(201)
        dense = CiresanMLP(recipe='residual', widths=(8, 12, 10, 8, 6, 4, 10))
        self.assertEqual(parameter_hash(dense), hashes[0])
        self.assertTrue(torch.equal(model(self.x, drop_probs=[0]*4), dense(self.x)))

    def test_single_branch_halfdrop_scale_and_bypass(self):
        model = self.model().train(); p = [0., .5, 0., 0.]
        for active in ((True, True, True, True), (True, False, True, True)):
            h = F.relu(model.layers[0](self.x*model.input_scale))
            for j, layer in enumerate(model.layers[1:-1]):
                skip = h[:, :layer.out_features]
                h = F.relu(skip+layer(h)*(2. if j == 1 else 1.)) if active[j] else F.relu(skip)
            expected = model.layers[-1](h)
            self.assertTrue(torch.equal(model(self.x, drop_probs=p, active=list(active)), expected))
            self.assertTrue(torch.equal(forward_with_mask(model, self.x, active, torch.tensor([1., 2., 1., 1.])), expected))

    def test_eval_always_full_and_unscaled_even_if_given_training_gates(self):
        model = self.model().eval(); baseline = model(self.x)
        for mask in range(16):
            actual = model(self.x, drop_probs=probabilities(mask), active=[False]*4)
            self.assertTrue(torch.equal(actual, baseline))
        score = evaluate(model, self.x, self.y, batch_size=9)
        self.assertEqual(score['accuracy'], int((baseline.argmax(1) == self.y).sum())/len(self.y))
        self.assertAlmostEqual(score['loss'], float(F.cross_entropy(baseline, self.y).detach()), places=6)

    def test_skipped_branch_does_not_execute_or_receive_sgd_momentum_update(self):
        model = self.model(); active = (True, False, True, True)
        def forbidden(*args): raise AssertionError('Skipped branch executed')
        handle = model.layers[2].register_forward_hook(forbidden)
        try: model(self.x, drop_probs=[0., .5, 0., 0.], active=list(active))
        finally: handle.remove()
        params = list(model.parameters()); saved = [p.detach().clone() for p in params]
        buffers = [torch.full_like(p, 2.) for p in params]
        for p in params: p.grad = torch.ones_like(p)
        indices = active_parameter_indices(model, active)
        update_active_sgd(params, buffers, indices, lr=.01, momentum=.9, shrinkage=0.)
        for i, (p, old) in enumerate(zip(params, saved)):
            if i not in indices:
                self.assertTrue(torch.equal(p, old)); self.assertTrue(torch.equal(buffers[i], torch.full_like(p, 2.)))
            else:
                torch.testing.assert_close(p, old-.028)

    def test_prediction_probabilities_margins_labels_and_error_ids(self):
        logits = np.random.default_rng(9).normal(size=(13, 10)).astype(np.float32)*7
        labels = np.arange(13)%10
        summary, arrays = prediction_arrays(logits, labels)
        reference = F.cross_entropy(torch.tensor(logits, dtype=torch.float64), torch.tensor(labels), reduction='none').numpy()
        np.testing.assert_allclose(arrays['ce'], reference, rtol=0, atol=1e-13)
        np.testing.assert_allclose(arrays['probs'].sum(1), np.ones(13), rtol=0, atol=1e-14)
        self.assertTrue(np.array_equal(arrays['pred'], logits.argmax(1)))
        self.assertTrue(np.all(arrays['margin'] >= 0))
        self.assertTrue(np.array_equal(arrays['true_margin'] > 0, arrays['pred'] == labels))
        self.assertEqual(summary['wrong_indices'], np.flatnonzero(logits.argmax(1) != labels).tolist())
        self.assertEqual(sum(c['n'] for c in summary['per_class']), 13)
        bad = logits.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): prediction_arrays(bad, labels)

    def test_dense_telemetry_probe_preserves_parameters_rng_and_persistent_grads(self):
        model = self.model().train()
        for p in model.parameters(): p.grad = torch.ones_like(p)*.3
        before = parameter_hash(model); rng = torch.get_rng_state().clone()
        gradients = [p.grad.clone() for p in model.parameters()]
        probe = probe_statistics(model, self.x, self.y, compute_gradients=True)
        self.assertEqual(probe['n'], 9)
        self.assertEqual(parameter_hash(model), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(model.training)
        for p, grad in zip(model.parameters(), gradients): self.assertTrue(torch.equal(p.grad, grad))


if __name__ == '__main__': unittest.main()
