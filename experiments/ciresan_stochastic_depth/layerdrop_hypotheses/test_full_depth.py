"""CPU-only qualification; never downloads data or allocates a cloud device."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_data import CiresanMLP
from layerdrop_hypotheses.audit import masked_forward as old_forward
from layerdrop_hypotheses.full_depth import (
    RECIPES, masked_forward, old_mask_to_full, mask_costs, enumerate_logits,
    summarize, verify_archived_panel, validate_spec,
)


class FullDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.x = torch.rand(11, 8)*255
        self.y = np.arange(11)%3

    def model(self, recipe='residual', output_relu=False):
        return CiresanMLP(recipe=recipe, pmax=.4, output_relu=output_relu,
                         widths=(8, 12, 10, 8, 6, 4, 3)).eval()

    def payload(self):
        model = self.model()
        z = enumerate_logits(model, self.x, batch_size=4).numpy()
        return summarize(z, self.y, model.widths, [2, 0])

    def test_full_and_all_old_masks_bitwise_every_recipe(self):
        for recipe in RECIPES:
            for relu in (False, True):
                model = self.model(recipe, relu)
                self.assertTrue(torch.equal(masked_forward(model, self.x, 63), model(self.x)))
                for old in range(16):
                    self.assertTrue(torch.equal(masked_forward(model, self.x, old_mask_to_full(old)),
                                                old_forward(model, self.x, old)))

    def test_head_absent_short_circuits_every_affine(self):
        model = self.model()
        def forbidden(*args):
            raise AssertionError('Abstention executed a learned affine')
        handles = [layer.register_forward_hook(forbidden) for layer in model.layers]
        try:
            for mask in range(32):
                self.assertIsNone(masked_forward(model, self.x, mask))
        finally:
            for handle in handles:
                handle.remove()

    def test_removed_stem_padding_and_body_crop_exact(self):
        model = self.model()
        h = F.pad(self.x*model.input_scale, (0, 4))
        expected = model.layers[-1](F.relu(h[:, :4]))
        self.assertTrue(torch.equal(masked_forward(model, self.x, 32), expected))
        # Keeping body1 really consumes the padded pixel representation.
        h = F.relu(h[:, :10]+model.layers[1](h))
        expected = model.layers[-1](h[:, :4])
        self.assertTrue(torch.equal(masked_forward(model, self.x, 34), expected))

    def test_exact_branch_call_inventory(self):
        model = self.model('plain')
        for mask in (32, 33, 34, 42, 62, 63):
            called = []
            handles = [layer.register_forward_hook(lambda module, inp, out, j=j: called.append(j))
                       for j, layer in enumerate(model.layers)]
            try:
                masked_forward(model, self.x, mask)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(called, [j for j in range(6) if mask & (1 << j)])

    def test_cost_distinguishes_nominal_switches_from_abstention(self):
        rows = mask_costs((784, 2500, 2000, 1500, 1000, 500, 10))
        self.assertEqual(rows[63]['raw_macs'], 11965000)
        self.assertEqual(rows[33]['raw_macs'], 1965000)
        self.assertEqual(rows[32]['raw_macs'], 5000)
        self.assertEqual(rows[31]['raw_macs'], 0)
        self.assertEqual(rows[31]['nominal_selected_macs'], 11960000)
        self.assertEqual(rows[0]['cost'], 0)
        self.assertEqual(rows[31]['executed_affines'], 0)
        for mask in range(16):
            costs = [5000000, 3000000, 1500000, 500000]
            expected = 1965000 + sum(c for j, c in enumerate(costs) if mask & (1 << j))
            self.assertEqual(rows[old_mask_to_full(mask)]['raw_macs'], expected)

    def test_mask_validation_and_no_training_dropout(self):
        model = self.model()
        for mask in (-1, 64, True, 2.5):
            with self.assertRaises(ValueError):
                masked_forward(model, self.x, mask)
        model.train()
        with self.assertRaises(ValueError):
            masked_forward(model, self.x, 63)
        self.assertEqual([old_mask_to_full(m) for m in range(16)], list(range(33, 64, 2)))

    def test_summary_abstention_perclass_gallery_and_npz(self):
        summary, arrays = self.payload()
        self.assertEqual(arrays['pred'].shape, (64, 11))
        self.assertEqual(arrays['pred'].dtype, np.int8)
        self.assertTrue((arrays['pred'][:32] == -1).all())
        self.assertTrue(np.isnan(arrays['ce'][:32]).all())
        self.assertTrue(np.isfinite(arrays['ce'][32:]).all())
        json.dumps(summary, allow_nan=False)
        self.assertEqual([e['index'] for e in summary['examples']], [2, 0])
        for row in summary['masks']:
            if row['id'] < 32:
                self.assertIsNone(row['accuracy']); self.assertIsNone(row['ce'])
                self.assertEqual(row['coverage'], 0); self.assertEqual(row['correct_output_rate'], 0)
                self.assertEqual(row['abstentions'], 11)
                self.assertTrue(all(c['accuracy'] is None for c in row['per_class']))
            else:
                self.assertEqual(row['coverage'], 1)
                self.assertEqual(row['accuracy'], row['correct_output_rate'])
                self.assertAlmostEqual(row['accuracy'], sum(c['n']*c['accuracy'] for c in row['per_class'])/11)
            self.assertAlmostEqual(summary['dense_accuracy']-row['harm']+row['repair'], row['correct_output_rate'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.npz'; np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertTrue(np.array_equal(loaded['pred'], arrays['pred']))

    def test_cross_entropy_and_invalid_gallery(self):
        z = np.zeros((32, 3, 3))
        summary, arrays = summarize(z, np.array([0, 1, 2]), (8, 12, 10, 8, 6, 4, 3))
        self.assertTrue(np.allclose(arrays['ce'][32:], np.log(3)))
        self.assertEqual(summary['masks'][32]['accuracy'], 1/3)
        for gallery in ([0, 0], [3], [True]):
            with self.assertRaises(ValueError):
                summarize(z, np.array([0, 1, 2]), (8, 12, 10, 8, 6, 4, 3), gallery)

    def test_archive_parity_and_corruption_failures(self):
        _, arrays = self.payload(); ids = [old_mask_to_full(m) for m in range(16)]
        archived = {'pred': arrays['pred'][ids].copy(), 'ce': arrays['ce'][ids].copy(),
                    'labels': arrays['labels'].copy()}
        ref = {'accuracy': float(np.mean(arrays['pred'][63] == arrays['labels'])),
               'loss': float(arrays['ce'][63].mean()),
               'wrong_indices': np.flatnonzero(arrays['pred'][63] != arrays['labels']).tolist()}
        self.assertTrue(verify_archived_panel(arrays, archived, ref)['all_16_predictions_exact'])
        for field in ('pred', 'labels', 'ce'):
            bad = {k: v.copy() for k, v in archived.items()}
            bad[field].flat[0] += 1
            with self.assertRaises(AssertionError):
                verify_archived_panel(arrays, bad, ref)
        bad = copy.deepcopy(ref); bad['wrong_indices'] = [] if ref['wrong_indices'] else [0]
        with self.assertRaises(AssertionError):
            verify_archived_panel(arrays, archived, bad)

    def test_frozen_panel_validation(self):
        spec = {'seed': 101, 'recipes': ['residual'], 'states': ['selected'], 'models': [
            {'id': 'residual-selected-s101', 'recipe': 'residual', 'state': 'selected',
             'seed': 101, 'epoch': 10, 'checkpoint_sha256': 'a'*64, 'gallery_indices': [3, 10]}]}
        self.assertEqual(validate_spec(spec)[0], ['residual'])
        for change in ('missing_model', 'bad_hash', 'bad_seed', 'duplicate_gallery', 'new_state'):
            bad = copy.deepcopy(spec)
            if change == 'missing_model': bad['models'] = []
            if change == 'bad_hash': bad['models'][0]['checkpoint_sha256'] = 'unfrozen'
            if change == 'bad_seed': bad['models'][0]['seed'] = 102
            if change == 'duplicate_gallery': bad['models'][0]['gallery_indices'] = [3, 3]
            if change == 'new_state': bad['states'] = ['selected', 'final']
            with self.assertRaises(ValueError):
                validate_spec(bad)


if __name__ == '__main__':
    unittest.main()
