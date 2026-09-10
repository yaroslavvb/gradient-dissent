"""Local fixtures only: no W&B SDK, network, login, or scientific-source edits."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.ciresan_stochastic_depth.halfdrop import upload_wandb as uploader


class UploadSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.directory = Path(self.tmp.name)
        self.manifest = self.directory/'main-manifest.json'
        self.jobs = [{'run_id': f'halfdrop-m{mask:02d}-s{seed}-v1', 'stage': 'evaluate',
                      'drop_mask': mask, 'seed': seed, 'epochs': 100, 'recipe': 'sd_constant',
                      'pmax': 0., 'lr': .01, 'batch_size': 64, 'momentum': .9,
                      'source_sha256': {'halfdrop/trainer.py': 'a'*64}}
                     for mask in range(16) for seed in (201, 202, 203)]
        self.write_manifest()
        self.patches = [patch.object(uploader, 'RESULTS', self.directory),
                        patch.object(uploader, 'MAIN_MANIFEST', self.manifest),
                        patch.object(uploader, 'SOURCES', tuple(j['run_id'] for j in self.jobs))]
        for p in self.patches: p.start()
        self.spec = self.jobs[6]; self.source = self.make_source(self.spec)

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.jobs))
        (self.directory/'main-source-freeze.json').write_text(json.dumps({
            'manifest_sha256': hashlib.sha256(self.manifest.read_bytes()).hexdigest()}))

    def make_source(self, spec):
        spec = copy.deepcopy(spec); mask = spec['drop_mask']
        score = {'n': 10000, 'accuracy': .98, 'loss': .08, 'errors': 200}
        train = {'n': 50000, 'accuracy': 1., 'loss': .0001}
        return {'run_id': spec['run_id'], 'spec': spec, 'complete': True, 'status': 'complete',
                'completed_epochs': 100, 'diverged': False, 'failure_reason': None, 'test_accessed': True,
                'source_sha256': copy.deepcopy(spec['source_sha256']),
                'drop_probabilities': [.5 if mask & (1 << j) else 0. for j in range(4)],
                'trained_final_parameters_sha256': 'b'*64, 'best_epoch': 10, 'best_validation_loss': .07,
                'selected_test': copy.deepcopy(score), 'final_test': copy.deepcopy(score),
                'selected_train': copy.deepcopy(train), 'final_train': copy.deepcopy(train),
                'training_seconds': 60., 'total_run_seconds': 70., 'telemetry': {'seconds': 2.},
                'telemetry_history': [{'epoch': e, 'training_seconds': 60.*e/100,
                    'run_elapsed_seconds': 5.+65.*e/100,
                    'metrics': {'validation/loss': .08, 'validation/accuracy_pct': 98.},
                    'diagnostics': {}} for e in (0, 100)]}

    def save(self, source):
        path = self.directory/(source['run_id']+'.json')
        path.write_text(json.dumps(source)); return path

    def test_complete_frozen_source_and_config_vector(self):
        path = self.save(self.source)
        uploader.validate_source(self.source, self.source['run_id'])
        payload = uploader.payload_for(self.source['run_id'])
        self.assertEqual(payload['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(payload['config']['eligible_mask'], 2)
        self.assertEqual(payload['config']['eligible_count'], 1)
        self.assertEqual(payload['config']['drop_probs'], [0., .5, 0., 0.])
        self.assertEqual(payload['config']['inference_keep_probability'], 1.)
        digest = hashlib.sha256(json.dumps({'config': payload['config'], 'rows': payload['rows']},
                                          sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.assertEqual(payload['content_sha256'], digest)
        self.assertEqual(payload['run_id'], 'halfdrop-'+digest[:20])
        self.assertEqual(payload['scientific_summary']['selected/test_accuracy_pct'], 98.)
        self.assertEqual(payload['scientific_summary']['final/test_loss'], .08)
        self.assertEqual(payload['scientific_summary']['time/training_seconds'], 60.)
        self.assertFalse(any('_runtime' in row or 'fps' in row for row in payload['rows']))
        self.assertTrue(all('test/loss' not in row and 'test/accuracy_pct' not in row for row in payload['rows']))

    def test_partial_failed_or_wrong_run_rejected(self):
        for key, value in [('complete', False), ('status', 'training'), ('completed_epochs', 99),
                           ('diverged', True), ('failure_reason', 'Nonfinite'), ('test_accessed', False),
                           ('run_id', 'other-run')]:
            bad = copy.deepcopy(self.source); bad[key] = value
            with self.subTest(key=key), self.assertRaises(uploader.IntegrityError):
                uploader.validate_source(bad, self.source['run_id'])

    def test_wrong_frozen_seed_mask_lr_or_code_rejected(self):
        for key, value in [('seed', 202), ('drop_mask', 4), ('drop_mask', True), ('lr', .03), ('stage', 'pilot')]:
            bad = copy.deepcopy(self.source); bad['spec'][key] = value
            with self.subTest(key=key), self.assertRaises(uploader.IntegrityError):
                uploader.validate_source(bad, self.source['run_id'])
        bad = copy.deepcopy(self.source); bad['source_sha256']['halfdrop/trainer.py'] = 'c'*64
        with self.assertRaises(uploader.IntegrityError): uploader.validate_source(bad, self.source['run_id'])

    def test_manifest_change_without_new_freeze_rejected(self):
        self.jobs[6]['lr'] = .03
        self.manifest.write_text(json.dumps(self.jobs))
        with self.assertRaises(uploader.IntegrityError): uploader.validate_source(self.source, self.source['run_id'])

    def test_both_test_endpoints_required_and_finite(self):
        for endpoint in ('selected_test', 'final_test'):
            for change in ({'n': 9999}, {'accuracy': float('nan')}, {'accuracy': 1.1},
                           {'loss': float('inf')}, {'loss': -1.}):
                bad = copy.deepcopy(self.source); bad[endpoint].update(change)
                with self.subTest(endpoint=endpoint, change=change), self.assertRaises(uploader.IntegrityError):
                    uploader.validate_source(bad, self.source['run_id'])
            bad = copy.deepcopy(self.source); del bad[endpoint]
            with self.assertRaises(uploader.IntegrityError): uploader.validate_source(bad, self.source['run_id'])

    def test_probability_mismatch_rejected_before_upload(self):
        bad = copy.deepcopy(self.source); bad['drop_probabilities'] = [0., .5, .5, 0.]
        self.save(bad)
        with self.assertRaises(uploader.IntegrityError): uploader.payload_for(bad['run_id'])

    def test_identical_telemetry_different_subset_has_different_content_identity(self):
        self.save(self.source); first = uploader.payload_for(self.source['run_id'])
        second_source = self.make_source(self.jobs[0]); self.save(second_source)
        second = uploader.payload_for(second_source['run_id'])
        self.assertEqual(first['rows'], second['rows'])
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertEqual(second['config']['drop_probs'], [0., 0., 0., 0.])

    def test_remote_history_prefix_checks_reject_conflicts(self):
        expected = [{'epoch': 0, 'training_seconds': 0., 'validation/loss': .1},
                    {'epoch': 100, 'training_seconds': 60., 'validation/loss': .08}]
        actual = [{'_step': i, '_runtime': 100+i, **row} for i, row in enumerate(expected)]
        self.assertEqual(uploader.verified_prefix(expected, actual), (2, 6))
        for corrupt in ([actual[1]], [actual[0], actual[0]],
                        [{**actual[0], 'validation/loss': .2}], [{**actual[0], 'unplanned_metric': 1.}]):
            with self.assertRaises(uploader.IntegrityError): uploader.verified_prefix(expected, corrupt)


if __name__ == '__main__': unittest.main()
