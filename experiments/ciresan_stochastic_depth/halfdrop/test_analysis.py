"""CPU-only synthetic statistical/leakage checks; no real outcomes or cloud use."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a


def validation_fixture(seed, mask):
    spec = {'run_id': f'synthetic-halfdrop-m{mask:02d}-s{seed}', 'stage': 'evaluate',
            'seed': seed, 'drop_mask': mask, 'epochs': 100, 'batch_size': 64,
            'lr': .01, 'momentum': .9, 'recipe': 'sd_constant', 'pmax': 0,
            'input_scale': 1/255, 'output_relu': False,
            'source_sha256': {'synthetic.py': 'a'*64, 'halfdrop/PROTOCOL.md': a.sha(a.HERE/'PROTOCOL.md')}}
    history = []
    for epoch in range(1, 101):
        row = {'epoch': epoch, 'training_seconds': float(epoch),
               'drop_probabilities': [.5 if mask & (1 << j) else 0 for j in range(4)]}
        if epoch == 1 or epoch % 5 == 0:
            row['validation'] = {'loss': .1+abs(epoch-10)*.001+mask*.0001, 'accuracy': .95+mask*.0001, 'n': 10000}
        history.append(row)
    selected = history[9]['validation']
    view = {'spec': spec, 'history': history, 'best_epoch': 10,
        'best_validation_loss': selected['loss'], 'selected_validation': copy.deepcopy(selected),
        'final_validation': copy.deepcopy(history[-1]['validation']), 'source_sha256': spec['source_sha256'],
        'dataset': {'train_size': 50000, 'val_size': 10000, 'test_included': False,
                    **{name: 'd'*64 for name in ('training_data_sha256', 'split_sha256', 'train_indices_sha256', 'val_indices_sha256')}},
        'steps_per_epoch': 781, 'training_examples_per_epoch': 49984,
        'initial_parameters_sha256': str(seed).zfill(64), 'initialization_rng_sha256': str(seed).zfill(64),
        'executed_epoch_order_sha256': str(seed).zfill(64), 'raw_mask_draws_sha256': str(seed).zfill(64),
        'executed_epoch_masks_sha256': str(mask).zfill(64), 'test_accessed': False,
        'status': 'complete', 'complete': True, 'diverged': False, 'failure_reason': None,
        'checkpoints': {name: {'sha256': 'b'*64} for name in ('best.pt', 'final.pt', 'checkpoint.pt')}}
    return spec, view


def write_commitment(manifest, jobs):
    a.write_json(manifest, jobs)
    a.write_json(manifest.with_name(manifest.name.replace('manifest', 'source-freeze')),
                 {'manifest_sha256': a.sha(manifest), 'source_sha256': jobs[0]['source_sha256']})


def archive_record(path):
    files = {'t10k-images-idx3-ubyte.gz': {'sha256': 'c'*64}, 't10k-labels-idx1-ubyte.gz': {'sha256': 'd'*64}}
    a.write_json(path.parent/'result.json', {'artifacts': {path.name: {'sha256': a.sha(path), 'bytes': path.stat().st_size}},
                                           'dataset': {'files': files}})
    return files


def panel_fixture(n=200):
    labels = (np.arange(n) % 10).astype(np.uint8)
    records, arrays = {}, {}
    for seed in a.SEEDS:
        for mask in range(16):
            records[(seed, mask)] = {'spec': {'run_id': f'synthetic-{seed}-{mask}'}, 'best_epoch': 10,
                'training_seconds': 100-mask, 'total_run_seconds': 102-mask,
                'checkpoints': {name: {'sha256': 'b'*64} for name in ('best.pt', 'final.pt')},
                **{endpoint+'_'+split: {'accuracy': .99, 'loss': .05, 'n': 50000 if split == 'train' else 10000}
                   for endpoint in a.ENDPOINTS for split in ('train', 'validation')}}
            for endpoint in a.ENDPOINTS:
                pred = labels.copy()
                wrong = np.arange((seed-201)+10, n, 11)
                pred[wrong] = (pred[wrong]+1)%10
                if mask:
                    i = (seed + mask) % n
                    pred[i] = (pred[i]+1)%10
                    repair = wrong[mask % len(wrong)]
                    pred[repair] = labels[repair]
                if endpoint == 'final':
                    pred[:mask] = labels[:mask]
                probs = np.full((n, 10), .1/9)
                probs[np.arange(n), pred] = .9
                ce = -np.log(probs[np.arange(n), labels])
                arrays[(seed, mask, endpoint)] = {'pred': pred, 'probs': probs, 'ce': ce,
                                                 'accuracy': float((pred == labels).mean()), 'mean_ce': float(ce.mean())}
    return records, arrays, labels


class AnalysisTests(unittest.TestCase):
    def test_t_interval_and_undefined_denominators(self):
        row = a.interval([1, 2, 3])
        self.assertEqual((row['mean'], row['sample_sd'], row['n'], row['df']), (2., 1., 3, 2))
        self.assertAlmostEqual(row['ci95_high'], 2+a.T95[3]/np.sqrt(3))
        self.assertIsNone(a.interval([None, None])['mean'])
        self.assertIsNone(a.mean_optional([1, None]))
        cmp = a.comparison(np.array([1]), np.array([1]), np.array([0]))
        self.assertIsNone(cmp['conditional_harm'])
        self.assertEqual(cmp['harm_rate'], 0)

    def test_paired_harm_repair_identity(self):
        labels = np.array([0, 0, 1, 1])
        baseline = np.array([0, 1, 1, 0])
        target = np.array([1, 0, 1, 1])
        c = a.comparison(target, baseline, labels)
        self.assertEqual((c['harm_count'], c['repair_count'], c['conditional_harm']), (1, 2, .5))
        self.assertEqual(c['accuracy_pp'], 25)
        self.assertEqual(c['accuracy_pp'], 100*(c['repair_rate']-c['harm_rate']))
        self.assertIsNone(a.comparison(target, baseline, labels, np.zeros(4, dtype=bool))['harm_rate'])

    def test_validation_paths_integer_error_ties_ce_then_lex(self):
        views = {(s, m): validation_fixture(s, m)[1] for s in a.SEEDS for m in range(16)}
        for (seed, mask), view in views.items():
            for row in view['history']:
                if 'validation' in row:
                    row['validation']['accuracy'] = .98
                    row['validation']['loss'] = .1+abs(row['epoch']-10)*.01
            view['best_validation_loss'] = .1
            view['selected_validation'] = copy.deepcopy(view['history'][9]['validation'])
            view['final_validation'] = copy.deepcopy(view['history'][-1]['validation'])
        # Equal integer error counts for all24 paths, one prefix has lower CE.
        for seed in a.SEEDS:
            views[(seed, 8)]['history'][9]['validation']['loss'] = .09
            views[(seed, 8)]['best_validation_loss'] = .09
            views[(seed, 8)]['selected_validation'] = copy.deepcopy(views[(seed, 8)]['history'][9]['validation'])
        paths = a.validation_paths(views)
        self.assertEqual(len(paths), 24)
        self.assertEqual({p['validation_error_count'] for p in paths}, {1800})
        self.assertEqual(next(p for p in paths if p['validation_selected'])['order'], [3, 0, 1, 2])
        self.assertEqual(a.prefixes([3, 0, 1, 2]), [0, 8, 9, 11, 15])

    def test_missing_validation_observation_and_changed_copy_are_rejected(self):
        _, view = validation_fixture(201, 0)
        del view['history'][4]['validation']
        with self.assertRaisesRegex(ValueError, 'cadence'):
            a.selected_validation(view)
        _, view = validation_fixture(201, 0)
        view['selected_validation']['accuracy'] = .5
        with self.assertRaisesRegex(ValueError, 'copies disagree'):
            a.selected_validation(view)

    def test_late_raw_download_preserves_canonical_validation_source(self):
        with tempfile.TemporaryDirectory(prefix='halfdrop-synthetic-') as td:
            root = Path(td); job, view = validation_fixture(201, 0)
            flat = root/(job['run_id']+'-validation.json')
            a.write_json(flat, view)
            self.assertEqual(a.find_file(root, job['run_id'], 'validation'), flat)
            raw = root/'raw'/job['run_id']/'validation.json'
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps(view, separators=(',', ':')))
            self.assertEqual(a.find_file(root, job['run_id'], 'validation'), flat)
            changed = copy.deepcopy(view); changed['best_epoch'] = 5
            a.write_json(raw, changed)
            with self.assertRaisesRegex(ValueError, 'copies disagree'):
                a.find_file(root, job['run_id'], 'validation')

    def test_aggregate_pairing_kmeans_edges_and_flat_ui_fields(self):
        records, payloads, labels = panel_fixture()
        out = a.analyze_arrays(records, payloads, labels)
        self.assertEqual((len(out['models']), len(out['by_k']), len(out['edges']), len(out['marginals'])), (32, 10, 64, 40))
        model = next(m for m in out['models'] if m['endpoint'] == 'selected' and m['drop_mask'] == 3)
        self.assertEqual(model['metrics']['accuracy']['n'], 3)
        self.assertEqual(model['per_class'][1]['accuracy'], model['per_class'][1]['metrics']['accuracy'])
        baseline = next(m for m in out['models'] if m['endpoint'] == 'selected' and m['drop_mask'] == 0)
        self.assertEqual(baseline['paired_vs_none']['accuracy_pp']['values'], [0., 0., 0.])
        krow = next(r for r in out['by_k'] if r['endpoint'] == 'selected' and r['k'] == 2)
        expected = [np.mean([payloads[(s, m, 'selected')]['accuracy'] for m in (3, 5, 6, 9, 10, 12)]) for s in a.SEEDS]
        np.testing.assert_allclose(krow['metrics']['accuracy']['values'], expected)
        marginal = next(m for m in out['marginals'] if m['endpoint'] == 'selected' and m['layer'] == 0 and m['context_size'] == 'all')
        edges = [e for e in out['edges'] if e['endpoint'] == 'selected' and e['branch'] == 0]
        self.assertEqual(len(edges), 8)
        np.testing.assert_allclose(marginal['accuracy_pp']['values'], [np.mean([e['seeds'][i]['accuracy_pp'] for e in edges]) for i in range(3)])

    def test_gallery_fixed_and_outcome_selection(self):
        _, payloads, labels = panel_fixture()
        fixed = sorted(int(i) for d in range(10) for i in np.flatnonzero(labels == d)[:10])
        images = np.zeros((len(labels), 28, 28), dtype=np.uint8)
        gallery = a.make_gallery(201, 'selected', payloads, labels, images, fixed)
        for row in gallery['models']:
            self.assertLessEqual(len(row['examples']), 140)
            self.assertTrue(set(fixed) <= {e['index'] for e in row['examples']})
            for e in row['examples']:
                self.assertIn(str(e['index']), gallery['images'])
                if e['index'] not in fixed:
                    self.assertTrue(any(reason.startswith('first_') for reason in e['selection']))
                if e['category'] == 'harmed':
                    self.assertEqual(e['baseline_pred'], e['label']); self.assertNotEqual(e['pred'], e['label'])
        self.assertTrue(all(e['category'] == 'unchanged' for e in gallery['models'][0]['examples']))

    def test_freeze_never_reads_new_test_results_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory(prefix='halfdrop-synthetic-') as td:
            root = Path(td); jobs = []
            for seed in a.SEEDS:
                for mask in range(16):
                    spec, view = validation_fixture(seed, mask); jobs.append(spec)
                    directory = root/'raw'/spec['run_id']; directory.mkdir(parents=True)
                    a.write_json(directory/'validation.json', view)
                    # A test/result reader would fail even to parse this poison.
                    (root/(spec['run_id']+'.json')).write_text('NEW TEST OUTCOMES MUST NOT BE READ')
            manifest = root/'manifest.json'; write_commitment(manifest, jobs)
            archive = root/'old-archive.npz'
            np.savez(archive, test_labels=(np.arange(10000)%10).astype(np.uint8))
            archive_record(archive)
            frozen = a.freeze_order(manifest, root, archive)
            self.assertFalse(frozen['frozen_inputs']['new_test_outcomes_read'])
            self.assertEqual(len(frozen['frozen_inputs']['gallery']['indices']), 100)
            self.assertEqual(a.freeze_order(manifest, root, archive), frozen)
            path = root/'raw'/jobs[0]['run_id']/'validation.json'
            changed = a.read_json(path); changed['history'][9]['validation']['accuracy'] = .5
            changed['selected_validation'] = copy.deepcopy(changed['history'][9]['validation'])
            a.write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                a.freeze_order(manifest, root, archive)

    def test_source_and_initial_pairing_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='halfdrop-synthetic-') as td:
            root = Path(td); jobs = []
            for seed in a.SEEDS:
                for mask in range(16):
                    spec, view = validation_fixture(seed, mask); jobs.append(spec)
                    a.write_json(root/(spec['run_id']+'-validation.json'), view)
            path = root/(jobs[1]['run_id']+'-validation.json')
            changed = a.read_json(path); changed['initial_parameters_sha256'] = 'e'*64
            a.write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'initial_parameters'):
                a.read_validation_views(jobs, root)

    def test_endpoint_prediction_consistency(self):
        labels = (np.arange(20)%10).astype(np.uint8)
        logits = np.eye(10, dtype=np.float32)[labels]*3
        shifted = logits.astype(np.float64)-3
        exp = np.exp(shifted); probs = exp/exp.sum(1, keepdims=True)
        ce = -np.log(probs[np.arange(20), labels])
        summary = {'n': 20, 'accuracy': 1., 'loss': float(ce.mean()), 'wrong_indices': []}
        with tempfile.TemporaryDirectory(prefix='halfdrop-synthetic-') as td:
            path = Path(td)/'test.npz'
            np.savez(path, indices=np.arange(20), labels=labels, pred=labels, logits=logits, probs=probs, ce=ce)
            self.assertEqual(a.read_endpoint(path, summary, labels)['accuracy'], 1.)
            changed = probs.copy(); changed[0] = .1
            np.savez(path, indices=np.arange(20), labels=labels, pred=labels, logits=logits, probs=changed, ce=ce)
            with self.assertRaisesRegex(ValueError, 'softmax'):
                a.read_endpoint(path, summary, labels)
            changed = probs.copy(); changed[0, 1] = -5e-7
            np.savez(path, indices=np.arange(20), labels=labels, pred=labels, logits=logits, probs=changed, ce=ce)
            with self.assertRaisesRegex(ValueError, 'probability bounds'):
                a.read_endpoint(path, summary, labels)

    def test_end_to_end_freeze_evaluate_and_compact_export_in_temporary_directory(self):
        exporter_path = a.REPO/'scripts/build_mnist_halfdrop_data.py'
        spec = importlib.util.spec_from_file_location('synthetic_halfdrop_exporter', exporter_path)
        exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory(prefix='halfdrop-SYNTHETIC-NOT-RESULTS-') as td:
            root = Path(td); labels = (np.arange(10000)%10).astype(np.uint8)
            archive = root/'old-archive.npz'
            np.savez_compressed(archive, test_labels=labels, test_images=np.zeros((10000, 28, 28), dtype=np.uint8))
            files = archive_record(archive)
            jobs = []
            # Create validation artifacts first; test files are still absent.
            for seed in a.SEEDS:
                for mask in range(16):
                    job, view = validation_fixture(seed, mask); jobs.append(job)
                    a.write_json(root/'raw'/job['run_id']/'validation.json', view)
            manifest = root/'manifest.json'; write_commitment(manifest, jobs)
            a.freeze_order(manifest, root, archive)
            self.assertFalse(list(root.glob('*-test.npz')))
            for job in jobs:
                directory = root/'raw'/job['run_id']
                view = a.read_json(directory/'validation.json')
                raw = copy.deepcopy(view)
                raw.update({'validation_view': copy.deepcopy(view), 'test_accessed': True,
                            'training_seconds': 100.-job['drop_mask'], 'total_run_seconds': 110., 'artifacts': {},
                            'test_labels_sha256': a.array_sha(labels.astype('<i8')), 'test_dataset': {'files': files}})
                for endpoint in a.ENDPOINTS:
                    pred = labels.copy()
                    offset = job['seed']-201
                    indices = np.arange(offset+job['drop_mask'], 10000, 37+job['drop_mask'])
                    pred[indices] = (pred[indices]+1)%10
                    logits = np.eye(10, dtype=np.float32)[pred]*4
                    z = logits.astype(np.float64)-4
                    exp = np.exp(z); probs = exp/exp.sum(1, keepdims=True)
                    ce = np.log(exp.sum(1))-z[np.arange(10000), labels]
                    path = directory/(endpoint+'-test.npz')
                    np.savez_compressed(path, indices=np.arange(10000), labels=labels, pred=pred, logits=logits, probs=probs, ce=ce)
                    raw['artifacts'][path.name] = {'sha256': a.sha(path), 'bytes': path.stat().st_size}
                    raw[endpoint+'_test'] = {'accuracy': float((pred == labels).mean()), 'loss': float(ce.mean()),
                                            'n': 10000, 'wrong_indices': np.flatnonzero(pred != labels).tolist()}
                    raw[endpoint+'_train'] = {'accuracy': .999, 'loss': .001, 'n': 50000}
                    raw[endpoint+'_validation'] = view['history'][9 if endpoint == 'selected' else 99]['validation']
                a.write_json(root/(job['run_id']+'.json'), raw)
            # Match the raw validation views exactly; endpoint train summaries
            # are source fields outside the view in this synthetic fixture.
            result = a.evaluate(manifest, root, archive)
            self.assertEqual(result['verification']['complete_runs'], 48)
            data, galleries = exporter.build(root)
            self.assertEqual((len(data['models']), len(data['paths']), len(galleries)), (32, 24, 6))
            self.assertTrue((root/'summary.md').exists())
            self.assertEqual(data['wandb_runs'], {})
            for text in galleries.values():
                self.assertEqual(len(json.loads(text)['models']), 16)
            # Evidence mutation must stop a subsequent publication build.
            (root/(jobs[0]['run_id']+'.json')).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed or missing evidence'):
                exporter.build(root)


if __name__ == '__main__':
    unittest.main()
