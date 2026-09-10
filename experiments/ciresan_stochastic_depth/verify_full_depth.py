"""Recompute the six-affine audit from saved NPZ files; CPU only, no network.

Run: uv run python experiments/ciresan_stochastic_depth/verify_full_depth.py
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/full-depth'
COSTS = [1960000, 5000000, 3000000, 1500000, 500000, 5000]


def read(path):
    return json.loads(path.read_text())


def verify():
    manifest_path = OUT / 'manifest-v2.json'
    freeze = read(OUT / 'source-freeze-v2.json')
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == freeze['manifest_sha256']
    assert hashlib.sha256((OUT / 'gallery-freeze.json').read_bytes()).hexdigest() == freeze['gallery_freeze_sha256']
    jobs = read(manifest_path)
    frozen = {m['id']: m for m in read(OUT / 'gallery-freeze.json')['models']}
    downloads = read(OUT / 'download-manifest.json')
    for row in downloads['files']:
        path = OUT / 'raw' / row['run_id'] / row['file']
        assert path.stat().st_size == row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    records = []; max_old = 0.; max_local = 0.
    counts = {'states': 0, 'mask_rows': 0, 'gallery_records': 0,
              'exact_original_prediction_comparisons': 0}
    for spec in jobs:
        result = read(OUT / (spec['run_id']+'.json'))
        rawdir = OUT / 'raw' / spec['run_id']
        assert {k:v for k,v in result.items() if k != 'local_dispatch_elapsed_seconds'} == read(rawdir/'result.json')
        assert result['status'] == 'complete' and result['passed'] and len(result['models']) == 10
        assert result['spec'] == spec and result['source_sha256'] == freeze['source_sha256']
        for m in result['models']:
            ident = m['id']; stem = m['recipe']+'-'+m['state']; fr = frozen[ident]
            assert m == read(rawdir/(stem+'.json'))
            assert m['checkpoint_sha256'] == fr['checkpoint_sha256'] and m['epoch'] == fr['epoch']
            with np.load(rawdir/(stem+'-test.npz'), allow_pickle=False) as a, np.load(
                    BASE/'results/hypotheses'/spec['archive_run_id']/(stem+'-test.npz'), allow_pickle=False) as old:
                pred = a['pred']; ce = a['ce']; labels = a['labels']; ids = np.arange(16)*2+33
                assert pred.shape == ce.shape == (64, 10000) and pred.dtype == np.int8
                assert np.array_equal(a['mask_ids'], np.arange(64))
                assert np.array_equal(a['coverage'], [0]*32+[1]*32)
                assert np.array_equal(labels, old['labels']) and np.array_equal(pred[ids], old['pred'])
                assert (pred[:32] == -1).all() and np.isnan(ce[:32]).all() and np.isfinite(ce[32:]).all()
                assert np.all((pred[32:] >= 0) & (pred[32:] <= 9))
                delta = float(np.max(np.abs(ce[ids].mean(1)-old['ce'].mean(1))))
                assert delta <= 3e-5
                max_old = max(max_old, delta)
                correct = pred == labels[None, :]; dense = correct[63]
                assert len(m['masks']) == 64
                for k, row in enumerate(m['masks']):
                    assert row['id'] == k and row['n'] == 10000
                    covered = k >= 32
                    selected = sum(c for i,c in enumerate(COSTS) if k & (1 << i))
                    assert row['coverage'] == float(covered)
                    assert row['raw_macs'] == (selected if covered else 0)
                    assert row['correct_outputs'] == int(correct[k].sum())
                    assert row['correct_output_rate'] == float(correct[k].mean())
                    assert row['accuracy'] == (float(correct[k].mean()) if covered else None)
                    if covered:
                        difference = abs(row['ce']-float(ce[k].mean()))
                        assert difference <= 1e-12
                        max_local = max(max_local, difference)
                    else:
                        assert row['ce'] is None
                    assert row['harm'] == float((dense & ~correct[k]).mean())
                    assert row['repair'] == float((~dense & correct[k]).mean())
                    assert len(row['per_class']) == 10
                    for cls in row['per_class']:
                        sel = labels == cls['digit']
                        assert cls['n'] == int(sel.sum())
                        assert cls['accuracy'] == (float(correct[k, sel].mean()) if covered else None)
                        assert cls['correct_output_rate'] == float(correct[k, sel].mean())
                assert [e['index'] for e in m['examples']] == fr['gallery_indices']
                for e in m['examples']:
                    i = e['index']
                    assert e['label'] == int(labels[i]) and e['pred'] == pred[:, i].tolist()
                    assert e['ce'] == [None]*32+ce[32:, i].tolist()
                counts['states'] += 1; counts['mask_rows'] += 64
                counts['gallery_records'] += len(m['examples'])
                counts['exact_original_prediction_comparisons'] += 160000
                records.append({'id': ident, 'all_16_predictions_exact': True,
                    'recomputed_64_summary_rows_match': True, 'gallery_records_match': len(m['examples']),
                    'max_old16_mean_ce_difference': delta,
                    'all_6_off': {'coverage': m['masks'][0]['coverage'], 'accuracy': m['masks'][0]['accuracy'],
                                  'correct_output_rate': m['masks'][0]['correct_output_rate']},
                    'five_dropped_head_only_accuracy': m['masks'][32]['accuracy'],
                    'stem_only_dropped_accuracy': m['masks'][62]['accuracy'],
                    'four_body_dropped_accuracy': m['masks'][33]['accuracy']})
    assert counts['states'] == 30 and counts['mask_rows'] == 1920
    record = {'status': 'verified', 'checked_unix': time.time(),
        'method': 'Independent local recomputation from downloaded NPZ plus archived old16 NPZ; no model execution or training.',
        'verification_script': 'verify_full_depth.py',
        'verification_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'downloaded_files_hash_checked': len(downloads['files']), **counts,
        'max_old16_mean_ce_difference': max_old, 'max_local_mean_ce_reduction_difference': max_local,
        'local_mean_ce_reduction_atol': 1e-12,
        'local_reduction_note': 'Strict scalar equality initially exposed ~1ulp mean-reduction differences between local and remote NumPy; predictions/counts/gallery values remain exact. This local summation tolerance does not alter the frozen original16 evaluator parity criterion.',
        'records': records}
    (OUT/'independent-verification.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return {k:record[k] for k in ('status', *counts, 'max_old16_mean_ce_difference', 'max_local_mean_ce_reduction_difference')}


if __name__ == '__main__':
    print(json.dumps(verify()))
