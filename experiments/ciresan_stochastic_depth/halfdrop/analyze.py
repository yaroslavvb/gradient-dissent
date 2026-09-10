"""Analyze the exhaustive half-drop training study without provisioning compute.

`--phase freeze` reads validation-only files and OLD archived labels, then writes
the immutable order manifest. `--phase evaluate` requires that freeze before
opening the new test summaries/arrays. Incomplete runs are never substituted.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RESULTS = HERE.parent / 'results' / 'halfdrop'
DEFAULT_ARCHIVE = HERE.parent / 'results' / 'hypotheses' / 'hypothesis-audit-s101-v1' / 'data.npz'
SEEDS = (201, 202, 203)
ENDPOINTS = ('selected', 'final')
T95 = {2: 12.7062047364, 3: 4.3026527299}
LAYER_MACS = (5_000_000, 3_000_000, 1_500_000, 500_000)
FULL_MACS = 11_965_000
CE_ATOL = 3e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(text)
        temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def interval(values):
    values = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in values), 'Nonfinite interval observation')
    n = len(values)
    if not n:
        return {'mean': None, 'values': [], 'n': 0, 'sample_sd': None, 'df': None, 'ci95_low': None, 'ci95_high': None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = T95[n] * sd / math.sqrt(n) if n in T95 else None
    return {'mean': mean, 'values': values, 'n': n, 'sample_sd': sd, 'df': n-1,
            'ci95_low': mean-half if half is not None else None,
            'ci95_high': mean+half if half is not None else None}


def mean_optional(values):
    values = list(values)
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def prefixes(order):
    masks = [0]
    for branch in order:
        masks.append(masks[-1] | (1 << branch))
    return masks


def manifest_jobs(path):
    value = read_json(path)
    jobs = value if isinstance(value, list) else value.get('jobs', value.get('runs'))
    require(isinstance(jobs, list), 'Manifest must be a list or contain jobs/runs')
    require(len(jobs) == 48, 'Require exactly 48 predeclared main runs')
    seen = set()
    for spec in jobs:
        key = (spec.get('seed'), spec.get('drop_mask'))
        require(type(key[0]) is int and key[0] in SEEDS and type(key[1]) is int and key[1] in range(16), 'Invalid main seed/eligibility mask')
        require(key not in seen, 'Duplicate seed/mask in manifest')
        seen.add(key)
        require(spec.get('stage') == 'evaluate', 'Main manifest may not include pilots or tuning')
        require(Path(spec['run_id']).name == spec['run_id'], 'Run ID must be a filename component')
        require(isinstance(spec.get('source_sha256'), dict) and spec['source_sha256'], 'Main run lacks a frozen source map')
        for name, expected in [('epochs', 100), ('batch_size', 64), ('lr', .01), ('momentum', .9), ('recipe', 'sd_constant'), ('pmax', 0)]:
            require(spec.get(name, expected) == expected, f'{spec["run_id"]}: unexpected fixed {name}')
        require(spec.get('input_scale', 1/255) == 1/255 and spec.get('output_relu', False) is False, 'Require normalized pixels and linear logits')
    require(len({j['run_id'] for j in jobs}) == 48, 'Duplicate run IDs')
    return sorted(jobs, key=lambda j: (j['seed'], j['drop_mask']))


def find_file(root, run_id, kind):
    root = Path(root)
    # The launcher creates this canonical flat view before asynchronous raw
    # downloads; preferring it keeps the freeze stable when raw copies arrive.
    variants = ([root/(run_id+'-validation.json'), root/'raw'/run_id/'validation.json', root/run_id/'validation.json']
                if kind == 'validation' else [root/run_id/'result.json', root/(run_id+'.json')])
    found = [p for p in variants if p.is_file()]
    require(found, f'Missing {kind} record for {run_id}')
    if len(found) > 1:
        first = read_json(found[0])
        require(all(read_json(p) == first for p in found[1:]), f'Duplicate {kind} copies disagree: {run_id}')
    return found[0]


def endpoint_path(root, run_id, endpoint):
    choices = [Path(root)/'raw'/run_id/(endpoint+'-test.npz'), Path(root)/run_id/(endpoint+'-test.npz')]
    found = [p for p in choices if p.is_file()]
    require(found, f'Missing {endpoint} predictions for {run_id}')
    require(all(sha(p) == sha(found[0]) for p in found[1:]), 'Duplicate prediction artifacts disagree')
    return found[0]


def selected_validation(view):
    rows = [r for r in view['history'] if isinstance(r.get('validation'), dict)]
    require([r['epoch'] for r in rows] == [1]+list(range(5, 101, 5)), 'Validation observation cadence differs from protocol')
    require(all(r['validation']['n'] == 10000 for r in rows), 'Validation panel must have 10000 examples')
    require(all(math.isfinite(r['validation']['loss']) and 0 <= r['validation']['accuracy'] <= 1 for r in rows), 'Invalid validation metrics')
    winner = min(rows, key=lambda r: (r['validation']['loss'], r['epoch']))
    require(winner['epoch'] == view['best_epoch'], 'Selected checkpoint is not earliest minimum validation CE')
    require(abs(winner['validation']['loss'] - view['best_validation_loss']) <= 1e-12, 'Selected validation CE differs')
    require(view['selected_validation'] == winner['validation'] and view['final_validation'] == rows[-1]['validation'], 'Top-level validation copies disagree with history')
    return winner


def read_validation_views(jobs, root):
    views, evidence = {}, []
    for spec in jobs:
        path = find_file(root, spec['run_id'], 'validation')
        view = read_json(path)
        if set(view) == {'validation_view'}:
            view = view['validation_view']
        require(not any(k in view for k in ('selected_test', 'final_test', 'test', 'test_predictions', 'test_logits')), 'Validation-only artifact contains test outcomes')
        require(view.get('test_accessed') is False and view['dataset'].get('test_included') is False, 'Validation view must precede official-test access')
        require(view['spec'] == spec, f'Validation spec mismatch: {spec["run_id"]}')
        require(view['source_sha256'] == spec['source_sha256'], 'Recorded executed sources differ from dispatch freeze')
        require(not view.get('diverged', False) and not view.get('failure_reason'), f'Failed/diverged run: {spec["run_id"]}')
        require(view.get('complete') is True and view.get('status') == 'complete', 'Incomplete validation-only training record')
        require([r['epoch'] for r in view['history']] == list(range(1, 101)), 'Require all100 epoch history rows')
        pvec = [.5 if spec['drop_mask'] & (1 << j) else 0 for j in range(4)]
        require(all(r['drop_probabilities'] == pvec for r in view['history']), 'Training eligibility/probabilities changed')
        require(view['dataset']['train_size'] == 50000 and view['dataset']['val_size'] == 10000, 'Wrong train/validation sizes')
        require(view['steps_per_epoch'] == 781 and view['training_examples_per_epoch'] == 49984, 'Wrong training exposure')
        selected_validation(view)
        for filename in ('best.pt', 'final.pt', 'checkpoint.pt'):
            digest = view['checkpoints'][filename]['sha256']
            require(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set('0123456789abcdef'), 'Malformed checkpoint SHA-256')
        require(view['checkpoints']['best.pt']['sha256'] == view['checkpoints']['checkpoint.pt']['sha256'], 'Selected checkpoint alias differs')
        views[(spec['seed'], spec['drop_mask'])] = view
        evidence.append({'run_id': spec['run_id'], 'path': relative(path), 'sha256': sha(path)})
    pairing = []
    for seed in SEEDS:
        baseline = views[(seed, 0)]
        for mask in range(16):
            other = views[(seed, mask)]
            for field in ('initial_parameters_sha256', 'initialization_rng_sha256', 'executed_epoch_order_sha256', 'raw_mask_draws_sha256', 'source_sha256'):
                require(other[field] == baseline[field], f'Matched-seed {field} differs: s{seed},mask{mask}')
            for field in ('training_data_sha256', 'split_sha256', 'train_indices_sha256', 'val_indices_sha256'):
                require(other['dataset'][field] == baseline['dataset'][field], f'Dataset pairing differs: {field}')
        pairing.append({'seed': seed, 'n': 16, 'initial_parameters_sha256': baseline['initial_parameters_sha256'],
                        'training_data_sha256': baseline['dataset']['training_data_sha256'],
                        'executed_epoch_order_sha256': baseline['executed_epoch_order_sha256'],
                        'raw_mask_draws_sha256': baseline['raw_mask_draws_sha256'], 'passed': True})
    source = views[(SEEDS[0], 0)]['source_sha256']
    require(all(v['source_sha256'] == source for v in views.values()), 'Executed sources differ across runs')
    dataset = views[(SEEDS[0], 0)]['dataset']
    require(all(v['dataset']['training_data_sha256'] == dataset['training_data_sha256'] for v in views.values()), 'Training split differs across seeds')
    return views, evidence, pairing


def validation_paths(views):
    rows = []
    for order in itertools.permutations(range(4)):
        masks = prefixes(order)
        selected = [selected_validation(views[(s, m)])['validation'] for s in SEEDS for m in masks[1:4]]
        counts = [round((1-v['accuracy'])*v['n']) for v in selected]
        require(all(abs(1-c/v['n']-v['accuracy']) <= 1e-12 for c, v in zip(counts, selected)), 'Validation accuracy is not an integer count fraction')
        errors, n = sum(counts), sum(v['n'] for v in selected)
        rows.append({'order': list(order), 'masks': masks,
                     'validation_error': errors/n, 'validation_error_count': errors, 'validation_observations': n,
                     'validation_ce': statistics.mean(v['loss'] for v in selected)})
    winner = min(rows, key=lambda r: (r['validation_error_count'], r['validation_ce'], r['order']))
    for row in rows:
        row['validation_selected'] = row['order'] == winner['order']
    return rows


def archived_gallery(path):
    # This is a pre-existing archive, not a new halfdrop test-outcome artifact.
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive['test_labels'], dtype=np.uint8)
    require(labels.shape == (10000,) and set(labels.tolist()) == set(range(10)), 'Wrong archived test labels')
    indices = sorted(int(i) for digit in range(10) for i in np.flatnonzero(labels == digit)[:10])
    require(len(set(indices)) == 100, 'Fixed gallery must contain100 distinct images')
    archive_manifest_path = Path(path).parent/'result.json'
    archive_manifest = read_json(archive_manifest_path)
    artifact = archive_manifest['artifacts'][Path(path).name]
    require(sha(path) == artifact['sha256'] and Path(path).stat().st_size == artifact['bytes'], 'Old archived data payload differs from original record')
    test_files = {name: entry['sha256'] for name, entry in archive_manifest['dataset']['files'].items() if name.startswith('t10k-')}
    require(set(test_files) == {'t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'}, 'Old archive lacks official test-file hashes')
    return labels, {'definition': 'First10 original test indices per true digit; no new model outcome used',
                    'archive': {'path': relative(path), 'sha256': sha(path)},
                    'archive_manifest': {'path': relative(archive_manifest_path), 'sha256': sha(archive_manifest_path)},
                    'official_test_files_sha256': test_files,
                    'labels_uint8_sha256': array_sha(labels), 'indices': indices}


def freeze_order(manifest_path, root, archive_path, output=None, source_freeze_path=None):
    jobs = manifest_jobs(manifest_path)
    source_freeze_path = Path(source_freeze_path or Path(manifest_path).with_name(Path(manifest_path).name.replace('manifest', 'source-freeze')))
    committed = read_json(source_freeze_path)
    require(committed['manifest_sha256'] == sha(manifest_path), 'Main manifest differs from pre-training source freeze')
    require(all(j['source_sha256'] == committed['source_sha256'] for j in jobs), 'Dispatch source maps differ from pre-training freeze')
    require(committed['source_sha256'].get('halfdrop/PROTOCOL.md') == sha(HERE/'PROTOCOL.md'), 'Protocol differs from executed pre-training commitment')
    views, evidence, pairing = read_validation_views(jobs, root)
    _, gallery = archived_gallery(archive_path)
    payload = {'schema_version': 1, 'criterion': 'Mean selected-checkpoint validation error over k1..3 and three seeds; tie mean validation CE then lexicographic branch indices',
               'manifest': {'path': relative(manifest_path), 'sha256': sha(manifest_path)},
               'source_freeze': {'path': relative(source_freeze_path), 'sha256': sha(source_freeze_path)},
               'protocol_sha256': sha(HERE/'PROTOCOL.md'), 'validation_files': evidence,
               'pairing': pairing, 'paths': validation_paths(views), 'gallery': gallery,
               'selected_endpoint': 'validation-CE-selected checkpoint',
               'reuse_order_for_final': True, 'new_test_outcomes_read': False}
    destination = Path(output or Path(root)/'order-manifest.json')
    if destination.exists():
        existing = read_json(destination)
        require(existing['frozen_inputs'] == payload, 'Existing ordering freeze differs; refusing overwrite')
        return existing
    result = {'created_utc': utc(), 'analyzer_sha256_at_freeze': sha(__file__), 'frozen_inputs': payload}
    write_json(destination, result, exclusive=True)
    return result


def comparison(pred, baseline, labels, selection=None):
    select = np.ones(len(labels), dtype=bool) if selection is None else np.asarray(selection, dtype=bool)
    n = int(select.sum())
    correct, base = pred == labels, baseline == labels
    harm = int((select & base & ~correct).sum())
    repair = int((select & ~base & correct).sum())
    base_correct = int((select & base).sum())
    return {'n': n, 'baseline_correct': base_correct, 'harm_count': harm, 'repair_count': repair,
            'harm_rate': harm/n if n else None, 'repair_rate': repair/n if n else None,
            'conditional_harm': harm/base_correct if base_correct else None,
            'accuracy_pp': 100*(repair-harm)/n if n else None}


def read_endpoint(path, summary, labels):
    with np.load(path, allow_pickle=False) as payload:
        data = {k: payload[k].copy() for k in ('indices', 'labels', 'pred', 'logits', 'probs', 'ce')}
    require(np.array_equal(data['labels'], labels), 'Test labels/order differ from frozen archive')
    n = len(labels)
    require(np.array_equal(data['indices'], np.arange(n)), 'Prediction indices are not original test order')
    require(data['pred'].shape == (n,) and data['ce'].shape == (n,), 'Wrong prediction/CE array shape')
    require(data['logits'].shape == (n, 10) and data['probs'].shape == (n, 10), 'Wrong ten-class score shape')
    require(all(np.isfinite(data[k]).all() for k in ('logits', 'probs', 'ce')), 'Nonfinite test payload')
    require(np.all((data['probs'] >= 0) & (data['probs'] <= 1)) and np.allclose(data['probs'].sum(1), 1, atol=1e-10, rtol=0), 'Invalid probability bounds/normalization')
    require(np.all(data['ce'] >= 0), 'Negative test cross-entropy')
    require(np.array_equal(data['pred'], data['logits'].argmax(1)), 'Prediction/logit argmax mismatch')
    z = data['logits'].astype(np.float64)
    z -= z.max(1, keepdims=True)
    exp = np.exp(z)
    expected_prob = exp/exp.sum(1, keepdims=True)
    expected_ce = np.log(exp.sum(1))-z[np.arange(n), labels]
    require(np.allclose(data['probs'], expected_prob, atol=1e-6, rtol=1e-5), 'Probabilities differ from full-logit softmax')
    require(np.max(np.abs(data['ce']-expected_ce)) <= 1e-4, 'Per-example CE/logit mismatch')
    accuracy = float(np.mean(data['pred'] == labels))
    ce = float(np.mean(data['ce']))
    require(summary['n'] == n and summary['accuracy'] == accuracy, 'Source test accuracy/count mismatch')
    require(abs(summary['loss']-ce) <= CE_ATOL, 'Source mean test CE differs')
    wrong = np.flatnonzero(data['pred'] != labels).tolist()
    require(summary['wrong_indices'] == wrong, 'Source test error indices differ')
    data.update({'accuracy': accuracy, 'mean_ce': ce})
    return data


def summarize_metrics(rows, fields):
    return {key: interval([r[key] for r in rows]) for key in fields}


def analyze_arrays(records, payloads, labels):
    """Pure complete-panel statistics; test payload access belongs to evaluate()."""
    models, by_k, edges, marginals = [], [], [], []
    keyed = {}
    for endpoint in ENDPOINTS:
        for mask in range(16):
            seeds = []
            for seed in SEEDS:
                raw, data, base = records[(seed, mask)], payloads[(seed, mask, endpoint)], payloads[(seed, 0, endpoint)]
                contrast = comparison(data['pred'], base['pred'], labels)
                contrast['ce'] = data['mean_ce']-base['mean_ce']
                cls = []
                for digit in range(10):
                    sel = labels == digit
                    c = comparison(data['pred'], base['pred'], labels, sel)
                    c.update({'digit': digit, 'accuracy': float((data['pred'][sel] == labels[sel]).mean()),
                              'ce': float(data['ce'][sel].mean())})
                    cls.append(c)
                epoch = raw['best_epoch'] if endpoint == 'selected' else 100
                row = {'seed': seed, 'run_id': raw['spec']['run_id'], 'accuracy': data['accuracy'],
                       'test_accuracy': data['accuracy'], 'ce': data['mean_ce'],
                       'training_seconds': raw['training_seconds'], 'selected_epoch': raw['best_epoch'],
                       'epoch': epoch, 'total_run_seconds': raw.get('total_run_seconds'),
                       'paired_vs_none': contrast, 'per_class': cls,
                       'hardware': raw.get('hardware'),
                       'checkpoint_sha256': raw['checkpoints']['best.pt' if endpoint == 'selected' else 'final.pt']['sha256'],
                       'train_accuracy': raw[endpoint+'_train']['accuracy'], 'train_ce': raw[endpoint+'_train']['loss'],
                       'validation_accuracy': raw[endpoint+'_validation']['accuracy'], 'validation_ce': raw[endpoint+'_validation']['loss']}
                seeds.append(row)
            contrast_fields = ('accuracy_pp', 'ce', 'harm_rate', 'repair_rate', 'conditional_harm', 'harm_count', 'repair_count')
            agg = {'endpoint': endpoint, 'drop_mask': mask, 'k': mask.bit_count(),
                   'drop_probabilities': [.5 if mask & (1 << j) else 0 for j in range(4)],
                   'nominal_training_affine_fraction': 1-.5*sum(c for j, c in enumerate(LAYER_MACS) if mask & (1 << j))/FULL_MACS,
                   'inference_affine_fraction': 1,
                   'metrics': summarize_metrics(seeds, ('accuracy', 'ce', 'training_seconds', 'selected_epoch', 'train_accuracy', 'train_ce', 'validation_accuracy', 'validation_ce')),
                   'paired_vs_none': summarize_metrics([s['paired_vs_none'] for s in seeds], contrast_fields),
                   'per_class': [], 'seeds': seeds}
            for name in ('harm_rate', 'repair_rate', 'conditional_harm'):
                agg['metrics'][name] = agg['paired_vs_none'][name]
            for digit in range(10):
                cs = [s['per_class'][digit] for s in seeds]
                metrics = summarize_metrics(cs, ('accuracy', 'ce', 'accuracy_pp', 'harm_rate', 'repair_rate', 'conditional_harm'))
                agg['per_class'].append({'digit': digit, 'n': int((labels == digit).sum()), **metrics, 'metrics': metrics})
            models.append(agg); keyed[(endpoint, mask)] = agg
        for k in range(5):
            masks = [m for m in range(16) if m.bit_count() == k]
            rows = [keyed[(endpoint, m)] for m in masks]
            ms = {key: interval([statistics.mean(r['seeds'][i][key] for r in rows) for i in range(3)])
                  for key in ('accuracy', 'ce', 'training_seconds')}
            paired = {key: interval([mean_optional(r['seeds'][i]['paired_vs_none'][key] for r in rows) for i in range(3)])
                      for key in ('accuracy_pp', 'ce', 'harm_rate', 'repair_rate', 'conditional_harm')}
            by_k.append({'endpoint': endpoint, 'k': k, 'masks': masks, 'metrics': ms, 'paired_vs_none': paired,
                         'subset_mean_range': {key: {'min': min(r['metrics'][key]['mean'] for r in rows),
                                                     'max': max(r['metrics'][key]['mean'] for r in rows)} for key in ('accuracy', 'ce')}})
        for branch in range(4):
            branch_edges = []
            for mask in range(16):
                if mask & (1 << branch):
                    continue
                target = mask | (1 << branch)
                values = []
                for seed in SEEDS:
                    base, other = payloads[(seed, mask, endpoint)], payloads[(seed, target, endpoint)]
                    comp = comparison(other['pred'], base['pred'], labels)
                    comp.update({'seed': seed, 'ce': other['mean_ce']-base['mean_ce']})
                    values.append(comp)
                row = {'endpoint': endpoint, 'branch': branch, 'from_mask': mask, 'to_mask': target, 'context_k': mask.bit_count(),
                       'metrics': summarize_metrics(values, ('accuracy_pp', 'ce', 'harm_rate', 'repair_rate', 'conditional_harm')),
                       'seeds': values}
                edges.append(row); branch_edges.append(row)
            for context in (None, 0, 1, 2, 3):
                subset = [e for e in branch_edges if context is None or e['context_k'] == context]
                metrics = {key: interval([mean_optional(e['seeds'][i][key] for e in subset) for i in range(3)])
                           for key in ('accuracy_pp', 'ce', 'harm_rate', 'repair_rate', 'conditional_harm')}
                marginals.append({'endpoint': endpoint, 'layer': branch, 'context_size': 'all' if context is None else context,
                                  'branch': branch, 'context_k': context, 'contexts': len(subset), **metrics, 'metrics': metrics})
    return {'models': models, 'by_k': by_k, 'edges': edges, 'marginals': marginals}


def make_gallery(seed, endpoint, payloads, labels, images, fixed):
    models, all_indices = [], set()
    base = payloads[(seed, 0, endpoint)]
    for mask in range(16):
        data = payloads[(seed, mask, endpoint)]
        selected = {i: ['fixed'] for i in fixed}
        base_ok, correct = base['pred'] == labels, data['pred'] == labels
        for digit in range(10):
            for category, condition in [('harmed', base_ok & ~correct), ('repaired', ~base_ok & correct)]:
                for index in np.flatnonzero((labels == digit) & condition)[:2]:
                    selected.setdefault(int(index), []).append('first_'+category)
        examples = []
        for index, reasons in sorted(selected.items()):
            category = 'harmed' if base_ok[index] and not correct[index] else 'repaired' if correct[index] and not base_ok[index] else 'unchanged'
            examples.append({'index': index, 'label': int(labels[index]), 'category': category, 'selection': reasons,
                'pred': int(data['pred'][index]), 'baseline_pred': int(base['pred'][index]),
                'probs': [float(v) for v in data['probs'][index]], 'baseline_probs': [float(v) for v in base['probs'][index]],
                'ce': float(data['ce'][index]), 'baseline_ce': float(base['ce'][index])})
        require(len(examples) <= 140, 'Gallery selection exceeded predeclared maximum')
        all_indices.update(selected)
        models.append({'drop_mask': mask, 'baseline_mask': 0, 'examples': examples})
    encoded = {str(i): base64.b64encode(np.asarray(images[i], dtype=np.uint8).reshape(784).tobytes()).decode() for i in sorted(all_indices)}
    return {'schema_version': 1, 'seed': seed, 'endpoint': endpoint, 'models': models, 'images': encoded,
            'fixed_indices': fixed, 'selection_note': 'Fixed first10 indices per digit plus outcome-selected first2 harmed/repaired per digit and comparison; unchanged means unchanged correctness, not necessarily identical prediction.'}


def evaluate(manifest_path, root, archive_path, order_path=None, source_freeze_path=None):
    root = Path(root); order_path = Path(order_path or root/'order-manifest.json')
    require(order_path.exists(), 'Freeze validation order before accessing new test outcomes')
    frozen = freeze_order(manifest_path, root, archive_path, order_path, source_freeze_path)
    jobs = manifest_jobs(manifest_path)
    views, _, _ = read_validation_views(jobs, root)
    labels, gallery = archived_gallery(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        images = archive['test_images'].copy()
    require(images.shape[0] == 10000 and images.size == 10000*784, 'Wrong archived image panel')
    records, payloads, raw_files = {}, {}, []
    for spec in jobs:
        result_path = find_file(root, spec['run_id'], 'result')
        raw = read_json(result_path)
        require('error' not in raw and not raw.get('diverged', False) and not raw.get('failure_reason'), f'Incomplete/failed run: {spec["run_id"]}')
        require(raw.get('complete') is True and raw.get('status') == 'complete' and raw.get('test_accessed') is True, 'Missing completed dense test evaluation')
        require(raw['spec'] == spec, 'Result spec differs from frozen manifest')
        require(raw['validation_view'] == views[(spec['seed'], spec['drop_mask'])], 'Validation record changed after freeze')
        for key, value in raw['validation_view'].items():
            if key not in ('test_accessed', 'training_and_selection_complete_unix'):
                require(raw.get(key) == value, 'Final result changed pre-test field: '+key)
        require(raw['test_labels_sha256'] == array_sha(labels.astype('<i8')), 'Recorded test label hash differs')
        for name, digest in gallery['official_test_files_sha256'].items():
            require(raw['test_dataset']['files'][name]['sha256'] == digest, 'Official test data file differs from archived panel')
        records[(spec['seed'], spec['drop_mask'])] = raw
        raw_files.append({'run_id': spec['run_id'], 'path': relative(result_path), 'sha256': sha(result_path)})
        for endpoint in ENDPOINTS:
            path = endpoint_path(root, spec['run_id'], endpoint)
            artifact = raw['artifacts'][path.name]
            require(sha(path) == artifact['sha256'] and path.stat().st_size == artifact['bytes'], 'Prediction artifact hash/size mismatch')
            data = read_endpoint(path, raw[endpoint+'_test'], labels)
            payloads[(spec['seed'], spec['drop_mask'], endpoint)] = data
            raw_files.append({'run_id': spec['run_id'], 'endpoint': endpoint, 'path': relative(path), 'sha256': sha(path)})
    summary = analyze_arrays(records, payloads, labels)
    # Preserve each paired outcome, beyond the compact illustrated gallery.
    harm = np.empty((2, 3, 16, len(labels)), dtype=bool)
    repair = np.empty_like(harm)
    for ei, endpoint in enumerate(ENDPOINTS):
        for si, seed in enumerate(SEEDS):
            base_correct = payloads[(seed, 0, endpoint)]['pred'] == labels
            for mask in range(16):
                correct = payloads[(seed, mask, endpoint)]['pred'] == labels
                harm[ei, si, mask] = base_correct & ~correct
                repair[ei, si, mask] = ~base_correct & correct
    pairs_path = root/'paired-errors.npz'
    np.savez_compressed(pairs_path, harm=harm, repair=repair, labels=labels,
                        seeds=np.array(SEEDS), masks=np.arange(16), endpoints=np.array(ENDPOINTS),
                        indices=np.arange(len(labels), dtype=np.int32))
    galleries = {}
    for seed in SEEDS:
        for endpoint in ENDPOINTS:
            name = f'gallery-s{seed}-{endpoint}.json'
            payload = make_gallery(seed, endpoint, payloads, labels, images, gallery['indices'])
            write_json(root/name, payload)
            galleries[f'{seed}-{endpoint}'] = name
    summary.update({'schema_version': 1, 'generated_utc': utc(), 'status': 'complete', 'seeds': list(SEEDS),
        'mask_bits': ['body1', 'body2', 'body3', 'body4'], 'paths': frozen['frozen_inputs']['paths'],
        'gallery_files': galleries, 'wandb_runs': {},
        'verification': {'expected_runs': 48, 'complete_runs': 48, 'endpoint_states': 96,
                         'test_examples_per_state': 10000, 'validation_order_frozen_before_test_analysis': True,
                         'pairing': frozen['frozen_inputs']['pairing'], 'predictions_logits_probabilities_ce_consistent': True},
        'provenance': {'manifest': {'path': relative(manifest_path), 'sha256': sha(manifest_path)},
                       'source_freeze': frozen['frozen_inputs']['source_freeze'],
                       'order_manifest': {'path': relative(order_path), 'sha256': sha(order_path)},
                       'protocol': {'path': relative(HERE/'PROTOCOL.md'), 'sha256': sha(HERE/'PROTOCOL.md')},
                       'analyzer': {'path': relative(__file__), 'sha256': sha(__file__)}, 'raw_files': raw_files,
                       'paired_errors': {'path': relative(pairs_path), 'sha256': sha(pairs_path),
                                         'array_order': 'endpoint[selected,final], seed[201,202,203], drop_mask[0..15], original_test_index[0..9999]'},
                       'gallery_files': {name: {'sha256': sha(root/name), 'path': relative(root/name)} for name in galleries.values()}},
        'units': {'accuracy': 'fraction', 'accuracy_pp': 'percentage points', 'ce': 'nats', 'training_seconds': 'full100-epoch training clock, including for the selected-checkpoint endpoint'},
        'limitations': ['Every test inference uses all branches at gain1; masks specify training eligibility only.',
                        'Three seeds, reused official test set, fixed LR, validation-selected checkpoints and unadjusted descriptive intervals.',
                        'Orders are paths through separately trained subsets, not temporal dropout curricula.',
                        'The selected order uses validation error then CE then lexicographic ties; it is not selected by test performance.',
                        'Fixed-gallery examples are separate from explicitly outcome-selected first harmed/repaired illustrations.',
                        'MAC fractions estimate training affine work, not measured latency or energy.']})
    write_json(root/'analysis.json', summary)
    (root/'summary.md').write_text(render_markdown(summary))
    return summary


def fmt_interval(value, scale=1, digits=3):
    if value['mean'] is None:
        return 'undefined'
    return f"{value['mean']*scale:.{digits}f} [{value['ci95_low']*scale:.{digits}f}, {value['ci95_high']*scale:.{digits}f}]" if value['ci95_low'] is not None else f"{value['mean']*scale:.{digits}f}"


def render_markdown(data):
    selected = next(p for p in data['paths'] if p['validation_selected'])
    lines = ['# Half-drop training: exhaustive four-branch eligibility results', '',
        'Complete 48-run panel: 16 independently trained eligibility subsets × seeds 201–203. Every test evaluation keeps all six affines at gain one. Eligibility means a 50% minibatch drop chance during training, not deleting a layer during inference.', '',
        'Validation-CE-selected checkpoints are primary; epoch 100 is secondary. Values are seed means with 95% t intervals (n=3, df=2), without multiple-comparison correction.', '',
        'Validation-selected static prefix order: '+ ' → '.join(str(j+1) for j in selected['order'])+'. This order was frozen before test analysis and reused for both endpoints. It is not a temporal training schedule.', '',
        '| Endpoint | Eligible branches | Mean full-test accuracy % [CI] | Accuracy delta vs none pp [CI] | Mean CE [CI] | Subset-mean accuracy range % |',
        '| --- | ---: | --- | --- | --- | --- |']
    for row in data['by_k']:
        r = row['subset_mean_range']['accuracy']
        lines.append(f"| {row['endpoint']} | {row['k']} | {fmt_interval(row['metrics']['accuracy'],100)} | {fmt_interval(row['paired_vs_none']['accuracy_pp'])} | {fmt_interval(row['metrics']['ce'],digits=4)} | {100*r['min']:.3f}–{100*r['max']:.3f} |")
    lines += ['', 'Each k averages subsets within each seed before forming an interval. The range describes the means of different subsets and is not a confidence interval.', '',
        '| Endpoint | Eligibility mask (shallow→deep) | Accuracy % [CI] | CE [CI] | Harm % of all images [CI] | Repair % of all images [CI] | Training seconds [CI] |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in data['models']:
        bits = ''.join(str((row['drop_mask']>>j)&1) for j in range(4))
        lines.append(f"| {row['endpoint']} | {bits} | {fmt_interval(row['metrics']['accuracy'],100)} | {fmt_interval(row['metrics']['ce'],digits=4)} | {fmt_interval(row['paired_vs_none']['harm_rate'],100)} | {fmt_interval(row['paired_vs_none']['repair_rate'],100)} | {fmt_interval(row['metrics']['training_seconds'],digits=2)} |")
    lines += ['', 'Harm and repair compare each run with mask 0 under the same seed and endpoint. Accuracy change equals repair minus harm. Conditional harm and all ten class-specific metrics, all 32 directed subset edges per endpoint, the 24 path scores, and every seed value are preserved in analysis.json. Paired harm/repair flags for every official test image are retained in paired-errors.npz.', '',
        'Training time refers to the complete 100 epochs even when a selected checkpoint is displayed. Source hardware is recorded per run; these values are not energy, job latency or invoice cost.', '',
        'Gallery files separate the fixed 100 label-selected original images from outcome-selected first two harmed/repaired examples per digit. Empty categories remain empty. No new test result selects a preferred subset or changes the frozen order.', '',
        'Limits: fixed learning rate 0.01, one architecture and reused MNIST test set; three new study seeds do not restore an untouched dataset. Any apparent best subset is conditional on this recipe and does not establish an optimally tuned or test-independent winner.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['freeze', 'evaluate'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--archive-data', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--order-manifest', type=Path)
    parser.add_argument('--source-freeze', type=Path, help='Defaults to the source-freeze filename corresponding to the manifest')
    args = parser.parse_args()
    if args.phase == 'freeze':
        result = freeze_order(args.manifest, args.results_dir, args.archive_data, args.order_manifest, args.source_freeze)
        chosen = next(p for p in result['frozen_inputs']['paths'] if p['validation_selected'])
        print(json.dumps({'phase': 'frozen', 'order': chosen['order'], 'paths': 24, 'new_test_outcomes_read': False}))
    else:
        result = evaluate(args.manifest, args.results_dir, args.archive_data, args.order_manifest, args.source_freeze)
        print(json.dumps({'phase': 'complete', 'runs': 48, 'aggregate_models': len(result['models']), 'edges': len(result['edges'])}))


if __name__ == '__main__':
    main()
