"""Two-phase exact-mask analysis; fit/freeze validation-only policies before test.

Default `--phase fit` never indexes a test member or opens a test NPZ/summary.
`--phase evaluate` requires an unchanged policy manifest and validation inputs.
No network, checkpoints, paid resources, or original training files are touched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

try:
    from . import policy, metrics
except ImportError:
    import policy, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results' / 'hypotheses'
RECIPES = ('plain', 'residual', 'sd_constant', 'sd_annealed', 'residual_unit_dropout')
ENDPOINTS = ('selected', 'final')
SEEDS = (101, 102, 103)
PRIMARY = tuple(m for m in range(16) if m.bit_count() == 2)
COST_ROWS = metrics.mask_costs((5000000, 3000000, 1500000, 500000), 1965000)
COSTS = np.asarray([r['cost'] for r in COST_ROWS])
ENERGY_FLOOR = 1e-24  # Matches the frozen audit writer; undefined is not zero.
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def finite(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def describe(values, ci=False):
    vals = [float(v) for v in values if finite(v)]
    n = len(vals); mean = statistics.mean(vals) if n else None
    sd = statistics.stdev(vals) if n > 1 else None
    result = {'n': n, 'mean': mean, 'sample_sd': sd, 'values': vals}
    if ci:
        t = T975.get(n-1)
        half = t*sd/math.sqrt(n) if t is not None and sd is not None else None
        result.update(df=n-1 if n else None, t_critical=t,
                      ci95_low=mean-half if half is not None else None,
                      ci95_high=mean+half if half is not None else None)
    return result


def job_dir(root, seed):
    return Path(root) / f'hypothesis-audit-s{seed}-v1'


def key_for(seed, recipe, endpoint):
    return f'{seed}/{recipe}-{endpoint}'


def state_path(root, seed, recipe, endpoint, split):
    return job_dir(root, seed) / f'{recipe}-{endpoint}-{split}.npz'


def load_validation_data(path):
    """NPZ is lazy: access only named validation members, never test members."""
    with np.load(path, allow_pickle=False) as z:
        x = z['val_features'].copy()
        y = z['val_labels'].copy()
        positions = z['val_positions'].copy()
    if x.shape != (10000, 49) or y.shape != (10000,) or positions.shape != (10000,):
        raise ValueError('Validation data must contain 10000 rows and 49 image features')
    policy._features(x)
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 9)):
        raise ValueError('Invalid validation labels')
    if not np.issubdtype(positions.dtype, np.integer) or not np.array_equal(np.sort(positions), np.arange(10000)):
        raise ValueError('val_positions must permute 0..9999')
    # Exact predeclared torch permutation, rather than merely any permutation.
    import torch
    expected = torch.randperm(10000, generator=torch.Generator().manual_seed(20260910)).numpy()
    if not np.array_equal(positions, expected):
        raise ValueError('Validation order differs from the frozen split')
    hashes = {'val_features': array_digest(x), 'val_labels': array_digest(y), 'val_positions': array_digest(positions)}
    return x, y, positions, hashes


def load_panel(path, n, require_interactions=False):
    with np.load(path, allow_pickle=False) as z:
        panel = {name: z[name].copy() for name in ('pred', 'ce', 'dense_margin', 'true_margin')}
        for name in ('higher_fraction', 'energy_by_degree', 'labels'):
            if name in z.files:
                panel[name] = z[name].copy()
    pred, ce = panel['pred'], panel['ce']
    if pred.shape != (16, n) or not np.issubdtype(pred.dtype, np.integer) or np.any((pred < 0) | (pred > 9)):
        raise ValueError(f'Invalid mask predictions in {Path(path).name}')
    if ce.shape != (16, n) or not np.isfinite(ce).all() or np.any(ce < 0):
        raise ValueError(f'Invalid CE in {Path(path).name}')
    for name in ('dense_margin', 'true_margin'):
        if panel[name].shape != (n,) or not np.isfinite(panel[name]).all():
            raise ValueError(f'Invalid {name}')
    if 'energy_by_degree' in panel:
        energy = panel['energy_by_degree']
        if energy.shape != (5, n) or not np.isfinite(energy).all() or np.any(energy < -1e-12):
            raise ValueError('Invalid per-example Walsh energies')
        denominator = energy[1:].sum(axis=0)
        expected = np.divide(energy[2:].sum(axis=0), denominator,
                             out=np.full(n, np.nan), where=denominator > ENERGY_FLOOR)
        if 'higher_fraction' not in panel:
            raise ValueError('Per-example higher_fraction is required alongside energies')
        fraction = panel['higher_fraction']
        if fraction.shape != (n,) or np.isinf(fraction).any():
            raise ValueError('Invalid higher_fraction')
        active = denominator > ENERGY_FLOOR
        if not np.allclose(fraction[active], expected[active], rtol=1e-6, atol=1e-8):
            raise ValueError('higher_fraction disagrees with per-example energies')
        if np.any(np.isfinite(fraction[~active]) & (fraction[~active] != 0)):
            raise ValueError('Nonzero fraction for a zero denominator')
    elif require_interactions:
        raise ValueError('Per-example interaction arrays are required for evaluation')
    return panel


def margin_bins(margin, interior):
    return np.searchsorted(np.asarray(interior), margin, side='right')


def freeze_class_adjustment(panel, labels):
    correct = panel['pred'][15] == labels
    fit_correct = correct[:5000]
    values = panel['true_margin'][:5000][fit_correct]
    if len(values):
        interior = np.unique(np.quantile(values, np.linspace(0, 1, 11)[1:-1])).tolist()
    else:
        interior = []
    bins = margin_bins(panel['true_margin'][5000:], interior)
    cal_correct = correct[5000:]; y = labels[5000:]; nb = len(interior)+1
    counts = np.zeros((10, nb), dtype=np.int64)
    for c in range(10):
        counts[c] = np.bincount(bins[cal_correct & (y == c)], minlength=nb)
    mass = np.bincount(bins[cal_correct], minlength=nb)
    supports = {}
    for name, classes in (('digits_1_4_9', (1, 4, 9)), ('all_digits', tuple(range(10)))):
        common = np.flatnonzero((counts[list(classes)] >= 20).all(axis=0)) if len(values) else np.asarray([], dtype=int)
        weights = mass[common] / mass[common].sum() if len(common) else np.asarray([])
        supports[name] = {'classes': list(classes), 'common_bins': common.tolist(), 'weights': weights.tolist(),
                          'calibration_common_correct_count': int(mass[common].sum()),
                          'calibration_correct_count': int(cal_correct.sum()),
                          'common_correct_fraction': float(mass[common].sum()/cal_correct.sum()) if cal_correct.any() else None}
    return {'covariate': 'true-class logit margin at full mask, restricted to full-correct examples',
            'fit_correct_count': len(values), 'interior_edges': interior, 'bin_count': nb,
            'edge_rule': 'Validation-fit deciles; duplicate interior cutpoints collapsed; outside values use outer bins',
            'calibration_counts_by_class_bin': counts.tolist(), 'calibration_mass_by_bin': mass.tolist(),
            'supports': supports, 'minimum_calibration_cell_count': 20}


def fit_manifest(root=ROOT, seeds=SEEDS):
    root = Path(root); path = root / 'policy-manifest.json'
    # Reusing a frozen manifest never refits policies after test unblinding.
    if path.exists():
        manifest = verify_manifest(root, seeds)
        return manifest
    source_files = {name: digest(HERE/name) for name in ('policy.py', 'PROTOCOL.md')}
    policies, adjustments, validation_sources = {}, {}, {}
    common_data_hashes = None
    for seed in seeds:
        folder = job_dir(root, seed)
        x, y, positions, hashes = load_validation_data(folder/'data.npz')
        if common_data_hashes is None:
            common_data_hashes = hashes
        elif hashes != common_data_hashes:
            raise ValueError('Shared validation arrays disagree across seed shards')
        validation_sources[str(seed)] = {'data_validation_members_sha256': hashes, 'panels': {}}
        for recipe in RECIPES:
            for endpoint in ENDPOINTS:
                name = key_for(seed, recipe, endpoint)
                panel_path = state_path(root, seed, recipe, endpoint, 'validation')
                panel = load_panel(panel_path, len(y))
                if 'labels' in panel and not np.array_equal(panel['labels'], y):
                    raise ValueError('Validation panel labels disagree with shared validation data')
                fitted = policy.fit_policy(x, panel['pred'], y, COSTS)
                fitted.update(seed=seed, recipe=recipe, endpoint=endpoint,
                              scope='plain_forced_crop_surgery' if recipe == 'plain' else 'secondary_unit_dropout' if recipe == 'residual_unit_dropout' else 'primary_residual_sd')
                policies[name] = fitted
                adjustments[name] = freeze_class_adjustment(panel, y)
                validation_sources[str(seed)]['panels'][f'{recipe}-{endpoint}'] = digest(panel_path)
    manifest = {'schema_version': 1, 'frozen_utc': stamp(), 'seeds': list(seeds), 'recipes': list(RECIPES),
                'endpoints': list(ENDPOINTS), 'policy_count': len(policies), 'policies': policies,
                'class_adjustments': adjustments, 'validation_sources': validation_sources,
                'fit_source_sha256': source_files, 'fit_analyzer_sha256': digest(Path(__file__)),
                'mask_costs': COST_ROWS, 'test_arrays_opened_during_fit': False,
                'interpretation': 'All policies and calibration-derived margin supports frozen before test NPZ or test members are accessed. Prior checkpoint/validation/test reuse is explicitly exploratory.'}
    atomic_json(path, manifest)
    (root/'policy-manifest.sha256').write_text(digest(path)+'\n')
    # An existence marker is persisted before any evaluation can start.
    return manifest


def verify_manifest(root=ROOT, seeds=SEEDS):
    root = Path(root); path = root/'policy-manifest.json'; checksum = root/'policy-manifest.sha256'
    if not path.exists() or not checksum.exists() or checksum.read_text().strip() != digest(path):
        raise ValueError('A frozen policy manifest with matching checksum is required before test access')
    manifest = read_json(path)
    expected = {key_for(s, r, e) for s in seeds for r in RECIPES for e in ENDPOINTS}
    if set(manifest['policies']) != expected or set(manifest['class_adjustments']) != expected or manifest['test_arrays_opened_during_fit'] is not False:
        raise ValueError('Policy manifest scope/gate mismatch')
    for name, sha in manifest['fit_source_sha256'].items():
        if name not in ('policy.py', 'PROTOCOL.md') or digest(HERE/name) != sha:
            raise ValueError('Policy/protocol source changed since fitting; do not silently refit after test access')
    for seed in seeds:
        _, _, _, hashes = load_validation_data(job_dir(root, seed)/'data.npz')
        expected_source = manifest['validation_sources'][str(seed)]
        if hashes != expected_source['data_validation_members_sha256']:
            raise ValueError('Validation data changed after manifest freeze')
        for recipe in RECIPES:
            for endpoint in ENDPOINTS:
                if digest(state_path(root, seed, recipe, endpoint, 'validation')) != expected_source['panels'][f'{recipe}-{endpoint}']:
                    raise ValueError('Validation predictions changed after manifest freeze')
    return manifest


def subset_statistics(panel, labels, masks, take=None):
    take = np.ones(len(labels), dtype=bool) if take is None else np.asarray(take, dtype=bool)
    n = int(take.sum())
    if not n:
        return {'n_examples': 0, 'mask_ids': list(masks), 'dense_correct_count': 0}
    pred = panel['pred'][:, take]; ce = panel['ce'][:, take]; y = labels[take]
    full = pred[15] == y; correct = pred[list(masks)] == y
    harm = ~correct & full; repair = correct & ~full
    return {'n_examples': n, 'mask_ids': list(masks), 'dense_correct_count': int(full.sum()),
            'dense_accuracy': float(full.mean()), 'masked_accuracy': float(correct.mean()),
            'accuracy_delta_pp': float(100*(correct.mean()-full.mean())),
            'dense_ce': float(ce[15].mean()), 'masked_ce': float(ce[list(masks)].mean()),
            'excess_ce': float((ce[list(masks)]-ce[15]).mean()),
            'harmful_flip_rate': float(harm.mean()), 'beneficial_flip_rate': float(repair.mean()),
            'resilience_given_dense_correct': float(correct[:, full].mean()) if full.any() else None,
            'harm_given_dense_correct': float(harm[:, full].mean()) if full.any() else None,
            'dense_prediction_agreement': float((pred[list(masks)] == pred[15]).mean()),
            'mean_mac_fraction': float(COSTS[list(masks)].mean()/COSTS[15])}


def mask_summary(panel, labels):
    full = panel['pred'][15] == labels
    counts = (panel['pred'][list(PRIMARY)] == labels).sum(axis=0)
    examples = []
    for c in range(10):
        for category, condition in (('always_preserved', counts == 6), ('mixed', (counts > 0) & (counts < 6)), ('always_lost', counts == 0)):
            for index in np.flatnonzero(full & (labels == c) & condition)[:2]:
                examples.append({'test_index': int(index), 'digit': c, 'category': category,
                                 'correct_mask_count': int(counts[index]),
                                 'failing_primary_masks': [m for m in PRIMARY if panel['pred'][m, index] != c]})
    return {'primary': subset_statistics(panel, labels, PRIMARY),
            'by_mask': [dict(mask_id=m, **subset_statistics(panel, labels, [m])) for m in range(16)],
            'by_retained_count': [dict(retained=k, **subset_statistics(panel, labels, [m for m in range(16) if m.bit_count() == k])) for k in range(5)],
            'all_nonfull': subset_statistics(panel, labels, range(15)),
            'per_class': [dict(class_id=c, **subset_statistics(panel, labels, PRIMARY, labels == c)) for c in range(10)],
            'resilience_counts_among_dense_correct': np.bincount(counts[full], minlength=7).tolist(),
            'resilience_denominator': int(full.sum()), 'example_selection': 'Lowest test index within digit and predeclared category, up to two each',
            'example_ids': examples,
            'dense_margin_correlation': metrics.spearman_stats(panel['dense_margin'][full], counts[full]/6)}


def _standardized(bins, labels, resilience, support, nb):
    classes = support['classes']; common = support['common_bins']; weights = np.asarray(support['weights'])
    counts = np.zeros((10, nb), dtype=np.int64); sums = np.zeros((10, nb))
    for c in range(10):
        choose = labels == c
        counts[c] = np.bincount(bins[choose], minlength=nb)
        sums[c] = np.bincount(bins[choose], weights=resilience[choose], minlength=nb)
    estimates = {}
    for c in classes:
        valid = bool(common) and bool(np.all(counts[c, common] > 0))
        estimates[str(c)] = float(np.dot(weights, sums[c, common]/counts[c, common])) if valid else None
    return estimates, counts, sums


def class_adjustment(panel, labels, frozen, permutations=1000):
    full = panel['pred'][15] == labels
    resilience = (panel['pred'][list(PRIMARY)] == labels).mean(axis=0)
    bins = margin_bins(panel['true_margin'][full], frozen['interior_edges']); y = labels[full]; r = resilience[full]
    nb = frozen['bin_count']; output = {'definition': frozen, 'test_dense_correct_count': int(full.sum()), 'supports': {}}
    for name, support in frozen['supports'].items():
        estimates, counts, sums = _standardized(bins, y, r, support, nb)
        values = list(estimates.values()); complete = bool(values) and all(v is not None for v in values)
        statistic = None
        if complete:
            statistic = estimates['1']-(estimates['4']+estimates['9'])/2 if name == 'digits_1_4_9' else float(np.var(values))
        common = support['common_bins']; chosen = np.isin(bins, common) & np.isin(y, support['classes'])
        result = {'standardized_resilience': estimates, 'test_counts_by_class_bin': counts.tolist(),
                  'test_retained_count': int(chosen.sum()),
                  'test_retained_fraction_of_dense_correct': float(chosen.mean()) if len(chosen) else None,
                  'identified': complete, 'statistic': statistic,
                  'statistic_name': 'digit1_minus_mean_digit4_digit9_resilience' if name == 'digits_1_4_9' else 'between_digit_standardized_resilience_variance',
                  'raw_resilience': {str(c): float(r[y == c].mean()) if np.any(y == c) else None for c in support['classes']}}
        if complete and permutations:
            rng = np.random.default_rng(20260913); targets = [np.flatnonzero(chosen & (bins == b)) for b in common]
            exceed = 0; null_values = []
            # Relabel within each common-support margin bin. Outcome vectors stay together.
            for _ in range(permutations):
                standard = {str(c): 0. for c in support['classes']}
                for indices, weight in zip(targets, support['weights']):
                    perm_y = rng.permutation(y[indices]); rr = r[indices]
                    for c in support['classes']:
                        standard[str(c)] += weight*float(rr[perm_y == c].mean())
                value = standard['1']-(standard['4']+standard['9'])/2 if name == 'digits_1_4_9' else float(np.var(list(standard.values())))
                null_values.append(value); exceed += value >= statistic-1e-15
            result['permutation'] = {'n_permutations': permutations, 'seed': 20260913,
                'one_sided_p': (1+exceed)/(permutations+1), 'null': describe(null_values),
                'interpretation': 'Exploratory conditional label-exchangeability reference within coarse fixed margin bins, not a causal or seed-replication test'}
            result['permutation']['null'].pop('values')
        else:
            result['permutation'] = None
        output['supports'][name] = result
    primary = output['supports']['digits_1_4_9']
    raw = primary['raw_resilience']
    primary['raw_digit1_minus_mean_4_9'] = raw['1']-(raw['4']+raw['9'])/2 if all(raw[c] is not None for c in ('1','4','9')) else None
    return output


def interaction_statistics(panel):
    energies = panel['energy_by_degree']; denominator = energies[1:].sum(axis=0)
    valid = denominator > ENERGY_FLOOR; fraction = panel['higher_fraction'][valid]
    energy = energies.mean(axis=1); nonconstant = float(energy[1:].sum())
    ce_values = panel['ce']-panel['ce'][15]
    return {'centered_logit_mean_example_higher_fraction': float(fraction.mean()) if len(fraction) else None,
            'higher_fraction_distribution': {'n': len(fraction), 'quantiles': np.quantile(fraction, [0,.1,.25,.5,.75,.9,1]).tolist() if len(fraction) else []},
            'zero_nonconstant_energy_count': int((denominator == 0).sum()),
            'undefined_nonconstant_energy_count_le_floor': int((~valid).sum()), 'denominator_floor': ENERGY_FLOOR,
            'near_zero_nonconstant_energy_count_le_1e_minus_12': int((denominator <= 1e-12).sum()),
            'mean_centered_logit_energy_by_degree': energy.tolist(),
            'centered_logit_ratio_of_pooled_energies': float(energy[2:].sum()/nonconstant) if nonconstant else None,
            'aggregation_warning': 'Primary is mean of each example higher-order fraction, excluding exact-zero denominators; ratio of pooled energies is separately named and can differ.',
            'ce_walsh': metrics.walsh_spectrum(ce_values), 'ce_mobius': metrics.interaction_summary(ce_values)}


def routed_ce(fitted, features, panel, labels):
    ids = policy.apply_policy(fitted, features)
    static = np.full(len(labels), fitted['static_comparator']['selected_mask_id'])
    random = policy.random_cost_matched_masks(ids)
    routes = {'routed': ids, 'static_calibrated': static, 'dense': np.full(len(labels),15), 'random_cost_matched': random}
    scored = policy.evaluate_policy(fitted, features, panel['pred'], labels)
    for name, masks in routes.items():
        scored[name]['ce'] = float(panel['ce'][masks, np.arange(len(labels))].mean())
        scored[name]['excess_ce'] = scored[name]['ce']-float(panel['ce'][15].mean())
        scored[name]['test_empirical_extra_error_tolerance_met'] = scored[name]['errors']-scored[name]['dense_errors'] <= .002*len(labels)+1e-12
        scored[name]['mask_assignment_sha256'] = array_digest(masks)
    return scored


def mechanism_summary(mechanism):
    if mechanism.get('status') != 'complete':
        return {'status': mechanism.get('status', 'unavailable')}
    blocks = []
    for b in mechanism['blocks']:
        blocks.append({k: b[k] for k in ('branch_index', 'deleted_mask_id', 'post_relu_width', 'all_examples', 'full_correct_examples', 'deletion_minus_gaussian_response_l2')})
    interior = [r for r in mechanism['fractional_gate_interpolation'] if r['alpha'] in (.25, .5, .75)]
    average = lambda values: statistics.mean(values) if values else None
    return {'status': 'complete', 'n_examples': mechanism['n_examples'], 'full_correct_count': mechanism['full_correct_count'],
            'blocks': blocks,
            'mean_block_gradient_damage_rho': average([b['all_examples']['gradient_vs_damage_spearman']['rho'] for b in blocks if b['all_examples']['gradient_vs_damage_spearman']['rho'] is not None]),
            'mean_block_norm_damage_rho': average([b['all_examples']['relative_norm_vs_damage_spearman']['rho'] for b in blocks if b['all_examples']['relative_norm_vs_damage_spearman']['rho'] is not None]),
            'mean_block_angle_damage_rho': average([b['all_examples']['angle_vs_damage_spearman']['rho'] for b in blocks if b['all_examples']['angle_vs_damage_spearman']['rho'] is not None]),
            'mean_block_first_order_r2_vs_zero': average([b['all_examples']['first_order_r2_vs_zero_change'] for b in blocks if b['all_examples']['first_order_r2_vs_zero_change'] is not None]),
            'mean_interior_centered_logit_multilinear_error_rms': average([r['summary']['centered_logit_multilinear_error_rms']['mean'] for r in interior]),
            'mean_interior_centered_logit_degree1_error_rms': average([r['summary']['centered_logit_degree1_error_rms']['mean'] for r in interior]),
            'fractional_gate_interpolation': [{k: v for k, v in r.items() if k != 'per_example'} for r in mechanism['fractional_gate_interpolation']],
            'survival_mean_bias': [{k: v for k, v in r.items() if k != 'per_example'} for r in mechanism['survival_mean_bias']],
            'interpretation': 'Block means are descriptive within a checkpoint, not independent trained-model replicates. First-order score is 1−SSE/SSE(zero change), not mean-baseline R².'}


def verify_original_reference(record, panel, labels, reference_dir):
    if not record['direct_forward_parity'].get('bitwise_equal') or record['direct_forward_parity']['max_abs_logit_error'] != 0:
        raise ValueError('All-kept forward parity failed')
    for split in ('validation', 'test'):
        if record[split+'_parity'].get('passed') is not True:
            raise ValueError('Recorded dense checkpoint parity failed')
    if record['test_parity'].get('error_indices_checked') is not True:
        raise ValueError('Test parity must check error indices')
    n = len(labels); pred = panel['pred'][15]
    error_ids = np.flatnonzero(pred != labels).tolist()
    if len(error_ids) != record['test_parity']['errors'] or abs(panel['ce'][15].mean()-record['test_parity']['loss']) > 1e-10:
        raise ValueError('Compact panel disagrees with recorded dense parity')
    path = Path(reference_dir)/f"graph-eval-{record['recipe']}-s{record['seed']}.json"
    source = read_json(path)
    if digest(path) != record['source_result_sha256']:
        raise ValueError('Original result source hash mismatch')
    ref = source[record['state']+'_test']
    if ref['n'] != n or sorted(ref['wrong_indices']) != error_ids or ref['errors'] != len(error_ids):
        raise ValueError('Test compact prediction errors differ from original checkpoint')
    if not math.isclose(ref['accuracy'], 1-len(error_ids)/n, abs_tol=1e-12, rel_tol=0):
        raise ValueError('Original accuracy/error count mismatch')
    if abs(ref['loss']-panel['ce'][15].mean()) > 3e-5:
        raise ValueError('Original checkpoint test CE mismatch')
    return {'passed': True, 'reference_file': path.name, 'reference_sha256': digest(path),
            'wrong_indices_verified': True, 'ce_absolute_difference': float(abs(ref['loss']-panel['ce'][15].mean()))}


def seed_aggregates(states):
    groups = defaultdict(list)
    for row in states:
        groups[(row['recipe'], row['endpoint'])].append(row)
    output = []
    for (recipe, endpoint), rows in groups.items():
        paths = {
            'dense_accuracy': lambda r: r['robustness']['primary']['dense_accuracy'],
            'primary_mask_accuracy': lambda r: r['robustness']['primary']['masked_accuracy'],
            'primary_excess_ce': lambda r: r['robustness']['primary']['excess_ce'],
            'primary_harmful_flip_rate': lambda r: r['robustness']['primary']['harmful_flip_rate'],
            'primary_conditional_resilience': lambda r: r['robustness']['primary']['resilience_given_dense_correct'],
            'mean_example_higher_interaction_fraction': lambda r: r['interactions']['centered_logit_mean_example_higher_fraction'],
            'raw_digit1_minus_mean4_9_resilience': lambda r: r['class_adjustment']['supports']['digits_1_4_9']['raw_digit1_minus_mean_4_9'],
            'adjusted_digit1_minus_mean4_9_resilience': lambda r: r['class_adjustment']['supports']['digits_1_4_9']['statistic'],
            'router_accuracy': lambda r: r['routing']['routed']['accuracy'],
            'router_accuracy_delta_pp': lambda r: r['routing']['routed']['accuracy_delta_pp'],
            'router_mac_saving_fraction': lambda r: r['routing']['routed']['mac_saving_fraction'],
            'router_minus_static_accuracy_pp': lambda r: 100*(r['routing']['routed']['accuracy']-r['routing']['static_calibrated']['accuracy']),
            'router_minus_random_accuracy_pp': lambda r: 100*(r['routing']['routed']['accuracy']-r['routing']['random_cost_matched']['accuracy']),
            'router_minus_static_cost_fraction': lambda r: r['routing']['routed']['mean_cost_fraction']-r['routing']['static_calibrated']['mean_cost_fraction'],
            'gradient_minus_norm_damage_rho': lambda r: r['mechanisms']['mean_block_gradient_damage_rho']-r['mechanisms']['mean_block_norm_damage_rho'] if finite(r['mechanisms'].get('mean_block_gradient_damage_rho')) and finite(r['mechanisms'].get('mean_block_norm_damage_rho')) else None,
            'gradient_minus_angle_damage_rho': lambda r: r['mechanisms']['mean_block_gradient_damage_rho']-r['mechanisms']['mean_block_angle_damage_rho'] if finite(r['mechanisms'].get('mean_block_gradient_damage_rho')) and finite(r['mechanisms'].get('mean_block_angle_damage_rho')) else None,
            'first_order_score_vs_zero_change': lambda r: r['mechanisms'].get('mean_block_first_order_r2_vs_zero'),
        }
        result = {'recipe': recipe, 'endpoint': endpoint, 'scope': rows[0]['scope'], 'seeds': [r['seed'] for r in rows],
                  'metrics': {key: describe([fn(r) for r in rows], ci=True) for key, fn in paths.items()},
                  'router_test_tolerance_met_count': sum(r['routing']['routed']['test_empirical_extra_error_tolerance_met'] for r in rows),
                  'router_n': len(rows)}
        output.append(result)
    return output


def paired_contrasts(states):
    keyed = {(r['recipe'], r['endpoint'], r['seed']): r for r in states}
    output = []
    for endpoint in ENDPOINTS:
        for recipe in RECIPES:
            if recipe == 'residual':
                continue
            seeds = sorted({r['seed'] for r in states if r['recipe'] == recipe and r['endpoint'] == endpoint and ('residual', endpoint, r['seed']) in keyed})
            pairs = [(keyed[(recipe, endpoint, s)], keyed[('residual', endpoint, s)]) for s in seeds]
            changes = {
                'primary_excess_ce_delta': lambda r: r['robustness']['primary']['excess_ce'],
                'primary_harmful_flip_rate_delta': lambda r: r['robustness']['primary']['harmful_flip_rate'],
                'primary_mask_accuracy_delta_pp': lambda r: 100*r['robustness']['primary']['masked_accuracy'],
                'mean_example_higher_fraction_delta': lambda r: r['interactions']['centered_logit_mean_example_higher_fraction'],
                'router_mac_saving_delta': lambda r: r['routing']['routed']['mac_saving_fraction'],
            }
            output.append({'recipe': recipe, 'control': 'residual', 'endpoint': endpoint, 'seeds': seeds,
                'scope': pairs[0][0]['scope'] if pairs else None,
                'metrics': {name: describe([fn(a)-fn(b) for a,b in pairs if finite(fn(a)) and finite(fn(b))], ci=True) for name, fn in changes.items()},
                'interpretation': 'Treatment minus residual, paired over checkpoints with the same seed. Plain contrasts are forced-surgery diagnostics, not equivalent trained bypasses. No pooled-example or mask-as-seed inference.'})
    return output


def directional_conclusion(interval, negative=True):
    if not interval.get('n'):
        return 'unidentified'
    lo, hi, mean = interval.get('ci95_low'), interval.get('ci95_high'), interval['mean']
    if lo is None or hi is None:
        return 'insufficient seed replication for an interval'
    if (hi < 0 if negative else lo > 0):
        return 'direction supported by the unadjusted three-seed interval'
    if (lo > 0 if negative else hi < 0):
        return 'opposite direction supported by the unadjusted three-seed interval'
    return 'interval includes zero; directional claim unresolved'


def evaluate(root=ROOT, seeds=SEEDS, reference_dir=HERE.parent/'results', permutations=1000):
    root = Path(root)
    manifest = verify_manifest(root, seeds)  # Entire validation/source check precedes test access.
    (root/'test-access-started.json').write_text(json.dumps({'utc': stamp(), 'policy_manifest_sha256': digest(root/'policy-manifest.json')})+'\n')
    states, artifacts, job_metadata = [], [], []
    for seed in seeds:
        folder = job_dir(root, seed)
        audit_result = read_json(folder/'result.json')
        if audit_result['run_id'] != f'hypothesis-audit-s{seed}-v1' or len(audit_result['records']) != 10:
            raise ValueError('Incomplete/wrong audit shard')
        if audit_result['source_sha256'] != audit_result['spec']['source_sha256']:
            raise ValueError('Executed audit sources disagree with the frozen job specification')
        job_metadata.append({'seed': seed, 'run_id': audit_result['run_id'], 'source_sha256': audit_result['source_sha256'],
                             'hardware': audit_result['hardware'], 'elapsed_seconds': audit_result['elapsed_seconds'],
                             'warmup': audit_result.get('warmup'), 'finished_unix': audit_result.get('finished_unix')})
        with np.load(folder/'data.npz', allow_pickle=False) as z:
            features = z['test_features'].copy(); labels = z['test_labels'].copy()
        policy._features(features)
        if len(features) != 10000 or labels.shape != (10000,):
            raise ValueError('Expected all 10000 official test examples')
        used = ['data.npz']
        for recipe in RECIPES:
            for endpoint in ENDPOINTS:
                panel_path = state_path(root, seed, recipe, endpoint, 'test')
                panel = load_panel(panel_path, len(labels), require_interactions=True)
                if 'labels' in panel and not np.array_equal(panel['labels'], labels):
                    raise ValueError('Test panel labels disagree with shared test labels')
                metadata_path = folder/f'{recipe}-{endpoint}.json'
                record = read_json(metadata_path)
                if (record['recipe'], record['state'], record['seed']) != (recipe, endpoint, seed):
                    raise ValueError('Checkpoint summary identity mismatch')
                reference = verify_original_reference(record, panel, labels, reference_dir)
                name = key_for(seed, recipe, endpoint); fitted = manifest['policies'][name]
                print(f'ANALYZE {name}', flush=True)
                row = {'key': name, 'seed': seed, 'recipe': recipe, 'endpoint': endpoint, 'scope': fitted['scope'],
                       'epoch': record['epoch'], 'checkpoint_sha256': record['checkpoint_sha256'],
                       'intervention': record['intervention'], 'dense_reference_verification': reference,
                       'robustness': mask_summary(panel, labels),
                       'class_adjustment': class_adjustment(panel, labels, manifest['class_adjustments'][name], permutations),
                       'interactions': interaction_statistics(panel),
                       'mechanisms': mechanism_summary(record.get('mechanisms', {})),
                       'routing': routed_ce(fitted, features, panel, labels),
                       'calibration_policy': fitted['calibration'], 'static_calibration': fitted['static_comparator'],
                       'source_files': [str(panel_path.relative_to(root)), str(metadata_path.relative_to(root))]}
                # Compact primary interaction statistic must agree with independent GPU reduction.
                reported = record['test']['per_example_logit_interaction_fraction']['mean']
                actual = row['interactions']['centered_logit_mean_example_higher_fraction']
                if (reported is None) != (actual is None) or reported is not None and not math.isclose(reported, actual, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError('Per-example interaction mean disagrees with audit JSON')
                states.append(row)
                used += [panel_path.name, metadata_path.name, f'{recipe}-{endpoint}-validation.npz']
        for filename in used:
            sha = digest(folder/filename)
            if audit_result['artifacts'].get(filename, {}).get('sha256') != sha:
                raise ValueError(f'Artifact hash mismatch: {filename}')
            artifacts.append({'file': str((folder/filename).relative_to(root)), 'sha256': sha})
        artifacts.append({'file': str((folder/'result.json').relative_to(root)), 'sha256': digest(folder/'result.json')})
    aggregates = seed_aggregates(states); contrasts = paired_contrasts(states)
    findings = []
    for row in contrasts:
        if row['recipe'] in ('sd_constant', 'sd_annealed') and row['endpoint'] == 'selected':
            findings.append({'hypothesis': 1, 'recipe': row['recipe'], 'endpoint': row['endpoint'],
                             'conclusion': directional_conclusion(row['metrics']['primary_excess_ce_delta']),
                             'metric': row['metrics']['primary_excess_ce_delta']})
            findings.append({'hypothesis': 4, 'recipe': row['recipe'], 'endpoint': row['endpoint'],
                             'conclusion': directional_conclusion(row['metrics']['mean_example_higher_fraction_delta']),
                             'metric': row['metrics']['mean_example_higher_fraction_delta']})
    for row in aggregates:
        if row['recipe'] in ('residual', 'sd_constant', 'sd_annealed') and row['endpoint'] == 'selected':
            findings.append({'hypothesis': 2, 'recipe': row['recipe'], 'endpoint': row['endpoint'],
                             'conclusion': directional_conclusion(row['metrics']['adjusted_digit1_minus_mean4_9_resilience'], negative=False),
                             'metric': row['metrics']['adjusted_digit1_minus_mean4_9_resilience']})
            findings.append({'hypothesis': 3, 'recipe': row['recipe'], 'endpoint': row['endpoint'],
                             'conclusion': directional_conclusion(row['metrics']['gradient_minus_norm_damage_rho'], negative=False),
                             'metric': row['metrics']['gradient_minus_norm_damage_rho'],
                             'metric_definition': 'Gradient/damage rank correlation minus increment-norm/damage rank correlation; blocks averaged within each checkpoint',
                             'gradient_minus_angle': row['metrics']['gradient_minus_angle_damage_rho'],
                             'first_order_score_vs_zero': row['metrics']['first_order_score_vs_zero_change'],
                             'qualification': 'A better rank correlation does not establish accurate Taylor prediction; inspect sign agreement, zero-change score, angles and real intermediate-gate errors.'})
            findings.append({'hypothesis': 5, 'recipe': row['recipe'], 'endpoint': row['endpoint'],
                             'test_tolerance_met': f"{row['router_test_tolerance_met_count']}/{row['router_n']}",
                             'mean_mac_saving': row['metrics']['router_mac_saving_fraction']['mean'],
                             'warning': 'Nominal MAC saving, not measured latency; calibration feasibility does not guarantee test feasibility. Compare static cost and random matched-cost quality.'})
    doc = {'schema_version': 1, 'generated_utc': stamp(),
           'verification': {'passed': True, 'complete': len(states) == len(seeds)*10, 'state_count': len(states),
                            'expected_states': len(seeds)*10, 'policy_manifest_sha256': digest(root/'policy-manifest.json'),
                            'validation_only_manifest_preceded_test_access': True,
                            'all_original_dense_error_indices_verified': True, 'issues': []},
           'primary_masks': list(PRIMARY), 'mask_costs': COST_ROWS, 'seeds': list(seeds),
           'policy_manifest': 'policy-manifest.json', 'states': states, 'aggregates': aggregates,
           'paired_contrasts': contrasts, 'findings': findings, 'jobs': job_metadata, 'artifacts': artifacts,
           'analysis_source_sha256': digest(Path(__file__)), 'permutation_count': permutations,
           'limitations': [
               'The test set, validation set and seed IDs were reused in earlier tasks. This is exploratory, conditional on fixed checkpoints and one dataset, not independent replication.',
               'Primary is selected checkpoint and uniform six two-retained masks; final endpoints, unit dropout and plain surgery are secondary or diagnostic.',
               'Three-seed paired t intervals have df=2. No multiple-comparison correction; examples, classes and masks are not additional trained-model replicates.',
               'Class adjustment uses calibration-frozen common support and test full-correct examples. Coarse margin bins leave residual confounding; zero support is unidentified, not evidence of equality.',
               'Primary logit interaction fraction averages per-example ratios; it differs from ratio of pooled energies. Loss/accuracy nonlinearities can introduce interactions without serial paths.',
               'Local derivatives, angular changes and finite differences are probes, not global contraction certificates; equality at binary corners does not establish interpolation linearity.',
               'Routing uses pooled raw-image features only. Dense logits/margins and true labels are analysis references, not free deployment inputs.',
               'A depth-three router and lambda are fitted/calibrated on previously used validation examples. Empirical 0.2pp tolerance is not a confidence guarantee. No test-based refitting.',
               'MAC savings omit routing/grouping overhead. Measured latency is a separate benchmark; no energy or invoice claim follows from this analysis.'
           ]}
    atomic_json(root/'analysis.json', doc)
    (root/'summary.md').write_text(markdown(doc))
    return doc


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if finite(value) else '—'


def interval(value, scale=1, digits=3):
    if not value.get('n'):
        return 'undefined'
    return f"{fmt(scale*value['mean'], digits)} [{fmt(scale*value['ci95_low'], digits) if value.get('ci95_low') is not None else '—'}, {fmt(scale*value['ci95_high'], digits) if value.get('ci95_high') is not None else '—'}]; n={value['n']}"


def markdown(d):
    lines = ['# Exact-mask layer-drop hypotheses: MNIST checkpoint audit', '',
             f"Generated {d['generated_utc']}. All {d['verification']['state_count']} checkpoint states verified. The policy manifest was frozen before test arrays were accessed by this postprocessor.", '',
             'Primary comparisons use the selected checkpoint, three seed pairs and six masks retaining exactly two of four body branches. Stem/head remain. Every mask has its own MAC cost. Earlier test/seed/validation reuse makes this an exploratory investigation.', '',
             '## Hypothesis outcomes', '']
    for f in d['findings']:
        if f['hypothesis'] != 5:
            lines.append(f"- H{f['hypothesis']} / {f['recipe']}: {f['conclusion']}; {interval(f['metric'])}.")
        else:
            lines.append(f"- H5 / {f['recipe']}: {f['test_tolerance_met']} selected-checkpoint routers meet the empirical test tolerance; mean nominal MAC saving {fmt(100*f['mean_mac_saving'],2)}%. This is not yet a measured-latency conclusion.")
    lines += ['', '## Robustness and interactions', '',
              'Mean values over seeds; confidence intervals below concern seed differences. Absolute mask accuracy/CE accompany excess losses in analysis.json. A lower-quality dense reference must not win solely by changing little.', '',
              '| Recipe / endpoint | Dense accuracy % | Two-retained accuracy % | Excess CE | Harmful flips % | Mean per-example high-order logit fraction |',
              '|---|---:|---:|---:|---:|---:|']
    for g in d['aggregates']:
        m = g['metrics']
        lines.append(f"| {g['recipe']} / {g['endpoint']} | {fmt(100*m['dense_accuracy']['mean'])} | {fmt(100*m['primary_mask_accuracy']['mean'])} | {fmt(m['primary_excess_ce']['mean'],4)} | {fmt(100*m['primary_harmful_flip_rate']['mean'])} | {fmt(m['mean_example_higher_interaction_fraction']['mean'],4)} |")
    lines += ['', '| Treatment / endpoint vs residual | Excess CE difference [95% CI] | Harmful-flip difference (pp) [95% CI] | Interaction-fraction difference [95% CI] |', '|---|---:|---:|---:|']
    for row in d['paired_contrasts']:
        m = row['metrics']
        lines.append(f"| {row['recipe']} / {row['endpoint']} | {interval(m['primary_excess_ce_delta'],digits=4)} | {interval(m['primary_harmful_flip_rate_delta'],scale=100)} | {interval(m['mean_example_higher_fraction_delta'],digits=4)} |")
    lines += ['', 'Plain deletions are forced crop surgery without a trained bypass. Unit dropout is secondary. These are not equivalent interventions to residual branch deletion.', '',
              '## Does digit 1 remain more resilient after margin adjustment?', '',
              'Contrast is digit 1 minus the mean of digits 4 and 9, on full-correct examples. Calibration freezes common-support bins and weights; sparse or empty support stays undefined. All ten digit profiles, counts, retained coverage, bin-specific counts and the 1,000-permutation reference are in analysis.json.', '',
              '| Recipe / endpoint | Raw resilience contrast (pp) [95% CI] | Adjusted contrast (pp) [95% CI] |', '|---|---:|---:|']
    for g in d['aggregates']:
        lines.append(f"| {g['recipe']} / {g['endpoint']} | {interval(g['metrics']['raw_digit1_minus_mean4_9_resilience'],scale=100)} | {interval(g['metrics']['adjusted_digit1_minus_mean4_9_resilience'],scale=100)} |")
    lines += ['', '## Local-direction and interpolation probes', '',
              'Each row describes one checkpoint on its fixed validation probe. Gradients, norms and angles are correlated with single-deletion damage. First-order score is 1−SSE/SSE(zero change), not mean-baseline R². Interior-gain RMS error is evaluated at real network states; binary reconstruction alone is not evidence of linearity.', '',
              '| State | Probe n | Gradient/damage rho | Norm/damage rho | Angle/damage rho | First-order score | Interior logit multilinear error RMS |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in d['states']:
        m = row['mechanisms']
        if m.get('status') == 'complete':
            lines.append(f"| {row['key']} | {m['n_examples']} | {fmt(m['mean_block_gradient_damage_rho'])} | {fmt(m['mean_block_norm_damage_rho'])} | {fmt(m['mean_block_angle_damage_rho'])} | {fmt(m['mean_block_first_order_r2_vs_zero'])} | {fmt(m['mean_interior_centered_logit_multilinear_error_rms'],4)} |")
    lines += ['', 'All branch-level errors, sign agreement, Gaussian-direction contrasts, interpolation paths and inverse-survival mixture/Jensen gaps are retained. No global Jacobian norm, contraction certificate or read/write circuit is inferred.', '',
              '## Frozen image-only routing', '',
              '49 pooled image features; depth-three multioutput tree; first 5,000 validation examples fit, remaining 5,000 calibrate. The 0.2pp extra-error allowance is empirical, not a confidence guarantee. The random policy shuffles exactly the routed mask multiset. Full calibration tables are in policy-manifest.json and analysis.json.', '',
              '| State | Router accuracy % | Full accuracy % | Static accuracy % | Same-cost random accuracy % | Router MAC saving % | Static MAC saving % | Test tolerance met |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in d['states']:
        r = row['routing']; p = r['routed']; static = r['static_calibrated']
        lines.append(f"| {row['key']} | {fmt(100*p['accuracy'])} | {fmt(100*r['dense']['accuracy'])} | {fmt(100*static['accuracy'])} | {fmt(100*r['random_cost_matched']['accuracy'])} | {fmt(100*p['mac_saving_fraction'],2)} | {fmt(100*static['mac_saving_fraction'],2)} | {p['test_empirical_extra_error_tolerance_met']} |")
    lines += ['', 'A lower-cost point is not an equal-cost victory. Static/dynamic costs can differ; same-cost random comparison isolates assignment association conditional on these fitted routers. Real feature/routing/gather/scatter latency is measured separately. No neural parameters were retrained.', '', '## Verification, raw evidence and limits', '',
              f"Policy manifest SHA-256: `{d['verification']['policy_manifest_sha256']}`. Original dense test error indices and CE, compact-artifact hashes and executed source-specification agreement all verify.", '',
              'Sources: [analysis.json](analysis.json), [frozen policies](policy-manifest.json), and the three seed audit folders listed in the JSON artifact ledger. No checkpoint is copied into this report.', '']
    lines += ['- '+s for s in d['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--phase', choices=('fit', 'evaluate', 'all'), default='fit')
    p.add_argument('--reference-dir', type=Path, default=HERE.parent/'results')
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = p.parse_args()
    if args.phase in ('fit', 'all'):
        manifest = fit_manifest(args.root, tuple(args.seeds))
        print(json.dumps({'phase': 'fit', 'policies': manifest['policy_count'], 'manifest_sha256': digest(args.root/'policy-manifest.json'), 'test_arrays_opened': False}), flush=True)
    if args.phase in ('evaluate', 'all'):
        doc = evaluate(args.root, tuple(args.seeds), args.reference_dir)
        print(json.dumps({'phase': 'evaluate', 'verification': doc['verification'], 'output': str(args.root/'analysis.json')}), flush=True)


if __name__ == '__main__':
    main()
