"""Six-affine inference interventions on frozen Ciresan checkpoints.

Bits 0..5 denote stem, four body branches, and classifier. A removed stem is
replaced by zero-padded normalized pixels; body deletion uses the old crop/ReLU
rule. A removed classifier ABSTAINS and short-circuits all affine computation.
This does not invent an untrained coordinate classifier or treat abstention as
chance classification. Only the 32 head-present masks produce logits.

run() consumes an already available CUDA device and never provisions hardware.
No training, checkpoint selection, existing artifact mutation, or timing claim.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

try:
    from ..model_data import CiresanMLP, load_mnist
except ImportError:
    from model_data import CiresanMLP, load_mnist

RECIPES = ('residual', 'sd_constant', 'sd_annealed', 'residual_unit_dropout', 'plain')
STATES = {'selected': 'best.pt', 'final': 'final.pt'}
FULL_MASK = 63
HEAD_BIT = 32
BATCH_SIZE = 2048
CE_MEAN_ATOL = 3e-5  # Same dense CE comparison tolerance as the original audit.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ('full_depth.py', 'FULL_DEPTH_PROTOCOL.md', 'audit.py')]
    paths += [here.parent / name for name in ('model_data.py', 'full_depth_app.py', 'telemetry_app.py')]
    return {str(p.relative_to(here.parent)): sha(p) for p in sorted(paths)}


def _integer_mask(mask, upper=64):
    if isinstance(mask, (bool, np.bool_)) or not isinstance(mask, (int, np.integer)) or not 0 <= mask < upper:
        raise ValueError(f'Mask must be an integer in 0..{upper-1}')
    return int(mask)


def old_mask_to_full(mask):
    """Embed an old four-body mask with the stem and head still present."""
    return 33 + 2 * _integer_mask(mask, 16)


def _check_model(model):
    if model.training:
        raise ValueError('Interventions require model.eval(); training dropout is forbidden')
    if len(model.layers) != 6 or len(model.widths) != 7:
        raise ValueError('Expected six affine layers and seven widths')
    if model.widths[1] < model.widths[0]:
        raise ValueError('Dropped-stem convention requires expansion or equal width')
    if any(b > a for a, b in zip(model.widths[1:-1], model.widths[2:])):
        raise ValueError('Body crops and classifier readout require decreasing widths')


def masked_forward(model, x, mask=FULL_MASK):
    """Return logits or None for a short-circuit abstention (head absent)."""
    mask = _integer_mask(mask)
    _check_model(model)
    if not mask & HEAD_BIT:
        return None
    h = x.reshape(-1, model.widths[0]) * model.input_scale
    if mask & 1:
        h = F.relu(model.layers[0](h))
    else:
        h = F.relu(F.pad(h, (0, model.widths[1] - model.widths[0])))
    for j, layer in enumerate(model.layers[1:-1], start=1):
        if not mask & (1 << j):
            h = F.relu(h[..., :layer.out_features])
        elif model.residual:
            h = F.relu(h[..., :layer.out_features] + layer(h))
        else:
            h = F.relu(layer(h))
    logits = model.layers[-1](h)
    return F.relu(logits) if model.output_relu else logits


def mask_costs(widths):
    widths = tuple(int(v) for v in widths)
    if len(widths) != 7 or any(v <= 0 for v in widths):
        raise ValueError('Expected seven positive widths')
    costs = [a*b for a, b in zip(widths[:-1], widths[1:])]
    full = sum(costs)
    result = []
    for mask in range(64):
        selected = sum(c for j, c in enumerate(costs) if mask & (1 << j))
        executed = selected if mask & HEAD_BIT else 0
        result.append({'id': mask, 'bits': [(mask >> j) & 1 for j in range(6)],
                       'retained': mask.bit_count(), 'selected_affines': mask.bit_count(),
                       'executed_affines': mask.bit_count() if mask & HEAD_BIT else 0,
                       'raw_macs': executed, 'cost': executed/full,
                       'nominal_selected_macs': selected, 'nominal_selected_cost': selected/full})
    return result


@torch.no_grad()
def enumerate_logits(model, x, batch_size=BATCH_SIZE):
    """Rows map to masks 32..63; all-kept logits must equal model() bitwise.

    The full official panel uses the same batch boundaries as the old audit.
    No batch-shape tolerance substitutes for all-kept forward equality.
    """
    _check_model(model)
    if not isinstance(batch_size, int) or batch_size <= 0 or not len(x):
        raise ValueError('Need nonempty examples and a positive batch size')
    result = torch.empty((32, len(x), model.widths[-1]), device=x.device, dtype=torch.float32)
    for mask in range(32, 64):
        for start in range(0, len(x), batch_size):
            batch = x[start:start+batch_size]
            logits = masked_forward(model, batch, mask)
            if mask == FULL_MASK and not torch.equal(logits, model(batch)):
                raise AssertionError('All-kept logits differ bitwise from model()')
            result[mask-32, start:start+batch_size] = logits
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite head-present logits')
    return result


def summarize(logits, labels, widths, gallery_indices=()):
    """Return JSON-safe summary and a compact NPZ-ready per-example payload.

    pred[64,N] uses -1 for abstention. ce[64,N] is NaN ONLY in head-absent
    rows, where cross-entropy is undefined. JSON represents those values null.
    Classification accuracy conditions on an emitted prediction. Correct-output
    rate divides by all inputs, so abstention has rate zero and accuracy null.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 3 or z.shape[0] != 32 or y.shape != (z.shape[1],) or not len(y):
        raise ValueError('Expected logits[32,N,C] and labels[N]')
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= z.shape[2]):
        raise ValueError('Invalid class labels')
    if not np.isfinite(z).all() or z.shape[2] != widths[-1] or z.shape[2] > 127:
        raise ValueError('Nonfinite logits or unsupported class count')
    gallery = list(gallery_indices)
    if len(set(gallery)) != len(gallery) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(y) for i in gallery):
        raise ValueError('Gallery indices must be unique in-range integers')
    n = len(y)
    pred = np.full((64, n), -1, dtype=np.int8)
    pred[32:] = z.argmax(-1).astype(np.int8)
    shifted = z - z.max(-1, keepdims=True)
    ce = np.full((64, n), np.nan, dtype=np.float64)
    ce[32:] = np.log(np.exp(shifted).sum(-1)) - shifted[:, np.arange(n), y]
    correct = pred == y[None, :]
    dense_correct = correct[63]
    masks = mask_costs(widths)
    for row in masks:
        m = row['id']; covered = bool(m & HEAD_BIT)
        row.update({'coverage': float(covered), 'n': n, 'predictions_emitted': n if covered else 0,
                    'correct_outputs': int(correct[m].sum()), 'correct_output_rate': float(correct[m].mean()),
                    'accuracy': float(correct[m].mean()) if covered else None,
                    'ce': float(ce[m].mean()) if covered else None,
                    'abstentions': 0 if covered else n,
                    'harm': float((dense_correct & ~correct[m]).mean()),
                    'repair': float((~dense_correct & correct[m]).mean()),
                    'per_class': []})
        for digit in range(z.shape[-1]):
            sel = y == digit; count = int(sel.sum())
            row['per_class'].append({'digit': digit, 'n': count, 'coverage': float(covered) if count else None,
                'dense_accuracy': float(dense_correct[sel].mean()) if count else None,
                'correct_output_rate': float(correct[m, sel].mean()) if count else None,
                'accuracy': float(correct[m, sel].mean()) if covered and count else None,
                'ce': float(ce[m, sel].mean()) if covered and count else None,
                'harm': float((dense_correct[sel] & ~correct[m, sel]).mean()) if count else None,
                'repair': float((~dense_correct[sel] & correct[m, sel]).mean()) if count else None})
    examples = [{'index': int(i), 'label': int(y[i]), 'pred': pred[:, i].astype(int).tolist(),
                 'ce': [None if m < 32 else float(ce[m, i]) for m in range(64)]} for i in gallery]
    arrays = {'pred': pred, 'ce': ce, 'labels': y.astype(np.uint8), 'mask_ids': np.arange(64, dtype=np.uint8),
              'coverage': np.array([0]*32+[1]*32, dtype=np.uint8)}
    return {'n': n, 'dense_accuracy': float(dense_correct.mean()), 'masks': masks, 'examples': examples}, arrays


def verify_archived_panel(arrays, archived, dense_reference):
    """Fail on any label/prediction mismatch, including all old error IDs."""
    indices = np.array([old_mask_to_full(m) for m in range(16)])
    if not np.array_equal(arrays['labels'], archived['labels']):
        raise AssertionError('Official test label/order mismatch with archived panel')
    previous = np.asarray(archived['pred'])
    current = arrays['pred'][indices]
    if previous.shape != current.shape or not np.array_equal(previous, current):
        raise AssertionError('An embedded old16-mask prediction differs from archived panel')
    old_ce = np.asarray(archived['ce'], dtype=np.float64)
    new_ce = arrays['ce'][indices]
    if old_ce.shape != new_ce.shape or not np.isfinite(old_ce).all():
        raise AssertionError('Archived CE payload is malformed')
    differences = np.abs(old_ce.mean(1)-new_ce.mean(1))
    if np.max(differences) > CE_MEAN_ATOL:
        raise AssertionError('An embedded old16-mask mean CE differs beyond frozen tolerance')
    wrong = np.flatnonzero(arrays['pred'][63] != arrays['labels']).tolist()
    accuracy = float(np.mean(arrays['pred'][63] == arrays['labels']))
    if accuracy != dense_reference['accuracy'] or wrong != dense_reference['wrong_indices']:
        raise AssertionError('Dense source accuracy/error indices differ')
    dense_delta = abs(float(arrays['ce'][63].mean()) - dense_reference['loss'])
    if dense_delta > CE_MEAN_ATOL:
        raise AssertionError('Dense source mean CE differs beyond frozen tolerance')
    return {'passed': True, 'embedded_masks': indices.tolist(), 'prediction_comparisons': int(current.size),
            'all_16_predictions_exact': True, 'labels_exact': True, 'dense_source_error_indices_exact': True,
            'max_abs_mean_ce_difference': float(differences.max()),
            'max_abs_per_example_ce_difference': float(np.abs(old_ce-new_ce).max()),
            'dense_source_mean_ce_difference': dense_delta, 'mean_ce_atol': CE_MEAN_ATOL}


def _gallery_for(spec, recipe, state):
    if 'models' in spec:
        return next(m['gallery_indices'] for m in spec['models'] if m['recipe'] == recipe and m['state'] == state)
    gallery = spec.get('gallery_indices', [])
    if isinstance(gallery, dict):
        gallery = gallery.get(f'{recipe}-{state}', gallery.get(f'{recipe}-{state}-s{spec["seed"]}', []))
    return gallery


def validate_spec(spec):
    """Validate the frozen complete panel before opening checkpoints or test data."""
    if spec.get('batch_size', BATCH_SIZE) != BATCH_SIZE:
        raise ValueError('Archived parity requires the frozen batch size 2048')
    seed = spec['seed']
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (101, 102, 103):
        raise ValueError('Only the original main seeds101..103 are eligible')
    recipes = spec.get('recipes', list(RECIPES)); states = spec.get('states', list(STATES))
    if not recipes or len(set(recipes)) != len(recipes) or any(r not in RECIPES for r in recipes):
        raise ValueError('Invalid/duplicate recipes')
    if not states or len(set(states)) != len(states) or any(s not in STATES for s in states):
        raise ValueError('Invalid/duplicate states')
    if 'models' not in spec:
        raise ValueError('Require frozen spec.models with checkpoint hashes and existing gallery indices')
    lookup = {}
    for model in spec['models']:
        key = (model['recipe'], model['state'])
        if key in lookup or model['seed'] != seed or model['id'] != f'{key[0]}-{key[1]}-s{seed}':
            raise ValueError('Frozen model IDs/seeds are invalid or duplicated')
        digest = model['checkpoint_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Require a lowercase SHA-256 checkpoint hash')
        if not isinstance(model['epoch'], int) or not 1 <= model['epoch'] <= 100:
            raise ValueError('Invalid frozen epoch')
        gallery = model['gallery_indices']
        if not isinstance(gallery, list) or len(set(gallery)) != len(gallery) or any(type(i) is not int or not 0 <= i < 10000 for i in gallery):
            raise ValueError('Frozen gallery requires unique original test indices')
        lookup[key] = model
    if set(lookup) != {(r, s) for r in recipes for s in states}:
        raise ValueError('Frozen models must cover the complete declared recipe/state panel')
    return recipes, states, lookup


def run(spec, root, progress_commit=None):
    """Evaluate one seed. Parent owns dispatch, reservations and source freeze."""
    start = time.perf_counter(); root = Path(root)
    if not torch.cuda.is_available():
        raise RuntimeError('run() requires an already available CUDA device; CPU tests use helpers')
    recipes, states, frozen_models = validate_spec(spec)
    sources = source_hashes()
    for name, digest in spec.get('source_sha256', {}).items():
        if sources.get(name) != digest:
            raise AssertionError(f'Frozen source mismatch: {name}')
    out = root/'runs'/spec['run_id']; out.mkdir(parents=True, exist_ok=True)
    with (out/'claimed.json').open('x') as f:
        json.dump({'spec': spec, 'unix': time.time()}, f)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    seed = int(spec['seed']); data = load_mnist(root/'data', include_test=True)
    if len(data['test_y']) != 10000:
        raise AssertionError('Expected all 10000 official test examples')
    x = data['test_x'].to('cuda'); y = data['test_y'].numpy()
    archived_dir = root/'runs'/spec.get('archive_run_id', f'hypothesis-audit-s{seed}-v1')
    archive_result = json.loads((archived_dir/'result.json').read_text())
    for filename, info in data['metadata']['files'].items():
        if archive_result['dataset']['files'][filename]['sha256'] != info['sha256']:
            raise AssertionError(f'Archived dataset file hash mismatch: {filename}')
    records = []; model_records = []
    for recipe in recipes:
        source_dir = root/'runs'/f'graph-eval-{recipe}-s{seed}'
        source = json.loads((source_dir/'result.json').read_text()); ms = source['spec']
        if source['dataset']['training_data_sha256'] != data['metadata']['training_data_sha256']:
            raise AssertionError('Training data hash mismatch')
        if ms['recipe'] != recipe or ms['seed'] != seed:
            raise AssertionError('Checkpoint source recipe/seed mismatch')
        model = CiresanMLP(recipe=recipe, pmax=ms.get('pmax', 0),
                          input_scale=ms['input_scale'], output_relu=ms['output_relu']).cuda().eval()
        for state in states:
            checkpoint = source_dir/STATES[state]
            archive_json = archived_dir/f'{recipe}-{state}.json'
            archived_record = json.loads(archive_json.read_text())
            checkpoint_hash = sha(checkpoint); source_hash = sha(source_dir/'result.json')
            if archived_record['checkpoint_sha256'] != checkpoint_hash or archived_record['source_result_sha256'] != source_hash:
                raise AssertionError('Checkpoint/source differs from original four-branch audit')
            frozen_model = frozen_models[(recipe, state)]
            epoch = source['best_epoch'] if state == 'selected' else ms['epochs']
            if frozen_model['checkpoint_sha256'] != checkpoint_hash or frozen_model['epoch'] != epoch:
                raise AssertionError('Checkpoint hash/epoch differs from frozen dispatch manifest')
            model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
            logits = enumerate_logits(model, x).cpu().numpy()
            summary, arrays = summarize(logits, y, model.widths, _gallery_for(spec, recipe, state))
            archived_npz = archived_dir/f'{recipe}-{state}-test.npz'
            with np.load(archived_npz, allow_pickle=False) as old:
                parity = verify_archived_panel(arrays, old, source['selected_test' if state == 'selected' else 'final_test'])
            record = {'id': f'{recipe}-{state}-s{seed}', 'recipe': recipe, 'state': state, 'seed': seed,
                      'epoch': epoch, 'source_run': ms['run_id'], 'checkpoint_sha256': checkpoint_hash,
                      'source_result_sha256': source_hash, 'archive_json_sha256': sha(archive_json),
                      'archive_npz_sha256': sha(archived_npz), 'source_sha256': sources,
                      'direct_forward_parity': {'all_kept_bitwise_equal': True, 'n': len(y), 'batch_size': BATCH_SIZE},
                      'archived_panel_parity': parity, 'frozen_model': frozen_model, **summary}
            np.savez_compressed(out/f'{recipe}-{state}-test.npz', **arrays)
            (out/f'{recipe}-{state}.json').write_text(json.dumps(record, allow_nan=False)+'\n')
            model_records.append(record)
            records.append({'recipe': recipe, 'state': state, 'epoch': epoch, 'id': record['id'],
                            'all_parity_passed': True, 'elapsed_seconds': time.perf_counter()-start})
            (out/'progress.json').write_text(json.dumps({'completed': records, 'elapsed_seconds': time.perf_counter()-start})+'\n')
            print('FULL_DEPTH', seed, recipe, state, 'complete', round(time.perf_counter()-start, 2), flush=True)
            if progress_commit:
                progress_commit()
            del logits, arrays, summary, record
        del model
    result = {'run_id': spec['run_id'], 'spec': spec, 'status': 'complete', 'passed': True,
              'source_sha256': sources, 'dataset': data['metadata'], 'archive_result_sha256': sha(archived_dir/'result.json'),
              'hardware': {'gpu': torch.cuda.get_device_name(), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                           'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
                           'precision': 'FP32 tensors; matmul precision high; no autocast', 'batch_size': BATCH_SIZE},
              'semantics': {'mask_bits': ['stem', 'body1', 'body2', 'body3', 'body4', 'head'],
                            'head_absent': 'short-circuit abstention; pred=-1; accuracy/CE undefined; correct-output rate/coverage/MACs zero',
                            'stem_absent': 'normalized flattened pixels zero-padded to2500 then ReLU; untrained intervention',
                            'body_absent': 'fixed prefix crop then ReLU; plain checkpoints were not trained with this bypass',
                            'npz_undefined_ce': 'NaN only for masks0..31 (abstention); JSON uses null',
                            'cost': 'Executed affine MACs only; excludes bias, ReLU, padding, cropping, indexing and memory work',
                            'harm': 'fraction of all examples correct in dense model but not correctly emitted under intervention; includes abstentions'},
              'records': records, 'models': model_records,
              'elapsed_seconds': time.perf_counter()-start, 'finished_unix': time.time()}
    result['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if progress_commit:
        progress_commit()
    return result
