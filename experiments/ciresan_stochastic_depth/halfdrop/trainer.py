"""Frozen-rate half-drop training on each subset of four residual branches.

All models share the original residual architecture, including subset0. Only
eligible branches receive training drop probability .5; inference keeps all
branches with gain1. Reuses the qualified CUDA graph SGD and epoch Recorder.
No resource provisioning, network logging, LR tuning or test-based selection.
"""
from pathlib import Path
import copy
import gc
import hashlib
import importlib
import io
import json
import math
import random
import shutil
import time
import unittest

import numpy as np
import torch
from torch.nn import functional as F

try:
    from ..model_data import CiresanMLP, load_mnist
    from ..optimization.stochastic_graph import ResidualGraphStep
    from ..telemetry.recorder import Recorder
except ImportError:
    from model_data import CiresanMLP, load_mnist
    from optimization.stochastic_graph import ResidualGraphStep
    from telemetry.recorder import Recorder

HERE = Path(__file__).resolve().parents[1]
DEFAULTS = {'recipe': 'sd_constant', 'pmax': 0., 'lr': .01, 'momentum': .9,
            'batch_size': 64, 'train_size': 50000, 'input_scale': 1/255,
            'output_relu': False, 'shrinkage': 0., 'precision': 'fp32', 'eval_every': 5,
            'telemetry': True, 'telemetry_every': 10, 'telemetry_probe_n': 128,
            'snapshot_epochs': []}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = ('halfdrop/trainer.py', 'halfdrop/test_trainer.py', 'halfdrop/PROTOCOL.md',
             'model_data.py', 'optimization/stochastic_graph.py', 'optimization/test_stochastic_graph.py',
             'telemetry/recorder.py', 'telemetry/metrics.py', 'halfdrop_app.py', 'telemetry_app.py')
    return {name: sha(HERE/name) for name in names if (HERE/name).exists()}


def probabilities(drop_mask):
    if type(drop_mask) is not int or not 0 <= drop_mask < 16:
        raise ValueError('drop_mask must be an integer0..15 (shallow-to-deep eligibility bits)')
    return [.5 if drop_mask & (1 << j) else 0. for j in range(4)]


def validate_spec(spec):
    probabilities(spec['drop_mask'])
    if spec.get('stage') not in ('pilot', 'evaluate'):
        raise ValueError('Only pilot and frozen evaluate stages are supported; no tuning')
    if type(spec.get('seed')) is not int or spec['seed'] < 0:
        raise ValueError('A nonnegative integer seed is required')
    if spec['stage'] == 'evaluate' and spec['seed'] not in (201, 202, 203):
        raise ValueError('Main panel requires new seeds201,202,203')
    if spec.get('epochs') != (2 if spec['stage'] == 'pilot' else 100):
        raise ValueError('Pilot must be2 epochs; main must be100 epochs')
    for key, expected in DEFAULTS.items():
        if spec.get(key, expected) != expected:
            raise ValueError(f'Frozen half-drop setting {key} must be {expected!r}')
    if 'drop_probabilities' in spec and spec['drop_probabilities'] != probabilities(spec['drop_mask']):
        raise ValueError('Explicit probability vector disagrees with eligibility mask')
    run_id = spec.get('run_id')
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError('run_id must be a nonempty filename component')
    return {**DEFAULTS, **spec}


def epoch_randomness(n, batch_size, probs, order_rng, mask_rng):
    """Always consume exactly four independent mask draws per training step."""
    order = torch.randperm(n, generator=order_rng)
    draws = torch.rand((n//batch_size, 4), generator=mask_rng)
    active = draws >= torch.tensor(probs, dtype=draws.dtype)
    return order, draws, active


def parameter_hash(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, x, y, batch_size=2048):
    """Full unscaled inference, using FP32 CE reduction like the prior study."""
    model.eval(); losses = []; correct = []; finite = []
    for start in range(0, len(y), batch_size):
        z = model(x[start:start+batch_size]); target = y[start:start+batch_size]
        losses.append(F.cross_entropy(z, target, reduction='sum'))
        correct.append((z.argmax(1) == target).sum()); finite.append(torch.isfinite(z).all())
    loss = float(torch.stack(losses).sum().item()/len(y))
    if not math.isfinite(loss) or not bool(torch.stack(finite).all()):
        raise FloatingPointError('Nonfinite full-inference logits/loss')
    return {'loss': loss, 'accuracy': float(torch.stack(correct).sum().item()/len(y)), 'n': len(y)}


def prediction_arrays(logits, labels):
    """All-example outputs; softmax/CE use stable float64 arithmetic on saved logits."""
    z = np.asarray(logits, dtype=np.float32); y = np.asarray(labels)
    if z.ndim != 2 or z.shape[1] != 10 or not len(z) or y.shape != (len(z),):
        raise ValueError('Expected logits[N,10] and labels[N]')
    if not np.isfinite(z).all() or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y > 9):
        raise ValueError('Invalid logits/labels')
    double = z.astype(np.float64); shifted = double-double.max(1, keepdims=True)
    exp = np.exp(shifted); probs = exp/exp.sum(1, keepdims=True)
    pred = z.argmax(1); ce = np.log(exp.sum(1))-shifted[np.arange(len(y)), y]
    top2 = np.partition(double, -2, axis=1)[:, -2:]
    margin = top2.max(1)-top2.min(1)
    other = double.copy(); other[np.arange(len(y)), y] = -np.inf
    true_margin = double[np.arange(len(y)), y]-other.max(1)
    arrays = {'logits': z, 'probs': probs, 'pred': pred.astype(np.uint8), 'labels': y.astype(np.uint8),
              'ce': ce, 'margin': margin, 'true_margin': true_margin,
              'confidence': probs[np.arange(len(y)), pred], 'indices': np.arange(len(y), dtype=np.int32)}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite prediction artifact')
    if not np.allclose(probs.sum(1), 1., rtol=0, atol=1e-14):
        raise AssertionError('Probabilities do not sum to1')
    correct = pred == y
    summary = {'loss': float(ce.mean()), 'accuracy': float(correct.mean()), 'n': len(y),
               'errors': int((~correct).sum()), 'wrong_indices': np.flatnonzero(~correct).tolist(),
               'per_class': []}
    for digit in range(10):
        mask = y == digit; count = int(mask.sum())
        summary['per_class'].append({'digit': digit, 'n': count,
            'accuracy': float(correct[mask].mean()) if count else None,
            'loss': float(ce[mask].mean()) if count else None,
            'errors': int((~correct[mask]).sum())})
    return summary, arrays


@torch.no_grad()
def export_predictions(model, x, y, destination):
    model.eval()
    logits = torch.cat([model(x[start:start+2048]) for start in range(0, len(x), 2048)]).cpu().numpy()
    summary, arrays = prediction_arrays(logits, y.detach().cpu().numpy())
    np.savez_compressed(destination, **arrays)
    summary['artifact'] = {'path': Path(destination).name, 'sha256': sha(destination),
                           'bytes': Path(destination).stat().st_size}
    return summary


def validate_kernels():
    package = __package__.rsplit('.', 1)[0]+'.' if __package__ and '.' in __package__ else ''
    module = importlib.import_module(package+'optimization.test_stochastic_graph')
    suite = unittest.defaultTestLoader.loadTestsFromModule(module); stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'passed': result.wasSuccessful() and not result.skipped,
              'output': stream.getvalue()}
    if not report['passed']:
        raise RuntimeError('CUDA kernel qualification failed: '+stream.getvalue())
    del suite, result, module; gc.collect(); torch.cuda.empty_cache()
    return report


def run(spec, root, progress_commit=None):
    cfg = validate_spec(spec)
    if not torch.cuda.is_available():
        raise RuntimeError('Requires an already provisioned CUDA device')
    sources = source_hashes()
    if not spec.get('source_sha256') or spec['source_sha256'] != sources:
        raise AssertionError('A complete matching source freeze is required before training')
    root = Path(root); directory = root/'runs'/spec['run_id']; directory.mkdir(parents=True, exist_ok=True)
    with (directory/'claimed.json').open('x') as f:
        json.dump({'spec': spec, 'unix': time.time()}, f)
    started = time.perf_counter(); device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    qualification = validate_kernels() if spec.get('validate_kernels', False) else None
    seed = spec['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    # All official-test access occurs after training and checkpoint selection.
    data = load_mnist(root/'data', train_size=50000, include_test=False)
    tensors = {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
    model = CiresanMLP(recipe='sd_constant', pmax=0., input_scale=1/255, output_relu=False)
    initial_hash = parameter_hash(model)
    init_rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    model.to(device); order_rng = torch.Generator().manual_seed(seed+100000)
    mask_rng = torch.Generator().manual_seed(seed+200000); torch.manual_seed(seed+300000)
    n = len(tensors['train_y']); steps = n//64; probs = probabilities(spec['drop_mask'])
    bx = torch.empty((64,)+tuple(tensors['train_x'].shape[1:]), device=device, dtype=torch.float32)
    by = torch.empty(64, device=device, dtype=torch.long)
    bx.copy_(tensors['train_x'][:64]); by.copy_(tensors['train_y'][:64])
    helper = ResidualGraphStep(model, .01, bx, by, momentum=.9, shrinkage=0., precision='fp32')
    if parameter_hash(model) != initial_hash:
        raise AssertionError('Graph setup modified initial model parameters')
    helper.update_scales(probs)
    # This mapping never receives test_x/test_y, even after test export.
    recorder = Recorder(model, cfg, directory, tensors, started, helper=helper)
    recorder.capture(0, 0., 0., force=True)
    epoch_losses = torch.empty(steps, device=device, dtype=torch.float32)
    history = []; train_seconds = 0.; prep_seconds = 0.; best_loss = math.inf; best_epoch = None
    skip_counts = np.zeros(4, dtype=np.int64)
    order_digest = hashlib.sha256(); mask_digest = hashlib.sha256(); draws_digest = hashlib.sha256()
    failed = False; failure_reason = None
    for epoch in range(1, cfg['epochs']+1):
        prep_start = time.perf_counter()
        order, draws, gates = epoch_randomness(n, 64, probs, order_rng, mask_rng)
        order_digest.update(order.numpy().tobytes()); draws_digest.update(draws.numpy().tobytes())
        mask_digest.update(gates.numpy().tobytes()); skip_counts += (~gates).sum(0).numpy()
        active_masks = gates.tolist(); order = order.to(device)
        model.train(); torch.cuda.synchronize(device); prep_seconds += time.perf_counter()-prep_start
        epoch_start = time.perf_counter()
        for step in range(steps):
            indices = order[step*64:(step+1)*64]
            torch.index_select(tensors['train_x'], 0, indices, out=bx)
            torch.index_select(tensors['train_y'], 0, indices, out=by)
            loss = helper(bx, by, active=active_masks[step])
            epoch_losses[step].copy_(loss)  # Graph loss storage aliases; copy before next replay.
        torch.cuda.synchronize(device); elapsed = time.perf_counter()-epoch_start; train_seconds += elapsed
        stochastic_loss = float(epoch_losses.mean().item())
        row = {'epoch': epoch, 'stochastic_training_loss': stochastic_loss if math.isfinite(stochastic_loss) else None,
               'training_seconds': train_seconds, 'epoch_training_seconds': elapsed,
               'drop_probabilities': probs, 'optimizer_steps': epoch*steps}
        if not math.isfinite(stochastic_loss):
            failed = True; failure_reason = 'Nonfinite stochastic training loss'
        if not failed and (epoch == 1 or epoch%5 == 0 or epoch == cfg['epochs']):
            try:
                row['validation'] = evaluate(model, tensors['val_x'], tensors['val_y'])
                row['dense_train_probe'] = evaluate(model, tensors['train_x'][:10000], tensors['train_y'][:10000])
                if row['validation']['loss'] < best_loss:
                    best_loss = row['validation']['loss']; best_epoch = epoch
                    torch.save(model.state_dict(), directory/'best.pt')
            except FloatingPointError as error:
                failed = True; failure_reason = str(error)
        history.append(row)
        if failed:
            row.update({'diverged': True, 'failure_reason': failure_reason}); break
        recorder.capture(epoch, train_seconds, elapsed, legacy_row=row)
        if epoch == 1 or epoch%5 == 0 or epoch == cfg['epochs']:
            (directory/'progress.json').write_text(json.dumps({'spec': spec, 'history': history}, allow_nan=False)+'\n')
            if epoch == 1 or epoch%10 == 0 or epoch == cfg['epochs']:
                print('HALFDROP', spec['run_id'], 'epoch', epoch, 'val', row['validation'],
                      'training_seconds', round(train_seconds, 3), flush=True)
    result = {'run_id': spec['run_id'], 'spec': spec, 'resolved_configuration': cfg,
              'drop_mask': spec['drop_mask'], 'drop_probabilities': probs,
              'seed': seed, 'recipe': 'sd_constant', 'history': history,
              'status': 'failed' if failed else 'complete', 'complete': not failed, 'diverged': failed,
              'failure_reason': failure_reason, 'best_epoch': best_epoch,
              'best_validation_loss': best_loss if math.isfinite(best_loss) else None,
              'initial_parameters_sha256': initial_hash, 'initialization_rng_sha256': init_rng_hash,
              'graph_setup_restored_initial_parameters': True, 'dataset': data['metadata'],
              'parameter_count': sum(p.numel() for p in model.parameters()), 'source_sha256': sources,
              'training_seconds': train_seconds, 'epoch_preparation_seconds': prep_seconds,
              'graph_setup_seconds': helper.setup_seconds, 'kernels': helper.metadata,
              'kernel_validation': qualification, 'steps_per_epoch': steps,
              'training_examples_per_epoch': steps*64, 'completed_epochs': len(history),
              'skipped_batch_counts': skip_counts.tolist(), 'executed_epoch_order_sha256': order_digest.hexdigest(),
              'executed_epoch_masks_sha256': mask_digest.hexdigest(), 'raw_mask_draws_sha256': draws_digest.hexdigest(),
              'mask_rng_scheme': 'CPU seed+200000; exactly4 draws per step for every subset; data order seed+100000',
              'hardware': {'gpu': torch.cuda.get_device_name(device), 'gpu_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
                           'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
                           'precision': 'FP32 tensors; torch.set_float32_matmul_precision(high); no autocast'}}
    recorder.finalize(result)
    if not failed:
        torch.save(model.state_dict(), directory/'final.pt')
        shutil.copyfile(directory/'best.pt', directory/'checkpoint.pt')
        result['trained_final_parameters_sha256'] = parameter_hash(model)
        result['checkpoints'] = {name: {'path': name, 'sha256': sha(directory/name)}
                                 for name in ('best.pt', 'final.pt', 'checkpoint.pt')}
        result['final_train'] = evaluate(model, tensors['train_x'], tensors['train_y'])
        model.load_state_dict(torch.load(directory/'best.pt', map_location=device, weights_only=True))
        result['selected_parameters_sha256'] = parameter_hash(model)
        result['selected_train'] = evaluate(model, tensors['train_x'], tensors['train_y'])
        model.load_state_dict(torch.load(directory/'final.pt', map_location=device, weights_only=True))
        selected_row = next(r for r in history if r['epoch'] == best_epoch)
        result['selected_validation'] = selected_row['validation']
        result['final_validation'] = history[-1]['validation']
        result['selected_train_probe'] = selected_row['dense_train_probe']
        result['final_train_probe'] = history[-1]['dense_train_probe']
    # Freeze a validation-only view before even opening official-test files.
    validation_view = copy.deepcopy(result)
    validation_view['test_accessed'] = False
    validation_view['training_and_selection_complete_unix'] = time.time()
    (directory/'validation.json').write_text(json.dumps(validation_view, allow_nan=False)+'\n')
    result['validation_view'] = validation_view
    result['validation_artifact'] = {'path': 'validation.json', 'sha256': sha(directory/'validation.json')}
    if progress_commit:
        progress_commit()
    if cfg['stage'] == 'evaluate' and not failed:
        test_started = time.perf_counter()
        test_data = load_mnist(root/'data', train_size=50000, include_test=True)
        if test_data['metadata']['training_data_sha256'] != data['metadata']['training_data_sha256']:
            raise AssertionError('Training split changed during test loading')
        tx = test_data['test_x'].to(device); ty = test_data['test_y'].to(device)
        result['test_dataset'] = test_data['metadata']
        result['test_labels_sha256'] = hashlib.sha256(test_data['test_y'].numpy().astype('<i8').tobytes()).hexdigest()
        result['test_order'] = 'All10000 official test examples in original IDX order'
        result['test_accessed'] = True
        try:
            result['final_test'] = export_predictions(model, tx, ty, directory/'final-test.npz')
            model.load_state_dict(torch.load(directory/'best.pt', map_location=device, weights_only=True))
            result['selected_parameters_sha256'] = parameter_hash(model)
            result['selected_test'] = export_predictions(model, tx, ty, directory/'selected-test.npz')
            result['prediction_definitions'] = {'evaluation': 'All four body branches present, gain1; no dropout',
                'probabilities': 'Stable float64 softmax of saved FP32 logits; columns digits0..9',
                'margin': 'Largest minus second-largest logit; label-free',
                'true_margin': 'True-class logit minus largest competing logit; label-dependent',
                'ce': 'Per-example CE from stable float64 arithmetic on saved logits',
              'confidence': 'Largest softmax probability', 'checkpoint.pt': 'Byte-identical copy of selected best.pt'}
        except FloatingPointError as error:
            result.update({'status': 'failed', 'complete': False, 'diverged': True, 'failure_reason': str(error)})
        result['test_export_seconds'] = time.perf_counter()-test_started
    else:
        result['test_accessed'] = False
    if source_hashes() != sources:
        raise RuntimeError('Frozen sources changed during execution')
    result['total_run_seconds'] = time.perf_counter()-started
    result['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                           for p in sorted(directory.iterdir()) if p.is_file()}
    (directory/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    if progress_commit:
        progress_commit()
    return result
