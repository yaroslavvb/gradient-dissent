"""GPU-resident MNIST training, explicit RNG streams, and held-out selection.

No curvature hooks, W&B writes, data augmentation, or test-set model selection.
Run locally with --spec JSON --root DIR; Modal calls the same run() function.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from model_data import CiresanMLP, load_mnist


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, x, y, batch_size=2048, include_errors=False):
    model.eval()
    losses, correct, predictions = [], [], []
    for start in range(0, len(y), batch_size):
        logits = model(x[start:start + batch_size])
        target = y[start:start + batch_size]
        losses.append(nn.functional.cross_entropy(logits, target, reduction='sum'))
        pred = logits.argmax(1)
        correct.append((pred == target).sum())
        if include_errors:
            predictions.append(pred)
    result = {'loss': float(torch.stack(losses).sum().item() / len(y)),
              'accuracy': float(torch.stack(correct).sum().item() / len(y)), 'n': len(y)}
    if include_errors:
        pred = torch.cat(predictions)
        result['wrong_indices'] = torch.where(pred != y)[0].cpu().tolist()
        result['errors'] = len(result['wrong_indices'])
    return result


def probabilities(spec, epoch):
    pmax = spec.get('pmax', 0.)
    if spec['recipe'] == 'sd_annealed':
        pmax *= 1 - epoch / max(1, spec['epochs'] - 1)
    if spec['recipe'] not in ('sd_constant', 'sd_annealed'):
        pmax = 0.
    return [pmax * i / 4 for i in range(1, 5)]


def run(spec, root, progress_commit=None):
    root = Path(root)
    run_dir = root / 'runs' / spec['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    # Never silently repeat a paid run after a driver interruption.
    claim = run_dir / 'claimed.json'
    with claim.open('x') as handle:
        json.dump({'spec': spec, 'unix': time.time()}, handle)
    if progress_commit:
        progress_commit()
    start_time = time.perf_counter()
    seed = int(spec['seed'])
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # TF32 preserves FP32 parameters/optimizer; no BF16 quantization confound.
    device = torch.device(spec.get('device', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
    data = load_mnist(root / 'data', train_size=spec.get('train_size', 50000),
                      include_test=spec['stage'] == 'evaluate')
    data_meta = data['metadata']
    tensors = {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
    model = CiresanMLP(recipe=spec['recipe'], pmax=spec.get('pmax', 0.),
                       input_scale=spec.get('input_scale', 1 / 255),
                       output_relu=spec.get('output_relu', False))
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec['lr'], momentum=spec.get('momentum', .9),
                                foreach=True)
    # Data order and Bernoulli masks are independent of initialization/dropout RNG.
    order_rng = torch.Generator(device='cpu').manual_seed(seed + 100000)
    mask_rng = torch.Generator(device='cpu').manual_seed(seed + 200000)
    torch.manual_seed(seed + 300000)
    n = len(tensors['train_y'])
    batch_size = spec.get('batch_size', 64)
    steps_per_epoch = n // batch_size
    eval_every = spec.get('eval_every', 5)
    history = []
    best_val_loss = float('inf')
    best_epoch = None
    train_seconds = 0.
    skipped_counts = np.zeros(4, dtype=np.int64)
    grad_observations = []
    failed = False
    probe_n = min(10000, n)
    synchronize(device)
    for epoch in range(spec['epochs']):
        probs = probabilities(spec, epoch)
        order = torch.randperm(n, generator=order_rng).to(device)
        # Batch-shared masks permit real compute skipping, generated once per epoch.
        active_masks = (torch.rand((steps_per_epoch, 4), generator=mask_rng) >= torch.tensor(probs)).tolist()
        skipped_counts += np.sum(np.logical_not(active_masks), axis=0)
        model.train()
        losses = []
        synchronize(device)
        epoch_start = time.perf_counter()
        for step in range(steps_per_epoch):
            ids = order[step * batch_size:(step + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            out = model(tensors['train_x'][ids], drop_probs=probs, active=active_masks[step])
            loss = nn.functional.cross_entropy(out, tensors['train_y'][ids])
            loss.backward()
            if step == 0 and (epoch % eval_every == 0 or epoch == spec['epochs'] - 1):
                grad_observations.append({'epoch': epoch + 1,
                    'layer_grad_norms': [float(layer.weight.grad.norm().item()) if layer.weight.grad is not None else None for layer in model.layers]})
            optimizer.step()
            if spec.get('shrinkage', 0.):
                # Exact historical convention, not torch.optim weight_decay.
                with torch.no_grad():
                    torch._foreach_mul_(list(model.parameters()), 1 - spec['shrinkage'])
            losses.append(loss.detach())
        synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        train_seconds += elapsed
        stochastic_loss = float(torch.stack(losses).mean().item())
        if not math.isfinite(stochastic_loss):
            failed = True
            history.append({'epoch': epoch + 1, 'diverged': True, 'training_seconds': train_seconds})
            break
        if epoch == 0 or (epoch + 1) % eval_every == 0 or epoch == spec['epochs'] - 1:
            val = evaluate(model, tensors['val_x'], tensors['val_y'])
            train_eval = evaluate(model, tensors['train_x'][:probe_n], tensors['train_y'][:probe_n])
            with torch.no_grad():
                logits = model(tensors['val_x'][:2048])
                predicted = torch.bincount(logits.argmax(1), minlength=10).cpu().tolist()
                zero_logit_fraction = float((logits == 0).float().mean().item())
            row = {'epoch': epoch + 1, 'stochastic_training_loss': stochastic_loss,
                   'dense_train_probe': train_eval, 'validation': val,
                   'training_seconds': train_seconds, 'epoch_training_seconds': elapsed,
                   'drop_probabilities': probs, 'zero_logit_fraction': zero_logit_fraction,
                   'validation_probe_prediction_counts': predicted}
            history.append(row)
            if val['loss'] < best_val_loss:
                best_val_loss, best_epoch = val['loss'], epoch + 1
                torch.save(model.state_dict(), run_dir / 'best.pt')
            (run_dir / 'progress.json').write_text(json.dumps({'spec': spec, 'history': history}, indent=2, allow_nan=False))
            print(json.dumps({'run_id': spec['run_id'], 'epoch': epoch + 1,
                              'val_accuracy': round(val['accuracy'], 5), 'val_loss': round(val['loss'], 5),
                              'training_seconds': round(train_seconds, 2)}), flush=True)
    result = {'spec': spec, 'run_id': spec['run_id'], 'history': history,
              'best_validation_loss': best_val_loss if math.isfinite(best_val_loss) else None,
              'best_epoch': best_epoch, 'diverged': failed,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'initial_parameters_sha256': initial_hash, 'dataset': data_meta,
              'training_seconds': train_seconds, 'gradient_observations': grad_observations,
              'skipped_batch_counts': skipped_counts.tolist(),
              'steps_per_epoch': steps_per_epoch, 'training_examples_per_epoch': steps_per_epoch * batch_size,
              'hardware': {'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                           'torch': str(torch.__version__), 'tf32': True, 'dtype': 'float32',
                           'cpu_threads': torch.get_num_threads()}}
    if spec['stage'] == 'evaluate' and not failed:
        result['final_train'] = evaluate(model, tensors['train_x'], tensors['train_y'])
        result['final_test'] = evaluate(model, tensors['test_x'], tensors['test_y'], include_errors=True)
        torch.save(model.state_dict(), run_dir / 'final.pt')
        model.load_state_dict(torch.load(run_dir / 'best.pt', weights_only=True, map_location=device))
        result['selected_train'] = evaluate(model, tensors['train_x'], tensors['train_y'])
        result['selected_test'] = evaluate(model, tensors['test_x'], tensors['test_y'], include_errors=True)
    result['total_run_seconds'] = time.perf_counter() - start_time
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.spec).read_text()), Path(args.root)), indent=2))
