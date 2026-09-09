"""Frozen-checkpoint four-branch intervention audit; no neural training.

Presence gates at inference have gain one, unlike training's inverse survival.
Plain MLP deletion is explicitly artificial crop surgery. All-kept parity is
required for every checkpoint; scientific outcomes are saved separately by split.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from model_data import CiresanMLP, load_mnist
from . import metrics
from .policy import pooled_image_features

RECIPES = ('residual', 'sd_constant', 'sd_annealed', 'residual_unit_dropout', 'plain')


def masked_forward(model, x, gains_or_mask=15):
    n = len(model.layers)-2
    if isinstance(gains_or_mask, (int, np.integer)):
        if not 0 <= gains_or_mask < (1 << n):
            raise ValueError('Mask outside body-branch range')
        gains = [(int(gains_or_mask) >> j) & 1 for j in range(n)]
    else:
        gains = tuple(float(v) for v in gains_or_mask)
        if len(gains) != n or not all(np.isfinite(v) and v >= 0 for v in gains):
            raise ValueError('Need one finite nonnegative gain per body branch')
    h = F.relu(model.layers[0](x.reshape(-1, model.widths[0])*model.input_scale))
    for gain, layer in zip(gains, model.layers[1:-1]):
        skip = h[..., :layer.out_features]
        if gain == 0:
            h = F.relu(skip)  # The affine is genuinely not executed.
        elif model.residual:
            branch = layer(h)
            h = F.relu(skip+branch if gain == 1 else skip+gain*branch)
        elif gain == 1:
            h = F.relu(layer(h))
        else:
            h = F.relu((1-gain)*skip+gain*layer(h))
    h = model.layers[-1](h)
    return F.relu(h) if model.output_relu else h


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    here = Path(__file__).resolve().parent
    paths = [here/name for name in ('audit.py','metrics.py','policy.py','mechanisms.py','PROTOCOL.md')] + [here.parent/'model_data.py', here.parent/'hypothesis_app.py']
    return {str(p.relative_to(here.parent)): sha(p) for p in sorted(paths) if p.exists()}


@torch.no_grad()
def enumerate_logits(model, x, batch_size=2048):
    result = torch.empty((16, len(x), model.widths[-1]), device=x.device, dtype=torch.float32)
    for mask in range(16):
        for start in range(0, len(x), batch_size):
            result[mask,start:start+batch_size] = masked_forward(model,x[start:start+batch_size],mask)
    return result


def parity_check(logits, labels, reference, reference_order=None, loss_atol=3e-5):
    """Old validation JSON has accuracy/loss but no error IDs; test has IDs."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    pred = z.argmax(-1)
    wrong = np.flatnonzero(pred != y)
    accuracy = float(np.mean(pred == y))
    ce = metrics.cross_entropy_by_example(np.broadcast_to(z,(16,)+z.shape), y)[15].mean()
    if accuracy != reference['accuracy']:
        raise AssertionError(f'Dense reference accuracy mismatch: {accuracy} != {reference["accuracy"]}')
    if abs(ce-reference['loss']) > loss_atol:
        raise AssertionError(f'Dense reference CE mismatch: {ce} != {reference["loss"]}')
    if 'wrong_indices' in reference:
        ids = wrong if reference_order is None else np.asarray(reference_order)[wrong]
        if sorted(ids.tolist()) != sorted(reference['wrong_indices']):
            raise AssertionError('Dense error indices differ from checkpoint source')
    return {'passed':True,'accuracy':accuracy,'loss':float(ce),
            'reference_loss':reference['loss'],'loss_absolute_error':float(abs(ce-reference['loss'])),
            'loss_tolerance':loss_atol,'errors':len(wrong),
            'error_indices_checked':'wrong_indices' in reference,
            'error_indices_note':'Source validation records omit wrong_indices; official test records include them.'}


def summarize(logits, labels, path):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    result = metrics.analyze_masks(z,y)
    # These diagnostics are derivable from compact predictions/margins below.
    result['classification'].pop('per_example',None)
    result['risk_coverage'].pop('dense_ranking_example_indices',None)
    centered = metrics.centered_dense_deltas(z)
    coef = metrics.walsh_transform(centered)
    energy = np.stack([np.sum(coef[[m for m in range(16) if m.bit_count()==d]]**2,axis=(0,2))/z.shape[-1]
                       for d in range(5)])
    nonconstant = energy[1:].sum(0)
    higher = np.divide(energy[2:].sum(0),nonconstant,out=np.full(len(y),np.nan),where=nonconstant>1e-24)
    other = z[15].copy(); other[np.arange(len(y)),y] = -np.inf
    arrays = {'pred':z.argmax(-1).astype(np.uint8),
              'ce':metrics.cross_entropy_by_example(z,y),
              'dense_margin':metrics.dense_margin(z).astype(np.float32),
              'true_margin':(z[15,np.arange(len(y)),y]-other.max(-1)).astype(np.float32),
              'energy_by_degree':energy,'higher_fraction':higher,
              'labels':y.astype(np.uint8)}
    if not all(np.isfinite(v).all() for k,v in arrays.items() if k != 'higher_fraction'):
        raise AssertionError('Nonfinite scientific output')
    result['per_example_logit_interaction_fraction'] = {'mean':float(np.nanmean(higher)) if np.isfinite(higher).any() else None,
                                                       'undefined_count':int(np.isnan(higher).sum()),'denominator_floor':1e-24}
    np.savez_compressed(path,**arrays)
    return result


def run(spec, root, progress_commit=None):
    start = time.perf_counter()
    root=Path(root); out=root/'runs'/spec['run_id']; out.mkdir(parents=True,exist_ok=True)
    with (out/'claimed.json').open('x') as f: json.dump({'spec':spec,'unix':time.time()},f)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available(): raise RuntimeError('Paid audit requires CUDA')
    device='cuda'; seed=int(spec['seed'])
    sources=source_hashes()
    for name,digest in spec.get('source_sha256',{}).items():
        if sources.get(name)!=digest: raise AssertionError(f'Frozen source mismatch: {name}')
    data=load_mnist(root/'data',include_test=True)
    order=torch.randperm(10000,generator=torch.Generator().manual_seed(20260910))
    val_x=data['val_x'].to(device); val_y=data['val_y']
    probe_x=val_x[order[:128].to(device)]; probe_y=val_y[order[:128]].to(device)
    test_x=data['test_x'].to(device); test_y=data['test_y']
    np.savez_compressed(out/'data.npz',val_features=pooled_image_features(data['val_x'][order].numpy()),
                        test_features=pooled_image_features(data['test_x'].numpy()),val_labels=val_y[order].numpy(),
                        test_labels=test_y.numpy(),val_positions=order.numpy(),test_images=data['test_x'].numpy().astype(np.uint8))
    # Timed validation-only warmup determines probe size before any mechanism outcomes.
    probe_n=128
    records=[]
    hardware={'gpu':torch.cuda.get_device_name(), 'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
              'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
              'precision':'FP32 tensors, torch.set_float32_matmul_precision(high), inference batch2048'}
    for recipe in RECIPES:
        source_dir=root/'runs'/f'graph-eval-{recipe}-s{seed}'
        source=json.loads((source_dir/'result.json').read_text())
        if source['dataset']['training_data_sha256'] != data['metadata']['training_data_sha256']:
            raise AssertionError('Training data hash mismatch')
        ms=source['spec']
        model=CiresanMLP(recipe=recipe,pmax=ms.get('pmax',0),input_scale=ms['input_scale'],output_relu=ms['output_relu']).to(device).eval()
        for state,filename in [('selected','best.pt'),('final','final.pt')]:
            checkpoint=source_dir/filename
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
            epoch=source['best_epoch'] if state=='selected' else ms['epochs']
            row=next(r for r in source['history'] if r['epoch']==epoch)
            record={'recipe':recipe,'state':state,'seed':seed,'epoch':epoch,'checkpoint_sha256':sha(checkpoint),
                    'source_result_sha256':sha(source_dir/'result.json'),'source_run':ms['run_id'],
                    'intervention':'forced crop surgery on plain MLP' if recipe=='plain' else 'unscaled residual branch deletion'}
            with torch.no_grad():
                a=masked_forward(model,val_x[:128],15); b=model(val_x[:128])
                err=float((a-b).abs().max())
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            record['direct_forward_parity']={'max_abs_logit_error':err,'bitwise_equal':True,'probe_n':128}
            if not records:
                with torch.no_grad():
                    for _ in range(2): masked_forward(model,val_x[:2048],15)
                    torch.cuda.synchronize(); warm=time.perf_counter()
                    for _ in range(10): masked_forward(model,val_x[:2048],15)
                    torch.cuda.synchronize(); batch_seconds=(time.perf_counter()-warm)/10
                # 1600 full-cost batches overestimates 10*2*16*5 heterogeneous masks.
                projected=1600*batch_seconds
                probe_n=128 if projected<70 else 64 if projected<100 else 32
                if projected>135: raise RuntimeError(f'Warmup projection {projected:.1f}s leaves insufficient bounded audit time')
                warmup={'dense_2048_seconds':batch_seconds,'conservative_mask_seconds':projected,'chosen_probe_n':probe_n}
            for split,x,y,ref in [('validation',val_x,val_y,row['validation']),
                                  ('test',test_x,test_y,source['selected_test' if state=='selected' else 'final_test'])]:
                t=time.perf_counter()
                z=enumerate_logits(model,x)
                logits=z.cpu().numpy()
                record[split+'_parity']=parity_check(logits[15],y.numpy(),ref)
                if split=='validation':
                    logits=logits[:,order.numpy()]
                    stored_y=y[order].numpy()
                else: stored_y=y.numpy()
                record[split]=summarize(logits,stored_y,out/f'{recipe}-{state}-{split}.npz')
                record[split+'_seconds']=time.perf_counter()-t
                if split=='validation': probe_logits=z[:,order[:probe_n].to(device)].detach().clone()
                del z,logits
            if recipe in ('residual','sd_constant','sd_annealed'):
                from .mechanisms import run_mechanisms
                t=time.perf_counter()
                bulk_probe_logits=probe_logits
                probe_logits=enumerate_logits(model,probe_x[:probe_n],batch_size=probe_n)
                record['probe_batch_numerics']={'batch_size':probe_n,'bulk_batch_size':2048,
                    'max_abs_logit_difference':float((probe_logits-bulk_probe_logits).abs().max()),
                    'prediction_disagreements':int((probe_logits.argmax(-1)!=bulk_probe_logits.argmax(-1)).sum()),
                    'note':'Mechanism corners recomputed at the same batch shape as fractional-gain and gradient probes.'}
                record['mechanisms']=run_mechanisms(model,probe_x[:probe_n],probe_y[:probe_n],probe_logits,epoch,recipe,masked_forward)
                record['mechanism_seconds']=time.perf_counter()-t
            else: record['mechanisms']={'status':'outside central mechanism panel'}
            (out/f'{recipe}-{state}.json').write_text(json.dumps(record,allow_nan=False)+'\n')
            records.append({'recipe':recipe,'state':state,'epoch':epoch,'validation_accuracy':record['validation_parity']['accuracy'],
                            'test_accuracy':record['test_parity']['accuracy'],'elapsed_seconds':time.perf_counter()-start})
            (out/'progress.json').write_text(json.dumps({'completed':records,'elapsed_seconds':time.perf_counter()-start}))
            print('AUDIT',seed,recipe,state,'complete',round(time.perf_counter()-start,2),flush=True)
        del model
    artifacts={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    result={'run_id':spec['run_id'],'spec':spec,'source_sha256':sources,'hardware':hardware,'dataset':data['metadata'],
            'warmup':warmup,'validation_order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest(),
            'records':records,'artifacts':artifacts,'elapsed_seconds':time.perf_counter()-start,'finished_unix':time.time()}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if progress_commit: progress_commit()
    return result
