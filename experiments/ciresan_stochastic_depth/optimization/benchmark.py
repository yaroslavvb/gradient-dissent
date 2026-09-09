"""A100 speed study: same-width baseline, transparent time-to-test-target.

This benchmark monitors the official test set to reach a user-requested target.
It is intentionally separate from the validation-only stochastic-depth study.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from model_data import CiresanMLP, _download, _decode_idx
from optimization.kernels import BaselineStep, make_sgd


def dataset(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    files = {}
    def load(name, images, n):
        blob, meta = _download(root, name); files[name] = meta
        return _decode_idx(blob, images, n)
    x = load('train-images-idx3-ubyte.gz', True, 60000).float().flatten(1)
    y = load('train-labels-idx1-ubyte.gz', False, 60000)
    tx = load('t10k-images-idx3-ubyte.gz', True, 10000).float().flatten(1)
    ty = load('t10k-labels-idx1-ubyte.gz', False, 10000)
    return x.cuda(), y.cuda(), tx.cuda(), ty.cuda(), files


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    errors = torch.zeros((), device=x.device, dtype=torch.long)
    loss = torch.zeros((), device=x.device)
    for start in range(0, len(y), 2048):
        logits = model(x[start:start+2048]).float()
        target = y[start:start+2048]
        errors += (logits.argmax(1) != target).sum()
        loss += F.cross_entropy(logits, target, reduction='sum')
    nerrors = int(errors.item())
    result = {'errors': nerrors, 'accuracy': 1-nerrors/len(y), 'loss': float(loss.item()/len(y)), 'n':len(y)}
    model.train()
    return result


def model_and_step(spec, x, y):
    torch.manual_seed(spec.get('seed', 1))
    model = CiresanMLP(recipe=spec.get('recipe','plain'), pmax=spec.get('dropout',0.), input_scale=spec.get('input_scale', 1.),
                       output_relu=spec.get('output_relu', True))
    digest = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    model.cuda().train()
    optimizer = make_sgd(model, lr=spec.get('lr', .001), momentum=.9,
                         fused=spec.get('fused', False), foreach=not spec.get('fused', False))
    step = BaselineStep(model, optimizer, x[:spec['batch_size']], y[:spec['batch_size']],
                        mode=spec.get('mode','eager'), precision=spec.get('precision','fp32'),
                        compile_mode=spec.get('compile_mode','reduce-overhead'),
                        shrinkage=spec.get('shrinkage', .00002))
    return model, optimizer, step, digest


@torch.no_grad()
def class_diagnostics(model, x, y):
    model.eval(); pre=[]; logits=[]
    hook=model.layers[-1].register_forward_hook(lambda module,args,out:pre.append(out.detach()))
    try:
        for start in range(0,len(y),2048):logits.append(model(x[start:start+2048]))
    finally:hook.remove()
    z=torch.cat(logits);a=torch.cat(pre);pred=z.argmax(1)
    result={'head_preactivation_max_by_class':a.amax(0).cpu().tolist(),
        'positive_logit_fraction_by_class':(z>0).float().mean(0).cpu().tolist(),
        'prediction_class_counts':torch.bincount(pred,minlength=10).cpu().tolist(),
        'true_class_counts':torch.bincount(y,minlength=10).cpu().tolist(),
        'class_accuracy':[float((pred[y==k]==k).float().mean().item()) for k in range(10)],
        'last_minibatch_head_row_gradient_norms':model.layers[-1].weight.grad.norm(dim=1).cpu().tolist()}
    model.train();return result


def run(spec, root, invocation_start=None, progress_commit=None):
    start = time.perf_counter() if invocation_start is None else invocation_start
    root=Path(root)
    run_dir=root/'optimization'/spec['run_id'];run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/'claim.json').open('x') as f: json.dump(spec,f)
    if progress_commit:progress_commit()
    torch.set_num_threads(2)
    # Inductor 2.14 still calls the legacy precision getter, which raises after
    # setting the new backend-specific API (reproduced on CPU and A100). Use
    # only this compatible API in a fresh process; do not mix precision APIs.
    torch.set_float32_matmul_precision('high')
    x,y,tx,ty,files=dataset(root/'data')
    result={'run_id':spec['run_id'],'spec':spec,'dataset':{'train_n':60000,'test_n':10000,'files':files},
            'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['benchmark.py','kernels.py']},
            'test_monitoring':'Official test is monitored as a speed target; this is not an unbiased generalization study.',
            'hardware':{'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
                        'tf32':True,'precision_api':'torch.set_float32_matmul_precision(high)',
                        'cpu_threads':torch.get_num_threads()}}
    if spec.get('kind')=='microbench' or spec.get('validate_kernels'):
        import unittest
        from optimization import test_kernels
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_kernels)
        checked=unittest.TextTestRunner(verbosity=2).run(suite)
        if not checked.wasSuccessful():
            raise RuntimeError('Kernel equivalence/restoration tests failed on A100.')
        result['kernel_tests']={'tests_run':checked.testsRun,'skipped':len(checked.skipped),'passed':True}
    if spec.get('kind')=='microbench':
        rows=[]
        for variant in spec['variants']:
            setting={**spec,**variant}
            print('BENCHMARK',variant,flush=True)
            variant_start=time.perf_counter()
            model,optimizer,step,digest=model_and_step(setting,x,y)
            setup=time.perf_counter()-variant_start
            # Actual examples are used, but the repeated batch is a kernel test,
            # never a convergence result. Accuracy is evaluated in separate runs.
            bx,by=x[:setting['batch_size']],y[:setting['batch_size']]
            for _ in range(10):step(bx,by)
            torch.cuda.synchronize()
            times=[]
            for _ in range(3):
                begin=time.perf_counter()
                for _ in range(spec.get('measure_steps',100)):step(bx,by)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-begin)/spec.get('measure_steps',100))
            row={'variant':variant,'setup_seconds':setup,'step_seconds':times,
                 'median_step_seconds':float(np.median(times)), 'initial_parameters_sha256':digest}
            rows.append(row);print(json.dumps(row),flush=True)
            del step,optimizer,model
            torch.cuda.empty_cache()
        result['benchmarks']=rows
    else:
        model,optimizer,step,digest=model_and_step(spec,x,y)
        result['initial_parameters_sha256']=digest
        torch.cuda.synchronize()
        result['setup_seconds']=time.perf_counter()-start
        order_rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',1)+100000)
        history=[]; training_seconds=0.;threshold=None
        schedule_events=[]
        batch=spec['batch_size'];nsteps=len(y)//batch
        # Inputs copied outside the graph, without allocation on the replay path.
        bx=torch.empty_like(x[:batch]);by=torch.empty_like(y[:batch])
        for epoch in range(spec.get('max_epochs',100)):
            if str(epoch+1) in spec.get('lr_schedule',{}):
                factor=spec['lr_schedule'][str(epoch+1)]
                for group in optimizer.param_groups:group['lr']=spec['lr']*factor
                step=BaselineStep(model,optimizer,bx,by,mode=spec.get('mode','eager'),
                    precision=spec.get('precision','fp32'),shrinkage=spec.get('shrinkage',.00002)*factor)
                schedule_events.append({'epoch':epoch+1,'factor':factor,'recapture_seconds':step.setup_seconds})
            order=torch.randperm(len(y),generator=order_rng).cuda()
            torch.cuda.synchronize(); epoch_start=time.perf_counter()
            loss=None
            for i in range(nsteps):
                ids=order[i*batch:(i+1)*batch]
                torch.index_select(x,0,ids,out=bx)
                torch.index_select(y,0,ids,out=by)
                loss=step(bx,by)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-epoch_start;training_seconds+=elapsed
            train_loss=float(loss.item())
            if not math.isfinite(train_loss):
                result['diverged']=True;break
            if (epoch+1)%spec.get('eval_every',1)==0:
                score=evaluate(model,tx,ty)
                elapsed_to_score=time.perf_counter()-start
                row={'epoch':epoch+1,'test':score,'last_minibatch_loss':train_loss,
                     'training_seconds':training_seconds,'run_wall_seconds':elapsed_to_score,
                     'epoch_training_seconds':elapsed}
                history.append(row)
                if epoch == 0 or (epoch+1)%10 == 0 or score['errors']<=137:
                    print(json.dumps({'run_id':spec['run_id'],**row}),flush=True)
                (run_dir/'progress.json').write_text(json.dumps({'spec':spec,'history':history},indent=2))
                # Exact integer criterion: 98.63% means <=137 errors of10000.
                if threshold is None and score['errors']<=137:
                    threshold=row.copy()
                    torch.save(model.state_dict(),run_dir/'target.pt')
                    if spec.get('stop_at_target',True):break
        result.update({'history':history,'threshold':threshold,'training_seconds':training_seconds,
                       'setup_helper_seconds':step.setup_seconds,'steps_per_epoch':nsteps,
                       'train_examples_per_epoch':nsteps*batch,'parameter_count':sum(p.numel() for p in model.parameters()),
                       'last_evaluated_test':history[-1]['test'] if history else None,
                       'schedule_events':schedule_events,
                       'final_test':history[-1]['test'] if history and not result.get('diverged') else None})
        if not result.get('diverged'):
            result['last_class_diagnostics']=class_diagnostics(model,tx,ty)
    result['total_run_seconds']=time.perf_counter()-start
    (run_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
