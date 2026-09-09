"""Actual grouped image-only inference timing; no compilation or training.

run() needs an existing CUDA device and downloaded frozen checkpoints. This
module never provisions hardware. Tensor helpers also run on CPU for tests.

Frozen numerical handling: CPU/GPU routing decisions and dense reference
predictions must match exactly. Grouped GEMMs use different batch shapes and
may change FP32/TF32 decisions. Such differences are reported with actual and
cached functional accuracy, example IDs, and reconstructed logit differences;
they do not silently become an equivalence claim or invalidate measured time.
Median of all3 measured passes is primary; median of passes2/3 is secondary.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import numpy as np
import torch

try:
    from . import policy
    from ..model_data import CiresanMLP, load_mnist
except ImportError:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from layerdrop_hypotheses import policy
    from model_data import CiresanMLP, load_mnist

METHODS=('dense','static','routed','random_cost_matched')
BATCH_SIZE=2048
WARMUP_BATCHES=2
TIMED_PASSES=3


def torch_image_features(raw_images):
    """Same49 features as policy.pooled_image_features, including rounding.

    Float64 reduction/division followed by float32 avoids one-ULP changes
    near tree thresholds from backend-specific reciprocal multiplication.
    The feature calculation remains inside the routed inference timer.
    """
    images=raw_images.reshape(-1,28,28)
    return (images.reshape(-1,7,4,7,4).sum(dim=(2,4),dtype=torch.float64)/4080.).reshape(-1,49).float()


class TorchPolicy:
    """Serialized policy application, with explicit float64 decisions."""
    def __init__(self,model,device):
        self.model=model;self.device=torch.device(device);self.penalty=model['chosen_lambda']
        self.cost=torch.tensor(model['cost_fractions'],dtype=torch.float64,device=device)
        priority=np.lexsort((-np.arange(16),-np.asarray(model['cost_fractions'])))
        if 'mask_tie_priority' in model and model['mask_tie_priority']!=priority.tolist():
            raise ValueError('Serialized tie priority disagrees with quality-conservative rule')
        self.priority=torch.tensor(priority,dtype=torch.int64,device=device)
        tree=model['tree']
        self.left=torch.tensor(tree['children_left'],dtype=torch.int64,device=device)
        self.right=torch.tensor(tree['children_right'],dtype=torch.int64,device=device)
        self.feature=torch.tensor(tree['feature'],dtype=torch.int64,device=device)
        self.threshold=torch.tensor(tree['threshold'],dtype=torch.float64,device=device)
        self.values=torch.tensor(tree['values'],dtype=torch.float64,device=device)
        # CPU-only metadata validation; no CUDA synchronization in tree loops.
        if self.values.shape!=(len(tree['children_left']),16):raise ValueError('Malformed tree values')
        def check(node,depth,ancestors):
            if node<0 or node>=len(tree['children_left']) or node in ancestors:raise ValueError('Malformed tree child')
            left=tree['children_left'][node]
            if left==-1:return
            if depth>=3 or not 0<=tree['feature'][node]<49:raise ValueError('Tree exceeds fixed depth or feature bounds')
            check(left,depth+1,ancestors|{node});check(tree['children_right'][node],depth+1,ancestors|{node})
        check(0,0,set())

    def route_features(self,features):
        features=features.float()
        if self.penalty is None:return torch.full((len(features),),15,dtype=torch.int64,device=features.device)
        nodes=torch.zeros(len(features),dtype=torch.int64,device=features.device)
        rows=torch.arange(len(features),device=features.device)
        for _ in range(3):
            left=self.left[nodes];internal=left>=0
            coordinate=self.feature[nodes].clamp(min=0)
            go_left=features[rows,coordinate].double()<=self.threshold[nodes]
            candidate=torch.where(go_left,left,self.right[nodes])
            nodes=torch.where(internal,candidate,nodes)
        predicted=self.values[nodes].clone();predicted[:,15]=0.
        score=predicted+float(self.penalty)*self.cost[None,:]
        return self.priority[score[:,self.priority].argmin(dim=1)]

    def __call__(self,raw_images):
        # Even dense fallback runs the common feature path, so its benchmark
        # does not silently omit image-only routing overhead.
        return self.route_features(torch_image_features(raw_images))


def grouped_forward(model,images,mask_ids,masked_forward):
    """Group, gather, execute true branch skips, scatter original row order.

    A single16-count device-to-host transfer decides the Python group loop.
    This synchronization, sorting, all gathers and scatters are timed.
    """
    if mask_ids.ndim!=1 or len(mask_ids)!=len(images):raise ValueError('Mask/image count mismatch')
    order=torch.argsort(mask_ids,stable=True)
    counts=torch.bincount(mask_ids,minlength=16).cpu().tolist()
    if len(counts)!=16:raise ValueError('Mask IDs outside0..15')
    output=None;offset=0
    for mask,count in enumerate(counts):
        if count:
            indices=order[offset:offset+count]
            logits=masked_forward(model,images.index_select(0,indices),mask)
            if output is None:output=torch.empty((len(images),logits.shape[-1]),dtype=logits.dtype,device=logits.device)
            output.index_copy_(0,indices,logits)
            offset+=count
    if output is None:raise ValueError('Empty grouped batch')
    return output


def _record_identity(key,model):
    if all(k in model for k in ('recipe','seed','endpoint')):
        recipe,seed,endpoint=model['recipe'],int(model['seed']),model['endpoint']
    else:
        match=re.fullmatch(r'(residual|sd_constant|sd_annealed)-selected-s(101|102|103)',key)
        alternate=re.fullmatch(r'(101|102|103)/(residual|sd_constant|sd_annealed)-selected',key)
        if match:recipe,seed=match.group(1),int(match.group(2))
        elif alternate:seed,recipe=int(alternate.group(1)),alternate.group(2)
        else:raise ValueError('Unknown policy identity: '+key)
        endpoint='selected'
    if recipe not in ('residual','sd_constant','sd_annealed') or seed not in (101,102,103) or endpoint!='selected':
        raise ValueError('Routing benchmark is restricted to primary9 selected checkpoints')
    return recipe,seed


def _sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _predict_pass(method,model,x,router,static_mask,random_masks,masked_forward,batch_size=BATCH_SIZE):
    predictions=torch.empty(len(x),dtype=torch.int64,device=x.device)
    routed_masks=torch.empty(len(x),dtype=torch.int64,device=x.device)
    for start in range(0,len(x),batch_size):
        images=x[start:start+batch_size]
        if method=='dense':
            logits=masked_forward(model,images,15);masks=torch.full((len(images),),15,dtype=torch.int64,device=x.device)
        elif method=='static':
            logits=masked_forward(model,images,static_mask);masks=torch.full((len(images),),static_mask,dtype=torch.int64,device=x.device)
        elif method=='routed':
            masks=router(images);logits=grouped_forward(model,images,masks,masked_forward)
        elif method=='random_cost_matched':
            masks=random_masks[start:start+batch_size];logits=grouped_forward(model,images,masks,masked_forward)
        else:raise ValueError('Unknown method')
        predictions[start:start+len(images)]=logits.argmax(dim=-1)
        routed_masks[start:start+len(images)]=masks
    return predictions,routed_masks


def _score(predicted,labels,masks,cost):
    wrong=predicted!=labels
    return {'n':len(labels),'errors':int(wrong.sum()),'accuracy':float(1-wrong.mean()),
            'prediction_sha256':hashlib.sha256(np.ascontiguousarray(predicted,dtype=np.int64).tobytes()).hexdigest(),
            'predictions':predicted.tolist(),'mask_counts':np.bincount(masks,minlength=16).tolist(),
            'mac_saving_fraction':float(1-np.asarray(cost)[masks].mean())}


def _logit_diagnostics(model,x,masks,indices,masked_forward,expected):
    """Reproduce affected full batches outside timers, preserving group shape."""
    rows=[]
    for start in sorted({(int(i)//BATCH_SIZE)*BATCH_SIZE for i in indices}):
        batch=x[start:start+BATCH_SIZE];ids=masks[start:start+len(batch)]
        actual=grouped_forward(model,batch,torch.from_numpy(ids).to(x.device),masked_forward)
        local=[int(i)-start for i in indices if start<=i<start+len(batch)]
        for mask in sorted({int(ids[i]) for i in local}):
            reference=masked_forward(model,batch,mask)
            take=[i for i in local if ids[i]==mask]
            a=actual[take].cpu().numpy();r=reference[take].cpu().numpy()
            for offset,index in enumerate(take):
                row=index+start;difference=a[offset]-r[offset]
                rows.append({'example_index':row,'mask_id':mask,'actual_logits':a[offset].tolist(),
                             'full_batch_mask_logits':r[offset].tolist(),'max_absolute_logit_difference':float(np.max(np.abs(difference))),
                             'logit_difference_l2':float(np.linalg.norm(difference)),
                             'reconstructed_full_batch_prediction':int(r[offset].argmax()),
                             'cached_npz_prediction':int(expected[mask,row]),
                             'reconstructed_full_batch_matches_cached':bool(r[offset].argmax()==expected[mask,row])})
    return rows


def run(spec,root,progress_commit=None):
    if spec.get('mode')!='routing':raise ValueError('Expected mode=routing')
    if not torch.cuda.is_available():raise RuntimeError('Requires an existing CUDA device; no provisioning performed')
    if spec.get('batch_size',BATCH_SIZE)!=BATCH_SIZE or spec.get('warmup_batches',WARMUP_BATCHES)!=WARMUP_BATCHES or spec.get('timed_passes',TIMED_PASSES)!=TIMED_PASSES:
        raise ValueError('Frozen benchmark uses batch2048,2 warmup batches,3 full-test passes')
    policies=spec['policies'];identities={key:_record_identity(key,model) for key,model in policies.items()}
    if set(identities.values())!={(r,s) for r in ('residual','sd_constant','sd_annealed') for s in (101,102,103)} or len(policies)!=9:
        raise ValueError('Expected exactly the9 primary selected-checkpoint policies')
    try:from .audit import masked_forward
    except ImportError:from layerdrop_hypotheses.audit import masked_forward
    started=time.perf_counter();root=Path(root);out=root/'runs'/spec['run_id'];out.mkdir(parents=True,exist_ok=True)
    with (out/'claimed.json').open('x') as handle:json.dump({'spec':spec,'unix':time.time()},handle)
    if progress_commit:progress_commit()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda',torch.cuda.current_device())
    source_paths=[Path(__file__),Path(policy.__file__),Path(__file__).with_name('audit.py'),Path(__file__).parents[1]/'model_data.py']
    sources={str(path.relative_to(Path(__file__).parents[1])):_sha_file(path) for path in source_paths}
    for name,digest in spec.get('source_sha256',{}).items():
        local=Path(__file__).parents[1]/name
        if not local.exists() or _sha_file(local)!=digest:raise AssertionError('Frozen source mismatch: '+name)
    data=load_mnist(root/'data',include_test=True)
    cpu_x=data['test_x'];labels=data['test_y'].numpy();dataset_metadata=data['metadata']
    if len(labels)!=10000:raise ValueError('Expected official10000 test examples')
    features=policy.pooled_image_features(cpu_x.numpy())
    x=cpu_x.to(device);del data
    result={'run_id':spec['run_id'],'spec':spec,'source_sha256':sources,'dataset':dataset_metadata,
            'hardware':{'gpu':torch.cuda.get_device_name(device),'torch':str(torch.__version__),'cuda':torch.version.cuda,
                        'float32_matmul_precision':torch.get_float32_matmul_precision(),'dtype':'float32'},
            'timing_protocol':{'batch_size':BATCH_SIZE,'warmup_batches_per_method':WARMUP_BATCHES,'whole_test_timed_passes':TIMED_PASSES,
                               'method_order':'Cyclic rotation by model index plus pass index',
                               'timed_scope':'GPU-resident raw images through prediction and mask output; routed includes feature calculation, tree decisions, sorting, count readback, gathers, true skipped forwards and scatters; synchronize before/after each full pass',
                               'excluded':'Imports, dataset loading/transfers, checkpoint loading, tree tensor preparation, CPU verification/scoring, result writes and remote startup; no compile or CUDA graph setup',
                               'first_pass_scope':'First measured full-test pass after2 batch warmups; not a cold-start measurement',
                               'primary_median_scope':'Median of all3 measured whole-test passes',
                               'warm_median_scope':'Secondary median of measured passes2/3; all3 durations also retained'},
            'models':{},'verification_passed':True,'all_reference_predictions_exact':True}
    with torch.inference_mode():
        for index,(key,serialized) in enumerate(sorted(policies.items())):
            recipe,seed=identities[key]
            checkpoint=root/'runs'/f'graph-eval-{recipe}-s{seed}'/'best.pt'
            reference=root/'runs'/f'hypothesis-audit-s{seed}-v1'/f'{recipe}-selected-test.npz'
            reference=Path(spec.get('reference_npz',{}).get(key,reference))
            with np.load(reference,allow_pickle=False) as reference_data:
                expected=reference_data['pred'].copy()
                if 'labels' in reference_data:np.testing.assert_array_equal(reference_data['labels'],labels)
            if expected.shape!=(16,len(labels)):raise ValueError('Reference prediction shape mismatch')
            setup_start=time.perf_counter()
            model=CiresanMLP(recipe=recipe,pmax={'residual':0.,'sd_constant':.4,'sd_annealed':.8}[recipe]).to(device)
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True);model.eval()
            router=TorchPolicy(serialized,device)
            static_mask=int(serialized['static_comparator']['selected_mask_id'])
            cpu_masks=policy.apply_policy(serialized,features)
            gpu_features=torch_image_features(x)
            np.testing.assert_array_equal(gpu_features.cpu().numpy(),features)
            gpu_masks=router.route_features(gpu_features).cpu().numpy()
            np.testing.assert_array_equal(gpu_masks,cpu_masks)
            random_cpu=policy.random_cost_matched_masks(cpu_masks)
            random_gpu=torch.from_numpy(random_cpu).to(device)
            torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup_start
            # All-kept semantic parity at the actual dense benchmark batch.
            if not torch.equal(masked_forward(model,x[:BATCH_SIZE],15),model(x[:BATCH_SIZE])):
                raise AssertionError('masked_forward(mask15) differs from dense model')
            for method in METHODS:
                _predict_pass(method,model,x[:WARMUP_BATCHES*BATCH_SIZE],router,static_mask,random_gpu[:WARMUP_BATCHES*BATCH_SIZE],masked_forward)
            torch.cuda.synchronize()
            timings={m:[] for m in METHODS};outputs={};orders=[]
            for repeat in range(TIMED_PASSES):
                offset=(index+repeat)%len(METHODS);order=METHODS[offset:]+METHODS[:offset];orders.append(list(order))
                for method in order:
                    torch.cuda.synchronize();tick=time.perf_counter()
                    pred,masks=_predict_pass(method,model,x,router,static_mask,random_gpu,masked_forward)
                    torch.cuda.synchronize();timings[method].append(time.perf_counter()-tick)
                    current=(pred.cpu().numpy(),masks.cpu().numpy())
                    if method in outputs:
                        np.testing.assert_array_equal(current[0],outputs[method][0]);np.testing.assert_array_equal(current[1],outputs[method][1])
                    else:outputs[method]=current
            details={};reference_pass=True
            for method in METHODS:
                pred,masks=outputs[method]
                wanted=expected[masks,np.arange(len(labels))]
                mismatches=np.flatnonzero(pred!=wanted)
                reference_pass=reference_pass and not len(mismatches)
                if method in ('dense','static') and len(mismatches):
                    raise AssertionError(f'{method} full-batch predictions differ from frozen reference at{len(mismatches)} examples')
                details[method]={**_score(pred,labels,masks,serialized['cost_fractions']),
                                 'cached_functional_accuracy':float(np.mean(wanted==labels)),
                                 'actual_minus_cached_accuracy_pp':float(100*(np.mean(pred==labels)-np.mean(wanted==labels))),
                                 'full_test_seconds':timings[method],'first_pass_seconds':timings[method][0],
                                 'all_pass_median_seconds':statistics.median(timings[method]),
                                 'primary_median_seconds':statistics.median(timings[method]),
                                 'warm_median_seconds':statistics.median(timings[method][1:]),
                                 'reference_prediction_disagreements':len(mismatches),
                                 'reference_disagreement_indices':mismatches.tolist(),
                                 'reference_logit_diagnostics':_logit_diagnostics(model,x,masks,mismatches,masked_forward,expected) if len(mismatches) else []}
            if outputs['routed'][1].tolist()!=cpu_masks.tolist():raise AssertionError('Timed GPU routes differ from CPU')
            if outputs['random_cost_matched'][1].tolist()!=random_cpu.tolist():raise AssertionError('Timed random routes differ')
            for method in METHODS:
                details[method]['primary_speedup_vs_dense']=details['dense']['primary_median_seconds']/details[method]['primary_median_seconds']
                details[method]['warm_speedup_vs_dense']=details['dense']['warm_median_seconds']/details[method]['warm_median_seconds']
            result['models'][key]={'recipe':recipe,'seed':seed,'endpoint':'selected','checkpoint_sha256':_sha_file(checkpoint),
                                    'reference_npz_sha256':_sha_file(reference),'policy_sha256':hashlib.sha256(json.dumps(serialized,sort_keys=True).encode()).hexdigest(),
                                    'setup_and_verification_seconds':setup_seconds,'method_orders':orders,'methods':details,
                                    'correctness':{'all_kept_matches_dense':True,'gpu_features_equal_cpu':True,'gpu_routes_equal_cpu_all_examples':True,
                                                   'predictions_repeat_across_all_passes':True,'random_exact_mask_multiset':True,
                                                   'reference_predictions_exact':reference_pass},
                                    'reference_warning':None if reference_pass else 'Different GEMM batch shapes may change FP32/TF32 predictions. Exact parity failed; report actual measured accuracy and mismatches, and do not silently assert NPZ equivalence.'}
            result['all_reference_predictions_exact'] &= bool(reference_pass)
            result['total_run_seconds']=time.perf_counter()-started
            (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            if progress_commit:progress_commit()
            print(json.dumps({'run_id':spec['run_id'],'model':key,'reference_predictions_exact':reference_pass,
                              'warm_seconds':{m:details[m]['warm_median_seconds'] for m in METHODS}}),flush=True)
            del model,router,gpu_features,random_gpu
    if sources!={str(path.relative_to(Path(__file__).parents[1])):_sha_file(path) for path in source_paths}:
        raise RuntimeError('Source changed during routing benchmark')
    result['total_run_seconds']=time.perf_counter()-started
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
