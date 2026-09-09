"""Export the30-state interactive report AFTER the policy freeze/readiness gate.

No fitting, training, paid calls, or automatic outcome reads at import. Running
requires explicit --ready and a matching policy-manifest.json/.sha256 pair.
Pure build_model() permits synthetic unit tests before outcomes are unblinded.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

RECIPES=('residual','sd_constant','sd_annealed','residual_unit_dropout','plain')
STATES=('selected','final')
SEEDS=(101,102,103)
TWO_MASKS=tuple(m for m in range(16) if m.bit_count()==2)
CATEGORIES=('preserved','mixed','lost')
BRANCH_MACS=(5000000,3000000,1500000,500000)
MASK_MACS=[1965000+sum(c for i,c in enumerate(BRANCH_MACS) if m & (1<<i)) for m in range(16)]
MASK_COSTS=[c/MASK_MACS[15] for c in MASK_MACS]
METHOD_LABELS={'residual':'Residual dense','sd_constant':'Constant stochastic depth',
               'sd_annealed':'Decreasing stochastic depth','residual_unit_dropout':'Residual unit dropout',
               'plain':'Plain MLP (forced crop surgery)'}
HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=HERE.parent/'results'/'hypotheses'
REPO=HERE.parents[2]
DEFAULT_OUTPUT=REPO/'docs'/'ciresan-stochastic-depth'/'hypotheses'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def reject(value):raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def _validate_panel(panel,labels):
    labels=np.asarray(labels);pred=np.asarray(panel['pred']);ce=np.asarray(panel['ce']);n=len(labels)
    if labels.shape!=(n,) or not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>9)):
        raise ValueError('Invalid digit labels')
    if pred.shape!=(16,n) or not np.issubdtype(pred.dtype,np.integer) or np.any((pred<0)|(pred>9)):
        raise ValueError('Need all16 integer mask predictions for every example')
    if ce.shape!=(16,n) or not np.isfinite(ce).all() or np.any(ce<0):raise ValueError('Invalid per-example CE')
    margin=np.asarray(panel['dense_margin']);energy=np.asarray(panel['energy_by_degree']);higher=np.asarray(panel['higher_fraction'])
    if margin.shape!=(n,) or not np.isfinite(margin).all() or np.any(margin<0):raise ValueError('Invalid dense margins')
    if energy.shape!=(5,n) or not np.isfinite(energy).all() or np.any(energy<0):raise ValueError('Invalid degree energies')
    if higher.shape!=(n,) or np.isinf(higher).any() or np.any((higher[np.isfinite(higher)]<0)|(higher[np.isfinite(higher)]>1)):
        raise ValueError('Invalid higher-order fractions')
    if 'labels' in panel:np.testing.assert_array_equal(panel['labels'],labels)
    if n==0:raise ValueError('Empty test panel')
    return pred,ce,margin,energy,higher


def representative_indices(predictions,labels):
    """At most2 lowest original indices per digit/full-correct resilience class."""
    pred=np.asarray(predictions);labels=np.asarray(labels)
    correct=pred==labels[None,:];full=correct[15];counts=correct[list(TWO_MASKS)].sum(axis=0)
    selected=[];available=[]
    for digit in range(10):
        row={'digit':digit}
        for category in CATEGORIES:
            state=(counts==6) if category=='preserved' else (counts==0) if category=='lost' else ((counts>0)&(counts<6))
            qualifying=np.flatnonzero((labels==digit)&full&state)
            row[category]=int(len(qualifying))
            selected.extend((int(index),category,int(counts[index])) for index in qualifying[:2])
        available.append(row)
    return selected,available


def build_model(record,panel,labels):
    """Pure numerical schema builder; accepts synthetic panels of any N>0."""
    pred,ce,margin,energy,higher=_validate_panel(panel,labels)
    labels=np.asarray(labels);n=len(labels);correct=pred==labels[None,:];dense=correct[15]
    recipe=record['recipe'];state=record['state'];seed=int(record['seed'])
    if recipe not in RECIPES or state not in STATES or seed not in SEEDS:raise ValueError('Unknown model identity')
    masks=[]
    for mask in range(16):
        harm=dense & ~correct[mask];repair=~dense & correct[mask];classes=[]
        for digit in range(10):
            take=labels==digit;count=int(take.sum())
            classes.append({'digit':digit,'n':count,'dense_accuracy':float(dense[take].mean()) if count else None,
                            'accuracy':float(correct[mask,take].mean()) if count else None,
                            'harm':float(harm[take].mean()) if count else None,'repair':float(repair[take].mean()) if count else None,
                            'harm_count':int(harm[take].sum()),'repair_count':int(repair[take].sum()),
                            'ce':float(ce[mask,take].mean()) if count else None})
        masks.append({'id':mask,'retained':mask.bit_count(),'cost':MASK_COSTS[mask],'macs':MASK_MACS[mask],'raw_macs':MASK_MACS[mask],
                      'accuracy':float(correct[mask].mean()),'ce':float(ce[mask].mean()),
                      'harm':float(harm.mean()),'repair':float(repair.mean()),
                      'harm_given_dense_correct':float(harm.sum()/dense.sum()) if dense.any() else None,
                      'per_class':classes})
    by_k=[]
    for k in range(5):
        ids=[mask for mask in range(16) if mask.bit_count()==k]
        by_k.append({'k':k,'mask_count':len(ids),'accuracy':float(correct[ids].mean()),'ce':float(ce[ids].mean()),
                     'harm':float((dense & ~correct[ids]).mean()),'repair':float((~dense & correct[ids]).mean()),
                     'cost':float(np.mean([MASK_COSTS[mask] for mask in ids]))})
    chosen,available=representative_indices(pred,labels)
    examples=[{'index':index,'label':int(labels[index]),'category':category,'pred':pred[:,index].tolist(),
               'ce':ce[:,index].tolist(),'dense_margin':float(margin[index]),'retained_two_correct_count':count}
              for index,category,count in chosen]
    finite=higher[np.isfinite(higher)]
    return {'id':f'{recipe}-{state}-s{seed}','recipe':recipe,'state':state,'seed':seed,'epoch':int(record['epoch']),
            'n':n,'dense_accuracy':float(dense.mean()),'dense_ce':float(ce[15].mean()),'dense_correct_count':int(dense.sum()),
            'masks':masks,'by_k':by_k,'higher_fraction':float(finite.mean()) if len(finite) else None,
            'higher_fraction_undefined_count':int(n-len(finite)),'energy_degrees':energy.mean(axis=1).tolist(),
            'examples':examples,'representative_category_counts':available,
            'intervention':record.get('intervention','forced crop surgery on plain MLP' if recipe=='plain' else 'unscaled residual branch deletion'),
            'checkpoint_sha256':record.get('checkpoint_sha256'),
            'example_selection':'Full-correct only; first2 original test indices per digit/category, based on correctness over all6 masks retaining2 branches; empty categories remain empty'}


def policy_gate(root,ready=False):
    """Run before opening any outcome file. Readiness is caller-authorized."""
    if not ready:raise RuntimeError('Export withheld: explicit readiness after the policy freeze is required')
    root=Path(root);path=root/'policy-manifest.json';checksum=root/'policy-manifest.sha256'
    if not path.is_file() or not checksum.is_file() or checksum.read_text().strip()!=sha(path):
        raise RuntimeError('A matching frozen policy manifest/checksum is required before outcome access')
    manifest=read_json(path)
    expected={f'{seed}/{recipe}-{state}' for seed in SEEDS for recipe in RECIPES for state in STATES}
    if set(manifest.get('policies',{}))!=expected or manifest.get('test_arrays_opened_during_fit') is not False:
        raise ValueError('Policy manifest must freeze all30 states without test-array fitting access')
    return path,sha(path)


def _write_atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(text);temporary.replace(path)


def export_report(root=DEFAULT_ROOT,output=DEFAULT_OUTPUT,*,ready=False,analysis_path=None,routing_path=None):
    root=Path(root);output=Path(output)
    manifest_path,manifest_sha=policy_gate(root,ready)
    models=[];images={};sources={};shared_labels=None;shared_pixels=None
    def register(path):sources[str(Path(path).relative_to(root))]=sha(path)
    register(manifest_path)
    for seed in SEEDS:
        directory=root/f'hypothesis-audit-s{seed}-v1';data_path=directory/'data.npz'
        result_path=directory/'result.json';job=read_json(result_path);register(result_path)
        if job.get('run_id')!=directory.name or len(job.get('records',[]))!=10:raise ValueError('Incomplete audit job')
        if job.get('artifacts',{}).get(data_path.name,{}).get('sha256')!=sha(data_path):
            raise ValueError('Audit test image/label artifact checksum mismatch')
        with np.load(data_path,allow_pickle=False) as data:
            labels=data['test_labels'].copy();pixels=data['test_images'].copy()
        if labels.shape!=(10000,) or pixels.shape!=(10000,28,28) or pixels.dtype!=np.uint8:
            raise ValueError('Expected official10000 raw28x28 uint8 test images')
        if shared_labels is None:shared_labels=labels;shared_pixels=pixels
        else:
            np.testing.assert_array_equal(labels,shared_labels);np.testing.assert_array_equal(pixels,shared_pixels)
        register(data_path)
        for recipe in RECIPES:
            for state in STATES:
                record_path=directory/f'{recipe}-{state}.json';panel_path=directory/f'{recipe}-{state}-test.npz'
                record=read_json(record_path)
                if (record.get('recipe'),record.get('state'),record.get('seed'))!=(recipe,state,seed):raise ValueError('Record identity mismatch')
                if not record.get('direct_forward_parity',{}).get('bitwise_equal') or not record.get('test_parity',{}).get('passed'):
                    raise ValueError('Source/dense prediction parity did not pass')
                with np.load(panel_path,allow_pickle=False) as stored:
                    panel={key:stored[key].copy() for key in ('pred','ce','dense_margin','energy_by_degree','higher_fraction','labels')}
                model=build_model(record,panel,labels)
                if abs(model['dense_accuracy']-record['test_parity']['accuracy'])>1e-12:raise ValueError('Dense accuracy differs from verified state record')
                if abs(model['dense_ce']-record['test_parity']['loss'])>1e-10:raise ValueError('Dense CE differs from verified state record')
                for path in (record_path,panel_path):
                    register(path)
                    expected=job.get('artifacts',{}).get(path.name,{}).get('sha256')
                    if expected is None or sha(path)!=expected:raise ValueError('Audit artifact checksum mismatch: '+path.name)
                models.append(model)
                for example in model['examples']:
                    key=str(example['index'])
                    encoded=base64.b64encode(pixels[example['index']].tobytes()).decode('ascii')
                    if key in images and images[key]!=encoded:raise ValueError('Test image index collision')
                    images[key]=encoded
    if len(models)!=30 or len({m['id'] for m in models})!=30:raise ValueError('Incomplete/duplicate state export')
    payload={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'models':models,'images':images,
             'mask_costs':MASK_COSTS,'mask_macs':MASK_MACS,'methodslabels':METHOD_LABELS,'method_labels':METHOD_LABELS,
             'units':{'accuracy':'fraction','harm':'unconditional rate of full-correct to mask-wrong flips, per-example or within digit',
                      'repair':'unconditional rate of full-wrong to mask-correct flips, per-example or within digit',
                      'ce':'nats per example','cost':'fraction of11965000 dense affine MACs','energy_degrees':'mean per-example centered-logit-delta Walsh energy, degrees0..4',
                      'images':'base64 raw784 uint8 grayscale pixels in row-major28x28 order'},
             'primary_panel':{'state':'selected','recipes':['residual','sd_constant','sd_annealed'],'masks':list(TWO_MASKS)},
             'provenance':{'policy_manifest_sha256':manifest_sha,'source_sha256':sha(__file__),'raw_file_sha256':sources},
             'limitations':['All16 masks are exact functional interventions; masks/examples are not independent seed replicates.',
                            'Final checkpoints, unit dropout and plain surgery are secondary; plain deletion is untrained crop surgery.',
                            'Representative examples follow a frozen lowest-index rule and are illustrations, not a random sample.',
                            'The dataset and trained checkpoints were used in earlier work; the investigation is exploratory.']}
    for name,path in [('analysis',analysis_path),('routing',routing_path)]:
        if path is not None:
            payload[name]=read_json(path);payload['provenance'][name+'_sha256']=sha(path)
    # Detect unexpected changes to the policy or raw bytes during export.
    if sha(manifest_path)!=manifest_sha:raise ValueError('Policy manifest changed during export')
    for relative,digest in sources.items():
        if sha(root/relative)!=digest:raise ValueError('Raw source changed during export: '+relative)
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False)
    _write_atomic(output/'data.json',encoded+'\n')
    script=encoded.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    _write_atomic(output/'data.js','window.HYPOTHESIS_DATA = '+script+';\n')
    return {'models':len(models),'representative_images':len(images),'representative_examples':sum(len(m['examples']) for m in models),
            'json_bytes':len(encoded.encode()),'output':str(output),'policy_manifest_sha256':manifest_sha}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--ready',action='store_true',help='Explicit authorization after the policy freeze to read/export outcomes')
    parser.add_argument('--analysis-json',type=Path)
    parser.add_argument('--routing-json',type=Path)
    args=parser.parse_args()
    print(json.dumps(export_report(args.root,args.output,ready=args.ready,analysis_path=args.analysis_json,routing_path=args.routing_json)))


if __name__=='__main__':main()
