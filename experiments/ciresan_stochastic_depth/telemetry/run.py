"""Bounded telemetry runner dispatch and paired instrumentation qualification."""
from pathlib import Path
import gc,hashlib,json,time
import torch


def hashes():
    base=Path(__file__).resolve().parents[1]
    names=['model_data.py','optimization/stochastic_graph.py','optimization/kernels.py','telemetry/controlled_train.py','telemetry/baseline_train.py','telemetry/recorder.py','telemetry/metrics.py','telemetry/run.py']
    return {n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}


def execute(spec,root,progress_commit=None):
    for n,d in spec.get('source_sha256',{}).items():
        if hashes().get(n)!=d:raise AssertionError('Frozen source mismatch '+n)
    if spec['runner']=='baseline':
        from .baseline_train import run
    elif spec['runner']=='controlled':
        from .controlled_train import run
    else:raise ValueError('Unknown training runner')
    result=run(spec,root,progress_commit=progress_commit)
    result['telemetry_source_sha256']=hashes()
    (Path(root)/'runs'/spec['run_id']/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    if progress_commit:progress_commit()
    return result


def qualification(spec,root,progress_commit=None):
    base=spec['base_specs'];results=[];start=time.perf_counter()
    for index,s in enumerate(base):
        print('QUALIFY',s['run_id'],flush=True)
        results.append(execute(s,root,progress_commit))
        gc.collect();torch.cuda.empty_cache()
    pairs=[]
    for name in sorted(set(s['qualification_pair'] for s in base)):
        rows=[r for r in results if r['spec']['qualification_pair']==name]
        on=next(r for r in rows if r['spec']['telemetry']);off=next(r for r in rows if not r['spec']['telemetry'])
        key='trained_final_parameters_sha256' if on.get('trained_final_parameters_sha256') else 'final_parameters_sha256'
        exact=on[key]==off[key]
        assert exact,'Telemetry changed trained parameters: '+name
        assert on['initial_parameters_sha256']==off['initial_parameters_sha256']
        for a,b in zip(on['history'],off['history']):
            for k in ['validation','test','dense_train_probe']:
                if k in a:assert a[k]==b[k],f'Telemetry changed {name} {k}'
        pairs.append({'name':name,'parameters_bitwise_equal':exact,'on_training_seconds':on['training_seconds'],'off_training_seconds':off['training_seconds'],
                      'on_total_seconds':on['total_run_seconds'],'off_total_seconds':off['total_run_seconds'],
                      'telemetry_seconds':on['telemetry']['seconds'],'overhead_fraction_of_training':on['telemetry']['seconds']/on['training_seconds'],
                      'paired_order':[r['spec']['telemetry'] for r in rows]})
    result={'run_id':spec['run_id'],'spec':spec,'pairs':pairs,'runs':results,'passed':True,'total_run_seconds':time.perf_counter()-start}
    out=Path(root)/'runs'/spec['run_id'];out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    if progress_commit:progress_commit()
    return result


def run(spec,root,progress_commit=None):
    if spec['runner']=='qualification':return qualification(spec,root,progress_commit)
    if spec['runner']=='curvature':
        from .offline_curvature import run as curvature
        return curvature(spec,root,progress_commit)
    return execute(spec,root,progress_commit)
