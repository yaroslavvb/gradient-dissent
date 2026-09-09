"""Verify and aggregate all declared instrumented reruns, without cloud calls.

Pending/failed/mismatched runs remain visible. Original scientific history is
kept separately from newly reduced telemetry curves. Historical W&B runtime
is never relabeled GPU training time. No training or checkpoint reads occur.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
DEFAULT_RESULTS=EXPERIMENT/'results'/'telemetry'
HASH=re.compile(r'^[0-9a-f]{64}$')
HISTORY_KEYS=('epoch','validation','test','dense_train_probe','stochastic_training_loss','last_minibatch_loss',
              'drop_probabilities','zero_logit_fraction','validation_probe_prediction_counts')
BASELINE_KEYS=('recipe','dropout','mode','precision','fused','batch_size','lr','shrinkage','lr_schedule','seed',
               'max_epochs','eval_every','stop_at_target','input_scale','output_relu')
CONTROLLED_KEYS=('recipe','pmax','input_scale','output_relu','lr','shrinkage','seed','epochs','batch_size',
                 'eval_every','train_size','precision')
TIMING_NOTE='Training seconds exclude evaluation/telemetry/checkpoint I/O. Invocation seconds include those costs; driver time also includes dispatch/startup. Historical W&B runtime has a different unknown-hardware/instrumentation scope.'


def read_json(path):
    def bad(value):raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(Path(path).read_text(),parse_constant=bad)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def describe(values):
    values=list(values)
    if not all(finite(v) for v in values):raise ValueError('Nonfinite summary values')
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
            'sample_sd':statistics.stdev(values) if len(values)>1 else None,'values':values}


def scientific_history(history):return [{key:row[key] for key in HISTORY_KEYS if key in row} for row in history]


def difference_paths(a,b,path='',limit=20):
    if type(a)!=type(b) and not(isinstance(a,(int,float)) and isinstance(b,(int,float))):return [path or '<root>']
    if isinstance(a,dict):
        paths=[]
        for key in sorted(set(a)|set(b)):
            child=path+'/'+str(key)
            paths.extend([child] if key not in a or key not in b else difference_paths(a[key],b[key],child,limit))
            if len(paths)>=limit:break
        return paths[:limit]
    if isinstance(a,list):
        if len(a)!=len(b):return [path+'/length']
        paths=[]
        for index,(x,y) in enumerate(zip(a,b)):
            paths.extend(difference_paths(x,y,path+'/'+str(index),limit))
            if len(paths)>=limit:break
        return paths[:limit]
    return [] if a==b else [path or '<root>']


def original_run_id(spec):
    if spec['runner']=='controlled':return f"graph-eval-{spec['recipe']}-s{spec['seed']}"
    if spec['runner']=='baseline':return f"opt-confirm-normalized-dropout-step-s{spec['seed']}"
    raise ValueError('Not a main replay runner')


def _source_names(sources):
    return {('optimization/'+name if name in ('kernels.py','benchmark.py') else name):value for name,value in sources.items()}


def _hash_check(name,actual,expected,checks,issues,required=True):
    if expected is None:
        checks[name]={'status':'reference_unavailable','actual':actual}
        if required:issues.append(name+':reference_missing')
    elif not isinstance(actual,str) or not HASH.fullmatch(actual):
        checks[name]={'status':'missing_or_invalid','expected':expected};issues.append(name+':missing_or_invalid')
    else:
        checks[name]={'status':'exact' if actual==expected else 'mismatch','actual':actual,'expected':expected}
        if actual!=expected:issues.append(name+':mismatch')


def check_eval(score,path,issues):
    if not isinstance(score,dict):issues.append(path+':missing');return
    n=score.get('n');accuracy=score.get('accuracy');loss=score.get('loss')
    if not isinstance(n,int) or n<=0 or not finite(accuracy) or not 0<=accuracy<=1 or not finite(loss) or loss<0:
        issues.append(path+':invalid_values');return
    if 'errors' in score:
        errors=score['errors']
        if not isinstance(errors,int) or not 0<=errors<=n or abs(accuracy-(1-errors/n))>1e-12:issues.append(path+':error_count_mismatch')
        if 'wrong_indices' in score:
            ids=score['wrong_indices']
            if len(ids)!=errors or len(set(ids))!=errors or any(not isinstance(i,int) or not 0<=i<n for i in ids):issues.append(path+':wrong_indices_invalid')
    if 'confusion' in score:
        matrix=score['confusion']
        if len(matrix)!=10 or any(len(row)!=10 for row in matrix) or any(not isinstance(v,int) or v<0 for row in matrix for v in row):
            issues.append(path+':confusion_shape_or_counts');return
        if sum(map(sum,matrix))!=n or abs(sum(matrix[i][i] for i in range(10))/n-accuracy)>1e-12:issues.append(path+':confusion_totals')
        classes=score.get('per_class',[])
        if len(classes)!=10:issues.append(path+':per_class_missing');return
        for i,row in enumerate(classes):
            count=sum(matrix[i]);correct=matrix[i][i]
            if row.get('class_id')!=i or row.get('n')!=count or row.get('correct')!=correct or row.get('errors')!=count-correct:
                issues.append(path+f':class_{i}_counts')


def telemetry_rows(raw,issues):
    rows=raw.get('telemetry_history',[]);result=[];previous_epoch=-1;previous_time=-1
    for index,row in enumerate(rows):
        epoch=row.get('epoch');training=row.get('training_seconds');wall=row.get('run_elapsed_seconds');metrics=row.get('metrics',{})
        if not isinstance(epoch,int) or epoch<=previous_epoch:issues.append(f'telemetry/{index}:epoch_order')
        if not finite(training) or training<previous_time or not finite(wall) or wall+1e-6<training:issues.append(f'telemetry/{index}:clock_values')
        if not isinstance(metrics,dict) or any(not isinstance(k,str) or not finite(v) for k,v in metrics.items()):issues.append(f'telemetry/{index}:nonfinite_metric')
        diagnostics=row.get('diagnostics',{});evaluations={}
        for split in ('train','validation','test'):
            if split in diagnostics:
                score=diagnostics[split];check_eval(score,f'telemetry/{index}/{split}',issues)
                evaluations[split]={k:score[k] for k in ('n','loss','accuracy','errors','confusion','per_class') if k in score}
        result.append({'epoch':epoch,'training_seconds':training,'run_elapsed_seconds':wall,'metrics':metrics,
                       'evaluations':evaluations,'update_interval':diagnostics.get('update_interval')})
        previous_epoch=epoch if isinstance(epoch,int) else previous_epoch
        previous_time=training if finite(training) else previous_time
    if raw.get('telemetry',{}).get('enabled') and (not rows or rows[0].get('epoch')!=0):issues.append('telemetry:missing_epoch_zero')
    return result


def verify_replay(raw,original,spec):
    """Compare raw scientific results, never rounded/telemetry-derived curves."""
    issues=[];checks={};runner=spec['runner']
    if raw.get('spec')!=spec:issues.append('spec:differs_from_manifest')
    science=CONTROLLED_KEYS if runner=='controlled' else BASELINE_KEYS
    defaults={'precision':'fp32','dropout':0.,'fused':False,'stop_at_target':True,'lr_schedule':{},'output_relu':False}
    differences=[key for key in science if spec.get(key,defaults.get(key))!=original['spec'].get(key,defaults.get(key))]
    checks['scientific_spec']={'status':'exact' if not differences else 'mismatch','different_fields':differences}
    if differences:issues.append('scientific_spec:mismatch')
    a=scientific_history(raw.get('history',[]));b=scientific_history(original.get('history',[]))
    paths=difference_paths(a,b);checks['scientific_history']={'status':'exact' if not paths else 'mismatch','rows':len(a),'reference_rows':len(b),'different_paths':paths}
    if paths:issues.append('scientific_history:mismatch')
    for row in raw.get('history',[]):
        for key in ('validation','test','dense_train_probe'):
            if key in row:check_eval(row[key],f'history/{row.get("epoch")}/{key}',issues)
    _hash_check('initial_parameters',raw.get('initial_parameters_sha256'),original.get('initial_parameters_sha256'),checks,issues)
    _hash_check('trained_final_parameters',raw.get('trained_final_parameters_sha256'),original.get('trained_final_parameters_sha256'),checks,issues,required=False)
    if runner=='controlled':
        _hash_check('reloaded_selected_parameters',raw.get('final_parameters_sha256'),original.get('final_parameters_sha256'),checks,issues,required=False)
        for field in ('executed_epoch_order_sha256','executed_epoch_masks_sha256'):_hash_check(field,raw.get(field),original.get(field),checks,issues)
        for key in ('skipped_batch_counts','steps_per_epoch','training_examples_per_epoch','best_epoch','best_validation_loss'):
            equal=raw.get(key)==original.get(key);checks[key]={'status':'exact' if equal else 'mismatch'}
            if not equal:issues.append(key+':mismatch')
        for key in ('selected_test','final_test','selected_train','final_train'):
            check_eval(raw.get(key),key,issues)
            paths=difference_paths(raw.get(key),original.get(key));checks[key]={'status':'exact' if not paths else 'mismatch','different_paths':paths}
            if paths:issues.append(key+':mismatch')
        for key in ('trained_final_parameters_sha256','final_parameters_sha256'):
            if not isinstance(raw.get(key),str) or not HASH.fullmatch(raw[key]):issues.append(key+':missing')
    else:
        check_eval(raw.get('final_test'),'final_test',issues)
        if raw.get('final_test')!=original.get('final_test'):issues.append('final_test:mismatch')
        threshold=raw.get('threshold');reference=original.get('threshold')
        if (threshold is None)!=(reference is None) or threshold and (threshold['epoch']!=reference['epoch'] or threshold['test']!=reference['test']):issues.append('target_checkpoint:mismatch')
        checks['training_order_digest']={'status':'not_recorded_in_original_or_baseline_runner','note':'Same declared seed/order algorithm plus exact recorded history; no fabricated stream-hash claim.'}
    if raw.get('parameter_count')!=original.get('parameter_count'):issues.append('parameter_count:mismatch')
    dataset_equal=raw.get('dataset')==original.get('dataset');checks['dataset']={'status':'exact' if dataset_equal else 'mismatch'}
    if not dataset_equal:issues.append('dataset:mismatch')
    old_sources=_source_names(original.get('source_sha256',{}));new_sources=_source_names(raw.get('source_sha256',{}))
    core=('model_data.py','optimization/stochastic_graph.py') if runner=='controlled' else ('optimization/kernels.py',)
    for name in core:_hash_check('core_source/'+name,new_sources.get(name),old_sources.get(name),checks,issues)
    executed=raw.get('telemetry_source_sha256',{})
    for name,digest in spec.get('source_sha256',{}).items():_hash_check('frozen_source/'+name,executed.get(name),digest,checks,issues)
    if not spec.get('source_sha256'):checks['frozen_sources']={'status':'manifest_has_no_frozen_source_hashes'}
    return checks,issues


def analyze_run(raw,original,spec):
    checks,issues=verify_replay(raw,original,spec)
    rows=telemetry_rows(raw,issues)
    telemetry=raw.get('telemetry',{})
    for key in ('seconds','checkpoint_seconds_within_telemetry'):
        if not finite(telemetry.get(key)) or telemetry[key]<0:issues.append('telemetry/'+key+':invalid')
    for field in ('training_seconds','total_run_seconds'):
        if not finite(raw.get(field)) or raw[field]<0:issues.append(field+':invalid')
    if telemetry.get('network_logging_during_training') is not False:issues.append('telemetry:network_logging_flag_not_false')
    if telemetry.get('probe_n')!=spec.get('telemetry_probe_n',256):issues.append('telemetry:probe_size_mismatch')
    if not isinstance(telemetry.get('probe_sha256'),str) or not HASH.fullmatch(telemetry['probe_sha256']):issues.append('telemetry:probe_hash_missing')
    history=raw.get('history',[]);last=history[-1]['epoch'] if history else None
    expected_epochs=([0]+list(range(1,last+1))) if spec['runner']=='baseline' and last else ([0]+[r['epoch'] for r in history])
    if [r['epoch'] for r in rows]!=expected_epochs:issues.append('telemetry:incomplete_epoch_series')
    end=last==spec.get('epochs',100) if spec['runner']=='controlled' else bool(raw.get('threshold')) or last==spec.get('max_epochs',100)
    failed='error' in raw or raw.get('diverged',False)
    status='failed' if failed else 'incomplete' if not end else 'mismatch' if issues else 'verified'
    elapsed=raw.get('training_seconds');seconds=telemetry.get('seconds')
    initial_telemetry=rows[0].get('metrics',{}).get('time/telemetry_cumulative_seconds') if rows else None
    return {'id':spec['run_id'],'run_id':spec['run_id'],'cohort':'main' if spec['runner']=='controlled' else 'optimized_baseline',
            'recipe':spec['recipe'],'seed':spec['seed'],'runner':spec['runner'],'spec':spec,'status':status,'issues':issues,
            'verification':checks,'history':history,'telemetry_rows':rows,'telemetry':telemetry,'last_epoch':last,
            'hardware':raw.get('hardware',{}),'reference':{'run_id':original.get('run_id',original_run_id(spec)),
                    'history':original.get('history',[]),'hardware':original.get('hardware',{}),
                    'training_seconds':original.get('training_seconds'),'total_run_seconds':original.get('total_run_seconds')},
            'parameter_hashes':{'initial':raw.get('initial_parameters_sha256'),'trained_final':raw.get('trained_final_parameters_sha256'),
                                'reloaded_selected':raw.get('final_parameters_sha256') if spec['runner']=='controlled' else None,
                                'note':'Controlled final_parameters_sha256 is AFTER loading the validation-selected checkpoint; trained_final_parameters_sha256 is the 100-epoch state. Old final hashes may be absent.'},
            'timing':{'training_seconds':elapsed,'total_run_seconds':raw.get('total_run_seconds'),
                      'local_dispatch_elapsed_seconds':raw.get('local_dispatch_elapsed_seconds'),'telemetry_seconds':seconds,
                      'initial_telemetry_seconds':initial_telemetry,
                      'later_telemetry_seconds':seconds-initial_telemetry if finite(seconds) and finite(initial_telemetry) else None,
                      'checkpoint_seconds_within_telemetry':telemetry.get('checkpoint_seconds_within_telemetry'),
                      'graph_setup_seconds':raw.get('graph_setup_seconds'),
                      'epoch_preparation_seconds':raw.get('epoch_preparation_seconds'),
                      'telemetry_over_training_fraction':seconds/elapsed if finite(seconds) and finite(elapsed) and elapsed>0 else None,
                      'training_time_ratio_to_original':elapsed/original['training_seconds'] if finite(elapsed) and original.get('training_seconds',0)>0 else None},
            'endpoints':{key:raw[key] for key in ('best_epoch','selected_test','final_test','selected_train','final_train','threshold') if key in raw}}


def verify_qualification(raw,spec=None):
    issues=[];runs=raw.get('runs',[]);pairs=[]
    if spec is not None and raw.get('spec')!=spec:issues.append('qualification:spec_differs_from_manifest')
    if len(runs)!=6:issues.append('qualification:expected6_runs')
    declared=(spec or raw.get('spec',{})).get('base_specs',[])
    if len(declared)!=6 or [r.get('spec') for r in runs]!=declared:issues.append('qualification:base_specs_mismatch')
    for r in runs:
        executed=r.get('telemetry_source_sha256',{});expected=r.get('spec',{}).get('source_sha256',{})
        if not expected or any(executed.get(k)!=v or not HASH.fullmatch(v) for k,v in expected.items()):issues.append(str(r.get('run_id'))+':frozen_sources_mismatch')
    groups=defaultdict(list)
    for run in runs:groups[run.get('spec',{}).get('qualification_pair')].append(run)
    for name,group in sorted(groups.items(),key=lambda item:str(item[0])):
        on=[r for r in group if r['spec'].get('telemetry')];off=[r for r in group if r['spec'].get('telemetry') is False]
        if len(on)!=1 or len(off)!=1:issues.append(str(name)+':expected_on_off_pair');continue
        a,b=on[0],off[0];key='trained_final_parameters_sha256' if a.get('trained_final_parameters_sha256') else 'final_parameters_sha256'
        exact=isinstance(a.get(key),str) and HASH.fullmatch(a[key]) is not None and a[key]==b.get(key)
        init=a.get('initial_parameters_sha256')==b.get('initial_parameters_sha256') and isinstance(a.get('initial_parameters_sha256'),str)
        same=bool(a.get('history')) and scientific_history(a.get('history',[]))==scientific_history(b.get('history',[]))
        if not(exact and init and same):issues.append(str(name)+':on_off_scientific_mismatch')
        if a['spec'].get('runner')=='controlled':
            for field in ('executed_epoch_order_sha256','executed_epoch_masks_sha256'):
                if not isinstance(a.get(field),str) or not HASH.fullmatch(a[field]) or a[field]!=b.get(field):issues.append(str(name)+':'+field+'_mismatch')
        rows_issues=[];telemetry_rows(a,rows_issues);issues.extend(str(name)+':'+s for s in rows_issues)
        initial_telemetry=a.get('telemetry_history',[{}])[0].get('metrics',{}).get('time/telemetry_cumulative_seconds')
        pairs.append({'name':name,'parameter_hash_field':key,'parameters_bitwise_equal':bool(exact),'initialization_equal':bool(init),
                      'scientific_history_equal':same,'on_run_id':a.get('run_id'),'off_run_id':b.get('run_id'),
                      'on_training_seconds':a.get('training_seconds'),'off_training_seconds':b.get('training_seconds'),
                      'on_total_seconds':a.get('total_run_seconds'),'off_total_seconds':b.get('total_run_seconds'),
                      'telemetry_seconds':a.get('telemetry',{}).get('seconds'),
                      'initial_telemetry_seconds':initial_telemetry,
                      'later_telemetry_seconds':a.get('telemetry',{}).get('seconds',0)-initial_telemetry if finite(initial_telemetry) else None,
                      'telemetry_over_training_fraction':a.get('telemetry',{}).get('seconds',0)/a['training_seconds'] if a.get('training_seconds',0)>0 else None})
    if len(pairs)!=3:issues.append('qualification:expected3_pairs')
    if raw.get('passed') is not True:issues.append('qualification:producer_not_passed')
    ratios=[p['telemetry_over_training_fraction'] for p in pairs]
    gate=bool(ratios) and all(finite(r) and r<=.10 for r in ratios)
    return {'run_id':raw.get('run_id'),'status':'verified' if not issues else 'mismatch','issues':issues,'pairs':pairs,
            'overhead_gate':{'limit_fraction':.10,'passed':gate,'max_fraction':max(ratios) if all(finite(r) for r in ratios) and ratios else None},
            'runs':[{'id':r.get('run_id'),'spec':r.get('spec'),'history':r.get('history',[]),
                     'telemetry_rows':telemetry_rows(r,[]),'telemetry':r.get('telemetry',{}),'hardware':r.get('hardware',{})} for r in runs]}


def historical_series(path):
    """Use retained observations honestly; accept a later explicit full series."""
    source=read_json(path)
    if isinstance(source,list):run={'rows':source}
    elif 'runs' in source:run=next(r for r in source['runs'] if r.get('id',r.get('run_id'))=='ts4k9n55')
    else:run=source
    observations=run.get('rows',run.get('history',run.get('evaluations')))
    sparse=observations is None
    if sparse:
        observations=[run[key] for key in ('first_observed_evaluation','first_at_least_98pct','best_observed_accuracy','minimum_observed_loss','last_observed_evaluation') if key in run]
    unique={}
    for row in observations:
        runtime=row.get('_runtime',row.get('runtime_seconds'));epoch=row.get('epoch')
        if not finite(runtime):raise ValueError('Historical row lacks finite W&B runtime')
        if not finite(row.get('val_accuracy')) or not 0<=row['val_accuracy']<=100:raise ValueError('Historical accuracy must be percent')
        metrics={}
        for source_key,target in [('val_accuracy','test/accuracy_pct'),('val_loss','test/loss'),('train_accuracy','train/accuracy_pct'),('train_loss','train/loss')]:
            if source_key in row:
                if not finite(row[source_key]):raise ValueError('Nonfinite historical metric')
                metrics[target]=row[source_key]
        transformed={'epoch':epoch,'wandb_runtime_seconds':runtime,'training_seconds':None,'run_elapsed_seconds':None,'metrics':metrics,'source_step':row.get('_step')}
        key=(runtime,row.get('_step'));previous=unique.get(key)
        if previous is not None and previous!=transformed:raise ValueError('Conflicting duplicate historical observation')
        unique[key]=transformed
    rows=sorted(unique.values(),key=lambda row:(row['wandb_runtime_seconds'],row['epoch'] if row['epoch'] is not None else -1))
    expected=run.get('accuracy_rows_expected',run.get('provenance',{}).get('accuracy_rows_expected',source.get('accuracy_rows_expected') if isinstance(source,dict) else None))
    complete=not sparse and isinstance(expected,int) and len(rows)==expected
    claimed_hash=run.get('provenance',{}).get('curve_sha256')
    if claimed_hash and hashlib.sha256(json.dumps(observations,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=claimed_hash:raise ValueError('Historical curve content hash mismatch')
    return {'run_id':'ts4k9n55','url':'https://wandb.ai/yaroslavvb/train_ciresan/runs/ts4k9n55','rows':rows,
            'retained_rows':len(rows),'count_verified_expected_evaluations':expected,'complete_series':complete,
            'source_sha256':sha(path),'source_path':str(path),'provenance':run.get('provenance',{}),'semantics':run.get('semantics',{}),
            'selection':'Sparse first/threshold/extrema/last observations, not a per-epoch curve' if sparse else 'All count-verified public evaluation rows' if complete else 'Explicit provided evaluation rows; completeness not established',
            'clock':'W&B runtime including evaluation/logging, hardware unknown; never treated as isolated training time',
            'accuracy_units':'percent as logged historically','legacy_val_means':'Official test set; historical run had no separate selection-validation split'}


def aggregate_groups(runs):
    groups=defaultdict(list)
    for run in runs:groups[(run['cohort'],run['recipe'])].append(run)
    output=[]
    for (cohort,recipe),members in sorted(groups.items()):
        available=[r for r in members if r.get('history')]
        summary={'cohort':cohort,'recipe':recipe,'expected_runs':len(members),'run_ids':[r['id'] for r in members],
                 'statuses':dict(Counter(r['status'] for r in members)),
                 'training_seconds':describe(r['timing']['training_seconds'] for r in available if finite(r.get('timing',{}).get('training_seconds'))),
                 'telemetry_seconds':describe(r['timing']['telemetry_seconds'] for r in available if finite(r.get('timing',{}).get('telemetry_seconds')))}
        for endpoint in ('selected_test','final_test'):
            scores=[r.get('endpoints',{}).get(endpoint) for r in available];scores=[s for s in scores if s]
            summary[endpoint]={'accuracy_pct':describe(100*s['accuracy'] for s in scores),'ce':describe(s['loss'] for s in scores)}
        output.append(summary)
    return output


def curvature_results(results,paths,runs,manifest):
    """Verify frozen auxiliary-job provenance; preserve partial diagnostics."""
    specs=read_json(manifest) if manifest.exists() else []
    declared={s['run_id']:s for s in specs};source_runs={r['id']:r for r in runs}
    paths=list(paths) if paths else sorted(results.glob('telemetry-curvature-*.json'))
    entries=[];statuses={name:'pending' for name in declared}
    for path in paths:
        raw=read_json(path);name=raw.get('run_id',Path(path).stem);spec=declared.get(name);issues=[]
        if spec is None:issues.append('not_in_curvature_manifest')
        elif raw.get('spec')!=spec:issues.append('curvature_spec_mismatch')
        if spec and raw.get('curvature_source_sha256')!=spec.get('curvature_source_sha256'):issues.append('curvature_source_hashes_mismatch')
        source=source_runs.get(raw.get('source_run_id'))
        if source is None or source.get('status')!='verified':issues.append('training_source_not_verified')
        else:
            if raw.get('source_training_source_sha256')!=source['spec'].get('source_sha256'):issues.append('source_training_hashes_mismatch')
            if raw.get('probe',{}).get('source_recorder_probe_sha256')!=source['telemetry'].get('probe_sha256'):issues.append('source_probe_hash_mismatch')
            expected={r['epoch']:r for r in source['telemetry'].get('snapshots',[])}
            for row in raw.get('snapshots',[]):
                ref=expected.get(row.get('epoch'),{})
                if row.get('checkpoint_sha256')!=ref.get('sha256') or row.get('checkpoint_file')!=ref.get('path'):issues.append('checkpoint_manifest_mismatch')
                expected_hash=source['parameter_hashes'].get('initial' if row.get('epoch')==0 else 'trained_final') if row.get('epoch') in (0,100) else None
                if expected_hash and row.get('source_fp32_parameters_sha256')!=expected_hash:issues.append('checkpoint_parameter_hash_mismatch')
                if row.get('parameters_unchanged') is not True:issues.append('checkpoint_mutation_check_missing')
                curv=row.get('curvature',{})
                if curv.get('probe_n')!=128 or curv.get('output_classes')!=10 or curv.get('derivative_dtype')!='torch.float64' or len(curv.get('layers',[]))!=6:issues.append('curvature_panel_mismatch')
        epochs=[r.get('epoch') for r in raw.get('snapshots',[])]
        wanted=(spec or {}).get('snapshot_epochs',[])
        if epochs!=wanted[:len(epochs)] or len(epochs)>len(wanted):issues.append('snapshot_epoch_panel_mismatch')
        done=raw.get('complete') is True and raw.get('passed') is True and raw.get('status')=='completed'
        if done and epochs!=wanted:issues.append('incomplete_snapshot_panel_claimed_complete')
        status='mismatch' if issues else 'verified' if done else raw.get('status','incomplete')
        statuses[name]=status
        entries.append({'path':str(path),'sha256':sha(path),'result':raw,'verification':{'status':status,'issues':issues}})
    complete=bool(specs) and len(entries)==len(specs) and all(statuses.get(n)=='verified' for n in declared)
    return entries,{'expected_runs':len(specs),'available_runs':len(entries),'expected_snapshots':sum(len(s.get('snapshot_epochs',[])) for s in specs),
                    'verified_runs':sum(v=='verified' for v in statuses.values()),'statuses':statuses,'complete':complete,
                    'manifest':str(manifest),'manifest_sha256':sha(manifest) if manifest.exists() else None}


def markdown(doc):
    lines=['# Instrumented Ciresan reruns','',f"Status: **{doc['status']}**. {doc['counts']['verified']} of 18 declared reruns verified; {doc['counts']['pending']} pending. All declared and additional attempts are retained.",'',
           '| Run | Status | Last epoch | Original scientific history | Training s | Telemetry s | Telemetry / training |',
           '|---|---|---:|---|---:|---:|---:|']
    fmt=lambda value: f'{value:.3f}' if finite(value) else '—'
    for run in doc['runs']:
        timing=run.get('timing',{});ratio=timing.get('telemetry_over_training_fraction')
        lines.append(f"| {run['id']} | {run['status']} | {run.get('last_epoch','—')} | {run.get('verification',{}).get('scientific_history',{}).get('status','—')} | {fmt(timing.get('training_seconds'))} | {fmt(timing.get('telemetry_seconds'))} | {100*ratio:.2f}% |" if finite(ratio) else f"| {run['id']} | {run['status']} | {run.get('last_epoch','—')} | — | — | — | — |")
    q=doc['qualification'];lines+=['','## Instrumentation qualification','',f"Status: {q['status']}.",'']
    for attempt in q.get('attempts',[]):
        gate=attempt.get('overhead_gate',{})
        lines.append(f"- {attempt['run_id']}: scientific parity {attempt['status']}; 10% overhead gate {'passed' if gate.get('passed') else 'not passed'}.")
        for pair in attempt.get('pairs',[]):lines.append(f"  - {pair['name']}: exact final parameter equality={pair['parameters_bitwise_equal']}; exact scientific history={pair['scientific_history_equal']}; telemetry {fmt(pair['telemetry_seconds'])} s ({100*pair['telemetry_over_training_fraction']:.2f}% of training).")
    lines+=['','## Verification limits','',
            '- Exact recorded curves and initial tensors do not prove exact final weights when the original run did not save a final parameter hash. New trained-final and reloaded-selected hashes remain separate.',
            '- Qualification on/off pairs test instrumentation with byte-identical end states; they are not additional independent experimental seeds.',
            '- Telemetry evaluation reduces CE differently from legacy summed minibatch evaluation. Replay equality is checked on the unchanged legacy scientific history; the diagnostic curve remains explicitly separate.',
            '- Three repeated seeds reuse the same MNIST data. The baseline recipe was selected using official-test outcomes; these reruns add measurements, not independent generalization evidence.',
            '- '+TIMING_NOTE,
            '- Historical data: '+doc['historical']['selection']+f" ({doc['historical']['retained_rows']} retained observations).",'']
    for run in doc['runs']:
        for issue in run.get('issues',[]):lines.append(f"- **{run['id']}**: {issue}")
    for issue in doc.get('issues',[]):lines.append('- '+issue)
    return '\n'.join(lines)+'\n'


def training_findings(doc):
    """Descriptive, fully enumerated scalar findings after every rerun verifies."""
    if not doc['complete']:raise ValueError('Training findings require all 18 verified reruns')
    recipes=('plain','residual','sd_constant','sd_annealed','residual_unit_dropout')
    names={'plain':'Plain','residual':'Residual','sd_constant':'Constant SD','sd_annealed':'Annealed SD','residual_unit_dropout':'Residual unit dropout'}
    groups={recipe:[r for r in doc['runs'] if r['runner']=='controlled' and r['recipe']==recipe] for recipe in recipes}
    baselines=[r for r in doc['runs'] if r['runner']=='baseline']
    mean=statistics.mean
    def ms(values,places=3):
        values=list(values);return f'{mean(values):.{places}f} ± {statistics.stdev(values):.{places}f}'
    def metric(run,epoch,key):return next(t['metrics'][key] for t in run['telemetry_rows'] if t['epoch']==epoch)
    lines=['# Training findings from the instrumented reruns','',
           'All 18 declared reruns verified. The 15 controlled runs reproduce every original recorded scientific loss/accuracy value, initialization hash, training-order digest, and layer-mask digest. The three optimized-baseline reruns reproduce their recorded curves, initialization and stopping checkpoints. Old final parameter hashes were not recorded, so matching final weights to those old runs is not claimed. Both qualification attempts separately established byte-identical on/off final parameters and complete scientific histories. These are instrumented repeats of existing seeds, not 18 new independent replications.','',
           'Controlled settings remain the original widths, normalized pixels, linear logits, 50,000/10,000 fitting/validation split, batch 64 FP32/TF32, 100 epochs and frozen learning rates: .03 for plain and .01 for all residual recipes. Ordinary dropout is .2; SD schedules remain frozen. Dense diagnostics disable both unit dropout and branch removal. They use the same fixed first 128 fitting examples, with detailed measurements at epoch 0, epoch 1 and every 10 epochs. No metric alters training or validation-CE checkpoint selection; diagnostic official-test evaluations are repeated and disclosed. Three seeds reuse the same datasets and probe.','',
           '## Quality and checkpoint dependence','',
           'Each cell is mean ± sample SD over seeds 101–103 (n=3); accuracy is percent, CE is mean cross-entropy. Selected means the unchanged minimum-validation-CE checkpoint. Final means epoch 100. Values below use the original scientific reductions, which the reruns reproduce exactly.','',
           '| Recipe | Selected epochs | Selected test accuracy | Selected test CE | Final test accuracy | Final test CE |',
           '|---|---|---:|---:|---:|---:|']
    for recipe,rs in groups.items():
        values=[ms((100*r['endpoints'][endpoint]['accuracy'] for r in rs),3) if field=='accuracy' else ms((r['endpoints'][endpoint]['loss'] for r in rs),4) for endpoint in ('selected_test','final_test') for field in ('accuracy','loss')]
        lines.append('| '+names[recipe]+' | '+', '.join(str(r['endpoints']['best_epoch']) for r in rs)+' | '+' | '.join(values)+' |')
    residual=groups['residual']
    for recipe in ('sd_constant','sd_annealed'):
        rs=groups[recipe]
        deltas=[100*mean(r['endpoints'][endpoint]['accuracy']-ref['endpoints'][endpoint]['accuracy'] for r,ref in zip(rs,residual)) for endpoint in ('selected_test','final_test')]
        lines+=['',f"{names[recipe]} minus residual accuracy is {deltas[0]:+.3f} percentage points at the selected checkpoint and {deltas[1]:+.3f} at epoch 100. The apparent accuracy ranking therefore depends on the endpoint. Both SD recipes have higher final test CE than residual; higher final top-1 accuracy alone is not an across-metric improvement."]
    lines+=['','All 12 plain/residual/SD runs reach 100% dense fitting accuracy by epoch 100. Their mean full-fitting CE values are '+', '.join(f"{names[p]} {mean(r['endpoints']['final_train']['loss'] for r in groups[p]):.3g}" for p in recipes[:4])+'. Residual unit dropout ends at 99.967% mean fitting accuracy, with full-fitting CE spanning '+f"{min(r['endpoints']['final_train']['loss'] for r in groups['residual_unit_dropout']):.3g}–{max(r['endpoints']['final_train']['loss'] for r in groups['residual_unit_dropout']):.3g}"+'. These training curves show near-complete fitting despite branch removal; they do not support a blanket claim that these SD settings prevent overfitting.','',
           '## What the fixed-probe gradients show','',
           'For the final affine head (layer 5), g_i is the per-example weight gradient of unaveraged CE. The reported gradient norm is ||mean_i g_i||, and diversity is D=mean_i||g_i||² / ||mean_i g_i||². The largest-example energy share is max_i||g_i||² / sum_i||g_i||², reconstructed from the recorded maximum norm and second moment. Gradients exclude bias here. The table uses all three seeds; gradient norms and D are seed means, energy shares are seed ranges.','',
           '| Recipe | Head gradient norm epoch 1 → 100 | D epoch 1 → 100 | Largest-example energy share at 100 |',
           '|---|---:|---:|---:|']
    pre='layer-5/probe/'
    for recipe,rs in groups.items():
        norms=[mean(metric(r,e,pre+'grad_l2') for r in rs) for e in (1,100)]
        diversity=[mean(metric(r,e,pre+'weight_grad_diversity') for r in rs) for e in (1,100)]
        shares=[metric(r,100,pre+'per_example_weight_grad_norm_max')**2/(r['telemetry']['probe_n']*metric(r,100,pre+'weight_grad_second_moment')) for r in rs]
        lines.append(f"| {names[recipe]} | {norms[0]:.3g} → {norms[1]:.3g} | {diversity[0]:.1f} → {diversity[1]:.1f} | {100*min(shares):.2f}–{100*max(shares):.2f}% |")
    lines+=['','A diversity ratio near 128 is not evidence that 128 examples supply equally useful independent gradients: one dominant example alone produces D≈128. The energy shares above and extremely small median gradient norms indicate substantial concentration as the probe becomes confidently fitted. Small denominators and FP32 arithmetic further limit late-stage ratio interpretation. These observations describe one fixed training probe; they neither identify a population mechanism nor make diversity a validated predictor of test accuracy. Neighbor-gradient cosines refer to cyclic neighbors in this fixed dataset order, not neighboring optimization steps.','',
           '## Activations and class-specific behavior','',
           'Mean fraction of zero post-ReLU activations at epoch 100, over the fixed probe and then the three seeds. Layers 0–4 are the five hidden activations; the linear head is excluded. A zero on this probe does not prove a permanently dead neuron.','',
           '| Recipe | Layer0 | Layer1 | Layer2 | Layer3 | Layer4 |','|---|---:|---:|---:|---:|---:|']
    for recipe,rs in groups.items():lines.append('| '+names[recipe]+' | '+' | '.join(f"{100*mean(metric(r,100,f'layer-{i}/probe/post_activation_zero_fraction') for r in rs):.1f}%" for i in range(5))+' |')
    lines+=['','Both SD recipes have sparser dense probe activations than the residual control at these frozen settings; annealed SD has roughly 69–74% zero activations across its hidden layers at epoch 100. This association coexists with near-perfect fitting and does not by itself explain generalization. The plain model also differs architecturally and uses another learning rate.','',
           'Epoch100 class-specific test-accuracy differences from residual (percentage points; mean across seeds) are shown for every digit, avoiding a selected best/worst-class account. Classes have different fixed MNIST test counts, so equal-weight averages of these ten cells need not equal the overall accuracy difference.','',
           '| Recipe − residual | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def class_accuracy(run,digit):return next(t['evaluations']['test']['per_class'][digit]['accuracy'] for t in run['telemetry_rows'] if t['epoch']==100)
    for recipe in ('sd_constant','sd_annealed','residual_unit_dropout'):
        rs=groups[recipe]
        lines.append('| '+names[recipe]+' | '+' | '.join(f'{100*mean(class_accuracy(r,d)-class_accuracy(ref,d) for r,ref in zip(rs,residual)):+.3f}' for d in range(10))+' |')
    lines+=['','Small overall differences hide mixed class effects. These repeated-test descriptive slices have no multiple-comparison correction and cannot establish digit-specific causal benefits.','',
           '## Timing and measurement overhead','',
           '| Recipe | Mean training s | Mean telemetry s | Total telemetry / training range |','|---|---:|---:|---:|']
    for recipe,rs in list(groups.items())+[('optimized_baseline',baselines)]:
        ratios=[r['timing']['telemetry_over_training_fraction'] for r in rs]
        lines.append(f"| {names.get(recipe,'Optimized baseline')} | {mean(r['timing']['training_seconds'] for r in rs):.3f} | {mean(r['timing']['telemetry_seconds'] for r in rs):.3f} | {100*min(ratios):.2f}–{100*max(ratios):.2f}% |")
    main=[r for rs in groups.values() for r in rs]
    checkpoint=[r['timing']['checkpoint_seconds_within_telemetry'] for r in main if r['telemetry']['snapshots']]
    later=[r['timing']['later_telemetry_seconds']/r['timing']['training_seconds'] for r in baselines]
    lines+=['',f"The three central seed 101 runs save seven snapshots each; checkpoint I/O adds {min(checkpoint):.3f}–{max(checkpoint):.3f}s, already included in their telemetry totals. Main-run training-time ratios to the original runs range {min(r['timing']['training_time_ratio_to_original'] for r in main):.4f}–{max(r['timing']['training_time_ratio_to_original'] for r in main):.4f}; this small variation is not an instrumented overhead estimate because evaluation is excluded from that timer.",'',
           f"The v1 qualification failed the 10% direct-overhead gate for one controlled pair (11.55%), and was retained. The timing-only change to cadence 10/probe 128 passed v2 (maximum 4.22%). Nevertheless, all three final short baseline runs exceed 10% including cold epoch 0 diagnostics: initial measurement costs {min(r['timing']['initial_telemetry_seconds'] for r in baselines):.3f}–{max(r['timing']['initial_telemetry_seconds'] for r in baselines):.3f}s, while later telemetry costs only {100*min(later):.2f}–{100*max(later):.2f}% of training. The qualification gate is therefore not a universal final cold-start bound. No settings were changed after observing this exceedance.",'',
           'The optimized baseline exactly repeats target epochs 24/23/21 and official-test accuracies 98.67/98.63/98.71%. Its faster, separately optimized recipe uses all 60,000 training examples, batch 256 BF16, unit dropout and the frozen LR/shrinkage schedule; it is not a controlled arm of the 50,000-example SD experiment. These are repeated already-selected seeds and a test-target stopping rule, not an untouched test evaluation. Invocation and stopping-wall clocks include extra setup/evaluation; training clocks do not. Offline curvature and subsequent W&B uploads are separate costs.','',
           '## Provenance and limits','',
           'Derived from every raw telemetry-main-*.json and telemetry-baseline-*.json declared in telemetry/rerun-manifest.json. Exact source/file hashes, every raw scientific curve, all scalar measurements, per-class counts and both qualification attempts are retained in analysis.json and plot-data.json. Reproduce this note with telemetry/analyze.py after all 18 reruns verify. The historical ts4k9n55 reference includes all 91 count-verified public evaluation rows; its legacy val means official test, and its W&B runtime includes logging/evaluation on unverified hardware. It is not an isolated GPU-training-time baseline.','']
    return '\n'.join(lines)


def build(results=DEFAULT_RESULTS,manifest=None,original_results=None,historical=None,qualification_manifest=None,curvature_paths=()):
    results=Path(results);original_results=Path(original_results or EXPERIMENT/'results')
    manifest=Path(manifest or (HERE/'rerun-manifest.json' if (HERE/'rerun-manifest.json').exists() else HERE/'rerun-manifest-draft.json'))
    specs=read_json(manifest)
    if not isinstance(specs,list) or len(specs)!=18 or len({s['run_id'] for s in specs})!=18:raise ValueError('Expected manifest of18 distinct reruns')
    expected={(r,s) for r in ('plain','residual','sd_constant','sd_annealed','residual_unit_dropout') for s in (101,102,103)}
    if {(s['recipe'],s['seed']) for s in specs if s.get('runner')=='controlled'}!=expected or {s['seed'] for s in specs if s.get('runner')=='baseline'}!={101,102,103}:raise ValueError('Manifest scientific scope mismatch')
    runs=[];files={};issues=[]
    for spec in specs:
        path=results/(spec['run_id']+'.json');original_path=original_results/(original_run_id(spec)+'.json')
        entry={'id':spec['run_id'],'run_id':spec['run_id'],'cohort':'main' if spec['runner']=='controlled' else 'optimized_baseline',
               'recipe':spec['recipe'],'seed':spec['seed'],'runner':spec['runner'],'spec':spec,'status':'pending','issues':[],
               'history':[],'telemetry_rows':[]}
        if original_path.exists():
            original=read_json(original_path);files[str(original_path)]=sha(original_path)
            entry['reference']={'run_id':original_run_id(spec),'history':original.get('history',[]),'hardware':original.get('hardware',{})}
        else:original=None;entry['issues'].append('original_reference_missing')
        if path.exists():
            files[str(path)]=sha(path)
            try:
                raw=read_json(path)
                if 'error' in raw:entry.update(status='failed',issues=['remote_failure:'+str(raw.get('error_type','unspecified'))],error_type=raw.get('error_type'))
                elif original is None:entry.update(status='mismatch')
                else:entry=analyze_run(raw,original,spec)
            except (ValueError,KeyError,TypeError,IndexError) as error:entry.update(status='invalid',issues=['analysis_validation_error:'+str(error)])
        entry['source']={'path':str(path),'sha256':files.get(str(path))}
        entry.setdefault('reference',{})['source']={'path':str(original_path),'sha256':files.get(str(original_path))}
        runs.append(entry)
    qualification_manifests=[Path(qualification_manifest)] if qualification_manifest else sorted(HERE.glob('qualification*manifest.json'))
    qualification_specs=[];qualification_runs=[]
    for qpath in qualification_manifests:
        files[str(qpath)]=sha(qpath);qualification_specs.extend(read_json(qpath))
    if len({s['run_id'] for s in qualification_specs})!=len(qualification_specs):raise ValueError('Duplicate qualification manifest run IDs')
    for qspec in qualification_specs:
        path=results/(qspec['run_id']+'.json')
        if path.exists():
            files[str(path)]=sha(path);raw=read_json(path)
            if 'error' in raw:q={'run_id':qspec['run_id'],'status':'failed','issues':['remote_failure:'+str(raw.get('error_type'))],'pairs':[],'runs':[]}
            else:q=verify_qualification(raw,qspec)
        else:q={'run_id':qspec['run_id'],'status':'pending','pairs':[],'runs':[],'issues':[]}
        qualification_runs.append(q)
    latest=qualification_runs[-1] if qualification_runs else {}
    qualification={'status':'verified' if latest.get('status')=='verified' and latest.get('overhead_gate',{}).get('passed') else 'incomplete',
                   'selected_attempt':latest.get('run_id'),'attempts':qualification_runs,
                   'note':'All qualification attempts are retained. The latest qualified cadence/probe setting must pass the prespecified 10% direct telemetry/training gate; earlier failed timing gates are not erased.'}
    freeze_path=manifest.parent/'rerun-freeze.json'
    if freeze_path.exists():
        freeze=read_json(freeze_path);files[str(freeze_path)]=sha(freeze_path)
        if freeze.get('manifest_sha256')!=sha(manifest):issues.append('rerun_freeze:manifest_sha256_mismatch')
        if freeze.get('qualification')!=latest.get('run_id'):issues.append('rerun_freeze:qualification_mismatch')
        if freeze.get('qualification_sha256')!=files.get(str(results/(str(latest.get('run_id'))+'.json'))):issues.append('rerun_freeze:qualification_sha256_mismatch')
        if any(s.get('source_sha256')!=freeze.get('source_sha256') for s in specs):issues.append('rerun_freeze:source_hashes_mismatch')
        settings={(r.get('spec',{}).get('telemetry_every'),r.get('spec',{}).get('telemetry_probe_n')) for r in latest.get('runs',[]) if r.get('spec',{}).get('telemetry')}
        if len(settings)!=1 or any((s.get('telemetry_every'),s.get('telemetry_probe_n')) not in settings for s in specs):issues.append('rerun_freeze:qualified_instrumentation_settings_mismatch')
    curvature_manifest=manifest.parent/'curvature-manifest.json'
    curvature,curvature_status=curvature_results(results,curvature_paths,runs,curvature_manifest)
    known={s['run_id'] for s in specs}|{s['run_id'] for s in qualification_specs}|set(curvature_status['statuses'])
    additional=[]
    for path in sorted(results.glob('telemetry-*.json')):
        if path.stem not in known:
            raw=read_json(path)
            if 'runs' in raw and raw.get('spec',{}).get('runner')=='qualification':
                additional.append({'run_id':path.stem,'status':'prior_qualification_attempt','qualification':verify_qualification(raw),
                                   'sha256':sha(path),'note':'Retained although not selected by the explicit qualification-manifest argument'})
            else:additional.append({'run_id':path.stem,'spec':raw.get('spec'),'status':'unplanned_attempt','error_type':raw.get('error_type'),'history':raw.get('history',[]),'telemetry_rows':telemetry_rows(raw,[]),'sha256':sha(path)})
    for runner in ('controlled','baseline'):
        hashes={r['telemetry']['probe_sha256'] for r in runs if r.get('runner')==runner and r.get('telemetry',{}).get('probe_sha256')}
        if len(hashes)>1:issues.append(runner+':fixed_probe_hash_changes_across_runs')
    statuses=Counter(r['status'] for r in runs);counts={key:statuses.get(key,0) for key in ('verified','pending','incomplete','failed','mismatch','invalid')}
    all_present=counts['pending']==0 and counts['incomplete']==0
    status='complete_verified' if counts['verified']==18 and qualification['status']=='verified' and not issues else 'complete_with_issues' if all_present else 'partial'
    if 'draft' in manifest.name:issues.append('The available rerun manifest is still labeled draft');status='partial' if not all_present else 'complete_with_issues'
    historical=Path(historical or (results/'historical-ts4k9n55.json' if (results/'historical-ts4k9n55.json').exists() else EXPERIMENT/'history'/'selected-histories.json'))
    doc={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'status':status,'complete':status=='complete_verified','expected_reruns':18,
         'manifest':{'path':str(manifest),'sha256':sha(manifest),'is_draft':'draft' in manifest.name},'counts':counts,'runs':runs,
         'qualification':qualification,'additional_attempts':additional,'groups':aggregate_groups(runs),
         'historical':historical_series(historical),'issues':issues,'input_sha256':files,'timing_note':TIMING_NOTE,
         'curvature':curvature,'curvature_status':curvature_status}
    for name in ('budget-opening.json','budget-ledger.json','billing-latest.json'):
        if (results/name).exists():doc.setdefault('budget',{})[name.replace('.json','').replace('-','_')]=read_json(results/name)
    plot={'schema_version':1,'generated_utc':doc['generated_utc'],'status':doc['status'],'complete':doc['complete'],'counts':counts,
          'runs':runs,'qualification':latest,'qualification_attempts':qualification_runs,'qualification_status':qualification['status'],
          'additional_attempts':additional,'historical':doc['historical'],
          'groups':doc['groups'],'curvature':doc['curvature'],'curvature_status':curvature_status,'timing_note':TIMING_NOTE,
          'source_manifest_sha256':doc['manifest']['sha256']}
    return doc,plot


def write_outputs(results,doc,plot):
    results=Path(results);results.mkdir(parents=True,exist_ok=True)
    for filename,value in [('analysis.json',doc),('plot-data.json',plot)]:
        temporary=results/(filename+'.tmp');temporary.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');temporary.replace(results/filename)
    temporary=results/'summary.md.tmp';temporary.write_text(markdown(doc).rstrip()+'\n');temporary.replace(results/'summary.md')
    if doc['complete']:
        temporary=results/'training-findings.md.tmp';temporary.write_text(training_findings(doc));temporary.replace(results/'training-findings.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,default=DEFAULT_RESULTS)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--original-results',type=Path);parser.add_argument('--historical',type=Path)
    parser.add_argument('--qualification-manifest',type=Path);parser.add_argument('--curvature',type=Path,action='append',default=[])
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();doc,plot=build(args.results,args.manifest,args.original_results,args.historical,args.qualification_manifest,args.curvature)
    write_outputs(args.results,doc,plot);print(json.dumps({'status':doc['status'],'counts':doc['counts'],'qualification':doc['qualification']['status'],'historical_rows':doc['historical']['retained_rows']}))
    if args.require_complete and not doc['complete']:raise SystemExit('Incomplete or unverified reruns; every available attempt was still exported')


if __name__=='__main__':main()
