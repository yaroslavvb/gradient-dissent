"""Freeze validation-only LR choices and final manifests. Never starts compute."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent

def main():
    manifest=json.loads((HERE/'graph-tune-manifest.json').read_text())
    results=[]
    for spec in manifest:
        path=HERE/'results'/(spec['run_id']+'.json')
        r=json.loads(path.read_text())
        assert r['spec']==spec and not r.get('error'), spec['run_id']
        assert r.get('diverged') or r['history'][-1]['epoch']==100, spec['run_id']
        assert 'selected_test' not in r and 'final_test' not in r
        results.append(r)
    recipes=['plain','residual','sd_constant','sd_annealed','residual_unit_dropout']
    choices={}
    selected={}
    final=[]
    for recipe in recipes:
        candidates=[r for r in results if r['spec']['recipe']==recipe]
        assert len(candidates)==3
        eligible=[r for r in candidates if not r.get('diverged')]
        assert eligible, f'No stable full-horizon candidate for {recipe}'
        winner=min(eligible,key=lambda r:(r['best_validation_loss'],r['spec']['lr']))
        selected[recipe]=winner['spec']['lr']
        choices[recipe]={'run_id':winner['run_id'],'lr':winner['spec']['lr'],
                         'validation_ce':winner['best_validation_loss'],'epoch':winner['best_epoch']}
        for seed in (101,102,103):
            spec={**winner['spec'],'run_id':f'graph-eval-{recipe}-s{seed}','stage':'evaluate','seed':seed}
            final.append(spec)
    for variant,recipe,pmax in [('source-plain','plain',0.),('source-residual','residual',0.),('source-sd','sd_constant',.4)]:
        for seed in (101,102,103):
            final.append({'run_id':f'graph-fidelity-{variant}-s{seed}','stage':'evaluate','cohort':'fidelity',
                          'variant':variant,'recipe':recipe,'pmax':pmax,'input_scale':1.,'output_relu':True,
                          'lr':.001,'momentum':.9,'shrinkage':.00002,'seed':seed,'epochs':100,'batch_size':64,
                          'eval_every':5,'train_size':50000,'timeout_seconds':180,'gpu':'A100','precision':'fp32'})
    assert len(final)==24
    now=datetime.now(timezone.utc).isoformat()
    selection={'frozen_utc':now,'criterion':'among completed nondiverged candidates, minimum validation CE across scheduled checkpoints, tie broken by lower LR',
               'excluded_diverged_candidates':[r['run_id'] for r in results if r.get('diverged')],
               'selected_learning_rates':selected,'winners':choices,
               'candidate_results_sha256':{r['run_id']:hashlib.sha256((HERE/'results'/(r['run_id']+'.json')).read_bytes()).hexdigest() for r in results}}
    # Fail rather than overwrite a frozen selection, even if a later rerun wins.
    with (HERE/'selection.json').open('x') as f:json.dump(selection,f,indent=2);f.write('\n')
    with (HERE/'final-manifest.json').open('x') as f:json.dump(final,f,indent=2);f.write('\n')
    freeze={'frozen_utc':now,'phase':'before official-test evaluation of resumed study','sha256':
            {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['selection.json','final-manifest.json','select_graph_runs.py','train_optimized.py','optimization/stochastic_graph.py','model_data.py','graph_app.py']}}
    with (HERE/'final-freeze.json').open('x') as f:json.dump(freeze,f,indent=2);f.write('\n')
    print(json.dumps({'selected_learning_rates':selected,'final_runs':len(final)},indent=2))

if __name__=='__main__':main()
