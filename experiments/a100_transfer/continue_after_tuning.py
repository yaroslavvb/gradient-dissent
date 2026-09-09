"""Wait for the frozen tuning grid, select on validation only, then run final seeds.

No retries, grid extensions, test-based selection, or extra paid invocations.
The Modal launcher enforces the reservation ledger again before final dispatch.
"""
from pathlib import Path
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
jobs=json.loads((HERE/'tuning-manifest.json').read_text())
print('WAITING for all',len(jobs),'locked tuning runs',flush=True)
while True:
    completed=[]
    for spec in jobs:
        p=OUT/(spec['run_id']+'.json')
        if not p.exists(): continue
        try: result=json.loads(p.read_text())
        except json.JSONDecodeError: continue  # The launcher is completing its write.
        if 'error' in result: raise RuntimeError('Tuning failed; no final jobs submitted: '+spec['run_id'])
        assert result['spec']==spec
        assert result['curve'][-1]['step']==spec['steps']
        assert result['rows']==[] and result['test_panel'] is None
        completed.append(result)
    if len(completed)==len(jobs): break
    time.sleep(10)
selected={}
for task in sorted({j['task'] for j in jobs}):
    for recipe in ('dense','constant_ild','decreasing_ild'):
        group=[r for r in completed if r['spec']['task']==task and r['spec']['recipe']==recipe]
        assert len(group)==3
        chosen=min(group,key=lambda r:(r['validation']['ce'],r['spec']['lr']))
        selected[task+'/'+recipe]={'lr':chosen['spec']['lr'],'validation_ce':chosen['validation']['ce'],
            'validation_grid':[{ 'lr':r['spec']['lr'],'ce':r['validation']['ce']} for r in group]}
(OUT/'selected-lrs.json').write_text(json.dumps(selected,indent=2)+'\n')
final=[]
for seed in (2000,2001,2002):
    for recipe in ('dense','constant_ild','decreasing_ild'):
        for task in sorted({j['task'] for j in jobs}):
            base=next(j for j in jobs if j['task']==task and j['recipe']==recipe)
            final.append(dict(base,run_id=f'eval-{task}-{recipe}-s{seed}',stage='evaluate',seed=seed,lr=selected[task+'/'+recipe]['lr']))
path=HERE/'evaluation-manifest.json'
if path.exists(): assert json.loads(path.read_text())==final,'Existing final manifest differs; refuse overwrite.'
else: path.write_text(json.dumps(final,indent=2)+'\n')
print('SELECTED',json.dumps({k:v['lr'] for k,v in selected.items()}),flush=True)
print('STARTING',len(final),'final runs; selected manifest is frozen.',flush=True)
with (OUT/'evaluation.log').open('w') as log:
    subprocess.run(['modal','run','-e','gradient-dissent-a100',str(HERE/'modal_app.py'),
        '--stage','evaluate','--manifest',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['python3',str(HERE/'analyze.py')],check=True)
print('ALL EXPERIMENTS AND INDEPENDENT DATA AUDIT COMPLETE',flush=True)
