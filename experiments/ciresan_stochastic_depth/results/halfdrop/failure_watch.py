"""CPU-only early-stop watchdog; examines status/provenance, never test metrics."""
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
jobs = json.loads((OUT/'main-manifest.json').read_text())
seen = set(); baselines = {}; start = time.time()
fields = ('initial_parameters_sha256', 'initialization_rng_sha256', 'executed_epoch_order_sha256', 'raw_mask_draws_sha256', 'source_sha256')

def halt(reason, runid):
    dispatches = list(OUT.glob('halfdrop-m*-dispatch.json'))
    app_ids = {json.loads(p.read_text())['app_id'] for p in dispatches}
    receipt = {'status':'protocol_stop','reason':reason,'run_id':runid,'app_ids':sorted(app_ids),'unix':time.time(),'test_metrics_inspected':False}
    (OUT/'watchdog-stop.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for appid in app_ids:
        subprocess.run([str(Path(sys.executable).with_name('modal')),'app','stop',appid,'-e','gradient-dissent-ciresan','--yes'],check=False,capture_output=True)
    print(json.dumps(receipt),flush=True)
    raise SystemExit(1)

while len(seen)<len(jobs):
    if time.time()-start>3600:
        halt('Watchdog one-hour wall limit reached','watchdog')
    for spec in jobs:
        ident=spec['run_id']; path=OUT/(ident+'.json')
        if ident in seen or not path.exists():continue
        result=json.loads(path.read_text())
        if 'error' in result or result.get('complete') is not True or result.get('status')!='complete':
            halt('Run failed or returned incomplete status',ident)
        view=json.loads((OUT/(ident+'-validation.json')).read_text())
        if view.get('spec')!=spec or view.get('test_accessed') is not False or view.get('source_sha256')!=spec['source_sha256']:
            halt('Validation protocol/source mismatch',ident)
        if [row['epoch'] for row in view['history']]!=list(range(1,101)):
            halt('Incomplete fixed100-epoch protocol',ident)
        comparable={key:view[key] for key in fields}
        comparable['training_data_sha256']=view['dataset']['training_data_sha256']
        comparable['split_sha256']=view['dataset']['split_sha256']
        seed=spec['seed']
        if seed in baselines and comparable!=baselines[seed]:
            halt('Matched-seed initialization/order/draw/source/data pairing mismatch',ident)
        baselines.setdefault(seed,comparable)
        seen.add(ident)
        print(json.dumps({'verified_complete':len(seen),'expected':48,'run_id':ident,'test_metrics_inspected':False}),flush=True)
    time.sleep(.5)
receipt={'status':'complete','verified_runs':len(seen),'matched_seed_provenance_passed':True,'test_metrics_inspected':False,'unix':time.time()}
(OUT/'watchdog-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
