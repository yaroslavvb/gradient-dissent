"""Historical CPU-only reconstruction in two existing capped GPU workers.

The strict tolerances were fixed before state comparisons: max absolute
difference <=1e-7 and whole-state relative L2 <=1e-6. No test outcomes are read.
Container IDs are historical; do not replay this script as a training command.
"""
import concurrent.futures
import json
from pathlib import Path
import subprocess

JOBS = [('avx512','ta-01M23WB92FS5Z37SCMH90CCT9R'),
        ('avx2','ta-01M23WAQAXSHTTGRF7WK218R0R')]
CODE = """import sys,torch,hashlib,json,gc
from pathlib import Path
sys.path.insert(0,'/opt/a100_transfer')
from vision import build_model
kind=sys.argv[1];out=Path('/work/vit-initialization-audit')/kind;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);rows=[]
for seed in (1000,2000,2001,2002):
 torch.manual_seed(seed);m=build_model('vit_cifar100',image_size=32);h=hashlib.sha256()
 for n,p in m.state_dict().items():h.update(n.encode());h.update(p.numpy().tobytes())
 path=out/(str(seed)+'.pt');torch.save(m.state_dict(),path)
 rows.append({'seed':seed,'state_sha256':h.hexdigest(),'file':str(path),'rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()});del m;gc.collect()
result={'kind':kind,'cpu_capability':torch.backends.cpu.get_cpu_capability(),'torch':str(torch.__version__),'source_sha256':hashlib.sha256(Path('/opt/a100_transfer/vision.py').read_bytes()).hexdigest(),'rows':rows}
(out/'manifest.json').write_text(json.dumps(result,indent=2))
import modal
modal.Volume.from_name('gradient-dissent-a100-20260909').commit();result['committed']=True
print(json.dumps(result))
"""

def run(job):
    p=subprocess.run(['/Users/yaroslavvb/.local/bin/modal','container','exec',job[1],
                      '--','python','-c',CODE,job[0]],capture_output=True,text=True,timeout=90)
    if p.returncode: return {'kind':job[0],'error':p.stderr[-2000:]}
    return dict(container_id=job[1], **json.loads(p.stdout))

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    rows=list(pool.map(run,JOBS))
out=Path(__file__).resolve().parents[1]/'results/vit-initialization-variant-manifests.json'
out.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
