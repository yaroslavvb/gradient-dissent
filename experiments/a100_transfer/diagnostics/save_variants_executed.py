import concurrent.futures,json,subprocess
from pathlib import Path
jobs=[('avx512','ta-01M23TVGB126DEJ6TT00J32A5R'),('avx2','ta-01M23TWH4TDZWB2X8E0CRRGZNR')]
code="""import sys,torch,hashlib,json,gc
from pathlib import Path
sys.path.insert(0,'/opt/a100_transfer')
from vision import build_model
kind=sys.argv[1];out=Path('/work/initialization-audit')/kind;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);rows=[]
for seed in (1000,2000,2001,2002):
 torch.manual_seed(seed);m=build_model('convnext_cifar100',image_size=64);h=hashlib.sha256()
 for n,p in m.state_dict().items():h.update(n.encode());h.update(p.numpy().tobytes())
 path=out/(str(seed)+'.pt');torch.save(m.state_dict(),path)
 rows.append({'seed':seed,'state_sha256':h.hexdigest(),'file':str(path),'rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()});del m;gc.collect()
result={'kind':kind,'cpu_capability':torch.backends.cpu.get_cpu_capability(),'torch':str(torch.__version__),'source_sha256':hashlib.sha256(Path('/opt/a100_transfer/vision.py').read_bytes()).hexdigest(),'rows':rows}
(out/'manifest.json').write_text(json.dumps(result,indent=2))
try:
 import modal
 modal.Volume.from_name('gradient-dissent-a100-20260909').commit();result['committed']=True
except Exception as e:result['commit_error']=str(e)
print(json.dumps(result))
"""
def run(j):
 p=subprocess.run(['modal','container','exec',j[1],'--','python','-c',code,j[0]],capture_output=True,text=True,timeout=90)
 if p.returncode:return {'kind':j[0],'error':p.stderr[-2000:]}
 return json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
Path('experiments/a100_transfer/results/initialization-variant-manifests.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(json.dumps(r))
