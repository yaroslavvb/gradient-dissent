import concurrent.futures,json,subprocess
from pathlib import Path
code="""import sys,torch,hashlib,json,os
sys.path.insert(0,'/opt/a100_transfer');from vision import build_model
torch.set_num_threads(2);torch.manual_seed(1000)
t=torch.empty(1000000);stages={}
for name,fn in [('uniform',lambda:t.uniform_(-1,1)),('erfinv',lambda:t.erfinv_()),('scale',lambda:t.mul_(.02*2**.5))]:
 fn();stages[name]=hashlib.sha256(t.numpy().tobytes()).hexdigest()
torch.manual_seed(1000);m=build_model('convnext_cifar100',image_size=64);h=hashlib.sha256()
for n,p in m.state_dict().items():h.update(n.encode());h.update(p.numpy().tobytes())
print(json.dumps({'cpu_capability':torch.backends.cpu.get_cpu_capability(),'mkl_available':torch.backends.mkl.is_available(),'mkl_enable_instructions':os.environ.get('MKL_ENABLE_INSTRUCTIONS'),'stages':stages,'model_hash':h.hexdigest(),'torch':str(torch.__version__)}))
"""
jobs=[('avx512-default','ta-01M23TVGB126DEJ6TT00J32A5R',[]),('avx512-mkl-avx2','ta-01M23TVGB126DEJ6TT00J32A5R',['env','MKL_ENABLE_INSTRUCTIONS=AVX2']),('avx2-default','ta-01M23TWH4TDZWB2X8E0CRRGZNR',[])]
def run(j):
 p=subprocess.run(['modal','container','exec',j[1],'--',*j[2],'python','-c',code],capture_output=True,text=True,timeout=90)
 if p.returncode:return {'label':j[0],'error':p.stderr[-1500:]}
 r=json.loads(p.stdout);r['label']=j[0];r['container_id']=j[1];return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,jobs))
Path('experiments/a100_transfer/results/initialization-kernel-probes.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(json.dumps(r))
