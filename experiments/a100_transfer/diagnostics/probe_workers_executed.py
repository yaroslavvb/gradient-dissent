import concurrent.futures,json,subprocess
from pathlib import Path
ids=['ta-01M23TWH4T3Y6BJDB1SZ6VZ3HR','ta-01M23TWH4TDZWB2X8E0CRRGZNR','ta-01M23TWDNMGYYH71SJFN14NSCR','ta-01M23TWDNNM22N1PBE4ZBZ38BR','ta-01M23TVGB126DEJ6TT00J32A5R','ta-01M23TVG5F7FQEYWK7BBXAKNVR','ta-01M23TVGB1J3J45AY6DZE213RR','ta-01M23TVG5FT8R1SX6SSXWBZCMR','ta-01M23TVGB4HC1VX4N4TBGR021R']
code="""import sys,torch,hashlib,json
from pathlib import Path
sys.path.insert(0,'/opt/a100_transfer')
from vision import build_model
torch.set_num_threads(2);torch.manual_seed(1000)
m=build_model('convnext_cifar100',image_size=64);h=hashlib.sha256();rows=[]
for n,p in m.state_dict().items():
 b=p.numpy().tobytes();h.update(n.encode());h.update(b);rows.append({'name':n,'shape':list(p.shape),'sha256':hashlib.sha256(b).hexdigest()})
torch.save(m.state_dict(),'/tmp/gradient-init-seed1000.pt')
print(json.dumps({'seed':1000,'state_sha256':h.hexdigest(),'cpu_capability':torch.backends.cpu.get_cpu_capability(),'cpu_models':sorted(set(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))),'torch':str(torch.__version__),'source_sha256':hashlib.sha256(Path('/opt/a100_transfer/vision.py').read_bytes()).hexdigest(),'rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'tensors':rows}))
"""
def run(cid):
 p=subprocess.run(['modal','container','exec',cid,'--','python','-c',code],capture_output=True,text=True,timeout=90)
 if p.returncode: return {'container_id':cid,'error':p.stderr[-1000:]}
 d=json.loads(p.stdout);d['container_id']=cid;return d
with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool: rows=list(pool.map(run,ids))
Path('experiments/a100_transfer/results/initialization-worker-probes.json').write_text(json.dumps({'purpose':'CPU-only posthoc initialization diagnosis in existing containers; no new GPU calls','rows':rows},indent=2)+'\n')
for r in rows:print(r['container_id'],r.get('state_sha256'),r.get('cpu_models'),r.get('error'))
