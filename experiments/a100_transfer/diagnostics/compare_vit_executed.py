"""Historical full-tensor comparison; CPU-only in an existing GPU worker."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = 'ta-01M23TVG5F7FQEYWK7BBXAKNVR'
prefix = """import modal
from pathlib import Path
v=modal.Volume.from_name('gradient-dissent-a100-20260909')
for kind in ('avx512','avx2'):
 for seed in (1000,2000,2001,2002):
  dest=Path('/tmp/vit-initialization-audit')/kind/(str(seed)+'.pt')
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.stat().st_size==340918266: continue
  with dest.open('wb') as out:
   for chunk in v.read_file('/vit-initialization-audit/'+kind+'/'+str(seed)+'.pt'):
    out.write(chunk)
"""
# This worker predates the commits. Read committed files through the API rather
# than reloading a mount used by the ongoing training process.
code = prefix + (ROOT/'compare_vit_initialization_variants.py').read_text().replace(
    "ROOT=Path('/work/vit-initialization-audit')", "ROOT=Path('/tmp/vit-initialization-audit')")
code = code.replace('print(json.dumps(result))',
    "Path('/tmp/vit-initialization-numerical-audit.json').write_text(json.dumps(result));print(json.dumps(result))")
p = subprocess.run(['/Users/yaroslavvb/.local/bin/modal','container','exec',CONTAINER,
                    '--','python','-c',code],capture_output=True,text=True,timeout=600)
(ROOT/'results/vit-initialization-comparison-stderr.txt').write_text(p.stderr)
if p.returncode:
    raise RuntimeError(p.stderr[-3000:])
if not p.stdout.strip():
    raise RuntimeError('Remote comparison produced no JSON: '+p.stderr[-3000:])
result = json.loads(p.stdout)
(ROOT/'results/vit-initialization-numerical-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in result.items() if k!='rows'},
                  'rows':[{k:v for k,v in r.items() if k!='tensors'} for r in result['rows']]},indent=2))
