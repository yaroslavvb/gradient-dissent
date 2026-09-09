"""CPU-only diagnostic inside already allocated experiment workers.

No new Modal function/GPU call is launched. Records runtime CPU capabilities
before selecting workers for an independent untrained ViT initialization audit.
"""
import concurrent.futures
import json
from pathlib import Path
import subprocess

MODAL = '/Users/yaroslavvb/.local/bin/modal'
OUT = Path(__file__).resolve().parents[1] / 'results'
containers = json.loads(subprocess.check_output([
    MODAL, 'container', 'list', '-e', 'gradient-dissent-a100', '--json'], text=True))
code = """import torch,json,platform
print(json.dumps({'cpu_capability':torch.backends.cpu.get_cpu_capability(),
 'torch':str(torch.__version__),'machine':platform.machine(),
 'mkl_available':torch.backends.mkl.is_available()}))
"""

def probe(row):
    cid = row['container_id']
    p = subprocess.run([MODAL, 'container', 'exec', cid, '--', 'python', '-c', code],
                       capture_output=True, text=True, timeout=45)
    return dict(container_id=cid, **(json.loads(p.stdout) if p.returncode == 0
                else {'error':p.stderr[-1000:]}))

with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
    rows = list(pool.map(probe, containers))
(OUT / 'vit-initialization-cpu-probes.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
