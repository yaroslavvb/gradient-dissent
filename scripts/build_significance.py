"""Build the overview's small data bundle from the verified A100 summary. No training."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'experiments/a100_transfer/results/summary.json'
s=json.loads(p.read_text())
assert s['verification']['passed'] and s['verification']['complete_final_runs']==27
x={k:s[k] for k in ['families','curves','primary_comparisons']}
x['source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
x['source']='../a100-transfer/data.json'
out=ROOT/'docs/significance'
(out/'data.js').write_text('window.SIGNIFICANCE_DATA = '+json.dumps(x,separators=(',',':'))+';\n')
print('Built significance data from 27 verified final runs.')
