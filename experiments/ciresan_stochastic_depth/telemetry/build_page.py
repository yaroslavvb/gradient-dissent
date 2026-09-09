"""Regenerate the self-contained Pages dataset and rendered research text."""
from pathlib import Path
import argparse,json,re,shutil
import markdown

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
RESULTS=HERE.parent/'results'/'telemetry'
PAGE=REPO/'docs'/'ciresan-stochastic-depth'/'telemetry'

def main():
 p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
 data=json.loads((RESULTS/'plot-data.json').read_text())
 if not args.allow_partial:
  assert data['complete'] and data['counts']['verified']==18,'All18 reruns must verify before publishing'
  assert data['curvature_status']['complete'] and len(data['curvature'])==3 and all(c['result'].get('complete') and len(c['result']['snapshots'])==7 for c in data['curvature']),'All21 offline checkpoints must complete'
 # Keep every plotted scalar, but avoid shipping duplicate reference histories,
 # per-class dictionaries already flattened into metrics, and qualification runs.
 payload={key:data[key] for key in ('status','complete','counts','historical')}
 payload['runs']=[{**{key:r[key] for key in ('id','run_id','recipe','seed','runner','status','history','telemetry','endpoints')},
  'telemetry_rows':[{key:t[key] for key in ('epoch','training_seconds','run_elapsed_seconds','metrics')} for t in r['telemetry_rows']]} for r in data['runs'] if 'telemetry' in r]
 payload['qualification']={key:data['qualification'][key] for key in ('run_id','status','pairs','overhead_gate')}
 payload['curvature']=[{'result':{key:c['result'][key] for key in ('recipe','seed','status','complete','snapshots')}} for c in data['curvature']]
 # No inline-script end tags can escape this external JavaScript payload.
 (PAGE/'data.js').write_text('window.TELEMETRY_DATA='+json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')+';\n')
 source=REPO/'research'/'ciresan-telemetry-rerun.md'
 if source.exists():
  report=source.read_text();(PAGE/'report.md').write_text(report)
  body=markdown.markdown(report,extensions=['tables','fenced_code']);body=body.replace('<h1>','<h2>').replace('</h1>','</h2>')
  template=(PAGE/'index.html').read_text()
  template=re.sub(r'<article id="report">.*?</article>',lambda m:'<article id="report">'+body+'</article>',template,flags=re.S)
  (PAGE/'index.html').write_text(template)
 plan=(HERE/'METRIC_PLAN.md').read_text()
 body=markdown.markdown(plan,extensions=['tables','fenced_code'])
 (PAGE/'metric-plan.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciresan metric audit</title><link rel="stylesheet" href="../style.css"></head><body><main><a href="./">Back to training plots</a><article>'+body+'</article></main></body></html>')
 print(json.dumps({'runs':len(data['runs']),'verified':data['counts']['verified'],'curvature_jobs':len(data['curvature']),'data_bytes':(PAGE/'data.js').stat().st_size}))
if __name__=='__main__':main()
