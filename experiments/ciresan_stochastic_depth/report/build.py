"""Build the practical report from verified experiments and actual W&B registry."""
from pathlib import Path
import hashlib,json,math,statistics
import markdown

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/ciresan_stochastic_depth'
OUT=ROOT/'docs/ciresan-stochastic-depth/conclusions'
NAMES={'plain':'Plain MLP','residual':'Residual dense','sd_constant':'Constant SD','sd_annealed':'Decreasing SD','residual_unit_dropout':'Residual + unit dropout','unit_dropout':'Optimized unit-dropout baseline'}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 training=read(EXP/'results/telemetry/analysis.json');audit=read(EXP/'results/hypotheses/analysis.json');registry=read(EXP/'results/telemetry/wandb-upload-manifest.json')
 assert training['complete'] and training['counts']['verified']==18
 assert audit['verification']['passed'] and audit['verification']['state_count']==30
 assert registry['expected_run_count']==18 and len(registry['runs'])==18
 assert registry['project']=='gradient-dissent' and registry['complete'] and registry['verified_run_count']==18,'Publishing this report requires all18 verified W&B runs in the requested project'
 source=ROOT/'research/ciresan-stochastic-depth-conclusions.md'
 body=markdown.markdown(source.read_text(),extensions=['tables','fenced_code'])
 body=body.replace('<h1>','<h2>').replace('</h1>','</h2>')
 template=(OUT/'template.html').read_text();assert '<!-- ARTICLE -->' in template
 if registry['complete']:
  assert registry['verified_run_count']==18 and all(r['status']=='verified' and r['verified_wandb_url'] for r in registry['runs'])
  body=body.replace('once authenticated uploads are verified, actual W&B runs','verified W&B runs')
 (OUT/'index.html').write_text(template.replace('<!-- ARTICLE -->',body))
 (OUT/'report.md').write_text(source.read_text())
 core=['residual','sd_constant','sd_annealed']
 aggregates=[a for a in audit['aggregates'] if a['recipe'] in core]
 rows=[]
 registry_by_id={r['scientific_run_id']:r for r in registry['runs']}
 for r in training['runs']:
  record=registry_by_id[r['run_id']]
  assert sha(ROOT/record['source_file'])==record['source_sha256']
  url=record['verified_wandb_url'] if record['status']=='verified' else None
  if url:
   assert url.startswith('https://wandb.ai/'+registry['entity']+'/'+registry['project']+'/runs/')
  rows.append({'run_id':r['run_id'],'recipe':r['recipe'],'name':NAMES[r['recipe']],'seed':r['seed'],'runner':r['runner'],
      'training_seconds':r['timing']['training_seconds'],'total_seconds':r['timing']['total_run_seconds'],'endpoints':r['endpoints'],
      'wandb_url':url,'wandb_status':record['status'],'raw_url':'https://github.com/yaroslavvb/gradient-dissent/blob/main/'+record['source_file']})
 payload={'aggregates':aggregates,'runs':rows,'wandb':{k:registry[k] for k in ['status','complete','entity','project','verified_run_count','expected_run_count']},
  'sources':{'training_analysis_sha256':sha(EXP/'results/telemetry/analysis.json'),'mask_analysis_sha256':sha(EXP/'results/hypotheses/analysis.json'),'wandb_registry_sha256':sha(EXP/'results/telemetry/wandb-upload-manifest.json')},
  'names':NAMES}
 (OUT/'data.js').write_text('window.CIRESAN_CONCLUSIONS='+json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')+';\n')
 print(json.dumps({'training_runs':len(rows),'mask_aggregates':len(aggregates),'verified_wandb_runs':registry['verified_run_count'],'data_bytes':(OUT/'data.js').stat().st_size}))
if __name__=='__main__':main()
