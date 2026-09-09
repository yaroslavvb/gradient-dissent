"""Regenerate manuscript A100 tables from the frozen verified result summary."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'experiments/a100_transfer/results/summary.json'; s=json.loads(src.read_text())
assert s['verification']['passed'] and s['verification']['complete_final_runs']==27
out=ROOT/'paper'; rows=[]; effects=[]; records=[]
for task,label in [('gpt_wikitext103','GPT'),('convnext_cifar100','ConvNeXt'),('vit_cifar100','ViT')]:
 for recipe,name in [('dense','Dense'),('constant_ild','Constant ILD'),('decreasing_ild','Decreasing ILD')]:
  a=next(c for c in s['curves'] if c['task']==task and c['recipe']==recipe and c['mask_name']=='full')
  b=next(c for c in s['curves'] if c['task']==task and c['recipe']==recipe and c['mask_name']=='primary_two_thirds')
  rows.append(f"{label} & {name} & {a['ce']['mean']:.3f} & {b['ce']['mean']:.3f} & {b['ce']['mean']-a['ce']['mean']:+.3f} & {100*a['accuracy']['mean']:.2f} & {100*b['accuracy']['mean']:.2f} \\\\")
  for mask,c in [('full',a),('primary_two_thirds',b)]:
   for metric in ['ce','accuracy']:
    for seed,value in zip(c['seeds'],c[metric]['values']):records.append([task,recipe,mask,metric,seed,value])
  if recipe!='dense':
   e=next(c for c in s['primary_comparisons'] if c['task']==task and c['recipe']==recipe)
   effects.append(f"{label} & {name} & ${e['mean']:+.3f}$ & $[{e['ci95_low']:+.3f},\\ {e['ci95_high']:+.3f}]$ \\\\")
 rows.append('\\addlinespace')
(out/'results-table.tex').write_text('\\newcommand{\\ResultRows}{%\n'+'\n'.join(rows[:-1])+'%\n}\n')
(out/'effects-table.tex').write_text('\\newcommand{\\EffectRows}{%\n'+'\n'.join(effects)+'%\n}\n')
with (out/'table-data.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['task','recipe','mask','metric','seed','value']);w.writerows(records)
(out/'provenance.json').write_text(json.dumps({'experiment_commit':'28f9db3bdc2e8910a105c7e40944b521c8bf0c1f','summary_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scientific_input_sha256':'5bad33d925dd3eb17e43351a69239f8a661df59c0a61a2916a1079dbd3c7d64f','final_runs':27,'tuning_runs':27,'no_new_training':True},indent=2)+'\n')
print('Built 9 result rows and 6 paired-effect rows; exported 108 seed-level values.')
