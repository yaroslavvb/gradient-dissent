#!/usr/bin/env python3
"""Independently validate saved measurements and evaluate a saved checkpoint."""
import csv,hashlib,itertools,json,math,statistics
from pathlib import Path
import torch
import run as experiment
ROOT=Path(__file__).resolve().parent
result=json.loads((ROOT/'results/result.json').read_text())
assert hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()==result['metadata']['source_sha256']
rows=list(csv.DictReader((ROOT/'results/pruning.csv').open()))
assert len(rows)==1335
runs=list(csv.DictReader((ROOT/'results/evaluation_runs.csv').open()))
assert len(runs)==25
assert len(list(csv.DictReader((ROOT/'results/tuning.csv').open())))==50
for r in rows:
 assert len(set(r['layers'].split(',')))==int(r['retained'])
 assert abs(float(r['excess_ce'])-(float(r['ce'])-float(r['full_ce'])))<1e-12
for name in experiment.TREATMENTS:
 depth=experiment.TREATMENTS[name]['depth']
 for seed in experiment.CFG['evaluation_seeds']:
  rs=[r for r in rows if r['config']==name and int(r['seed'])==seed]
  assert len(rs)==2**depth-1
  assert len({r['layers'] for r in rs})==len(rs)
# Recalculate the primary paired intervals from raw individual-submodel values.
comparisons={}
for name in ['constant_ild','decreasing_ild','alternating']:
 diff=[]
 for seed in experiment.CFG['evaluation_seeds']:
  def primary(c):return float(next(r['excess_ce'] for r in rows if r['config']==c and int(r['seed'])==seed and r['layers']=='1,2,3,4'))
  diff.append(primary(name)-primary('dense6'))
 mean=statistics.mean(diff);margin=2.7764451052*statistics.stdev(diff)/math.sqrt(5)
 summary=next(r for r in result['paired_comparisons'] if r['config']==name and r['mode']=='prefix' and r['retained']==4)['excess_ce_difference_vs_dense6']
 assert abs(summary['mean']-mean)<1e-12
 assert abs(summary['ci95_low']-(mean-margin))<1e-12
 comparisons[name]={'mean':mean,'low':mean-margin,'high':mean+margin}
train,val,test,vocab=experiment.prepare_data()
# No-drop forward and gradient equivalence on the same model/data.
torch.manual_seed(67);m=experiment.Model(vocab,6);x=test[0][:2];y=test[1][:2]
def gradients(scales):
 m.zero_grad(set_to_none=True);o=m(x,scales=scales);loss=torch.nn.functional.cross_entropy(o.flatten(0,1),y.flatten());loss.backward();return o.detach(),[p.grad.clone() for p in m.parameters()]
a,ga=gradients(None);b,gb=gradients(torch.ones(2,6));assert torch.equal(a,b);assert all(torch.equal(x,y) for x,y in zip(ga,gb))
# The per-sequence multiplier is shared across both residual branches; dropped block is identity.
h=torch.randn(2,64,64);assert torch.equal(m.blocks[0](h,torch.zeros(2,1,1)),h)
checkpoint_checks=[]
for name in ['dense6','decreasing_ild']:
 checkpoint=ROOT/f'checkpoints/{name}-100.pt'
 if checkpoint.exists():
  model=experiment.Model(vocab,6);model.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'])
  measured=experiment.evaluate(model,test,keep=(0,1,2,3))['ce']
  expected=float(next(r['ce'] for r in rows if r['config']==name and int(r['seed'])==100 and r['layers']=='1,2,3,4'))
  assert measured==expected;checkpoint_checks.append({'config':name,'seed':100,'prefix4_ce_exact_match':True})
output={'passed':True,'source_sha256':result['metadata']['source_sha256'],'raw_mask_rows':len(rows),'tuning_runs':50,'evaluation_runs':25,'complete_subset_enumeration':True,'independent_primary_ci_recalculation':comparisons,'zero_dropout_forward_and_gradients_equal':True,'dropped_block_identity':True,'checkpoint_readback':checkpoint_checks,'note':'Readback uses local ignored checkpoints when available; no new hyperparameter selection.'}
(ROOT/'results/verification.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
