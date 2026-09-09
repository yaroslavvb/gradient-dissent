#!/usr/bin/env python3
"""Depth robustness in a tiny causal character transformer. Local CPU only.
This is a qualitative mechanism reproduction, not the paper's LLM recipe.
"""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,platform,statistics,subprocess,time,sys,urllib.request
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIR=Path(__file__).resolve().parent
DATA=DIR/'data';OUT=DIR/'results';CKPT=DIR/'checkpoints'
for folder in [DATA,OUT,CKPT]:folder.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
CFG={'depth':6,'width':64,'heads':4,'context':64,'batch':16,'steps':800,
 'learning_rates':[.001,.003,.01,.02,.03],'tuning_seeds':[10,11],'evaluation_seeds':[100,101,102,103,104],
 'concurrency':'Up to4 isolated worker processes; each uses1 PyTorch CPU thread and its own RNG state.',
 'validation_windows':64,'test_windows':128,'split':[.8,.1,.1],
 'primary_endpoint':'Prefix4 CE minus full CE, paired difference vs dense6; report raw CE as well.',
 'tuning_endpoint':'Full-depth validation CE only. No pruning masks or test data used for selection. Initial partial three-rate sweep showed improving validation at upper boundary; uniformly extended all five treatments to five rates before any final evaluation.',
 'budget':'Matched steps and tokens, not matched FLOPs. Compute then mask; no actual training speedup claim.',
 'optimizer':'AdamW lr chosen per treatment; weight_decay .01 for matrices only, betas .9/.999, eps1e-8; gradient clipping1;20-step warmup then cosine to10%.',
 'mask_semantics':'Shared per-sequence attention/FFN Bernoulli mask, inverse-survival scaling separately on both residual branches (paper Eq6). All parameters participate in autograd; zero gradients still get AdamW updates.',
 'evaluation':'No fine-tuning, no auxiliary exit head/loss, no adapters, no rescaling after pruning. Original absolute positional embeddings, final LayerNorm and linear head reused. Exact all nonempty subsets, one-based layer IDs. Random-mask results mean uniform over exact subsets.',
 'model':'Pre-LN causal transformer, learned absolute positions,4-head attention,GELU FFN4x,untied char embedding/head,output projections init std .02/sqrt(2L); all other linear/embed std .02. No CompleteP, ALiBi, squared ReLU or tokenization from target paper.'}
TREATMENTS={'dense6':{'depth':6,'schedule':'none','pmax':0.},
 'constant_ild':{'depth':6,'schedule':'constant','pmax':.4},
 'decreasing_ild':{'depth':6,'schedule':'decreasing','pmax':.8},
 'alternating':{'depth':6,'schedule':'alternating','pmax':.4},
 'dense4':{'depth':4,'schedule':'none','pmax':0.}}

def dump(file,obj):file.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def write_csv(file,rows):
 with file.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(b):return hashlib.sha256(b).hexdigest()
def metadata():
 return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if sys.platform=='darwin' else platform.processor(),'threads':1,'device':'cpu','precision':'float32','source_sha256':sha(Path(__file__).read_bytes()),'cloud_spend_usd':0,'modal_invocations':0}
def intervals(values):
 n=len(values);mean=statistics.mean(values);sd=statistics.stdev(values) if n>1 else 0.;crit={5:2.7764451052,2:12.7062047364}.get(n,1.96)
 return {'n':n,'mean':mean,'std':sd,'ci95_low':mean-crit*sd/math.sqrt(n),'ci95_high':mean+crit*sd/math.sqrt(n)}
def prepare_data():
 source='https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt'
 path=DATA/'input.txt'
 if not path.exists():
  with urllib.request.urlopen(source,timeout=60) as r:path.write_bytes(r.read())
 raw=path.read_bytes();assert sha(raw)=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed','Dataset hash mismatch';text=raw.decode('utf-8');n=len(text);a=int(.8*n);b=int(.9*n)
 # Vocabulary derived from training text only. Fail if a later split has unseen chars.
 chars=sorted(set(text[:a]));stoi={c:i for i,c in enumerate(chars)}
 assert not (set(text)-set(chars)),'Held-out unseen char: need explicit UNK mapping before experiment.'
 ids=torch.tensor([stoi[c] for c in text],dtype=torch.long)
 parts={'train':ids[:a],'validation':ids[a:b],'test':ids[b:]}
 def windows(part,nwin):
  starts=np.linspace(0,len(part)-CFG['context']-1,nwin,dtype=int)
  assert np.all(np.diff(starts)>CFG['context']),'Held-out target windows must not overlap.'
  x=torch.stack([part[i:i+CFG['context']] for i in starts]);y=torch.stack([part[i+1:i+CFG['context']+1] for i in starts])
  return x,y,starts.tolist()
 vx,vy,vs=windows(parts['validation'],CFG['validation_windows']);tx,ty,ts=windows(parts['test'],CFG['test_windows'])
 manifest={'source_url':source,'sha256':sha(raw),'bytes':len(raw),'characters':n,'vocabulary':chars,'vocabulary_size':len(chars),'split_ranges':{'train':[0,a],'validation':[a,b],'test':[b,n]},'split_description':'Contiguous disjoint80/10/10 character ranges; train-derived vocabulary; windows wholly within a split; deterministic evenly spaced nonoverlapping held-out target windows. Corpus is small and repeatedly sampled, unlike modern single-epoch LLM pretraining.','validation_window_starts_relative':vs,'test_window_starts_relative':ts,'validation_target_tokens':vy.numel(),'test_target_tokens':ty.numel()}
 dump(OUT/'dataset.json',manifest)
 return parts['train'],(vx,vy),(tx,ty),len(chars)

class Block(nn.Module):
 def __init__(self):
  super().__init__();d=CFG['width'];self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.ln2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
 def forward(self,x,scale):
  b,s,d=x.shape;q,k,v=self.qkv(self.ln1(x)).chunk(3,dim=-1)
  def heads(t):return t.reshape(b,s,CFG['heads'],d//CFG['heads']).transpose(1,2)
  y=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),is_causal=True,dropout_p=0).transpose(1,2).reshape(b,s,d)
  x=x+scale*self.proj(y);x=x+scale*self.ff(self.ln2(x));return x
class Model(nn.Module):
 def __init__(self,vocab,depth):
  super().__init__();self.depth=depth;d=CFG['width'];self.embed=nn.Embedding(vocab,d);self.pos=nn.Embedding(CFG['context'],d);self.blocks=nn.ModuleList([Block() for _ in range(depth)]);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,vocab,bias=False);self.apply(self.init)
  for block in self.blocks:
   nn.init.normal_(block.proj.weight,std=.02/math.sqrt(2*depth));nn.init.normal_(block.ff[-1].weight,std=.02/math.sqrt(2*depth))
 @staticmethod
 def init(m):
  if isinstance(m,(nn.Linear,nn.Embedding)):
   nn.init.normal_(m.weight,std=.02)
   if getattr(m,'bias',None) is not None:nn.init.zeros_(m.bias)
 def forward(self,tokens,scales=None,keep=None):
  x=self.embed(tokens)+self.pos(torch.arange(tokens.shape[1]))
  for l,block in enumerate(self.blocks):
   if keep is not None and l not in keep:continue
   x=block(x,1. if scales is None else scales[:,l,None,None])
  return self.head(self.norm(x))
def rates(treatment,step,steps):
 spec=TREATMENTS[treatment];d=spec['depth'];p=spec['pmax']
 if spec['schedule']=='none':return torch.zeros(d)
 if spec['schedule']=='alternating':return torch.tensor([p*(l%2) for l in range(d)])
 spatial=torch.arange(d)/(d-1)*p
 return spatial*(1-step/(steps-1)) if spec['schedule']=='decreasing' else spatial
@torch.no_grad()
def evaluate(model,xy,keep=None):
 model.eval();x,y=xy;loss=0.;correct=0;n=y.numel()
 for xx,yy in zip(x.split(16),y.split(16)):
  logits=model(xx,keep=keep);loss+=F.cross_entropy(logits.flatten(0,1),yy.flatten(),reduction='sum').item();correct+=(logits.argmax(-1)==yy).sum().item()
 return {'ce':loss/n,'accuracy':correct/n,'perplexity':math.exp(loss/n)}
def sample(train,g):
 starts=torch.randint(0,len(train)-CFG['context']-1,(CFG['batch'],),generator=g)
 offsets=torch.arange(CFG['context'])
 x=train[starts[:,None]+offsets];y=train[starts[:,None]+offsets+1];return x,y

def train_run(name,seed,lr,train,val,vocab,steps=None,checkpoint=False):
 steps=steps or CFG['steps'];torch.manual_seed(seed);model=Model(vocab,TREATMENTS[name]['depth'])
 data_g=torch.Generator().manual_seed(10000+seed);mask_g=torch.Generator().manual_seed(20000+seed)
 groups=[{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.01},{'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
 opt=torch.optim.AdamW(groups,lr=lr);curve=[];active=0;clipped=0;maxnorm=0.;start=time.perf_counter()
 for step in range(steps):
  model.train();x,y=sample(train,data_g);p=rates(name,step,steps);mask=(torch.rand((CFG['batch'],model.depth),generator=mask_g)>=p);active+=mask.sum().item();scales=mask.float()/(1-p)
  progress=max(0,(step-19)/(steps-20));rate=lr*min(1,(step+1)/20)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
  for group in opt.param_groups:group['lr']=rate
  opt.zero_grad(set_to_none=True);logits=model(x,scales=scales);loss=F.cross_entropy(logits.flatten(0,1),y.flatten());assert torch.isfinite(loss);loss.backward();norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm);clipped+=norm>1.;maxnorm=max(maxnorm,norm);opt.step()
  if (step+1)%100==0 or step==0 or step+1==steps:curve.append({'step':step+1,'train_ce':loss.item(),'mean_dropout':p.mean().item(),'realized_active_sequence_blocks':active})
 seconds=time.perf_counter()-start;metrics=evaluate(model,val)
 run={'config':name,'seed':seed,'lr':lr,'steps':steps,'tokens':steps*CFG['batch']*CFG['context'],'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':seconds,'validation_ce':metrics['ce'],'active_sequence_blocks':active,'dense_sequence_blocks':steps*CFG['batch']*model.depth,'nominal_mean_dropout':float(torch.stack([rates(name,s,steps) for s in range(steps)]).mean()),'clipped_steps':clipped,'max_preclip_grad_norm':maxnorm}
 if checkpoint:torch.save({'state_dict':model.state_dict(),'run':run,'config':CFG},CKPT/f'{name}-{seed}.pt')
 return model,run,curve

def correctness(vocab):
 torch.manual_seed(555);m=Model(vocab,6);m.eval();x=torch.randint(vocab,(2,CFG['context']));original=m(x);full=m(x,keep=tuple(range(6)));assert torch.equal(original,full)
 assert torch.equal(original,m(x,scales=torch.ones(2,6)))
 # Causality: modifying later tokens cannot affect logits on preceding positions.
 x2=x.clone();x2[:,32:]=torch.randint(vocab,x2[:,32:].shape);err=float((m(x)[:,:32]-m(x2)[:,:32]).abs().max().detach());assert err<1e-6
 for name in TREATMENTS:
  mean=float(torch.stack([rates(name,t,CFG['steps']) for t in range(CFG['steps'])]).mean());expect=0 if name.startswith('dense') else .2;assert abs(mean-expect)<1e-6
 assert float(rates('decreasing_ild',CFG['steps']-1,CFG['steps']).max())==0
 # Real pruning equals explicit zeroing of the same two residual branches in eval.
 keep=(0,2,4);scale=torch.zeros(2,6);scale[:,list(keep)]=1.;pruned=m(x,keep=keep);masked=m(x,scales=scale);assert torch.equal(pruned,masked)
 checks={'all_keep_equals_full':True,'unit_scale_equals_eval':True,'pruning_equals_zeroed_branches':True,'causal_future_token_test_max_abs_error':err,'schedule_mean_and_endpoints':True}
 dump(OUT/'checks.json',checks);return checks

def prune_run(model,xy,run):
 depth=model.depth;rows=[];full=evaluate(model,xy)
 for k in range(1,depth+1):
  for subset in itertools.combinations(range(depth),k):
   metric=full if k==depth else evaluate(model,xy,keep=subset)
   rows.append({'config':run['config'],'seed':run['seed'],'retained':k,'layers':','.join(str(l+1) for l in subset),'keeps_first':0 in subset,'is_prefix':subset==tuple(range(k)),'is_odd_layers':subset==tuple(range(0,depth,2)),'is_even_layers':subset==tuple(range(1,depth,2)),'ce':metric['ce'],'accuracy':metric['accuracy'],'perplexity':metric['perplexity'],'full_ce':full['ce'],'excess_ce':metric['ce']-full['ce'],'perplexity_ratio_vs_full':math.exp(metric['ce']-full['ce'])})
 return rows

def aggregate(rows):
 summaries=[];paired=[];seed_metrics={}
 for name,spec in TREATMENTS.items():
  for k in range(1,spec['depth']+1):
   for mode in ['prefix','all_subsets','keep_first']:
    sel=[r for r in rows if r['config']==name and r['retained']==k and (mode!='prefix' or r['is_prefix']) and (mode!='keep_first' or r['keeps_first'])]
    if not sel:continue
    stats={}
    for metric in ['ce','excess_ce','accuracy','perplexity_ratio_vs_full']:
     vals={seed:statistics.mean(r[metric] for r in sel if r['seed']==seed) for seed in CFG['evaluation_seeds']};stats[metric]=intervals(vals.values());seed_metrics[(name,k,mode,metric)]=vals
    summaries.append({'config':name,'retained':k,'mode':mode,'masks_per_seed':len(sel)//len(CFG['evaluation_seeds']),**stats})
 for name in TREATMENTS:
  if name=='dense6':continue
  for k in range(1,TREATMENTS[name]['depth']+1):
   for mode in ['prefix','all_subsets','keep_first']:
    stats={}
    for metric in ['ce','excess_ce','accuracy']:
     a=seed_metrics[(name,k,mode,metric)];b=seed_metrics[('dense6',k,mode,metric)];stats[metric+'_difference_vs_dense6']=intervals([a[s]-b[s] for s in CFG['evaluation_seeds']])
    paired.append({'config':name,'retained':k,'mode':mode,**stats})
 return summaries,paired

def tuning_job(args):
 name,seed,lr,train,val,vocab=args
 _,run,_=train_run(name,seed,lr,train,val,vocab)
 return run
def evaluation_job(args):
 name,seed,lr,train,val,test,vocab=args
 model,run,curve=train_run(name,seed,lr,train,val,vocab,checkpoint=True)
 prunes=prune_run(model,test,run)
 return run,curve,prunes

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['pilot','tune','evaluate','all'],default='all');args=parser.parse_args()
 train,val,test,vocab=prepare_data();checks=correctness(vocab);dump(OUT/'protocol.json',{'protocol':CFG,'treatments':TREATMENTS,'metadata':metadata()})
 if args.part=='pilot':
  _,run,curve=train_run('dense6',999,.003,train,val,vocab,steps=120);dump(OUT/'pilot.json',{'run':run,'curve':curve});print(json.dumps(run),flush=True);return
 if args.part in ['all','tune']:
  tune=[]
  jobs=[(name,seed,lr,train,val,vocab) for name in TREATMENTS for lr in CFG['learning_rates'] for seed in CFG['tuning_seeds']]
  with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
   for run in pool.map(tuning_job,jobs):
    tune.append(run);print('tune',run['config'],run['lr'],run['seed'],'val',round(run['validation_ce'],4),'seconds',round(run['train_seconds'],2),flush=True);write_csv(OUT/'tuning.csv',tune)
  selected={name:min(CFG['learning_rates'],key=lambda lr:statistics.mean(r['validation_ce'] for r in tune if r['config']==name and r['lr']==lr)) for name in TREATMENTS}
  dump(OUT/'selected_lrs.json',selected);print('selected',selected,flush=True)
 if args.part in ['all','evaluate']:
  selected=json.loads((OUT/'selected_lrs.json').read_text());runs=[];rows=[];curves=[];sweep_start=time.perf_counter()
  jobs=[(name,seed,selected[name],train,val,test,vocab) for seed in CFG['evaluation_seeds'] for name in TREATMENTS]
  with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
   for run,curve,prunes in pool.map(evaluation_job,jobs):
    name=run['config'];seed=run['seed'];runs.append(run);rows.extend(prunes);curves.append({'config':name,'seed':seed,'points':curve});print('eval',name,seed,'full',round(prunes[-1]['ce'],4),'prefix4',round(next(r['ce'] for r in prunes if r['is_prefix'] and r['retained']==4),4),flush=True)
    write_csv(OUT/'evaluation_runs.csv',runs);write_csv(OUT/'pruning.csv',rows)
  summaries,paired=aggregate(rows);tune=list(csv.DictReader((OUT/'tuning.csv').open()));wall=time.perf_counter()-sweep_start
  result={'metadata':metadata(),'protocol':CFG,'treatments':TREATMENTS,'dataset':json.loads((OUT/'dataset.json').read_text()),'selected_lrs':selected,'checks':checks,'runs':runs,'curves':curves,'summaries':summaries,'paired_comparisons':paired,'cost':{'cloud_spend_usd':0,'modal_invocations':0,'local_training_seconds':sum(r['train_seconds'] for r in runs)+sum(float(r['train_seconds']) for r in tune),'evaluation_sweep_elapsed_seconds':wall,'cost_scope':'No paid cloud service launched. Electricity and hardware ownership costs not estimated.'},'sample_counts':{'tuning_runs':len(tune),'evaluation_runs':len(runs),'test_model_subset_evaluations':len(rows)},'limitations':['One tiny repeated corpus and one model family; not a reproduction of 271M–8.2B pretraining.','Only learning rate tuned; small grid; any boundary choice must be disclosed.','Five seeds on one fixed test split; t intervals not multiplicity-adjusted and exclude dataset variation.','Compute-then-mask training; matched training tokens, no total-FLOPs/time/energy claim.','Exact subset average summarizes robustness under a uniform subset distribution, not a learned routing policy.','No generation-speed, calibration, or lossless speculative-decoding claim.']}
  dump(OUT/'result.json',result);dump(OUT/'curves.json',curves);print('DONE',result['sample_counts'],'local seconds',result['cost']['local_training_seconds'],flush=True)
if __name__=='__main__':main()
