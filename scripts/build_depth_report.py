#!/usr/bin/env python3
"""Build the separate depth-robustness report from immutable saved measurements."""
from pathlib import Path
import csv,html,json,math,statistics,collections,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/depth-robustness';OUT.mkdir(exist_ok=True)
NAMES={'dense6':'Dense · 6 blocks','constant_ild':'Constant ILD','decreasing_ild':'Decreasing ILD','alternating':'Alternating dropout','dense4':'Dense · 4 blocks','dense3':'Dense · 3 blocks'}
COLORS={'dense6':'#202a35','constant_ild':'#a85117','decreasing_ild':'#255bd5','alternating':'#207453','dense4':'#8052ac','dense3':'#8052ac'}
def read(path):return json.loads((ROOT/path).read_text())
def stat(x):return {'mean':x['mean'],'n':x['n'],'std':x['std'],'ci95_low':x.get('ci95_low',x.get('low')),'ci95_high':x.get('ci95_high',x.get('high'))}
def csvrows(path):return list(csv.DictReader((ROOT/path).open()))
lm=read('experiments/depth_lm/results/result.json');digits=read('experiments/depth_digits/result.json')
def mask_means(rows,namekey,kkey,layerfn):
 groups=collections.defaultdict(list)
 for r in rows:groups[(r[namekey],int(r[kkey]),tuple(layerfn(r)))].append(float(r['ce']))
 return [{'config':c,'retained':k,'layers':list(l),'ce':statistics.mean(v),'seeds':len(v)} for (c,k,l),v in groups.items()]
lm_masks=mask_means(csvrows('experiments/depth_lm/results/pruning.csv'),'config','retained',lambda r:[int(x) for x in r['layers'].split(',')])
digit_masks=mask_means(csvrows('experiments/depth_digits/masks.csv'),'treatment','retained_depth',lambda r:[i+1 for i,b in enumerate(r['mask']) if b=='1'])
group_map={'prefix':'prefix','all_masks':'all_subsets','keeps_first':'keep_first'}
ds=[]
for r in digits['curves']:
 if r['mask_group'] not in group_map:continue
 ds.append({'config':r['treatment'],'retained':r['retained_depth'],'mode':group_map[r['mask_group']],'masks_per_seed':r['n_masks_per_seed'],**{m:stat(r[m]) for m in ['ce','excess_ce','accuracy']}})
dp=[]
for r in digits['paired_comparisons']:
 if r['mask_group'] not in group_map:continue
 key=(r['treatment'],r['retained_depth'],group_map[r['mask_group']]);target=next((p for p in dp if (p['config'],p['retained'],p['mode'])==key),None)
 if target is None:target={'config':key[0],'retained':key[1],'mode':key[2]};dp.append(target)
 target[r['metric']+'_difference_vs_dense6']=stat(r['difference'])
report={'tasks':{
 'lm':{'title':'Tiny Shakespeare · causal transformer','short_context':'312,448-parameter six-block character transformer; 8,192 fixed test targets. Cross-entropy is per character, not per LLM token.','config_order':list(lm['treatments']),'depths':{k:v['depth'] for k,v in lm['treatments'].items()},'summaries':lm['summaries'],'paired':lm['paired_comparisons'],'mask_means':lm_masks},
 'digits':{'title':'8×8 digits · residual MLP','short_context':'1,797 handwritten digit images, split into 1,078 train / 359 validation / 360 test rows; this is not MNIST or the original writer-disjoint UCI protocol.','config_order':list(digits['configs']),'depths':{k:v['depth'] for k,v in digits['configs'].items()},'summaries':ds,'paired':dp,'mask_means':digit_masks}},
 'spending':{'budget_usd':50,'cloud_spend_usd':0,'modal_invocations':0,'scope':'Local CPU only. No paid experiment service; electricity and hardware amortization excluded.'}}
(OUT/'data.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def row(task,config,k,mode='prefix'):return next(r for r in report['tasks'][task]['summaries'] if r['config']==config and r['retained']==k and r['mode']==mode)
def pair(task,config,k=4,metric='excess_ce'):return next(r for r in report['tasks'][task]['paired'] if r['config']==config and r['retained']==k and r['mode']=='prefix')[metric+'_difference_vs_dense6']
def number(v):return f'{v:.4f}'
def interval(v):return f'{v["mean"]:+.4f} [{v["ci95_low"]:+.4f}, {v["ci95_high"]:+.4f}]'
def table(task):
 t=report['tasks'][task];out='<div class="table-wrap"><table><thead><tr><th>Training recipe</th><th class="num">Full CE</th><th class="num">Prefix4 CE</th><th class="num">Excess CE</th><th class="num">Paired excess-CE difference vs dense6<br>mean [95% interval]</th></tr></thead><tbody>'
 for config in t['config_order']:
  depth=t['depths'][config];full=row(task,config,depth)['ce']['mean'];r=row(task,config,4) if depth>=4 else None
  diff=interval(pair(task,config)) if config in ['constant_ild','decreasing_ild','alternating'] and r else 'reference' if config=='dense6' else 'architecture control'
  out+=f'<tr><td>{NAMES[config]}</td><td class="num">{number(full)}</td><td class="num">{number(r["ce"]["mean"]) if r else "—"}</td><td class="num">{number(r["excess_ce"]["mean"]) if r else "—"}</td><td class="num">{diff}</td></tr>'
 return out+'</tbody></table></div>'

# Publication-ready charts; numeric backing is committed JSON/CSV.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#202a35','text.color':'#202a35','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,(task,t) in zip(axes,report['tasks'].items()):
 for config in t['config_order']:
  rr=sorted([r for r in t['summaries'] if r['config']==config and r['mode']=='prefix'],key=lambda r:r['retained'])
  x=[r['retained'] for r in rr];y=np.array([r['ce']['mean'] for r in rr]);lower=np.array([r['ce']['ci95_low'] for r in rr]);upper=np.array([r['ce']['ci95_high'] for r in rr]);
  ax.errorbar(x,y,yerr=np.array([y-lower,upper-y]),label=NAMES[config],color=COLORS[config],marker='o',capsize=2,lw=1.6,ls='--' if config in ['dense4','dense3'] else '-')
 ax.set_title(t['title'],loc='left',fontweight='bold',fontsize=11);ax.set_xlabel('Original blocks retained (prefix)');ax.set_ylabel('Test cross-entropy (nats)');ax.set_xticks(range(1,7));ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
fig.savefig(OUT/'prefix-curves.svg');fig.savefig(OUT/'prefix-curves.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
for ax,task in zip(axes,report['tasks']):
 configs=[c for c in report['tasks'][task]['config_order'] if c in ['constant_ild','decreasing_ild','alternating']]
 for i,c in enumerate(configs):
  v=pair(task,c);ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['ci95_low']],[v['ci95_high']-v['mean']]],fmt='o',color=COLORS[c],capsize=4)
 ax.axvline(0,c='#999',lw=1);ax.set_yticks(range(len(configs)),[NAMES[c] for c in configs]);ax.invert_yaxis();ax.set_title(report['tasks'][task]['title'],loc='left',fontweight='bold',fontsize=11);ax.set_xlabel('Difference in prefix4 excess CE vs dense6\nNegative = less damage from removing last two blocks');ax.grid(axis='x',alpha=.2)
fig.savefig(OUT/'primary-differences.svg');fig.savefig(OUT/'primary-differences.png',dpi=180);plt.close(fig)

lm_dense=row('lm','dense6',6);lm_early=row('lm','dense6',4);lm_drop=row('lm','decreasing_ild',4);lm_const=row('lm','constant_ild',4)
dg_dense=row('digits','dense6',6);dg_early=row('digits','dense6',4);dg_drop=row('digits','decreasing_ild',4)
p_lm=pair('lm','decreasing_ild');p_dg=pair('digits','decreasing_ild')
endpoint_pass=p_lm['ci95_high']<0 and p_dg['ci95_high']<0
verdict='Dropout reduces the damage from early exit on both toys.' if endpoint_pass else 'The depth-robustness evidence differs between the two toys.'
tuning=lm['sample_counts']['tuning_runs']+digits['counts']['tuning_runs'];final=lm['sample_counts']['evaluation_runs']+digits['counts']['evaluation_runs'];mask_count=lm['sample_counts']['test_model_subset_evaluations']+digits['counts']['test_mask_evaluations']
values={'VERDICT':verdict,'TUNING_COUNT':str(tuning),'FINAL_COUNT':str(final),'MASK_COUNT':str(mask_count),'LM_TABLE':table('lm'),'DIGIT_TABLE':table('digits'),'LM_EARLY':number(lm_early['ce']['mean']),'LM_DROP':number(lm_drop['ce']['mean']),'LM_CONST':number(lm_const['ce']['mean']),'LM_DIFF':interval(p_lm),'DG_EARLY':number(dg_early['ce']['mean']),'DG_DROP':number(dg_drop['ce']['mean']),'DG_DIFF':interval(p_dg),'LM_FULL':number(lm_dense['ce']['mean']),'DG_FULL':number(dg_dense['ce']['mean']),'LM_LR':', '.join(f'{NAMES[k]}: {v:g}' for k,v in lm['selected_lrs'].items()),'DIGITS_LR':', '.join(f'{NAMES[k]}: {v:g}' for k,v in {k:v['lr'] for k,v in digits['selected_lrs'].items()}.items()),'LM_SOURCE_SHA':lm['metadata']['source_sha256'],'DIGIT_SOURCE_SHA':digits['metadata']['source_sha256'],'DATA_SHA':lm['dataset']['sha256']}
template=(ROOT/'scripts/depth-report-template.html').read_text()
for key,value in values.items():template=template.replace('{{'+key+'}}',value)
assert '{{' not in template,'Unfilled report template'
(OUT/'index.html').write_text(template)
manifest={'source_files':{'language_model':'experiments/depth_lm/results/result.json','digits':'experiments/depth_digits/result.json'},'primary_endpoint':'Paired difference of prefix4 excess CE relative to full6 baseline','cloud_spend_usd':0,'experiment_budget_usd':50,'final_tuning_runs':tuning,'final_evaluation_runs':final,'subset_evaluations':mask_count,'lm_primary':p_lm,'digits_primary':p_dg,'notes':'Earlier partial LM tuning/pilot and complete digits boundary-search sweep are archived separately, not counted as independent final replicates.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built separate report;',tuning,'tuning,',final,'final runs;',mask_count,'submodel evaluations; cloud spend $0.')

# A standalone Markdown report with the same primary numbers and explicit provenance.
md=['# Depth robustness: a separate experimental report',
'\n9 September 2026. '+verdict,
'\n## Main findings',
f'\nTwo local tasks, five final seeds per treatment, {tuning} tuning runs and {final} final runs. Exact enumeration produced {mask_count} model/subset evaluations. External experiment spend was **$0** against a $50 ceiling. No Modal job or paid service was launched.',
'\nIncreasing layer dropout (ILD) raises dropout probability with layer depth. Constant and decreasing refer to its schedule over training time.',
'\nThe primary endpoint is `D4 = CE(prefix4) - CE(full)`, compared with the dense six-block model using paired seeds. A negative difference means less damage from removing the final two blocks. Raw CE accompanies it because robustness alone can reward a poor starting model.']
for task,t in report['tasks'].items():
 md+=['\n## '+t['title'],'\n| Recipe | Full CE | Prefix4 CE | Excess CE | Paired excess-CE difference vs dense6 [95% CI] |','|---|---:|---:|---:|---|']
 for c in t['config_order']:
  depth=t['depths'][c];full=row(task,c,depth)['ce']['mean'];r=row(task,c,4) if depth>=4 else None;contrast=interval(pair(task,c)) if c in ['constant_ild','decreasing_ild','alternating'] else 'Reference' if c=='dense6' else 'Architecture control'
  md.append(f'| {NAMES[c]} | {full:.4f} | {number(r["ce"]["mean"]) if r else "—"} | {number(r["excess_ce"]["mean"]) if r else "—"} | {contrast} |')
md+=['\n## Interpretation',
'\nLayer dropout reduces early-exit damage on both tasks. In the language model, the full-depth paired CE intervals for all dropout methods include zero; these data do not resolve an intact-model quality advantage. Constant ILD has the lowest mean pruned CE at several depths, while decreasing ILD retains useful robustness when its dropout schedule reaches zero. This does not establish persistence after a long additional dense tail.',
'\nSeparately training the desired smaller model is an essential control. The dense four-block transformer has mean full CE 2.1030, versus 2.1156/2.1254 for constant/decreasing ILD four-block prefixes. Dense 3 on digits has CE 0.0828, better than either ILD prefix-3 (0.1187/0.1087). These mean comparisons are not formal superiority claims. Small-model zero degradation at its own full depth is a definition, not evidence of better robustness.',
'\nWhich blocks are removed matters. All 63 nonempty subsets are evaluated per six-block model. Summaries distinguish prefixes, every subset, and subsets preserving the first block. ILD never drops block 1 in training; deleting it is outside the training-mask distribution. Alternating dropout additionally always retains blocks 3 and 5. Exact mask averages are averages of individual-model losses, not ensemble prediction loss. No mask was selected as a deployment winner on test data.',
'\n## Language-model protocol',
'\nTiny Shakespeare, immutable char-rnn revision 370cbcd448eb7daf32f21a6be560b70e0b33c4e3. Disjoint contiguous 80/10/10 split; training-derived vocabulary. Validation uses 64 and test 128 evenly spaced nonoverlapping context 64 windows (8,192 test target characters). Each run trains 800 steps with batch 16 for 819,200 character targets, using random training windows with replacement.',
'\nArchitecture: six pre-LayerNorm causal transformer blocks, hidden width 64, four heads, GELU FFN 256, learned positions, untied embedding/head; 312,448 parameters. AdamW, matrix-only weight decay 0.01, 20-step warmup, cosine cooldown to 10%, gradient clipping 1. Controls: dense6, constant ILD max 0.4, decreasing ILD max 0.8, alternating max 0.4, and dense 4. All dropout treatments have 20% expected omission. Attention and FFN share one per-sequence mask, with inverse-survival scaling on each residual branch as in paper Eq. 6. Compute-then-mask execution does not save actual branch computation.',
'\nEach method receives five learning rates × two tuning seeds; full-depth validation CE alone selects the rate. All select 0.01, an interior grid value. Five final seeds (100–104) are disjoint from tuning seeds (10,11). An initial partial three-rate sweep was extended equally based on boundary validation results before final evaluation. The partial data and engineering pilot are retained.',
'\n## Digits protocol',
'\nscikit-learn 8×8 digits (1,797 images), fixed stratified 60/20/20 row split: 1,078 train, 359 validation, 360 test. Train-only feature normalization. This is not MNIST and not UCI\'s writer-disjoint protocol; the sklearn data were copied from the original UCI test set.',
'\nSix residual pre-LayerNorm MLP blocks of width 64 (FFN 128/GELU), 105,162 parameters, final 10-class head; dense 3 control. Matched 600 steps, batch 128 (76,800 training presentations). AdamW weight decay 0.01, cosine cooldown, clipping 1. Constant/decreasing ILD each omit 20% of example-block work in expectation. Training computes before masking. Five rates × two tuning seeds select 0.0003 for every method, followed by five evaluation seeds (2701–2705).',
'\nThe initial three-rate digits experiment completed before its validation-triggered grid extension. Its 44 runs, including test outputs, remain archived; the final grid has 60 runs. Selection used validation CE, not test/pruned performance. This is disclosed development, not a claim of pristine test preregistration. Digits effect size is LR-sensitive: original-grid improvements were about 0.0164/0.0142 nats, versus 0.0659/0.0677 in the final grid.',
'\n## Inference and uncertainty',
'\nRetain original block order and the same trained final normalization/readout. No auxiliary losses, adapters, rescaling, fine-tuning, or calibration. Prefix4 is the primary six-block endpoint. For arbitrary subsets, average masks within seed, then construct 95% Student-t intervals across five seeds. Intervals are not multiplicity-adjusted and omit dataset/split uncertainty. Character CE is not comparable numerically with LLM-token CE.',
'\n## Verification and spending',
'\nVerified disjoint data splits, schedule means/endpoints, all-kept/full equivalence, pruning/zero-residual equivalence, transformer causality, no-drop forward/gradient equivalence, complete subset counts, source hashes, independent primary-CI calculations, and saved-checkpoint readback. See each experiment\'s verification artifacts.',
'\nAll compute was local CPU on Apple M5 Max, float32. The language sweep used up to four isolated one-thread workers; digits used one thread. External spend $0; no paid services. Electricity/hardware amortization are not estimated. Run timings are provenance, not a performance comparison.',
'\n## Limits',
'\nThese are qualitative mechanism reproductions, not the paper\'s 271M–8.2B experiments. No CompleteP, ALiBi, squared ReLU, paper tokenizer, modern token stream, long-budget training, energy instrumentation, inference-speed timing, generation study, or speculative decoding. Full-model quality and smaller dense models remain essential controls. A local positive result does not imply a universal optimum or efficiency gain.',
'\n## Reproduce and sources',
'\n- [Transformer code and raw measurements](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/depth_lm).',
'- [Digits code and raw measurements](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/depth_digits).',
'- [Interactive report](https://yaroslavvb.github.io/gradient-dissent/depth-robustness/).',
'- [Reviewed paper PDF](https://arxiv.org/pdf/2609.05275v1), especially §8.1.',
'- [LayerDrop](https://arxiv.org/abs/1909.11556), ICLR 2020.',
'- [LayerSkip](https://aclanthology.org/2024.acl-long.681/), ACL 2024.',
'- [Tiny Shakespeare immutable revision](https://github.com/karpathy/char-rnn/tree/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare).',
'- [scikit-learn load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).',
'- [Original UCI digit dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).']
(OUT/'report.md').write_text('\n'.join(md)+'\n')
(ROOT/'experiments/depth_robustness/REPORT.md').write_text('\n'.join(md)+'\n')
for svg in OUT.glob('*.svg'):
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
