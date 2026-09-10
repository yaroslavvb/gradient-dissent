"""Publish a read-back-verified report for the complete 48-run half-drop group."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.ciresan_stochastic_depth.telemetry.build_wandb_report import read_visibility

ROOT=Path(__file__).resolve().parents[1]/'results/halfdrop'
ENTITY='yaroslavvb';PROJECT='gradient-dissent';GROUP='ciresan-halfdrop-20260909'
TITLE='MNIST: 50% training dropout, full-depth testing · 48 runs'
MARKER='gradient-dissent-halfdrop-48-v1'


def signature(report):
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces.reports.v2.interface import expr
    def metric(value):
        if isinstance(value,list):return [metric(v) for v in value]
        if isinstance(value,str) or value is None:return value
        # The SDK round-trip wraps ordinary history-field strings as Metric.
        # Config and SummaryMetric remain distinct namespaces.
        if type(value) is wr.Metric:return value.name
        return {'type':type(value).__name__,'name':value.name}
    grids=[]
    for block in report.blocks:
        if not isinstance(block,wr.PanelGrid):continue
        runsets=[]
        for r in block.runsets:
            filters=getattr(r,'_stashed_filters_v2',None)
            if filters is None:filters=expr.filters_tree_to_v2(expr.expr_to_filters(r.filters))
            runsets.append({'entity':r.entity,'project':r.project,'filters':filters})
        panels=[]
        for p in block.panels:
            row={'type':type(p).__name__,'title':p.title,'x':metric(p.x),'y':metric(p.y)}
            for k in ('z','aggregate','smoothing_type','max_runs_to_show','regression'):
                if hasattr(p,k):row[k]=metric(getattr(p,k)) if k=='z' else getattr(p,k)
            panels.append(row)
        grids.append({'runsets':runsets,'panels':panels})
    return {'entity':report.entity,'project':report.project,'title':report.title,'description':report.description,'grids':grids}


def make_report(registry):
    import wandb_workspaces.reports.v2 as wr
    records=registry['runs'];source_ids=sorted(r['scientific_run_id'] for r in records)
    colors=['#6c8794','#177c84','#268565','#b38125','#ad6251']
    byid={s['run_id']:s for s in json.loads((ROOT/'main-manifest.json').read_text())}
    runset=wr.Runset(entity=ENTITY,project=PROJECT,name='All 16 eligibility subsets × 3 paired seeds',
        filters=f"Group == {GROUP!r} and State == 'finished' and Config('scientific_run_id') in {source_ids!r}",
        custom_run_colors={r['wandb_run_id']:colors[byid[r['scientific_run_id']]['drop_mask'].bit_count()] for r in records},
        visible_columns=['config:eligible_count.value','config:eligible_mask.value','config:seed.value',
                         'summary:selected/test_accuracy_pct','summary:final/test_accuracy_pct','summary:time/training_seconds'],
        pinned_columns=['run:displayName'],lock_columns=True)
    panels=[]
    for title,y in [('Primary full-test accuracy (%) vs droppable branch count','selected/test_accuracy_pct'),
                    ('Epoch 100 full-test accuracy (%) vs droppable branch count','final/test_accuracy_pct'),
                    ('100-epoch training seconds vs droppable branch count','time/training_seconds'),
                    ('Epoch 100 test CE vs droppable branch count','final/test_loss')]:
        i=len(panels)
        panels.append(wr.ScatterPlot(title=title,x=wr.Config('eligible_count'),y=wr.SummaryMetric(y),
            z=wr.Config('eligible_mask'),regression=False,range_x=(-.2,4.2),
            layout=wr.Layout(x=12*(i%2),y=9*(i//2),w=12,h=9)))
    for title,metric in [('Validation accuracy over training','validation/accuracy_pct'),
                         ('Validation cross-entropy over training','validation/loss'),
                         ('Dense full-training-set accuracy','train/accuracy_pct'),
                         ('Stochastic minibatch training loss','train/stochastic_minibatch_loss')]:
        i=len(panels)
        panels.append(wr.LinePlot(title=title,x='epoch',y=[metric],title_x='Measured completed epoch',
            smoothing_type='none',smoothing_factor=0.,aggregate=False,groupby=None,
            max_runs_to_show=48,ignore_outliers=False,groupby_rangefunc='none',
            layout=wr.Layout(x=12*(i%2),y=9*(i//2),w=12,h=9)))
    return wr.Report(entity=ENTITY,project=PROJECT,title=TITLE,description=MARKER+'; exact verified 48-run group; train-time eligibility only.',width='fluid',blocks=[
        wr.MarkdownBlock(text='Every eligible middle branch is dropped with probability **50% during training**. **Every layer runs at testing.** Each point is a separately trained run: all 16 subsets × seeds 201–203, same residual architecture, learning rate 0.01, batch 64, 100 epochs. Colors in the learning curves indicate the number of eligible branches. No smoothing or pooled confidence bands. [Interactive paired-example and order report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/training-50/).'),
        wr.P('The primary checkpoint minimizes validation cross-entropy; epoch 100 is secondary. The first four charts use frozen endpoint summaries; the remaining charts use measured training histories. The official test set is evaluated only after training and checkpoint selection. Training time excludes telemetry and setup; W&B runtime is the post-training uploader. Three seeds and an already reused test set limit interpretation.'),
        wr.PanelGrid(runsets=[runset],panels=panels)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    registry_path=ROOT/'wandb-upload-manifest.json';registry=json.loads(registry_path.read_text())
    assert registry['complete'] and registry['verified_run_count']==48 and registry['group']==GROUP
    assert registry['entity']==ENTITY and registry['project']==PROJECT
    assert all(r['status']=='verified' and r['verification']['finished'] and r['verified_wandb_url'] for r in registry['runs'])
    desired=make_report(registry);expected=signature(desired)
    (ROOT/'wandb-report-plan.json').write_text(json.dumps(expected,indent=2)+'\n')
    if not args.publish:print('Eight W&B panels prepared; no network writes.');return
    import wandb
    import wandb_workspaces.reports.v2 as wr
    api=wandb.Api(timeout=30);existing=[]
    for i,r in enumerate(api.reports(ENTITY+'/'+PROJECT,per_page=50)):
        if i>=200:raise ValueError('Report inventory exceeds bounded duplicate check')
        if r.display_name==TITLE:
            if MARKER not in (r.description or ''):raise ValueError('Conflicting report title')
            existing.append(r)
    if len(existing)>1:raise ValueError('Ambiguous existing report')
    if existing:
        loaded=wr.Report.from_url(existing[0].url);loaded.blocks=desired.blocks;loaded.description=desired.description;desired=loaded
    desired.save(draft=False,clone=False)
    actual=wr.Report.from_url(desired.url)
    if signature(actual)!=expected:raise ValueError('Published report axis/filter read-back mismatch')
    receipt={'status':'verified','verified_report_url':actual.url,'panel_count':8,'verified_runs':48,
             'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest(),
             'signature':expected,'visibility':read_visibility(api)}
    (ROOT/'wandb-report-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'verified','report_url':actual.url,'panels':8}))


if __name__=='__main__':main()
