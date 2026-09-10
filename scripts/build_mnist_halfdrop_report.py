"""Build the 50%-training-dropout report from complete, verified outcomes."""
from pathlib import Path
import hashlib
import json
import shutil
import datetime
import markdown
from build_mnist_halfdrop_data import build as verified_data

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'experiments/ciresan_stochastic_depth/results/halfdrop'
OUT=ROOT/'docs/ciresan-stochastic-depth/training-50'
GIT='https://github.com/yaroslavvb/gradient-dissent/blob/main/'


def main():
    data,gallery_text=verified_data(RAW,ROOT)
    assert data['status']=='complete' and data['verification']['complete_runs']==48
    assert len(data['models'])==32 and len(data['paths'])==24
    registry=json.loads((RAW/'wandb-upload-manifest.json').read_text())
    assert registry['complete'] and registry['verified_run_count']==48
    assert registry['entity']=='yaroslavvb' and registry['project']=='gradient-dissent'
    data['wandb_runs']={r['scientific_run_id']:r['verified_wandb_url'] for r in registry['runs']}
    assert len(data['wandb_runs'])==48 and all(data['wandb_runs'].values())
    closing=json.loads((RAW/'budget-closing.json').read_text())
    assert closing['complete'] and closing['main_runs_complete']==48 and closing['all_owned_apps_stopped']
    dashboard=json.loads((RAW/'wandb-report-manifest.json').read_text())
    assert dashboard['status']=='verified' and dashboard['verified_runs']==48 and dashboard['panel_count']==8
    assert dashboard['registry_sha256']==hashlib.sha256((RAW/'wandb-upload-manifest.json').read_bytes()).hexdigest()
    figures=json.loads((RAW/'figure-manifest.json').read_text())
    assert figures['analysis_sha256']==hashlib.sha256((RAW/'analysis.json').read_bytes()).hexdigest()
    for figure in figures['outputs'].values():
        assert hashlib.sha256((RAW/figure['file']).read_bytes()).hexdigest()==figure['sha256']
    data['publication_provenance']={
        'wandb_registry_sha256':hashlib.sha256((RAW/'wandb-upload-manifest.json').read_bytes()).hexdigest(),
        'budget_closing_sha256':hashlib.sha256((RAW/'budget-closing.json').read_bytes()).hexdigest(),
        'wandb_report_sha256':hashlib.sha256((RAW/'wandb-report-manifest.json').read_bytes()).hexdigest(),
        'figure_manifest_sha256':hashlib.sha256((RAW/'figure-manifest.json').read_bytes()).hexdigest()}
    data['wandb_report_url']=dashboard['verified_report_url']
    OUT.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    (OUT/'data.json').write_text(encoded+'\n')
    (OUT/'data.js').write_text('window.HALFDROP_DATA = '+encoded+';\n')
    for name,content in gallery_text.items():
        (OUT/name).write_text(content)
    for figure in figures['outputs'].values():
        shutil.copyfile(RAW/figure['file'],OUT/figure['file'])
    get=lambda e,m:next(r for r in data['models'] if r['endpoint']==e and r['drop_mask']==m)
    base,all_drop=get('selected',0),get('selected',15)
    interval=all_drop['paired_vs_none']['accuracy_pp']
    p=lambda v:f'{100*v:.3f}%'
    chosen=next(o for o in data['paths'] if o['validation_selected'])
    order=' → '.join(str(j+1) for j in chosen['order'])
    findings=f'<p>At the primary checkpoint, making all four middle branches droppable gives <strong>{p(all_drop["metrics"]["accuracy"]["mean"])}</strong> full-network test accuracy, versus <strong>{p(base["metrics"]["accuracy"]["mean"])}</strong> without training dropout. The paired change is <strong>{interval["mean"]:+.3f} percentage points</strong> (95% interval {interval["ci95_low"]:+.3f} to {interval["ci95_high"]:+.3f}).</p><p>The order chosen using validation is <strong>{order}</strong>. Explore every subset and all 24 orders below; selection on validation does not guarantee the best test ordering.</p>'
    final_base,final_drop=get('final',0),get('final',15)
    f,b=final_drop['metrics'],final_base['metrics']
    saving=100*(1-f['training_seconds']['mean']/b['training_seconds']['mean'])
    findings+=f'<p>At epoch 100, accuracy is <strong>{p(f["accuracy"]["mean"])}</strong> versus <strong>{p(b["accuracy"]["mean"])}</strong>; its paired interval also includes zero. Measured training time falls <strong>{b["training_seconds"]["mean"]:.1f} → {f["training_seconds"]["mean"]:.1f} seconds</strong> ({saving:.1f}% less), while test cross-entropy worsens <strong>{b["ce"]["mean"]:.4f} → {f["ce"]["mean"]:.4f}</strong>. Similar accuracy hides both new mistakes and repaired errors.</p>'
    overview=(ROOT/'research/mnist-halfdrop-training.md').read_text()
    report=overview+'\n\n'+(RAW/'summary.md').read_text()
    report+='\n\n[W&B comparison dashboard]('+dashboard['verified_report_url']+') · [Download the accuracy graph (SVG)](accuracy-vs-drop-count.svg) · [PNG](accuracy-vs-drop-count.png)\n'
    report+='\n\n## Individual W&B runs\n\nAll 48 runs were uploaded after training and read back to verify their scientific histories. Plot against measured `epoch` or `training_seconds`; W&B importer runtime is not GPU training time.\n\n| Droppable branches | Seed | Run |\n|---|---:|---|\n'
    for row in sorted(registry['runs'],key=lambda r:r['scientific_run_id']):
        source=json.loads((RAW/(row['scientific_run_id']+'.json')).read_text())
        spec=source['spec'];mask=spec['drop_mask']
        label=', '.join(str(i+1) for i in range(4) if mask&(1<<i)) or 'none'
        report+=f'| {label} | {spec["seed"]} | [{row["scientific_run_id"]}]({row["verified_wandb_url"]}) |\n'
    report+='\n\n## Execution and reproducibility\n\n'
    report+=(ROOT/'research/mnist-halfdrop-execution.md').read_text()
    report+=f'\n\n[Full analysis]({GIT}experiments/ciresan_stochastic_depth/results/halfdrop/analysis.json) · [Validation-order freeze]({GIT}experiments/ciresan_stochastic_depth/results/halfdrop/order-manifest.json) · [Protocol]({GIT}experiments/ciresan_stochastic_depth/halfdrop/PROTOCOL.md) · [Closing billing and shutdown receipt]({GIT}experiments/ciresan_stochastic_depth/results/halfdrop/budget-closing.json)\n'
    (OUT/'report.md').write_text(report)
    template=(ROOT/'scripts/mnist-halfdrop-template.html').read_text()
    page=template.replace('<!-- FINDINGS -->',findings).replace('<!-- WANDB_REPORT -->',dashboard['verified_report_url']).replace('<!-- REPORT -->',markdown.markdown(report,extensions=['tables','fenced_code']))
    assert '<!-- FINDINGS -->' not in page and '<!-- REPORT -->' not in page
    (OUT/'index.html').write_text(page)
    files=['index.html','style.css','app.js','data.js','data.json','report.md']+list(data['gallery_files'].values())+[f['file'] for f in figures['outputs'].values()]
    manifest={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_analysis_sha256':hashlib.sha256((RAW/'analysis.json').read_bytes()).hexdigest(),
        'outputs':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built complete 48-run half-drop report with verified W&B links.')


if __name__=='__main__':main()
