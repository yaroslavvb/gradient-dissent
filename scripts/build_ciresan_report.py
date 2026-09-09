#!/usr/bin/env python3
"""Package measured learning curves and a readable report; never launches training."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import markdown

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/ciresan_stochastic_depth'
OUT = ROOT / 'docs/ciresan-stochastic-depth'
parser = argparse.ArgumentParser()
parser.add_argument('--final', action='store_true')
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
files = list((EXP / 'results').glob('pilot-*.json'))
if args.final:
    manifest = json.loads((EXP / 'final-manifest.json').read_text())
    files += [EXP / 'results' / (s['run_id'] + '.json') for s in manifest]
runs = []
raw = []
for path in files:
    r = json.loads(path.read_text())
    assert 'error' not in r and not r.get('diverged'), path
    raw.append({'run_id': r['run_id'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    runs.append({k: r[k] for k in ['run_id','spec','history','best_epoch','best_validation_loss','training_seconds','total_run_seconds','hardware','initial_parameters_sha256','parameter_count','steps_per_epoch','skipped_batch_counts','final_train','selected_train','final_test','selected_test'] if k in r})
payload = {'final': args.final, 'runs': runs, 'raw_files': raw,
           'analysis': json.loads((EXP / 'results/analysis.json').read_text()) if args.final else None}
serialized = json.dumps(payload, separators=(',', ':'), allow_nan=False)
(OUT / 'data.json').write_text(serialized + '\n')
(OUT / 'data.js').write_text('window.CIRESAN_DATA=' + serialized + ';\n')
if args.final:
    source = (ROOT / 'research/ciresan-stochastic-depth.md').read_text()
else:
    source = '''## What the pilots established

Six measured A100 pilots completed five training epochs each. The source-style
plain network reached **97.88% validation accuracy**, but adding residual shortcuts
to the raw-pixel, ReLU-output variant produced **all-zero logits and 10.22% accuracy**.
The normalized, linear-logit variants all learned. No official test files were
opened in these pilots. The stochastic-depth sweep is paused while the baseline is optimized on A100s to reach the historical 98.63% target faster.

## The controlled comparison

The original widths are **784 → 2500 → 2000 → 1500 → 1000 → 500 → 10**. Each model has
11,972,510 parameters. The four middle transitions gain a fixed prefix-crop bypass;
the input stem and classifier remain mandatory. A dropped branch skips its matrix
multiplication. Evaluation restores all branches without training-time scaling.

The study separates the architectural change from dropout: plain network,
residual control, constant stochastic depth, decreasing stochastic depth, and
ordinary unit dropout. Learning-rate selection uses validation data only.
'''
(OUT / 'report.md').write_text(source)
md = markdown.Markdown(extensions=['tables','fenced_code','toc'])
body = md.convert(source).replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
title = 'Stochastic depth in your MNIST MLP'
status = '100 epochs · 3 paired seeds · A100' if args.final else 'Measured pilots · baseline speed optimization in progress'
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Controlled Ciresan-width MNIST experiments: what stochastic depth changes about fitting, generalization, and runtime."><title>''' + title + ''' · Gradient dissent</title><link rel="stylesheet" href="style.css"></head><body><a class="skip" href="#main">Skip to results</a><header><a class="brand" href="../">GRADIENT DISSENT</a><nav aria-label="Related reports"><a href="hypotheses/">Which digits can skip layers?</a><a href="optimization/">A100 speed results</a><a href="../stochastic-depth-history/">Stochastic depth history</a><a href="../dropout-animation/">How dropout works</a></nav></header><main id="main"><div class="heading"><p class="eyebrow">''' + status + '''</p><h1>''' + title + '''</h1><p id="headline"></p></div><section class="explorer" aria-labelledby="curve-title"><div class="chart-heading"><h2 id="curve-title">Watch the training curves</h2><p>Every line comes from a recorded run.</p></div><div class="controls"><label>Experiment<select id="cohort"><option value="main">Controlled adaptation · three seeds</option><option value="fidelity">Historical source settings · three seeds</option><option value="pilot">Five-epoch pilots</option></select></label><label>Measure<select id="metric"><option value="validation_accuracy">Validation accuracy</option><option value="validation_loss">Validation cross entropy</option><option value="train_accuracy">Training-probe accuracy</option><option value="train_loss">Training-probe cross entropy</option><option value="stochastic_loss">Stochastic training loss</option></select></label><label>Horizontal axis<select id="xaxis"><option value="epoch">Epochs</option><option value="seconds">Training seconds</option></select></label><label class="check"><input type="checkbox" id="individual" checked>Show individual seeds</label></div><div id="legend" class="legend"></div><svg id="curves" viewBox="0 0 1000 410" role="img" aria-labelledby="svg-title svg-desc"><title id="svg-title">Measured MNIST learning curves</title><desc id="svg-desc">Select an experiment and metric. Bold lines show seed means; thin lines show individual runs.</desc></svg><p id="curve-note" class="note"></p><div class="table-wrap"><table id="curve-table"><caption>Last recorded validation scores</caption><thead><tr><th>Method</th><th>Runs</th><th>Validation accuracy</th><th>Validation CE</th><th>Training time</th></tr></thead><tbody></tbody></table></div></section><article id="report">''' + body + '''</article><footer><a href="report.md" download>Download report</a><a href="data.json" download>Measured curves and results</a><a href="https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth">Code, protocol &amp; raw data ↗</a></footer></main><script src="data.js"></script><script src="app.js"></script></body></html>'''
(OUT / 'index.html').write_text(page)
(OUT / 'manifest.json').write_text(json.dumps({'final':args.final,'data_sha256':hashlib.sha256((OUT/'data.json').read_bytes()).hexdigest(),'raw_files':raw},indent=2)+'\n')
print(f'Built Ciresan report: {len(runs)} measured runs, final={args.final}')
