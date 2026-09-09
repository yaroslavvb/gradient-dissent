#!/usr/bin/env python3
"""Build the measured A100 time-to-quality report without launching compute."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import markdown

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/ciresan_stochastic_depth'
OUT = ROOT / 'docs/ciresan-stochastic-depth/optimization'
OUT.mkdir(parents=True, exist_ok=True)
GITHUB = 'https://github.com/yaroslavvb/gradient-dissent/blob/main/'
analysis = json.loads((EXP / 'results/optimization-analysis.json').read_text())
runs = []
for path in sorted((EXP / 'results').glob('opt-*.json')):
    r = json.loads(path.read_text())
    if r.get('spec', {}).get('kind') != 'accuracy' or 'error' in r:
        continue
    keep = ['run_id','spec','history','threshold','hardware','setup_seconds','training_seconds',
            'total_run_seconds','local_dispatch_elapsed_seconds','parameter_count','last_class_diagnostics']
    runs.append({**{k:r[k] for k in keep if k in r}, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
fresh = [r for r in runs if r['spec'].get('confirmation_group') == 'normalized-dropout-step']
assert len(fresh) == 3 and all(r['threshold']['test']['errors'] <= 137 for r in fresh)
payload = {'runs':runs, 'analysis':analysis}
data = json.dumps(payload, separators=(',',':'), allow_nan=False)
(OUT / 'data.json').write_text(data+'\n')
(OUT / 'data.js').write_text('window.OPTIMIZATION_DATA='+data+';\n')
source = (ROOT / 'research/ciresan-optimization.md').read_text()
source = re.sub(r'\]\(\.\./(experiments/[^)]+)\)', lambda m: ']('+GITHUB+m[1]+')', source)
(OUT / 'report.md').write_text(source)
body = markdown.markdown(source, extensions=['tables','fenced_code','toc'])
body = body.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
rows = ''
for r in fresh:
    t = r['threshold']
    rows += f'''<tr><td>{r['spec']['seed']}</td><td>{r['hardware']['gpu'].replace('NVIDIA ','')}</td><td>{t['epoch']}</td><td>{100*t['test']['accuracy']:.2f}%</td><td>{t['training_seconds']:.2f}</td><td>{t['run_wall_seconds']:.2f}</td><td>{r['local_dispatch_elapsed_seconds']:.2f}</td></tr>'''
train = statistics.median(r['threshold']['training_seconds'] for r in fresh)
wall = statistics.median(r['threshold']['run_wall_seconds'] for r in fresh)
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>98.63% MNIST in seconds · A100 optimization</title><meta name="description" content="Measured Ciresan-width MNIST optimization on A100: CUDA Graphs, fused SGD, BF16, and three confirmation runs."><link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="style.css"><script defer src="data.js"></script><script defer src="app.js"></script></head><body>
<a class="skip" href="#main">Skip to report</a><header><a class="brand" href="../../">GRADIENT DISSENT</a><nav><a href="../">Stochastic-depth experiment</a><a href="../../stochastic-depth-history/">Research lineage</a><a href="https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth">Code &amp; runs</a></nav></header>
<main id="main"><p class="eyebrow">A100 · MNIST · 11.97 million parameters · measured September 9, 2026</p><h1>98.63% in seconds.<br>Speed comes from kernels <em>and</em> training.</h1>
<p id="headline">The original Ciresan widths reach the requested accuracy on all three confirmation seeds. The recommended recipe takes <strong>WALL seconds from invocation to score</strong> at the median, including setup and evaluation. Its synchronized training loop takes <strong>TRAIN seconds</strong>.</p>
<div class="stats"><div><strong>WALL s</strong><span>Median invocation → score</span></div><div><strong>TRAIN s</strong><span>Median training loop</span></div><div><strong>3 / 3</strong><span>Confirmation seeds reached 98.63%</span></div><div><strong>2.345×</strong><span>Matched kernel-only speedup</span></div></div>
<p class="note">Three assigned A100s include 40GB SXM4 and 80GB PCIe devices; these medians describe the observed service runs, not a hardware-controlled recipe comparison. The matched kernel comparison below uses the same reported 40GB SXM4 GPU model and unchanged learning curves.</p>
<section class="explorer" aria-labelledby="curve-title"><div class="chart-heading"><h2 id="curve-title">Watch time turn into accuracy</h2><p>Every point is a measured epoch</p></div><div class="controls"><label>Comparison<select id="group"><option value="recommended">Recommended recipe · confirmation seeds</option><option value="kernels">Kernel-only · identical recipe</option><option value="fragile">Original output ReLU · confirmation seeds</option><option value="raw">Linear output + dropout · raw pixels</option></select></label><label>Time axis<select id="clock"><option value="run_wall_seconds">Invocation to score · seconds</option><option value="training_seconds">Training loop · seconds</option><option value="epoch">Epoch</option></select></label><label>Accuracy range<select id="zoom"><option value="detail">Detail · 97–99%</option><option value="full">Full · 0–100%</option></select></label></div><div id="legend" class="legend"></div><svg id="chart" viewBox="0 0 1000 400" role="img" aria-label="Measured accuracy learning curves with a 98.63 percent target"></svg><p id="chart-note" class="note" aria-live="polite"></p><details><summary>Accessible chart data</summary><div class="table-wrap"><table id="chart-data"></table></div></details></section>
<h2>The actual times, without hiding setup</h2><div class="table-wrap"><table><caption>First qualifying epoch, three confirmation seeds. Seconds; all models use the same recommended recipe.</caption><thead><tr><th>Seed</th><th>Assigned A100</th><th>Epoch</th><th>Test accuracy</th><th>Training s</th><th>Invocation → score s</th><th>Dispatch → result s</th></tr></thead><tbody>ROWS</tbody></table></div>
<p class="note">Training excludes shuffling, initialization, evaluation and graph capture. Invocation-to-score includes those costs and imports; it starts inside the remote function, after container startup. Dispatch-to-result includes queueing, startup, output and final persistence. Cache/container reuse was not reliably recorded; none of these is claimed to be a controlled cold-start measurement.</p>
<section class="recipe"><h2>The recommended baseline</h2><p><strong>784 → 2500 → 2000 → 1500 → 1000 → 500 → 10.</strong> Same six affine layers, 11,972,510 parameters. Hidden ReLUs, linear output logits, pixels divided by 255. Ordinary unit dropout at probability 0.2 after all five hidden ReLUs.</p><p>SGD momentum 0.9, batch 256, learning rate 0.12, direct parameter shrinkage 0.00008 per update. At the start of epoch 21, multiply both learning rate and shrinkage by 0.1. FP32 parameters and momentum, BF16 autocast, fused SGD, full-step CUDA Graphs. PyTorch 2.14.0, CUDA 13, Triton 3.8. Data stay on the GPU. Each epoch shuffles the 60,000-example pool and uses 59,904 examples in complete batches; the remainder changes with the shuffle.</p><p>This is an optimized Ciresan-width recipe with explicit input, head and regularization changes. It uses no augmentation. Test accuracy was monitored and used for tuning and stopping. Seeds 101–103 had already been examined with the earlier raw-input/ReLU recipe; these are confirmation runs, not untouched holdout seeds. This is a time-to-quality study, not an unbiased generalization estimate.</p></section>
<section class="takeaways"><h2>What made the difference?</h2><div class="findings"><div><h3>Remove launch overhead</h3><p>Full-step graph replay plus fused SGD cut identical 150-epoch work from 248.93 to 106.16 training seconds. Both traces peaked at 98.61%, so this comparison establishes speed, not time to the target.</p></div><div><h3>Fix the output head</h3><p>With the original output ReLU, only 1 of 5 confirmation seeds reached the target. A matched seed-104 diagnostic gave digit 6 recall of 0/958; changing only the head to linear raised it to 98.64% after five epochs.</p></div><div><h3>Reach useful accuracy sooner</h3><p>Dropout and the combined LR/shrinkage schedule reached the target in fewer epochs in exploratory runs. BF16 formed part of the selected batch-256 recipe; at batch 64, its matched microsteps were slower.</p></div></div><p>The tested compiler candidate needed 229.41 seconds of setup. For this short job, full-step CUDA Graphs were the practical choice. That compiler test compiled the model, with optimizer orchestration outside; it was not an exhaustive comparison of compiler strategies.</p></section>
<details class="audit"><summary>Read the complete audit: all attempts, failures, clocks, sources and caveats</summary><article>BODY</article></details>
<footer><a href="data.json">Measured curves &amp; analysis JSON</a><a href="report.md">Full report in Markdown</a><a href="https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth/optimization">Reproduce the benchmarks</a><a href="../">Stochastic-depth comparison</a></footer></main></body></html>'''
page = page.replace('WALL',f'{wall:.2f}').replace('TRAIN',f'{train:.2f}').replace('ROWS',rows).replace('BODY',body)
(OUT/'index.html').write_text(page)
print(f'Built {len(runs)} measured accuracy traces; {len(fresh)} recommended confirmations')
