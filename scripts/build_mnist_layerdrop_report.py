"""Build the hypothesis report from frozen outcomes. No training/cloud calls."""
from pathlib import Path
import hashlib
import json
import statistics
from datetime import datetime, timezone
import markdown

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/ciresan-stochastic-depth/hypotheses'
RAW=ROOT/'experiments/ciresan_stochastic_depth/results/hypotheses'
GIT='https://github.com/yaroslavvb/gradient-dissent/blob/main/'
EXP=GIT+'experiments/ciresan_stochastic_depth/'

def main():
    analysis=json.loads((RAW/'analysis.json').read_text())
    assert analysis['verification']['passed'] and analysis['verification']['state_count']==30
    timing=json.loads((RAW/'routing-summary.json').read_text())
    billing=json.loads((RAW/'billing-latest.json').read_text())
    opening=json.loads((RAW/'opening-billing-latest.json').read_text())
    ledger=json.loads((RAW.parent/'budget-ledger.json').read_text())
    assert ledger['reserved_upper_usd']<24
    source=(ROOT/'research/mnist-layerdrop-hypotheses.md').read_text()
    extra=['','## Measured A100 latency: static masks win this implementation','',
           'The nine primary selected-checkpoint models were timed on one **NVIDIA A100-SXM4-40GB**, using PyTorch 2.14.0+cu130, CUDA 13.0, FP32 tensors with TF32-enabled `high` matmul precision. The full 10,000-image test set is already on the GPU. Batch size is 2,048; every method receives two warmup batches, then three synchronized whole-test passes in rotating order. The table uses the median of all three passes, in **milliseconds per 10,000 images**. It is throughput timing, not single-image or cold-start latency.','',
           'The input-dependent implementation includes average pooling, the decision tree, sorting, GPU-to-CPU mask counts, per-group gathers, genuine skipped affine calls and output scattering. The random comparator includes grouping but has preassigned masks. Checkpoint loading, initial host-to-device data transfer, fitting/calibration and verification are outside these inference timers. No compilation or graph-capture cost is hidden.','',
           '| Method / seed | Dense ms | Static ms | Routed ms | Same-cost random ms | Static test accuracy | Routed test accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in timing['models']:
        m=row['methods'];ms=lambda name:f"{1000*m[name]['primary_median_seconds']:.3f}"
        extra.append(f"| {row['recipe']} / {row['seed']} | {ms('dense')} | {ms('static')} | {ms('routed')} | {ms('random_cost_matched')} | {100*m['static']['accuracy']:.2f}% | {100*m['routed']['accuracy']:.2f}% |")
    med=timing['median_across_nine_models_ms'];slow=timing['routed_slowdown_range'];speed=timing['static_speedup_range']
    extra+=['',f"Across the nine per-model medians, dense inference takes **{med['dense']:.2f} ms**, static masks **{med['static']:.2f} ms**, and image-dependent routing **{med['routed']:.2f} ms**. The router is **{slow[0]:.2f}–{slow[1]:.2f}× slower** than dense inference on every model. Static masks are **{speed[0]:.2f}–{speed[1]:.2f}× faster**, at the accuracy/cost points shown above. These masks were selected using validation; several exceed the empirical 0.2pp error tolerance on test, so speed is not an unconditional quality-preservation claim.",'',
           'All nine CPU/GPU routing decisions and all 36 method/checkpoint prediction vectors match the exact-mask functional references. Observed numerical disagreement count is zero. Each raw timing pass is retained. These three repetitions on one assigned GPU do not constitute an independent hardware benchmark or a tuned lower bound on routing overhead. A fused or coarse-bucket router could behave differently; that remains untested.','',
           f"[Raw timing run]({EXP}results/hypothesis-routing-v1.json) · [Compact timing summary]({EXP}results/hypotheses/routing-summary.json) · [Timing implementation]({EXP}layerdrop_hypotheses/routing_benchmark.py)",'',
           '## Compute cost and execution','']
    jobs=[json.loads((RAW.parent/f'hypothesis-audit-s{s}-v1.json').read_text()) for s in (101,102,103)]
    extra+=['| Audit seed | GPU | Function execution seconds | Dispatch to result seconds |','|---|---|---:|---:|']
    for j in jobs:
        extra.append(f"| {j['spec']['seed']} | {j['hardware']['gpu']} | {j['elapsed_seconds']:.2f} | {j['local_dispatch_elapsed_seconds']:.2f} |")
    delta=billing['metered_cost_usd']-opening['metered_cost_usd'];stamp=datetime.fromtimestamp(billing['queried_unix'],timezone.utc).isoformat()
    extra+=['',f"The fourth routing job took {timing['remote_seconds']:.2f} seconds inside the function and {timing['dispatch_seconds']:.2f} seconds dispatch-to-result. All four paid calls completed; all cloud apps are stopped. No neural training was performed. The shared MNIST environment’s metered snapshot rose from **${opening['metered_cost_usd']:.5f} to ${billing['metered_cost_usd']:.5f}**, an increase of about **${delta:.2f}**. Snapshot checked {stamp}; this is metering, not a final invoice.",'',
           f"The original **$30** cap still applies. Immutable worst-case reservations rose from $23.10357 to **${ledger['reserved_upper_usd']:.5f}**, including $0.77743 for these four 180-second A100 allocations plus startup/idle allowances. Old reservations were not released or reset. Reservations and actual metering are different quantities.",'',
           f"[Billing snapshot]({EXP}results/hypotheses/billing-latest.json) · [Shared reservation ledger]({EXP}results/budget-ledger.json) · [Stopped-app check]({EXP}results/hypotheses/apps-stopped.json)",'',
           '## Reproduce without new compute','',
           'From the repository root, use the existing Python environment with NumPy, PyTorch and scikit-learn. The stored policy manifest is immutable: evaluation verifies its validation data and source hashes before reading test arrays. The CPU full-width qualification used 32 original validation examples only to check API/runtime; its outputs are excluded from the experiment.','',
           '```bash','experiments/.venv/bin/python experiments/ciresan_stochastic_depth/layerdrop_hypotheses/analyze.py \\','  --phase evaluate \\','  --reference-dir experiments/ciresan_stochastic_depth/results/hypotheses/source-results',
           'experiments/.venv/bin/python experiments/ciresan_stochastic_depth/layerdrop_hypotheses/export_report.py --ready',
           'python3 scripts/build_mnist_layerdrop_report.py',
           'node experiments/ciresan_stochastic_depth/layerdrop_hypotheses/test_page.cjs','```','',
           f"[Frozen experimental protocol]({EXP}layerdrop_hypotheses/PROTOCOL.md) · [Pre-audit hashes/specifications]({EXP}layerdrop_hypotheses/audit-freeze.json) · [Frozen policies]({EXP}results/hypotheses/policy-manifest.json) · [Full analysis]({EXP}results/hypotheses/analysis.json) · [Independent verification]({EXP}results/hypotheses/independent-check.json)",
           '',f"[Fable’s argument: mathematical audit]({GIT}research/fable-layerdrop-audit.md) · [Executed counterexamples]({EXP}results/hypotheses/theory-counterexamples.json)",'']
    report=source+'\n'.join(extra)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.md').write_text(report)
    findings='<p>Stochastic depth makes missing layers less damaging. At the validation-selected checkpoints, the fraction of nonconstant logit interaction energy in higher-order terms falls from <strong>13.31% to about 2.3%</strong>. At epoch 100, keeping two of four body branches still gives <strong>98.34–98.35% accuracy</strong>.</p><p>A fixed reduced network is fast. This image-dependent router adds enough grouping overhead to become slower than the full network, despite using fewer nominal MACs.</p>'
    verdicts='''<h2>Five hypotheses. Five different answers.</h2><div class="verdict-grid"><div><p class="badge">H1 · partly supported</p><h3>Training with omissions buys tolerance</h3><p>Decreasing SD reduces selected-checkpoint deletion loss by 0.0588 nats versus residual training; the three-seed interval excludes zero. Constant SD has the same direction, with an interval crossing zero.</p></div><div><p class="badge">H2 · unresolved</p><h3>There is no stable “easy digit” rule</h3><p>Digit 1’s apparent advantage weakens after margin adjustment. All primary adjusted intervals include zero; final decreasing SD reverses the ordering.</p></div><div><p class="badge">H3 · local support</p><h3>Direction predicts more than size</h3><p>At selected checkpoints, gradient-based sensitivity tracks damage better than increment norm or angle. Final-checkpoint rank correlations deteriorate; finite-difference Gaussian controls are numerically unresolved.</p></div><div><p class="badge">H4 · supported, with limits</p><h3>Layer effects become more additive</h3><p>Exact binary-mask interactions fall sharply under SD. Real fractional-gain interpolation still has error, so this does not establish an affine network or an “if and only if” theorem.</p></div><div><p class="badge">H5 · no practical win here</p><h3>The cheap router is not cheap enough</h3><p>Only 4 of 9 primary routers keep the test error increase within 0.2 percentage points. All nine run 2.07–3.25× slower than full inference with this implementation.</p></div></div>'''
    template=(ROOT/'scripts/mnist-layerdrop-template.html').read_text()
    page=template.replace('<!-- FINDINGS -->',findings).replace('<!-- VERDICTS -->',verdicts).replace('<!-- REPORT -->',markdown.markdown(report,extensions=['tables','fenced_code']))
    (OUT/'index.html').write_text(page)
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'policy_manifest_sha256':analysis['verification']['policy_manifest_sha256'],
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'research/mnist-layerdrop-hypotheses.md',RAW/'analysis.json',RAW/'routing-summary.json',RAW/'billing-latest.json']},
              'outputs':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['index.html','report.md','data.json','data.js','app.js','style.css']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built MNIST layer-drop report from verified frozen results.')

if __name__=='__main__':main()
