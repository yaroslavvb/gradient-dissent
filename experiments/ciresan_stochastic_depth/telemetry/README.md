# Instrumented Ciresan reruns

This adds epoch-boundary diagnostics to the already optimized CUDA-graph trainers. It does not change the frozen original trainers or retune the scientific recipes. Detailed instrumentation is fixed at epochs0,1 and every10, with a common128-example training probe. Original evaluation cadence is retained.

## Reproduce

The checked-in manifests retain all measured source hashes. The original source files must match those hashes; changing an instrumented trainer requires a new qualification and new run IDs.

```bash
# Local correctness checks; no cloud charges.
experiments/.venv/bin/python -m unittest \
  experiments.ciresan_stochastic_depth.telemetry.test_metrics \
  experiments.ciresan_stochastic_depth.telemetry.test_curvature \
  experiments.ciresan_stochastic_depth.telemetry.test_offline_curvature \
  experiments.ciresan_stochastic_depth.telemetry.test_analyze \
  experiments.ciresan_stochastic_depth.telemetry.test_wandb_export

# Paid A100 runs require the existing Modal account/environment and unused IDs.
modal run -e gradient-dissent-ciresan experiments/ciresan_stochastic_depth/telemetry_app.py --stage pilot --manifest experiments/ciresan_stochastic_depth/telemetry/qualification-v2-manifest.json
modal run -e gradient-dissent-ciresan experiments/ciresan_stochastic_depth/telemetry_app.py --stage evaluate --manifest experiments/ciresan_stochastic_depth/telemetry/main-manifest.json
modal run -e gradient-dissent-ciresan experiments/ciresan_stochastic_depth/telemetry_app.py --stage pilot --manifest experiments/ciresan_stochastic_depth/telemetry/baseline-manifest.json
modal run -e gradient-dissent-ciresan experiments/ciresan_stochastic_depth/telemetry_app.py --stage evaluate --manifest experiments/ciresan_stochastic_depth/telemetry/curvature-manifest.json

# Read-only aggregation and static report generation.
python3 experiments/ciresan_stochastic_depth/telemetry/analyze.py --require-complete
python3 experiments/ciresan_stochastic_depth/telemetry/build_page.py
node experiments/ciresan_stochastic_depth/telemetry/test_page.cjs
```

The paid commands above document the exact invocations. They deliberately refuse duplicate run IDs. Do not reset the existing budget ledger or overwrite old outputs to rerun them: make an explicitly budgeted new phase with new IDs and reviewed source pins.

## Results and statistics

See `../results/telemetry/` for all18 reruns, both qualification attempts, historical W&B data, offline curvature results and post-training W&B exports. Large model snapshots remain on the existing Modal volume, with hashes in each training result. The snapshot files can be regenerated from deterministic recipes; the checked-in scientific histories and measurements are directly usable without Modal.

[METRIC_PLAN.md](METRIC_PLAN.md) maps the historical logger to corrected modern definitions, including omitted buggy or unjustified values. [WANDB.md](WANDB.md) explains optional authenticated replay. No network logging occurs while the training clock runs.

Full training/validation/test scores, per-class confusion statistics, activation/backprop summaries, parameter/gradient/update norms and exact fixed-probe per-example gradient moments are available in the main runs. Sparse offline snapshots add Fisher, CE-GGN, Jacobian-Gram, KFAC factors/spectra, gradient-noise contractions and output mismatch diagnostics. Exact probe diagonals are distinguished from KFAC approximations; none of these is a full network Hessian.

## Interpretation

The15 controlled reruns retain the original50k/10k split, five recipes, seeds101–103 and validation checkpoint rule. Test data are evaluated repeatedly for diagnostic curves, without changing the frozen recipe. The three fast baseline runs use60k training images, a test-selected recipe and test-threshold stopping. These are repeated measurements of previously used seeds/data, not new independent generalization evidence.

Synchronized training time, diagnostic time, invocation elapsed and Modal dispatch clocks are separate. Historical W&B `_runtime` is not isolated GPU training time. Qualification compares instrumentation on/off on the same device and checks bitwise identical learned parameters; comparisons to older result files verify their recorded scientific histories, initialization, data and RNG digests rather than claiming unavailable old final parameter hashes.
