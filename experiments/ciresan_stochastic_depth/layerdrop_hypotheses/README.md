# Fable-inspired MNIST layer-drop audit

[Interactive results](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/) · [Experimental report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/report.md) · [Mathematical audit](../../../research/fable-layerdrop-audit.md)

This is an inference audit of existing optimized-run checkpoints. No neural model is retrained. See [PROTOCOL.md](PROTOCOL.md) for the pre-evaluation hypotheses, fixed masks, validation fit/calibration split, statistical choices and caveats. Every source checkpoint uses the normalized-input/linear-head controlled cohort, not the separate test-target baseline search.

Five training methods × three seeds × two endpoints = 30 states. Every state receives all 16 masks on all 10,000 validation and 10,000 test examples. The 18 central residual/SD states also receive a fixed 128-example validation mechanism probe. Selected checkpoints are primary; epoch100 is secondary. Plain MLP deletion is deliberately labeled forced crop surgery. Official test, validation and seed reuse make the study exploratory.

## Reanalyze saved results (no paid compute)

From the repository root, with NumPy, Torch2.14 and scikit-learn1.9 installed:

```bash
experiments/.venv/bin/python experiments/ciresan_stochastic_depth/layerdrop_hypotheses/analyze.py \
  --phase evaluate \
  --reference-dir experiments/ciresan_stochastic_depth/results/hypotheses/source-results
experiments/.venv/bin/python experiments/ciresan_stochastic_depth/layerdrop_hypotheses/export_report.py --ready
npm run build:layerdrop
npm run check:layerdrop
PYTHONPATH=experiments/ciresan_stochastic_depth experiments/.venv/bin/python \
  -m unittest discover -s experiments/ciresan_stochastic_depth/layerdrop_hypotheses -p 'test_*.py'
```

The existing policy manifest is never overwritten. `--phase fit` was run once, without opening test arrays; it fits 30 depth-three image-only trees and freezes validation-defined class bins. `--phase evaluate` verifies that manifest, source hashes and validation files before test access. Original remote training results are retained in `source-results/`; the earlier local launch summaries contain an additional dispatch-time field and therefore have different byte hashes.

Raw artifacts live in `../results/hypotheses/`. Each audit folder contains `data.npz` (49 image features, labels, validation permutation, raw test images), 20 split-separated mask NPZs, 10 checkpoint JSON summaries and an integrity manifest. NPZ masks are ordered0..15, bit0=first body branch present. Arrays retain predictions, float64 per-example CE, margins, per-example Walsh energy and higher-order fractions. Mechanism JSONs retain all probe examples, gradients, norms, angles, finite differences and fractional-gain errors. Full network weights remain on the preexisting Modal volume and are not published.

## Repeating cloud work

The completed launch specifications and source hashes are in `audit-manifest.json`, `audit-freeze.json`, `routing-manifest.json` and `routing-freeze.json`. The four calls used explicit `A100`, two CPU cores,8GiB RAM, no retries,180-second timeouts and2-second scale-down. They share the immutable original $30 budget ledger. **Do not rerun these IDs:** they are already reserved and the remote claim files prevent overwrites. Any future paid repeat needs a new run ID and sufficient remaining reservation budget; the current four-call envelope is complete.

`hypothesis_app.py` owns reservation/dispatch; `audit.py` performs deterministic unscaled deletion and dense-reference parity. `routing_benchmark.py` times actual image features, decision trees, grouping, gathers/scatters and skipped affine calls. No cached-logit simulation is labeled GPU latency. The old model/training/kernel files are unchanged.

## Interpretation limits

Three seeds are three model replicates; masks and examples do not increase that count. Paired t intervals are exploratory and unadjusted for multiple comparisons. A validation extra-error allowance is not a test/population guarantee. MAC reduction is not latency or energy. Small finite-difference probes exhibit numerical instability even through the linear head, and late-checkpoint losses approach FP32 resolution; retain these failed diagnostics instead of claiming precise Jacobian estimates.
