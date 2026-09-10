# 50% training dropout, full-depth testing

This folder contains the exhaustive MNIST follow-up: every subset of four residual branches, three paired seeds, and 100 epochs per run. All six learned affine layers are used at testing. Read [PROTOCOL.md](PROTOCOL.md) for the frozen design and [the interactive report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/training-50/) for results.

## Files

- `trainer.py`: optimized A100/TF32 CUDA-graph training, per-minibatch branch skipping, paired random streams, validation-only checkpoint selection, and full test predictions.
- `../halfdrop_app.py`: bounded Modal dispatch, source verification, budget reservations, and no automatic training retries.
- `plot.py`: standalone SVG/PNG accuracy figures with seed-level uncertainty.
- `analyze.py`: separate validation-order freeze and test-analysis phases; all 16 subsets, 24 orders, 32 marginal additions, and paired example comparisons.
- `upload_wandb.py`: post-training upload to `yaroslavvb/gradient-dissent`; content-derived run identities and read-back verification of every exported scalar.
- `build_wandb_report.py`: eight W&B panels for endpoint comparisons and training histories.
- `../results/halfdrop/`: frozen manifests, qualification and transport receipts, scalar histories, selected/final test arrays, analysis, order selection, and budget closure. No credentials are needed to inspect the committed results.

## Rebuild from recorded results

From the repository root, use the existing uv environment:

```bash
uv sync
uv run python experiments/ciresan_stochastic_depth/halfdrop/analyze.py --phase evaluate --manifest experiments/ciresan_stochastic_depth/results/halfdrop/main-manifest.json
uv run python experiments/ciresan_stochastic_depth/halfdrop/plot.py
uv run python scripts/build_mnist_halfdrop_data.py
uv run python scripts/build_mnist_halfdrop_report.py
npm run check:halfdrop
```

The recorded order manifest is required. Do not reselect an order based on test accuracy. Its validation inputs and fixed reference-panel indices are hash-bound to the committed artifacts. The archival MNIST input data referenced by the analyzer remains part of the earlier layer-drop study in this repository.

Local tests do not launch paid compute:

```bash
uv run python -m unittest experiments.ciresan_stochastic_depth.halfdrop.test_trainer experiments.ciresan_stochastic_depth.test_halfdrop_app experiments.ciresan_stochastic_depth.halfdrop.test_analysis experiments.ciresan_stochastic_depth.halfdrop.test_upload_wandb
node experiments/ciresan_stochastic_depth/halfdrop/test_page.cjs --fixture
```

For a new training replication, create a new immutable run namespace, freeze the desired manifest and executed source hashes, qualify the GPU kernels, and use the existing launcher's budget guards. The committed launch receipts are an archive of an already completed cohort; the launcher intentionally refuses to overwrite or silently relaunch it. All MNIST wrappers share the existing budget ledger.

## W&B

The original training process does not communicate with W&B. Upload after training with:

```bash
uv run python experiments/ciresan_stochastic_depth/halfdrop/upload_wandb.py --upload
uv run python experiments/ciresan_stochastic_depth/halfdrop/build_wandb_report.py --publish
```

These commands use local W&B authentication, preserve scientific run IDs, and reconcile existing content before any continuation. Plot against `epoch` or measured `training_seconds`; the W&B run runtime is the uploader's runtime. All eligible configurations are included, including unfavorable results.

## Checkpoints

The full prediction arrays and hashes are committed. The ~48 MB weight files remain on the owner's Modal volume, with their hashes and exact paths in each `*-download.json` receipt. For example, an authenticated owner can retrieve the no-dropout seed-201 validation-selected checkpoint:

```bash
uv run modal volume get -e gradient-dissent-ciresan gradient-dissent-ciresan-20260909 /runs/halfdrop-m00-s201-v1/best.pt ./halfdrop-m00-s201-best.pt
```

`best.pt` is the validation-selected checkpoint, `final.pt` is epoch 100, and `checkpoint.pt` is a copy of `best.pt`. These volume files are not public GitHub downloads.
