# Ciresan-width MNIST: speed optimization and stochastic depth

Status on 2026-09-09: **complete**. Both final linear-output/unit-dropout baseline recipes reached98.63% in all three confirmation runs. The controlled study completed15 tuning attempts,15 main evaluation runs and9 source-style controls. Three high-LR tuning attempts diverged and remain reported. Stochastic depth reduced100-epoch training time by about14.4%, but did not improve primary mean test accuracy. The revised final scope uses three paired seeds; the five-seed initial plan remains archived in the [protocol](PROTOCOL.md).

[Interactive baseline optimization report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/optimization/) · [Controlled experiment report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/)

The two experiments answer different questions. The speed search fits the full official 60,000-example training pool and uses the official test set to guide recipes and stopping. The controlled stochastic-depth comparison uses a fixed 50,000/10,000 fitting/validation split, with its test-loading gate preserved during tuning. The speed search means MNIST test outcomes are already known to the research process; reused confirmation seeds do not restore a pristine, previously unseen benchmark.

## Source provenance and declared changes

The reference is Yaroslav Bulatov's public [train_ciresan_new.py](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py), with [util.py](https://github.com/yaroslavvb/stuff/blob/master/autotune/util.py). Local snapshots are in [reference/](reference/); [history/source-audit.json](history/source-audit.json) records source hashes. The [historical audit](history/README.md) explains the public W&B records and source behavior. This is a Ciresan-style tapered MLP, **not** a reproduction of the original augmentation-heavy Ciresan record-setting system.

All current variants retain six affine layers with biases, widths **784 → 2500 → 2000 → 1500 → 1000 → 500 → 10**, and **11,972,510 trainable parameters**. They use PyTorch's default `nn.Linear` initialization. There is no image augmentation or normalization layer.

| Detail | Public source snapshot | Stable normalized speed recipe | Controlled stochastic-depth study |
|---|---|---|---|
| Pixels | Raw float, 0–255 | Divide by 255 | Divide by 255 |
| Hidden activation | ReLU | ReLU, then unit dropout 0.2 | ReLU; treatment-specific dropout |
| Output | ReLU after final affine | Linear logits | Linear logits |
| Architecture | Plain affine chain | Same plain chain | Plain control plus matched residual variants |
| Training pool / batch | 60,000 / 64 | 60,000 / 256 | 50,000 / 64 |
| Initial learning rate | 0.001 | 0.12 | Tune 0.01, 0.03, 0.1 equally |
| Momentum | 0.9 | 0.9 | 0.9 |
| Direct shrinkage per update | 0.00002 | 0.00008 initially | Zero |
| Training arithmetic | Historical device/runtime not established | BF16 autocast, FP32 parameters | FP32 parameters and TF32 matrix multiplies |

Source shrinkage means `p *= 1 - shrinkage` **after** the SGD update, for all parameters including biases. It is not an SGD `weight_decay` argument. At epoch 21, the stable recipe multiplies both its initial LR and initial shrinkage by 0.1, giving LR 0.012 and shrinkage 0.000008. The graph is recaptured while preserving model and optimizer state. The raw-pixel stable alternative uses the same recipe except `input_scale=1` and initial LR 0.004.

Original-source options remain available in [model_data.py](model_data.py): `input_scale=1`, `output_relu=True`, `recipe='plain'`. The [local collapse diagnostic](results/collapse-diagnostic.json) and [script](diagnose_collapse.py) investigate the raw-input residual/ReLU-head failure without silently replacing it. An entirely inactive ReLU output head has zero instantaneous cross-entropy gradient; existing momentum can still move its parameters.

## Confirmed baseline results

The target is the first scheduled official-test score with **at most 137 errors out of 10,000**, i.e. at least 98.63%. These confirmations evaluate every epoch and stop at the first qualifying score or 100 epochs. At batch 256, each shuffled epoch makes 234 updates and presents 59,904 examples from the 60,000-example pool; the incomplete final minibatch is discarded.

The final [selection record](optimization/stable-selection.json) and [six-run confirmation manifest](optimization/stable-confirmation-manifest.json) lock two recipes and seeds 101–103. The normalized recipe has the shortest observed training-loop times among these confirmations:

| Seed / raw result | GPU reported by PyTorch | First target epoch | Test accuracy | Training to target | Invocation to score | Dispatch to returned result |
|---|---|---:|---:|---:|---:|---:|
| [101](results/opt-confirm-normalized-dropout-step-s101.json) | A100 80GB PCIe | 24 | 98.67% | 5.609 s | 17.236 s | 28.590 s |
| [102](results/opt-confirm-normalized-dropout-step-s102.json) | A100-SXM4-40GB | 23 | 98.63% | 5.819 s | 15.592 s | 25.146 s |
| [103](results/opt-confirm-normalized-dropout-step-s103.json) | A100-SXM4-40GB | 21 | 98.71% | 5.320 s | 14.897 s | 22.777 s |

The raw-pixel linear-head alternative also succeeded in 3/3 seeds: epochs 32/30/42, training 7.414/7.599/9.677 seconds, and invocation-to-score 18.294/15.807/20.551 seconds. Its [seed 101](results/opt-confirm-raw-linear-dropout-step-s101.json), [seed 102](results/opt-confirm-raw-linear-dropout-step-s102.json), and [seed 103](results/opt-confirm-raw-linear-dropout-step-s103.json) results report GPU variants separately. These are observations on three seeds, not guarantees or hardware-normalized comparisons.

The earlier raw-input/ReLU-head candidate in [selected-baseline.json](optimization/selected-baseline.json) reached the target in only **1/5** fresh seeds by 200 epochs; its same-seed repeat is separate. That earlier selection is retained as evidence, not treated as the reliable final recipe. All exploratory successes, censored attempts, compiler failures, preceding observations and timing definitions appear in the [experimental report](../../research/ciresan-optimization.md) and [machine-readable analysis](results/optimization-analysis.json).

There is a narrower implementation comparison: matched raw/ReLU, batch-64, FP32/TF32, seed-1 runs on reported A100-SXM4-40GB took **248.926 s eager versus 106.164 s full CUDA graph plus fused SGD** for 150 epochs, a 2.345× training-loop speedup with identical scored test trajectories. Neither reached the target. That supports a kernel execution speedup; the much faster normalized/BF16 recipe additionally changes optimization and regularization.

### What each clock includes

- `training_seconds`: synchronized minibatch loops, including index gathering, graph-input copies, forward/backward, SGD and shrinkage. Excludes epoch permutation, evaluation, checkpoint writes, imports, data loading and graph preparation/recapture.
- `run_wall_seconds`: invocation start before benchmark import through availability of the score, including setup, data transfer, all prior training/evaluation and schedule recapture. Excludes subsequent writes for that score, dispatch/container startup, image build and final volume commit.
- `total_run_seconds`: through the invocation's final timer after post-score work; still excludes dispatch/startup and final volume commit.
- `local_dispatch_elapsed_seconds`: dispatch to returned result, including queue/startup and result transfer. This is not the exact instant the target became available.

Container/cache reuse was not reliably recorded. No cold-start, energy, invoice, or direct speedup over historical W&B runtime is claimed. The historical [ts4k9n55](https://wandb.ai/yaroslavvb/train_ciresan/runs/ts4k9n55) first reached 98.63% at logged epoch 94 and 388.4867 seconds of W&B runtime; its hardware and timing scope do not establish a controlled speed comparison.

## Completed controlled stochastic-depth comparison

The graph trainer preserves the original controlled adaptation: 100 epochs, batch 64, FP32/TF32, SGD momentum 0.9, constant LR, no shrinkage, and five arms: plain, residual-dense, residual constant SD (`pmax=.4`), residual decreasing SD (`pmax=.8`), and residual unit dropout (`p=.2`). The final comparison uses **three paired seeds**, with equal three-rate tuning on seed 1. The frozen graph-study manifests specify15 tuning calls and24 evaluation calls:15 main runs and9 fidelity controls. Do not replay the old five-seed/fidelity counts as a current launch plan.

For the four middle transitions, the residual operator is `ReLU(h[:out_width] + M * Linear(h) / (1-p))`. Stem and head are mandatory. The fixed crop introduces no parameters but discards coordinates, so this is a substantive architectural change requiring the residual-dense control. One Bernoulli draw per transition is shared by the minibatch. A dropped branch's affine operation is skipped, and its SGD momentum is frozen. Inference retains all branches with scale one. Unit dropout instead acts independently on activations after each hidden ReLU, including shortcut activations in the residual arm.

Depthwise probabilities are `pmax * [1,2,3,4]/4`. Decreasing SD multiplies these by `1 - epoch/(E-1)` for zero-based epochs. Both schedules average 20% omission by hidden-affine-layer count over the full horizon; this is neither a 20% parameter reduction nor a promised wall-clock speedup. The preactivation branch expectation is preserved by inverted scaling; the nonlinear network output need not have the same expectation.

[load_mnist](model_data.py) uses a fixed random permutation with seed 20260909, taking the first 50,000 official training examples for fitting and the last 10,000 for validation; the split is not stratified. It verifies the OSSCI-hosted MNIST gzip files against known MD5s and records SHA-256 and split digests. `include_test=False` does not request or open test files. Batch 64 gives 781 updates and 49,984 presentations per epoch. Initialization, example order, and SD mask RNG streams are separate; paired methods share initialization and data order within seed.

LR selection uses lowest validation cross-entropy across scheduled checkpoints; each final seed's validation-selected checkpoint is the primary endpoint, with the fixed last-epoch model secondary. Official test scores do not select these controlled-study LRs or checkpoints. Seed-paired intervals, when available, must use the three seeds as replicates, not the repeated test examples. All24 final runs passed data, pairing, source and endpoint checks. Primary mean test accuracies were98.373% residual dense,97.803% constant SD and97.997% decreasing SD. At epoch100 they were98.400%,98.443% and98.510%; both SD accuracy-difference intervals include zero. The full report gives both metrics and endpoints.

## Implementations and reproduction

Run commands from the repository root. The local tests need Python, PyTorch and NumPy; the Modal wrappers pin Python 3.11, `torch==2.14.0`, and `numpy==2.2.6`. Qualification reports PyTorch `2.14.0+cu130`. Use the legacy-compatible `torch.set_float32_matmul_precision('high')` setting; mixing the newer backend precision setters with compiler legacy getters caused an earlier retained failure.

```bash
experiments/.venv/bin/python -m unittest \
  experiments.ciresan_stochastic_depth.test_model_data \
  experiments.ciresan_stochastic_depth.optimization.test_kernels \
  experiments.ciresan_stochastic_depth.optimization.test_stochastic_graph -v
```

Current local verification: **23 discovered, 19 passed, 4 CUDA-only tests skipped**. The tests never provision GPUs. They cover source-style initialization and gradients, split/test-loading boundaries, all-kept and all-skipped branches, actual branch omission, deterministic inference, dropout randomness, SGD/shrinkage parity, and graph state restoration. The [earlier baseline A100 kernel qualification](results/opt-kernels-214-v2.json) passed its then-current four tests. The [stable normalized baseline run](results/opt-stable-normalized-dropout-step-s1.json) also passed all five current baseline kernel tests on A100, including fresh graph-replay unit dropout. The [new SD A100 qualification](results/graph-qualification-v1.json) passed all six current stochastic-graph tests, including changing masks/scales, frozen inactive momentum, restored initialization/RNG, and fresh reproducible unit-dropout masks on graph replay. Its five-epoch pilot completed without test scoring; it does not replace final experimental validation.

| Entry point | Purpose and authoritative inputs |
|---|---|
| [optimization_app.py](optimization_app.py) → [optimization/benchmark.py](optimization/benchmark.py) | Full-60k test-target speed search; manifests under [optimization/](optimization/), especially `stable-confirmation-manifest.json`. |
| [graph_app.py](graph_app.py) → [train_optimized.py](train_optimized.py) | Optimized controlled study; [graph-qualification-manifest.json](graph-qualification-manifest.json) is the completed qualification input. |
| [modal_app.py](modal_app.py) → [train.py](train.py) | Archived original PyTorch 2.8 pilot/tuning implementation and its earlier manifests. |
| [optimization/kernels.py](optimization/kernels.py) | Eager, compiled-model, or full-step CUDA graph baseline; graph mode includes SGD and source shrinkage. |
| [optimization/stochastic_graph.py](optimization/stochastic_graph.py) | Up to 16 mask-specific graphs sharing parameter/gradient/momentum storage; active-only updates and an epoch-updated survival-scale buffer. |

For a new paid reproduction, first copy the desired JSON-list manifest and assign new `run_id` values. Existing IDs are reserved permanently and are intentionally rejected. Keep scientific settings unchanged when claiming a replication. Example launch forms, using that new manifest:

```bash
modal run -e gradient-dissent-ciresan \
  experiments/ciresan_stochastic_depth/optimization_app.py \
  --stage pilot --manifest /tmp/new-baseline-confirmations.json

modal run -e gradient-dissent-ciresan \
  experiments/ciresan_stochastic_depth/graph_app.py \
  --stage tune --manifest /tmp/new-graph-tuning.json
```

Manifest stages must match `--stage`. These commands use paid A100 resources; the wrappers reserve each call before dispatch, with explicit timeouts and no automatic retries. The shared Modal volume holds downloaded data and checkpoints; result JSON files are copied into `results/`. Checkpoints and datasets are not committed. Source hashes, initialization hashes, data hashes, executed data-order/mask digests, hardware, and kernel metadata are recorded where supported by each implementation; earlier results are not retroactively assigned new-source hashes.

To regenerate the optimization analysis locally without training:

```bash
experiments/.venv/bin/python \
  experiments/ciresan_stochastic_depth/optimization/summarize.py
```

This writes the aggregate JSON and research Markdown from raw results. Its optional `--finalize` flag explicitly checks and closes the declared search; it is not a launch command. Original results and failed attempts remain the evidence base.

## Spending evidence

The isolated Modal environment is `gradient-dissent-ciresan`. All three wrappers share [results/budget-ledger.json](results/budget-ledger.json): configured phase cap **$30**, dispatch reservation ceiling **$24**, and a **$6** buffer for build, egress, billing lag, and interruption overhead. Each call reserves its maximum timeout plus 130 seconds of startup/idle allowance before submission. Reservations are never released, including failures; a metering monitor requests cancellation at $24. Resource limits are two CPU cores and 8 GiB RAM per worker, at most six A100 workers, no scheduled training, and no automatic retries.

After all15 experiment apps stopped, metering recorded **$3.9768** at **2026-09-09T22:58:39.245245+00:00**. The ledger conservatively reserved **$23.1036**. Metering is a possibly delayed usage snapshot, not a final invoice; reservations are conservative dispatch bounds, not spend. This includes all current MNIST work and excludes earlier separate studies. No GPU apps remain running; [the final app inventory](results/modal-apps-final.json) records their stopped state.

Regenerate the completed controlled analysis and both reports without paid calls:

```bash
python3 experiments/ciresan_stochastic_depth/analyze.py --final --require-complete
npm run build:ciresan
npm run check:ciresan
```
