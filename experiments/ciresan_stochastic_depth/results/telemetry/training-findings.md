# Training findings from the instrumented reruns

All 18 declared reruns verified. The 15 controlled runs reproduce every original recorded scientific loss/accuracy value, initialization hash, training-order digest, and layer-mask digest. The three optimized-baseline reruns reproduce their recorded curves, initialization and stopping checkpoints. Old final parameter hashes were not recorded, so matching final weights to those old runs is not claimed. Both qualification attempts separately established byte-identical on/off final parameters and complete scientific histories. These are instrumented repeats of existing seeds, not 18 new independent replications.

Controlled settings remain the original widths, normalized pixels, linear logits, 50,000/10,000 fitting/validation split, batch 64 FP32/TF32, 100 epochs and frozen learning rates: .03 for plain and .01 for all residual recipes. Ordinary dropout is .2; SD schedules remain frozen. Dense diagnostics disable both unit dropout and branch removal. They use the same fixed first 128 fitting examples, with detailed measurements at epoch 0, epoch 1 and every 10 epochs. No metric alters training or validation-CE checkpoint selection; diagnostic official-test evaluations are repeated and disclosed. Three seeds reuse the same datasets and probe.

## Quality and checkpoint dependence

Each cell is mean ± sample SD over seeds 101–103 (n=3); accuracy is percent, CE is mean cross-entropy. Selected means the unchanged minimum-validation-CE checkpoint. Final means epoch 100. Values below use the original scientific reductions, which the reruns reproduce exactly.

| Recipe | Selected epochs | Selected test accuracy | Selected test CE | Final test accuracy | Final test CE |
|---|---|---:|---:|---:|---:|
| Plain | 5, 10, 10 | 97.913 ± 0.332 | 0.0853 ± 0.0092 | 98.573 ± 0.025 | 0.1116 ± 0.0064 |
| Residual | 10, 10, 10 | 98.373 ± 0.067 | 0.0738 ± 0.0072 | 98.400 ± 0.000 | 0.0972 ± 0.0056 |
| Constant SD | 5, 5, 10 | 97.803 ± 0.179 | 0.0790 ± 0.0056 | 98.443 ± 0.031 | 0.1336 ± 0.0225 |
| Annealed SD | 15, 5, 10 | 97.997 ± 0.289 | 0.0725 ± 0.0035 | 98.510 ± 0.046 | 0.1170 ± 0.0159 |
| Residual unit dropout | 15, 15, 10 | 98.330 ± 0.142 | 0.0714 ± 0.0042 | 98.507 ± 0.115 | 0.1031 ± 0.0055 |

Constant SD minus residual accuracy is -0.570 percentage points at the selected checkpoint and +0.043 at epoch 100. The apparent accuracy ranking therefore depends on the endpoint. Both SD recipes have higher final test CE than residual; higher final top-1 accuracy alone is not an across-metric improvement.

Annealed SD minus residual accuracy is -0.377 percentage points at the selected checkpoint and +0.110 at epoch 100. The apparent accuracy ranking therefore depends on the endpoint. Both SD recipes have higher final test CE than residual; higher final top-1 accuracy alone is not an across-metric improvement.

All 12 plain/residual/SD runs reach 100% dense fitting accuracy by epoch 100. Their mean full-fitting CE values are Plain 1.37e-06, Residual 7.64e-06, Constant SD 2.14e-06, Annealed SD 4.51e-06. Residual unit dropout ends at 99.967% mean fitting accuracy, with full-fitting CE spanning 4.56e-06–0.0044. These training curves show near-complete fitting despite branch removal; they do not support a blanket claim that these SD settings prevent overfitting.

## What the fixed-probe gradients show

For the final affine head (layer 5), g_i is the per-example weight gradient of unaveraged CE. The reported gradient norm is ||mean_i g_i||, and diversity is D=mean_i||g_i||² / ||mean_i g_i||². The largest-example energy share is max_i||g_i||² / sum_i||g_i||², reconstructed from the recorded maximum norm and second moment. Gradients exclude bias here. The table uses all three seeds; gradient norms and D are seed means, energy shares are seed ranges.

| Recipe | Head gradient norm epoch 1 → 100 | D epoch 1 → 100 | Largest-example energy share at 100 |
|---|---:|---:|---:|
| Plain | 0.146 → 6.55e-05 | 112.3 → 127.9 | 99.61–100.00% |
| Residual | 0.259 → 0.000162 | 116.3 → 127.9 | 89.55–98.99% |
| Constant SD | 0.188 → 8.35e-06 | 156.3 → 127.6 | 60.25–100.00% |
| Annealed SD | 0.17 → 0.000106 | 131.8 → 128.0 | 99.98–100.00% |
| Residual unit dropout | 0.191 → 0.00122 | 143.6 → 138.8 | 58.26–99.99% |

A diversity ratio near 128 is not evidence that 128 examples supply equally useful independent gradients: one dominant example alone produces D≈128. The energy shares above and extremely small median gradient norms indicate substantial concentration as the probe becomes confidently fitted. Small denominators and FP32 arithmetic further limit late-stage ratio interpretation. These observations describe one fixed training probe; they neither identify a population mechanism nor make diversity a validated predictor of test accuracy. Neighbor-gradient cosines refer to cyclic neighbors in this fixed dataset order, not neighboring optimization steps.

## Activations and class-specific behavior

Mean fraction of zero post-ReLU activations at epoch 100, over the fixed probe and then the three seeds. Layers 0–4 are the five hidden activations; the linear head is excluded. A zero on this probe does not prove a permanently dead neuron.

| Recipe | Layer0 | Layer1 | Layer2 | Layer3 | Layer4 |
|---|---:|---:|---:|---:|---:|
| Plain | 83.9% | 68.5% | 60.5% | 59.3% | 59.5% |
| Residual | 64.0% | 43.0% | 32.3% | 28.5% | 29.6% |
| Constant SD | 68.3% | 62.4% | 63.8% | 64.4% | 56.9% |
| Annealed SD | 73.0% | 68.6% | 71.9% | 74.2% | 73.2% |
| Residual unit dropout | 84.0% | 65.2% | 58.5% | 55.2% | 53.6% |

Both SD recipes have sparser dense probe activations than the residual control at these frozen settings; annealed SD has roughly 69–74% zero activations across its hidden layers at epoch 100. This association coexists with near-perfect fitting and does not by itself explain generalization. The plain model also differs architecturally and uses another learning rate.

Epoch100 class-specific test-accuracy differences from residual (percentage points; mean across seeds) are shown for every digit, avoiding a selected best/worst-class account. Classes have different fixed MNIST test counts, so equal-weight averages of these ten cells need not equal the overall accuracy difference.

| Recipe − residual | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant SD | +0.000 | +0.147 | +0.129 | +0.033 | +0.034 | -0.037 | +0.313 | +0.000 | -0.137 | -0.066 |
| Annealed SD | +0.170 | +0.029 | +0.129 | +0.297 | +0.136 | -0.037 | +0.104 | +0.259 | +0.068 | -0.066 |
| Residual unit dropout | +0.170 | +0.059 | -0.258 | +0.099 | +0.136 | +0.299 | +0.487 | +0.357 | -0.034 | -0.198 |

Small overall differences hide mixed class effects. These repeated-test descriptive slices have no multiple-comparison correction and cannot establish digit-specific causal benefits.

## Timing and measurement overhead

| Recipe | Mean training s | Mean telemetry s | Total telemetry / training range |
|---|---:|---:|---:|
| Plain | 62.130 | 1.458 | 2.12–2.79% |
| Residual | 66.218 | 1.899 | 1.30–5.94% |
| Constant SD | 56.849 | 1.753 | 1.51–5.57% |
| Annealed SD | 56.859 | 1.628 | 2.03–4.53% |
| Residual unit dropout | 68.716 | 1.100 | 1.24–2.26% |
| Optimized baseline | 5.716 | 0.913 | 14.13–18.54% |

The three central seed 101 runs save seven snapshots each; checkpoint I/O adds 0.705–0.924s, already included in their telemetry totals. Main-run training-time ratios to the original runs range 0.9914–1.0047; this small variation is not an instrumented overhead estimate because evaluation is excluded from that timer.

The v1 qualification failed the 10% direct-overhead gate for one controlled pair (11.55%), and was retained. The timing-only change to cadence 10/probe 128 passed v2 (maximum 4.22%). Nevertheless, all three final short baseline runs exceed 10% including cold epoch 0 diagnostics: initial measurement costs 0.592–0.860s, while later telemetry costs only 3.34–4.33% of training. The qualification gate is therefore not a universal final cold-start bound. No settings were changed after observing this exceedance.

The optimized baseline exactly repeats target epochs 24/23/21 and official-test accuracies 98.67/98.63/98.71%. Its faster, separately optimized recipe uses all 60,000 training examples, batch 256 BF16, unit dropout and the frozen LR/shrinkage schedule; it is not a controlled arm of the 50,000-example SD experiment. These are repeated already-selected seeds and a test-target stopping rule, not an untouched test evaluation. Invocation and stopping-wall clocks include extra setup/evaluation; training clocks do not. Offline curvature and subsequent W&B uploads are separate costs.

## Provenance and limits

Derived from every raw telemetry-main-*.json and telemetry-baseline-*.json declared in telemetry/rerun-manifest.json. Exact source/file hashes, every raw scientific curve, all scalar measurements, per-class counts and both qualification attempts are retained in analysis.json and plot-data.json. Reproduce this note with telemetry/analyze.py after all 18 reruns verify. The historical ts4k9n55 reference includes all 91 count-verified public evaluation rows; its legacy val means official test, and its W&B runtime includes logging/evaluation on unverified hardware. It is not an isolated GPU-training-time baseline.
