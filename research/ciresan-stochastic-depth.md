# Stochastic depth in a revived Ciresan-width MNIST network

This controlled experiment asks whether skipping residual branches improves optimization or generalization in a large, tapered MNIST multilayer perceptron. All 24 final runs, 15 tuning attempts and six pilots are accounted for, including three tuning divergences. Stochastic depth saves fixed-horizon training time but does not improve primary mean test accuracy; the secondary endpoint gives a different ranking. The nine historical-fidelity runs expose severe seed sensitivity and show that adding residual bypasses to the raw-input, ReLU-output source configuration can impair learning.

The study is exploratory. A separate effort already used the official MNIST test set to optimize a baseline and investigate its output head. The resumed stochastic-depth study keeps its five recipes fixed and selects learning rates and checkpoints using the separate validation split, but this does not restore a globally untouched test set. Seeds 101–103 were also used in baseline recipe checks; they are confirmation seeds relative to tuning seed 1, not seeds unknown to the overall investigation. See the [separate test-target optimization report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/optimization/) and [frozen resumed protocol](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/PROTOCOL.md).

## Main results: faster fixed-horizon training, no primary accuracy gain

All 15 main confirmation runs completed. Both SD arms reduced measured 100-epoch training time by about 14.3–14.4% relative to residual dense, but neither improved mean accuracy at the predeclared validation-CE-selected endpoint. Constant SD reduced primary accuracy by 0.570 percentage points, with a paired 95% t interval [−0.958, −0.182]. Decreasing SD reduced it by 0.377 points, with a wider interval [−0.931, +0.177] that includes zero. This result does not establish equivalence for decreasing SD.

At the secondary, final-epoch endpoint, the differences reverse direction: constant SD is +0.043 points [−0.033, +0.119], and decreasing SD is +0.110 points [−0.004, +0.224]. Both intervals include zero. Reporting only final-epoch accuracy would obscure the primary result; reporting only the primary result would conceal the endpoint dependence. These are conditional three-seed intervals on one reused test set, without multiple-comparison correction.

Values below are means ± sample SD over seeds 101–103. Test accuracy uses the same 10,000 examples for every run. CE is natural-log cross entropy.

| Arm | Primary accuracy (%) | Primary test CE | Final accuracy (%) | Final test CE |
|---|---:|---:|---:|---:|
| Plain MLP | 97.913 ± 0.332 | 0.0853 ± 0.0092 | 98.573 ± 0.025 | 0.1116 ± 0.0064 |
| Residual dense | 98.373 ± 0.067 | 0.0738 ± 0.0072 | 98.400 ± 0.000 | 0.0972 ± 0.0056 |
| Constant SD | 97.803 ± 0.179 | 0.0790 ± 0.0056 | 98.443 ± 0.031 | 0.1336 ± 0.0225 |
| Decreasing SD | 97.997 ± 0.289 | 0.0725 ± 0.0035 | 98.510 ± 0.046 | 0.1170 ± 0.0159 |
| Unit dropout | 98.330 ± 0.142 | 0.0714 ± 0.0042 | 98.507 ± 0.115 | 0.1031 ± 0.0055 |

Cross entropy and accuracy answer different questions here. Decreasing SD has slightly lower mean primary test CE than residual dense (0.07251 versus 0.07377), despite lower accuracy; its paired CE interval also includes zero. At epoch 100, mean test CE is higher for both SD arms despite their slightly higher accuracy. These observations alone do not identify a calibration mechanism.

The plain adapted MLP provides a particularly clear endpoint check: its primary accuracy is 97.913%, but its final accuracy is 98.573%, the highest final mean among the five arms. Its selected checkpoints are epochs 5, 10 and 10. The residual-dense checkpoints are all epoch 10. Plain versus residual additionally changes architecture and selected LR (0.03 versus 0.01), so this contrast does not isolate dropout or an architecture-only effect. The pattern cautions against interpreting a CE-selected accuracy ranking as a universal ranking of attainable classification accuracy.

### Generalization and optimization are separate

| Arm | Primary full-train CE | Primary validation CE | Final full-train CE | Final validation CE | Final full-train accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Plain MLP | 0.01626 | 0.09097 | 0.00000137 | 0.12921 | 100.000 |
| Residual dense | 0.00097 | 0.08215 | 0.00000764 | 0.11459 | 100.000 |
| Constant SD | 0.02365 | 0.08558 | 0.00000214 | 0.14494 | 100.000 |
| Decreasing SD | 0.02021 | 0.07871 | 0.00000451 | 0.13199 | 100.000 |
| Unit dropout | 0.00467 | 0.07853 | 0.00151501 | 0.12393 | 99.967 |

The final plain, residual-dense and both SD models all achieve 100% mean full-training accuracy, while their validation CE rises relative to the selected checkpoint. Constant SD does not prevent near-zero dense-inference training loss at this horizon. A smaller train/validation gap at an earlier selected checkpoint can reflect less fitting as well as useful regularization; it should be interpreted alongside absolute held-out quality.

All 15 main runs report NVIDIA A100-SXM4-40GB, PyTorch 2.14.0+cu130, CUDA runtime 13.0 and FP32/TF32. Their execution-source hashes and same-seed initialization/data/order hashes match, so all three speed pairs are retained. The intervals below do not incorporate worker-level uncertainty from randomized hardware placement or repeated timings.

| Arm | Full training seconds | Total run seconds | First observed epoch ≥98% validation | Training seconds to ≥98% validation |
|---|---:|---:|---:|---:|
| Plain MLP | 62.27 ± 0.34 | 64.63 ± 0.36 | 15.00 ± 5.00 | 9.37 ± 3.10 |
| Residual dense | 66.38 ± 0.28 | 69.04 ± 0.53 | 10.00 ± 0.00 | 6.69 ± 0.03 |
| Constant SD | 56.84 ± 0.18 | 58.25 ± 0.31 | 15.00 ± 0.00 | 8.57 ± 0.02 |
| Decreasing SD | 56.87 ± 0.18 | 58.55 ± 0.54 | 18.33 ± 2.89 | 9.04 ± 1.42 |
| Unit dropout | 69.03 ± 0.05 | 70.39 ± 0.07 | 13.33 ± 2.89 | 9.23 ± 2.00 |

Constant and decreasing SD take about 56.84 and 56.87 seconds of training, versus 66.38 for residual dense. Mean paired speedup ratios are 1.168 [1.146, 1.189] and 1.167 [1.149, 1.185], respectively. This is a real execution saving at fixed training exposure, rather than merely a masked-work estimate. It does not mean faster optimization to a quality target: residual dense first reaches the scheduled 98% validation threshold at epoch 10 in all three runs, constant SD at epoch 15, and decreasing SD at epochs 15, 20 and 20. All runs reach this threshold, so these particular summaries have no censored observations, but the five-epoch measurement cadence limits their precision.

The numerical record, including seed-paired error-count differences and disagreement counts, is in the [generated analysis](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/analysis.json) and [complete endpoint tables](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/summary.md). All nine fidelity outcomes are reported separately below.

## What was revived, and what changed

The supplied [historical script](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py) uses affine widths 784 → 2500 → 2000 → 1500 → 1000 → 500 → 10: 11,972,510 trainable parameters. It is a plain MLP, not a residual network, and it is not a reproduction of the augmentation-heavy original Ciresan record-setting system. The source normally uses raw 0–255 inputs and a ReLU on the classifier output. Its direct parameter shrinkage is applied after the optimizer update and is not equivalent to SGD weight decay with the same numerical coefficient. The [historical audit](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/history/README.md) records additional fidelity limits, including hardcoded seed 1 and historical use of official test accuracy as “validation.”

The main study deliberately uses pixels divided by 255 and unconstrained linear classifier logits. The plain arm measures this adapted architecture. Four other arms add a fixed prefix-crop bypass around each of the four middle transitions, without adding trainable parameters. If a transition reduces width from a to b, the bypass keeps the first b coordinates. It is therefore a lossy projection, not an identity map on the wider state. A kept branch adds its affine output to that crop before the ReLU; a skipped branch applies the ReLU to the crop alone. This architectural intervention matters independently of regularization: the identifying SD comparison is against residual dense, while plain versus residual measures the effect of adding bypasses.

The [model definition](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/model_data.py) has compulsory stem and classifier layers. A Bernoulli decision is shared across a minibatch for each eligible residual branch; kept branches are scaled by 1/(1−p). The expectation identity holds for the pre-ReLU branch contribution, not the nonlinear network's prediction or loss. Dense inference uses every branch at scale one.

## Main comparison and compute accounting

All five arms use SGD with momentum 0.9, batch 64, 100 epochs, constant learning rate, FP32 parameters and TF32 matrix multiplication. There is no augmentation, normalization layer or weight decay. A fixed permutation divides the original 60,000 MNIST training examples into training 50,000 and validation 10,000; the official 10,000 test examples are separate. Each epoch reshuffles training examples and drops the incomplete final batch: 781 updates and 49,984 presentations per epoch, or 78,100 updates and 4,998,400 presentations over the fixed horizon. These are repeated presentations of the same 50,000-example pool.

| Main recipe | Intervention |
|---|---|
| Plain MLP | Adapted plain architecture, no dropout |
| Residual dense | Four fixed crop bypasses, no dropout |
| Residual + constant SD | Branch drop probabilities 0.4 × [1,2,3,4]/4 |
| Residual + decreasing SD | Probabilities 0.8 × [1,2,3,4]/4 × (1−epoch/(E−1)) |
| Residual + unit dropout | Independent unit dropout with p=0.2 after all five hidden ReLUs, including shortcut activations |

The two SD schedules have the same expected integrated number of omitted branches: one of the five hidden affine layers per minibatch on average. They also match expected nominal affine work omitted over training. Count-based omission is 20%, while nominal affine multiply-accumulate omission is about 14.626%, because the largest eligible transition has the lowest drop probability. These are expectations, not measured wall-time savings. The decreasing schedule changes probability variance and peak strength as well as time ordering, so it does not isolate schedule ordering alone. It reaches zero dropout only at the final epoch, rather than having an extended dense tail. Unit dropout is an ordinary regularization comparator, not an equal-compute or equal-exposure control.

The resumed [graph runner](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/train_optimized.py) genuinely omits a skipped branch's affine work and its gradient/momentum update for that minibatch. Its [CUDA graph implementation](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/optimization/stochastic_graph.py) supports the 16 branch-mask topologies. This differs from computing a branch and multiplying its output by zero. The [qualification evidence](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/graph-qualification-v1.json) checks switching masks and scales, state restoration, and fresh unit-dropout randomness. The implementation replaces scalar division with FP32 reciprocal multiplication; numerical agreement was tested, but bitwise equivalence to the earlier eager runner is not claimed. The earlier Torch 2.8 tuning batch is archived and does not select resumed learning rates.

Training time is a synchronized sum of minibatch loops, including gathers, graph execution, optimizer updates and in-loop diagnostics. Graph setup and per-epoch order/mask/scaling preparation are measured separately. Data loading, evaluation, checkpointing and these setup/preparation costs enter total run time rather than training time. Neither measure is GPU energy or invoice cost. Reported A100 variants and software/runtime identifiers are preserved; speed comparisons require matching reported GPU/runtime and source within a seed pair and are stratified by hardware. An identical GPU model name still does not control clock, contention or thermal state.

## Completed validation selection

Each main arm received the same three-rate grid {0.01, 0.03, 0.1} at tuning seed 1 and the full 100-epoch horizon. Among completed, nondiverged candidates, the selected learning rate minimizes validation cross entropy across scheduled checkpoints; an exact tie is broken by the lower rate. Validation accuracy is descriptive. This nested selection chooses both the arm's learning rate and, later, each confirmation run's checkpoint. Its primary endpoint is validation-selected and can differ substantially from the fixed-horizon endpoint; both must be reported.

| Recipe | Selected LR | Tuning checkpoint epoch | Tuning validation CE |
|---|---:|---:|---:|
| Plain MLP | 0.03 | 5 | 0.08663254 |
| Residual dense | 0.01 | 5 | 0.07866967 |
| Residual + constant SD | 0.01 | 10 | 0.07920895 |
| Residual + decreasing SD | 0.01 | 15 | 0.08049963 |
| Residual + unit dropout | 0.01 | 15 | 0.08553691 |

The LR=0.1 constant-SD, decreasing-SD and residual-unit-dropout runs diverged at epochs 12, 1 and 21 respectively. These are scientific optimization failures, not failed transport or missing jobs. All three remain in the attempted-run accounting and are excluded from learning-rate selection; their last finite validation observations do not make them completed candidates. The common lower selected rate for residual dense and all three regularizers avoids a selected-LR difference within the central residual comparisons, although tuning remains conditional on one seed. The [frozen selection](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/selection.json) pins every candidate result's SHA-256 digest; the [validation analysis](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/analysis.json) independently verifies all 15 pins and all five CE choices.

## Endpoints, pairing and historical-fidelity controls

The [final manifest](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/final-manifest.json) freezes 15 main runs and nine fidelity runs. Each main recipe has three confirmation seeds, 101–103. The primary model minimizes validation CE over the scheduled checkpoints; the secondary model is the last fixed-epoch checkpoint. Report both test accuracy/error count and CE, together with full-training and validation metrics. The dense training probe recorded during training is a fixed subset and should not be mislabeled as full-training accuracy; full-training evaluation is performed at the selected and final endpoints.

The fidelity arms retain raw pixels, output ReLU, LR 0.001, momentum 0.9, batch 64 and direct shrinkage p←(1−0.00002)p per update. They compare plain, residual dense and constant SD at three seeds. They are descriptive failure/fidelity controls, not members of the equally tuned main comparison. A raw-input linear-head intervention was investigated separately on all 60,000 training examples in the test-monitored optimization study, so it is not a fourth matched 50,000-example fidelity arm. Historical W&B performance also used a different training-data scope and does not furnish a paired baseline for these controls.

The analyzer requires exact paired hashes for initialization, training data, split indices and the ordered stream of executed epoch permutations. Equal batch size and drop-last then determine minibatch membership/order. Branch-mask hashes are intentionally recipe-specific. The main and fidelity cohorts are analyzed separately. Treatment-minus-residual differences are paired by seed for accuracy in percentage points, test error counts, loss and eligible timing measurements. For three complete pairs, the 95% Student-t interval uses two degrees of freedom and critical value 4.302653. This interval concerns seed variation on one fixed split and one reused test set, conditional on learning-rate selection; it is not a population confidence interval or three independent replications of the dataset. No multiple-comparison correction is applied. An interval including zero does not establish equivalence.

The [analysis program](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/analyze.py) reads only filenames authorized by the original pilot manifest, resumed graph-tuning manifest and—when explicitly enabled—the final manifest. It does not mix abandoned tuning, optimization, qualification or ancillary JSON into the controlled comparison. Every test error count is checked against its recorded error-index set and accuracy. Scientific divergence is reported separately from malformed data or failed final verification. Missing or divergent final runs cannot be silently reported as a complete study.

## Historical fidelity: seed sensitivity and late escapes

All nine fidelity runs completed without nonfinite divergence. Completion is not successful learning: the raw-input, ReLU-output recipes can remain near chance or learn only part of the task. Per-seed values make this more visible than a cohort mean. The paired inputs and initial parameters match across the three fidelity variants for each seed.

| Raw-input / ReLU-output arm | Seed | Selected epoch | Primary test accuracy (%) | Final test accuracy (%) | Final test CE | Final full-training accuracy (%) |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 101 | 75 | 98.34 | 98.28 | 0.06091 | 100.000 |
| Plain | 102 | 50 | 87.00 | 86.87 | 0.32996 | 88.640 |
| Plain | 103 | 80 | 88.67 | 88.62 | 0.29184 | 90.006 |
| Residual dense | 101 | 1 | 9.80 | 9.80 | 2.30259 | 9.802 |
| Residual dense | 102 | 100 | 36.83 | 36.83 | 1.73210 | 36.294 |
| Residual dense | 103 | 100 | 39.05 | 39.05 | 1.66471 | 39.194 |
| Constant SD | 101 | 1 | 9.80 | 9.80 | 2.30259 | 9.802 |
| Constant SD | 102 | 1 | 9.80 | 9.80 | 2.30259 | 9.802 |
| Constant SD | 103 | 1 | 9.80 | 9.80 | 2.30259 | 9.802 |

The plain source-style model reaches 98.34% primary accuracy at seed 101, but only 87.00% and 88.67% at seeds 102 and 103. Its poor-seed outcomes are consistent with the seed sensitivity seen in the separate baseline optimization, but this 50,000-example study has a different recipe and training-data scope; it is not a reproduction of those 60,000-example timing runs.

Residual dense reaches 9.80%, 36.83% and 39.05% final test accuracy. Seed 103 first shows a clear escape from the near-chance validation plateau at the scheduled epoch-90 observation; seed 102 does so at epoch 100 after still being near chance at epoch 95. Thus it would be wrong to call all of these trajectories permanently absorbing zero-output states. At epoch 100, about 96.8% and 93.6% of logits in the recorded 2,048-example validation probe are still zero for seeds 103 and 102. Those probe fractions are observations on a subset, not a proof that every training example had zero gradients.

Constant SD remains at 9.80% selected and final test accuracy for all three seeds, with CE about log(10), throughout this finite 100-epoch observation. The exact agreement produces zero sample SD in these scores; it does not demonstrate low uncertainty about other seeds, longer training or different hyperparameters. Nor does it imply that every parameter is permanently frozen: momentum, shrinkage and changing hidden activations matter to the trajectory, and no intervention here isolates their mechanism.

These fidelity results support treating input/head adaptation as a substantive modeling choice before judging dropout. They do not support a universal claim that stochastic depth or residual networks fail. The bypass projection, raw input scale, ReLU classifier, untuned historical learning rate and direct shrinkage jointly define this particular failure-prone configuration.

## Verification and reproducibility

Final verification passed with 24/24 expected evaluation results, 15/15 resumed tuning outcomes and six pilots. All main and fidelity seed groups match exact initialization, training-data/split and executed epoch-order hashes. Every graph result, including the three scientifically divergent tuning attempts, matches the three frozen execution-core source hashes; the resumed tuning manifest and all 15 frozen selection-result digests also verify. No malformed record, missing final run or test error-index inconsistency remains. The three tuning divergences remain reported outcomes rather than being reclassified as successful candidates.

Reproduce the final aggregation with `python experiments/ciresan_stochastic_depth/analyze.py --final --require-complete`. This reads existing result JSON only and launches no compute. The [raw results directory](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth/results), [frozen execution manifest](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/graph-protocol-freeze.json), [complete generated tables](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/summary.md) and [analysis JSON](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/analysis.json) retain the audit trail. Archived initial tuning and baseline optimization remain separate from the resumed controlled comparison.


## Compute cost and completion

After all Modal apps had stopped, the dedicated MNIST environment reported
**$3.9768 of metered usage** at 2026-09-09T22:58:39.245245+00:00. This covers the
baseline optimization, kernel/compile trials, pilots, interrupted first tuning
batch, resumed tuning and all final SD/fidelity runs. It is a possibly delayed
metering snapshot, not a final invoice. The shared user cap was **$30**.
Conservative reservations totaled **$23.1036** and were
never released; these are dispatch bounds, not measured spending. All15 apps
in the dedicated environment are stopped with zero running containers.

The [budget ledger](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/budget-ledger.json)
and [metering snapshot](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/billing-latest.json)
retain the evidence. Earlier, separate A100 experiments in this repository are
not part of this MNIST environment or its $30 budget.
