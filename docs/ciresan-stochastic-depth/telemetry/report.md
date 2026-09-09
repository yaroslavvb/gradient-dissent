# Ciresan MNIST reruns with low-overhead training statistics

All **18 requested reruns are complete and verified**, with **401 logged checkpoints and 838 distinct scalar fields** available in the interactive explorer. The five-method comparison reproduces its original recorded learning curves. The optimized baseline again reaches the requested 98.63% threshold at epochs 24,23 and 21. The earlier layer-deletion findings remain available in the [hypothesis report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/).

The main practical result is that useful historical diagnostics fit comfortably beside these optimized trainers. Added telemetry costs **1.24–5.94% of training time in the 100-epoch comparison**. For the very short baseline, cold initial diagnostics cost enough to exceed our 10% target: **0.80–1.12 seconds total, or 14.13–18.54% of training**. After initialization, the added work is 0.19–0.26 seconds. Both successful qualification and this final cold-start exceedance are reported.

The revived statistics also expose two interpretation traps. A gradient-diversity ratio near the probe size can mean one example dominates nearly vanishing gradients. And cross-entropy curvature can fall even as sensitivity of the logits grows. Measuring these quantities separately is more informative than treating “flatness,” Fisher magnitude or gradient diversity as a single explanation of generalization.

## What is available

The [interactive report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/telemetry/) includes train/validation/test accuracy and CE against epoch, synchronized training time or elapsed time; all scalar layer statistics; a matching mean-gradient magnitude plot; every digit’s accuracy curve; Fisher/GGN/Jacobian diagnostics; exact factor spectra; and the complete 91-observation historical W&B reference curve. No smoothing is applied. Missing and undefined measurements remain missing.

The architecture remains 784→2500→2000→1500→1000→500→10. Cropped residual bypasses make the four body affine branches eligible for stochastic depth; the stem and classifier always run. The optimized CUDA-graph implementations and frozen scientific settings were reused. Statistics run at epoch boundaries, without optimizer-step hooks or network logging.

Historical statistics were audited definition by definition. We retain full-split scores, per-class counts/confusions, activation/backprop distributions, parameter and gradient norms, momentum and parameter-displacement summaries, and fixed-probe per-example gradient moments. Sparse saved checkpoints support the expensive curvature families afterward. The [metric audit](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/telemetry/metric-plan.html) identifies corrected labels/normalizations and excluded buggy or unjustified legacy fields. These are intentional corrections, not a byte-for-byte resurrection of the old logger.

## W&B and reproducibility

W&B was not authenticated in this environment. The plots are hosted on GitHub Pages, and **all 18 runs have complete post-training W&B JSONL exports**: 401 rows, with every original finite scalar and measured clock preserved. Authenticated replay creates new runs and uses measured epoch/training/elapsed axes; upload time is never substituted for training time. See the [export instructions](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/telemetry/WANDB.md), [exports](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth/results/telemetry/wandb-export), and [reproduction commands](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth/telemetry).

Verification includes 49 CPU tests for formulas, state/RNG preservation, replay checks, checkpoint provenance and export completeness. The report additionally exercises every scalar selector and the main cohort/split/time controls without a browser. Source hashes, all runs, both qualification attempts and budget records are committed alongside the code. Large model checkpoint files remain in the existing Modal volume, with exact hashes; all reported measurements and spectra are in GitHub.

## Training findings from the instrumented reruns

All 18 declared reruns verified. The 15 controlled runs reproduce every original recorded scientific loss/accuracy value, initialization hash, training-order digest, and layer-mask digest. The three optimized-baseline reruns reproduce their recorded curves, initialization and stopping checkpoints. Old final parameter hashes were not recorded, so matching final weights to those old runs is not claimed. Both qualification attempts separately established byte-identical on/off final parameters and complete scientific histories. These are instrumented repeats of existing seeds, not 18 new independent replications.

Controlled settings remain the original widths, normalized pixels, linear logits, 50,000/10,000 fitting/validation split, batch 64 FP32/TF32, 100 epochs and frozen learning rates: .03 for plain and .01 for all residual recipes. Ordinary dropout is .2; SD schedules remain frozen. Dense diagnostics disable both unit dropout and branch removal. They use the same fixed first 128 fitting examples, with detailed measurements at epoch 0, epoch 1 and every 10 epochs. No metric alters training or validation-CE checkpoint selection; diagnostic official-test evaluations are repeated and disclosed. Three seeds reuse the same datasets and probe.

### Quality and checkpoint dependence

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

### What the fixed-probe gradients show

For the final affine head (layer 5), g_i is the per-example weight gradient of unaveraged CE. The reported gradient norm is ||mean_i g_i||, and diversity is D=mean_i||g_i||² / ||mean_i g_i||². The largest-example energy share is max_i||g_i||² / sum_i||g_i||², reconstructed from the recorded maximum norm and second moment. Gradients exclude bias here. The table uses all three seeds; gradient norms and D are seed means, energy shares are seed ranges.

| Recipe | Head gradient norm epoch 1 → 100 | D epoch 1 → 100 | Largest-example energy share at 100 |
|---|---:|---:|---:|
| Plain | 0.146 → 6.55e-05 | 112.3 → 127.9 | 99.61–100.00% |
| Residual | 0.259 → 0.000162 | 116.3 → 127.9 | 89.55–98.99% |
| Constant SD | 0.188 → 8.35e-06 | 156.3 → 127.6 | 60.25–100.00% |
| Annealed SD | 0.17 → 0.000106 | 131.8 → 128.0 | 99.98–100.00% |
| Residual unit dropout | 0.191 → 0.00122 | 143.6 → 138.8 | 58.26–99.99% |

A diversity ratio near 128 is not evidence that 128 examples supply equally useful independent gradients: one dominant example alone produces D≈128. The energy shares above and extremely small median gradient norms indicate substantial concentration as the probe becomes confidently fitted. Small denominators and FP32 arithmetic further limit late-stage ratio interpretation. These observations describe one fixed training probe; they neither identify a population mechanism nor make diversity a validated predictor of test accuracy. Neighbor-gradient cosines refer to cyclic neighbors in this fixed dataset order, not neighboring optimization steps.

### Activations and class-specific behavior

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

### Timing and measurement overhead

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

### Provenance and limits

Derived from every raw telemetry-main-*.json and telemetry-baseline-*.json declared in telemetry/rerun-manifest.json. Exact source/file hashes, every raw scientific curve, all scalar measurements, per-class counts and both qualification attempts are retained in analysis.json and plot-data.json. Reproduce this note with telemetry/analyze.py after all 18 reruns verify. The historical ts4k9n55 reference includes all 91 count-verified public evaluation rows; its legacy val means official test, and its W&B runtime includes logging/evaluation on unverified hardware. It is not an isolated GPU-training-time baseline.

## Offline curvature: what the revived Ciresan statistics actually show

All **21 planned snapshots completed and passed verification**: residual dense, constant stochastic depth, and decreasing stochastic depth; seed 101; epochs 0, 1, 5, 10, 20, 50, and 100. The same 128 fitting-training examples, initial parameters, and dense gain-one inference convention were used throughout. These are three training trajectories from **one seed**, not three independent seed replications.

The clearest result is that loss curvature and raw logit sensitivity move differently. From initialization to epoch 100, residual-dense CE-GGN weight trace drops from 87.04 to 0.4235, while its Jacobian-Gram trace grows from 976.04 to 170,620.90. Both SD models finish with smaller CE-GGN and Jacobian traces than this residual control, but all three perfectly classify this small training probe. The measurements therefore do not establish a test-accuracy improvement or a universal “flatter means better” explanation.

A second result is a warning about the historical approximations: at epoch 100 the KFAC CE-GGN diagonal differs from the exact diagonal by **83–583% relative L1 error**, depending on recipe and layer. Its eigenspectrum is computed accurately, but it is the spectrum of an approximation that can be poor on this concentrated probe.

### Definitions, scope, and connection to the old names

The [metric inventory](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/telemetry/METRIC_PLAN.md) audits the supplied `train_ciresan_new.py` and `util.py`. The new implementation preserves useful mathematical quantities and corrects misleading historical labels and normalization bugs.

| Historical intent/name | Modern quantity and limitation |
| --- | --- |
| `layer-i/fisher/diag_trace` | Exact finite-probe mean squared observed-label CE weight-gradient norm; empirical Fisher, not generally CE curvature |
| `layer-i/hessian/diag_trace` | Exact affine-weight-block CE-GGN diagonal trace using all ten output-Hessian-factor directions; not the full parameter Hessian |
| `layer-i/jacobian/diag_trace` | Exact mean logit-Jacobian Gram weight trace, summed over ten outputs rather than averaged over classes |
| `kfac_trace`, `kfac_fro`, `{hess,jac,fish}_l2` | Correctly normalized KFAC trace, Frobenius norm and largest eigenvalue; the old Frobenius field omitted the `n²` normalization |
| `{hess,fish,jac}{A,B}_erank` | Factor trace divided by largest eigenvalue; not entropy rank. Stable rank is also reported separately |
| `hess_curv`, `hess_curv_grad` | Mean-gradient and per-example-gradient Rayleigh quotients of the KFAC CE-GGN block; different directions give different quantities |
| Gradient-noise ratios | Explicit covariance/curvature contractions, separating empirical activation/backprop dependence from factorized approximations; not an inferred optimal batch size |
| `mismatch/rho`, `rho_cheap` | Output-logit H/F Lyapunov/preconditioned-Fisher concentration on retained Hessian support; not a full-network stability certificate |

All diagnostic models and inputs were converted from recorded FP32 checkpoints to FP64. Output derivatives use a cancellation-resistant CE formulation, and moment/spectrum reductions are FP64. The original training/recorder arithmetic remains unchanged. These runs compute exact diagonals and exact **factor spectra**, not an exact full Fisher/GGN/Jacobian matrix. KFAC still assumes factor independence and omits cross-layer structure. Full output-direction enumeration removes stochastic output sampling error, but does not remove finite-probe or KFAC approximation error.

### The trajectories: Fisher, GGN and Jacobian are not interchangeable

The traces below sum exact weight diagonals over all six affine layers; biases are excluded. Off-diagonal blocks do not affect that summed trace. CE and accuracy refer only to the fixed 128 training examples. Values are descriptive at the seven saved epochs; unsaved intermediate extrema are unknown.

| Recipe | Epoch | Probe CE | Probe accuracy % | Empirical-Fisher trace | CE-GGN trace | Jacobian-Gram trace |
| --- | --- | --- | --- | --- | --- | --- |
| Residual dense | 0 | 2.3262 | 5.469 | 86.983 | 87.039 | 976.044 |
| Residual dense | 1 | 0.1182 | 97.656 | 115.88 | 130.61 | 16123.9 |
| Residual dense | 5 | 0.035437 | 99.219 | 61.367 | 17.842 | 38482.7 |
| Residual dense | 10 | 0.00025018 | 100.000 | 0.069652 | 4.8946 | 75265.5 |
| Residual dense | 20 | 4.5511e-05 | 100.000 | 0.0039747 | 1.4043 | 110343 |
| Residual dense | 50 | 1.5776e-05 | 100.000 | 0.0008853 | 0.75423 | 144778 |
| Residual dense | 100 | 7.2171e-06 | 100.000 | 0.00023162 | 0.4235 | 170621 |
| Constant SD | 0 | 2.3262 | 5.469 | 86.983 | 87.039 | 976.044 |
| Constant SD | 1 | 0.12145 | 97.656 | 64.978 | 85.315 | 12013.1 |
| Constant SD | 5 | 0.031464 | 99.219 | 44.244 | 10.591 | 27069.5 |
| Constant SD | 10 | 0.023937 | 99.219 | 49.043 | 33.964 | 39891.4 |
| Constant SD | 20 | 3.075e-05 | 100.000 | 0.00048844 | 0.44197 | 83840.4 |
| Constant SD | 50 | 2.951e-07 | 100.000 | 1.1807e-07 | 0.0092719 | 140424 |
| Constant SD | 100 | 1.471e-06 | 100.000 | 3.1185e-06 | 0.02206 | 111458 |
| Decreasing SD | 0 | 2.3262 | 5.469 | 86.983 | 87.039 | 976.044 |
| Decreasing SD | 1 | 0.15692 | 96.094 | 75.371 | 101.32 | 6146.92 |
| Decreasing SD | 5 | 0.051097 | 99.219 | 45.566 | 26.448 | 13356.6 |
| Decreasing SD | 10 | 0.039646 | 99.219 | 56.42 | 22.807 | 20318.2 |
| Decreasing SD | 20 | 0.028463 | 99.219 | 62.521 | 55.898 | 36949.6 |
| Decreasing SD | 50 | 9.4715e-06 | 100.000 | 6.0475e-05 | 0.069058 | 61928.3 |
| Decreasing SD | 100 | 9.2335e-07 | 100.000 | 1.0139e-06 | 0.013558 | 82068.6 |

At epoch 100, constant SD has 19.2× lower CE-GGN trace than residual dense, and decreasing SD has 31.2× lower trace. Their logit-Jacobian traces are only 1.53× and 2.08× lower. Thus the loss-curvature contrast is much larger than the raw sensitivity contrast. CE's output curvature shrinks as probabilities saturate; interpreting the entire trace reduction as a change in the underlying network Jacobian would be incorrect.

Empirical Fisher gives another answer again. For residual dense at epoch 5, its trace is 61.37 while CE-GGN is 17.84, and one probe image is wrong. At epoch 100, Fisher is about 1828× smaller than GGN. Squared observed-label gradients and expected output-loss curvature need not agree. Constant SD also has larger CE and GGN at epoch 100 than epoch 50 on this probe: the actual trajectories should not be forced monotonic.

The dense probe reaches 100% accuracy at the first observed epoch 10 for residual, epoch 20 for constant SD, and epoch 50 for decreasing SD. That is an observation about these saved training-probe evaluations, not the precise first crossing time or a whole-dataset convergence claim.

### Spectra and approximation error: lower magnitude can coexist with higher rank

The final-checkpoint table contrasts curvature scale and spectral concentration. Activation effective rank is computed from an **uncentered second moment**, not a centered covariance or count of usable features. The final affine layer is layer 5, the classifier.

| Recipe | GGN KFAC diagonal relative-L1 error, layers 0–5 | Stem KFAC top eigenvalue | Head KFAC top eigenvalue | Head input effective rank | Head GGN KFAC effective rank |
| --- | --- | --- | --- | --- | --- |
| Residual dense | 0.876–1.871 | 0.062115 | 0.0066085 | 1.981 | 2.647 |
| Constant SD | 0.833–4.065 | 0.0027933 | 0.00048267 | 3.441 | 3.903 |
| Decreasing SD | 0.833–5.832 | 0.0016471 | 0.00016115 | 4.021 | 6.008 |

At initialization, CE-GGN diagonal approximation errors range only 0.008–0.194 across layers. They grow substantially during fitting; the largest observed error across this panel is about 5.89, or 589%, for decreasing SD at epoch 50. Exact factor eigendecomposition does not validate the factorization assumption.

SD's final classifier-input effective rank is higher than the control's even as curvature magnitude is lower. Likewise the classifier's KFAC CE-GGN effective rank increases from 2.647 in residual dense to 3.903/6.008 in constant/decreasing SD. “Small curvature” is not the same as “low rank.” These values depend on parameterization, uncentered moments, finite-probe support, and the KFAC approximation; they are not evidence of a universal representation-capacity law.

The stored mean-gradient and per-example directional curvatures are useful descriptions of this KFAC operator. Given the large diagonal errors, they should not be promoted to accurate directional curvature of the exact GGN or to a Hessian-based learning-rate guarantee. The older buggy ratio/rate fields and duplicated eigenvalue histograms were not reproduced.

### Output mismatch: rising rho can accompany vanishing absolute noise

Here H is the mean CE Hessian with respect to ten output logits, and F is their observed-label gradient second moment. The solver retains H's numerical support and solves `H L + L H = 2F` there. Every snapshot retains the expected nine-dimensional softmax support; the discarded relative Fisher norm is below 2.2×10⁻¹⁵. No material discarded Fisher component explains these results.

The reported ambient convention is `rho = 10 / effective_rank(L)`. It is scale invariant and includes the constant-shift null direction in its dimension convention; it is not a divergence threshold. “Cheap rho” uses the spectrum of the symmetrically preconditioned Fisher, equivalent to the nonzero eigenvalues of the projected `F H⁺`.

| Checkpoint | Output H trace | Output F trace | Top eigenvalue of L | Effective rank of L | Rho | Cheap rho |
| --- | --- | --- | --- | --- | --- | --- |
| Common initialization | 0.89925 | 0.90484 | 1.4623 | 6.271 | 1.595 | 1.595 |
| Residual dense final | 1.4422e-05 | 7.9815e-09 | 0.00070445 | 1.297 | 7.711 | 7.796 |
| Constant SD final | 2.9412e-06 | 4.7443e-10 | 0.00018423 | 1.021 | 9.792 | 9.995 |
| Decreasing SD final | 1.8464e-06 | 1.433e-10 | 0.00011641 | 1.011 | 9.890 | 9.989 |

Rho approaches ten for the final SD models because the remaining mismatch is concentrated in roughly one direction, while the absolute scale of L and F has fallen sharply. Calling the higher rho “more instability” would ignore that scale change.

A related finite-probe warning appears in the gradient-noise statistics. At final checkpoints, the empirical covariance-to-mean-gradient quadratic ratio is about 127 for SD, close to `N−1` for N=128. If only one example contributes a nonzero gradient, that ratio equals N−1 even without an independently identified critical batch size. As a post hoc concentration check, `(sum ||g_i||)² / sum ||g_i||²` is only 1.003–1.005 across layers for constant SD and 1.013–1.018 for decreasing SD, versus 1.57–1.87 for residual. The norm mass is carried by very few of these already memorized examples. This is not a count of statistically independent samples, and these ratios should not drive an optimizer recommendation without a larger fresh probe.

### Runtime, verification, and limits

These were separate post-training A100 jobs. They added **zero optimizer steps** and did not extend the training clocks. Their own compute, checkpoint/data loading and volume commits still have real cost. All jobs used NVIDIA A100-SXM4-40GB, torch 2.14.0+cu130, Python 3.11.12, two CPU threads and FP64 diagnostics.

| Recipe | Curvature compute seconds | Recorded job elapsed seconds | Local dispatch elapsed seconds |
| --- | --- | --- | --- |
| Residual dense | 3.598 | 20.526 | 48.089 |
| Constant SD | 3.205 | 19.213 | 28.152 |
| Decreasing SD | 3.832 | 24.423 | 35.266 |

Curvature computation totals 10.636 seconds across 21 snapshots. Recorded job elapsed includes loading and prior volume commits but excludes its own final persistence operation; local dispatch includes setup, queue/wait and return overhead. Those clocks must not be interchanged or credited as a training speedup. The largest eigenproblem was 1,280, and every factor spectrum used the exact smaller-Gram solver; no job was censored or downgraded.

The wrapper verified original checkpoint bytes, training-source pins, separate curvature-source pins, training/split hashes and the recorder probe hash. This postprocessing independently rehashed all 21 diagnostic artifact payloads and checked all three current offline source-pin maps. Initial parameters agree across recipes; the diagnostic model's parameters were unchanged. FP64 and recorder-FP32 probe predictions agree at every snapshot where the recorder supplied a same-size probe, while loss differences are recorded. That agreement does not make their gradients numerically identical.

The limits are substantial: one seed, 128 reused fitting-training examples, changing confidence, seven observation times, a particular tapered residual architecture and structured KFAC approximations. Dense-mode probing turns off the SD masks and unit dropout; it describes the resulting trained weights, not the stochastic training objective's expected curvature. The results motivate separating geometry, confidence and gradient concentration, but do not isolate which mechanism caused branch-deletion resilience or establish better held-out generalization.

The [compact machine-readable summary](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/curvature-summary.json) contains every row and 378 recipe/epoch/layer/family records with raw-file SHA-256 links. Full spectra and provenance are in the [residual result](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/telemetry-curvature-residual-s101-v1.json), [constant-SD result](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/telemetry-curvature-sd_constant-s101-v1.json), and [decreasing-SD result](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/telemetry-curvature-sd_annealed-s101-v1.json). The [curvature implementation](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/telemetry/curvature.py) and [checkpoint wrapper](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/telemetry/offline_curvature.py) specify normalization, support projection and the fixed runtime rule.

## Compute budget

The new phase added **$0.93** of metered Modal usage. The full MNIST environment total is **$5.00 of the existing $30 cap**, as of the recorded closing query. All 23 environment apps, including the six new telemetry apps, are stopped with zero tasks. Metering can lag slightly; the retained conservative aggregate reservation is $9.26, below the $24 guard and $30 absolute cap.

This phase accounts for completed earlier work using its $4.08 metered closing balance plus every new worst-case invocation reservation. The old reservation ledger was preserved byte for byte. Both timing-only qualification attempts, all 18 reruns and all three offline jobs are included. No new spending allowance was inferred.
