# Offline curvature: what the revived Ciresan statistics actually show

All **21 planned snapshots completed and passed verification**: residual dense, constant stochastic depth, and decreasing stochastic depth; seed 101; epochs 0, 1, 5, 10, 20, 50, and 100. The same 128 fitting-training examples, initial parameters, and dense gain-one inference convention were used throughout. These are three training trajectories from **one seed**, not three independent seed replications.

The clearest result is that loss curvature and raw logit sensitivity move differently. From initialization to epoch 100, residual-dense CE-GGN weight trace drops from 87.04 to 0.4235, while its Jacobian-Gram trace grows from 976.04 to 170,620.90. Both SD models finish with smaller CE-GGN and Jacobian traces than this residual control, but all three perfectly classify this small training probe. The measurements therefore do not establish a test-accuracy improvement or a universal “flatter means better” explanation.

A second result is a warning about the historical approximations: at epoch 100 the KFAC CE-GGN diagonal differs from the exact diagonal by **83–583% relative L1 error**, depending on recipe and layer. Its eigenspectrum is computed accurately, but it is the spectrum of an approximation that can be poor on this concentrated probe.

## Definitions, scope, and connection to the old names

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

## The trajectories: Fisher, GGN and Jacobian are not interchangeable

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

## Spectra and approximation error: lower magnitude can coexist with higher rank

The final-checkpoint table contrasts curvature scale and spectral concentration. Activation effective rank is computed from an **uncentered second moment**, not a centered covariance or count of usable features. The final affine layer is layer 5, the classifier.

| Recipe | GGN KFAC diagonal relative-L1 error, layers 0–5 | Stem KFAC top eigenvalue | Head KFAC top eigenvalue | Head input effective rank | Head GGN KFAC effective rank |
| --- | --- | --- | --- | --- | --- |
| Residual dense | 0.876–1.871 | 0.062115 | 0.0066085 | 1.981 | 2.647 |
| Constant SD | 0.833–4.065 | 0.0027933 | 0.00048267 | 3.441 | 3.903 |
| Decreasing SD | 0.833–5.832 | 0.0016471 | 0.00016115 | 4.021 | 6.008 |

At initialization, CE-GGN diagonal approximation errors range only 0.008–0.194 across layers. They grow substantially during fitting; the largest observed error across this panel is about 5.89, or 589%, for decreasing SD at epoch 50. Exact factor eigendecomposition does not validate the factorization assumption.

SD's final classifier-input effective rank is higher than the control's even as curvature magnitude is lower. Likewise the classifier's KFAC CE-GGN effective rank increases from 2.647 in residual dense to 3.903/6.008 in constant/decreasing SD. “Small curvature” is not the same as “low rank.” These values depend on parameterization, uncentered moments, finite-probe support, and the KFAC approximation; they are not evidence of a universal representation-capacity law.

The stored mean-gradient and per-example directional curvatures are useful descriptions of this KFAC operator. Given the large diagonal errors, they should not be promoted to accurate directional curvature of the exact GGN or to a Hessian-based learning-rate guarantee. The older buggy ratio/rate fields and duplicated eigenvalue histograms were not reproduced.

## Output mismatch: rising rho can accompany vanishing absolute noise

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

## Runtime, verification, and limits

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
