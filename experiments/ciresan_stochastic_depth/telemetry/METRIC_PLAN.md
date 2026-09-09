# Metric inventory and low-overhead telemetry plan

Status: implementation proposal, 9 September 2026. This document inventories the local historical source and sanitized W&B evidence; it does not authorize compute, upload logs, or change the frozen experiments. New instrumented reruns must have new run IDs and source hashes. The root runner decides the final cadence and records any departure from this proposal.

## Recommendation

Keep the useful historical measurements: loss/accuracy, layer gradient magnitude, activation/backprop scale and zeros, and gradient variability. Add actual parameter updates, branch activity, class errors, and separate training/telemetry clocks. Collect most scalar measurements at existing evaluation boundaries, with a small fixed dense probe every ten epochs. Do not put Python hooks, histogram construction, matrix eigendecompositions, or W&B calls inside CUDA-graph replay.

Use accurately named dense-probe and sampled-training metrics instead of silently overlaying quantities with different semantics. In particular, a last masked minibatch's gradient is not the historical dense-statistics gradient; a Fisher/gradient second moment is not a Hessian eigenvalue; nominal skipped MACs are not measured time or energy saved.

## Sources and what can actually be recovered

The frozen source snapshots are [train_ciresan_new.py](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/reference/train_ciresan_new.py) (SHA-256 `60bfa9d8d8c4d1ff1a7689dd6089e391a6b557059bcaa20205e26bcf99c6cf73`) and [util.py](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/reference/util.py) (`2a3599417c650bd0bab7e6117895c2d89bdd2d3a761568ef25104564b121c8c6`). They were copied from the public `yaroslavvb/stuff/autotune` source. They do not establish the exact dirty checkout used for every historical run, and the snapshot does not include the underlying `autograd_lib` implementation.

The [history inventory](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/history/README.md) contains sanitized metadata for all 327 project runs. `run-metadata.json` deliberately whitelists only scalar performance/configuration fields. It does **not** retain per-layer diagnostic histories. `selected-histories.json` queried all count-verified evaluation rows for ten runs but saved only first, extrema, last, and first ≥98% records. Thus the metric names below are verified source emissions, not a claim that every diagnostic curve was recovered from W&B. No absent diagnostic curve should be invented from these caches.

The source explicitly links historical run [425pu650](https://wandb.ai/yaroslavvb/train_ciresan/runs/425pu650): configured `train_steps=100`, training batch 64, statistics batch 5,000, one statistics batch, `skip_stats=0`. Its config also includes `disable_hess=1`, a flag absent from this source version. Run [0ovwv4x6](https://wandb.ai/yaroslavvb/train_ciresan/runs/0ovwv4x6) uses 100 training steps per collection, statistics batch 1,000, `sampled=0`, `curv=kfac`, and `log_spectra=0`. The faster 2024 reference [ts4k9n55](https://wandb.ai/yaroslavvb/train_ciresan/runs/ts4k9n55) records `skip_stats=1`, `disable_hess=1`, and 1,000 training steps per evaluation. Its curvature curves therefore are not an appropriate reference for an instrumented training-speed claim. Config fields absent from the source demonstrate source-version uncertainty.

## Historical collection cadence and axes

Every outer iteration first evaluates the **entire official test set and configured training set**, then optionally collects diagnostics, then executes `train_steps` optimization steps. Defaults are 100 steps × batch 64 = 6,400 training-example presentations between evaluations. The first evaluation is at initialization. The loop has no final post-training evaluation. `epoch = token_count // 60000` is an integer exposure counter even if a smaller dataset was configured; it is not necessarily a completed DataLoader epoch. With `drop_last=True`, incomplete training batches are dropped.

`skip_stats=1` skips the large curvature block but leaves full train/test evaluation. `stats_batch_size × stats_num_batches` defines diagnostic sample count `n`; however the Fisher diagnostic code asserts that `stats_num_batches==1`. The statistics loader shuffles examples from the training dataset. After validation, the model remains in eval mode throughout statistics collection, so unit dropout is off. Training mode is restored afterward.

The source sends each scalar separately to TensorBoard with `global_step=gl.get_global_step()`; W&B is enabled through `wandb.tensorboard.patch(tensorboardX=False)`. The copied files do not define `gl.get_global_step()`. Historical `_step`, `global_step`, and `epoch` are not interchangeable: e.g. 425pu650 ends at `_step=32600`, `global_step=2086400`, epoch 34. The modern rerun should explicitly log `optimizer_step`, `examples_seen`, and `epoch`, with a documented axis for every plot.

### Performance and timing names

| Historical key | Exact meaning in supplied source | Proposed modern field |
| --- | --- | --- |
| `val_accuracy`, `val_loss` | Official-test percent accuracy and mean CE, despite the word “val” | `test/accuracy_pct`, `test/ce`; gated by the declared study protocol |
| `train_accuracy`, `train_loss` | Dense eval-mode accuracy in percent and mean CE over the entire configured training set | `train_full/*` only if full-set evaluation; otherwise explicitly `train_probe/*` with sample count |
| `epoch` | Integer `token_count//60000` before the next training chunk | `epoch`, `optimizer_step`, `examples_seen`, and `equivalent_passes` separately |
| `lr`, `momentum` | Current optimizer scalars | `optimizer/lr`, `optimizer/momentum`; also log shrinkage and schedules |
| `time/outer` | Milliseconds since start of the previous outer iteration | `time/run_elapsed_seconds`; keep interval and cumulative clocks distinct |
| `time/validate` | Python wall time around complete test and train evaluations, milliseconds | `time/evaluation_seconds` and cumulative total |
| `time/train` | Python wall time around `train_steps` updates, milliseconds | Synchronized `time/training_seconds` and interval training seconds |
| `time/stats_forward`, `time/backprop_H`, `time/compute_stats`, `time/spectrum` | Python wall time around diagnostic portions, in milliseconds; `compute_stats` logs repeatedly inside hooks | `time/telemetry_seconds` and optional coarse phase totals |

The historical timer uses `perf_counter` without an explicit CUDA synchronization. Transfers and scalar logging may incidentally synchronize, but these timers are not clean GPU kernel times. W&B `_runtime` includes evaluation, diagnostics, initialization and logging. Preserve it as historical run elapsed time, never call it isolated training time or assume the same GPU.

## Exact layer-statistics inventory

All original layer metrics use `layer-{i}` for affine layers 0 through 5. Let `A_i` be an example's **input to that affine layer**, `B_i` its backpropagated preactivation derivative, and `G_i = B_i A_iᵀ` its weight-gradient outer product. Bias coordinates are not appended to `A`; these are weight metrics, not weight-plus-bias metrics. The ordinary Fisher pass differentiates **sum CE**, so each `B_i` is the per-example derivative without a `1/n` batch-mean factor. Its mean weight gradient is `g = BᵀA/n`.

The source also obtains Hessian-loss and Jacobian backpropagation samples. Their factor normalization depends on the `autograd_lib` sampler convention; the source names alone do not establish an exact full parameter Hessian. Use “output-curvature/GGN-style block” and “Jacobian Gram block” unless the revived implementation explicitly defines and verifies them.

### Factor and diagonal metrics

For each of `hessian`, `jacobian`, and `fisher`, the source emits the same eight suffixes beneath `layer-{i}/{family}/`:

| Suffix | Source computation and qualification |
| --- | --- |
| `diag_l2` | Largest element of the estimated curvature diagonal; not the full matrix spectral norm |
| `diag_fro` | Frobenius norm of that diagonal array |
| `diag_trace` | Sum of the diagonal entries |
| `diag_average` | Mean diagonal entry |
| `kfac_trace` | `tr(AA/n) * tr(BB/n)` |
| `kfac_fro` | **Unnormalized** `||AA||F * ||BB||F`; scales as `n²` relative to the normalized factor product |
| `diversity` | Mean per-example squared outer-product norm divided by squared mean outer-product norm |
| `kfac_error` | Mean absolute difference between the KFAC diagonal and estimated diagonal, divided by mean absolute estimated diagonal |

Here `AA = Σ A_i A_iᵀ` and `BB = Σ B_i B_iᵀ`, with additional sampler accumulation for the curvature/Jacobian families. No parameter-norm or actual parameter-update/delta metric is emitted by this script; those would be useful **new** measurements, not preserved legacy fields.

### Gradient, activation and pairwise metrics

| Historical suffix under `layer-{i}/` | Exact meaning / needed correction |
| --- | --- |
| `grad_fro` | `||g||F`, the weight gradient of mean dense statistics-batch CE |
| `enorms` | `||g||F²` |
| `min_norm`, `median_norm`, `max_norm` | Statistics of `||G_i||F = ||A_i||₂ ||B_i||₂` |
| `mean_norm` | **`sqrt(Σ||G_i||F²)/n`**, not `mean(||G_i||F)` and not RMS; do not carry the misleading name forward |
| `norms_centered` | `mean(||G_i||F²) − ||g||F²`, the trace of the population-normalized gradient covariance |
| `a_sparsity`, `b_sparsity` | Fraction of affine-input activations / backprop elements **≤0**; negative backprops count as “sparse” in the legacy field |
| `mean_activation`, `msr_activation` | Mean and root-mean-square affine input; “msr” is a legacy spelling |
| `mean_backprop`, `msr_backprop` | Mean and root-mean-square preactivation derivative |
| `mean_dot_product`, `median_dot_product`, `mean_cosine`, `median_cosine` | Summaries from `autograd_lib.offset_dotprod/offset_cosines` on within-statistics-batch example pairs; exact pairing requires the unarchived helper implementation |
| `train_regret`, `test_regret1`, `test_regret2`, `train_regret_opt`, `test_regret_opt`, `regret_ratio` | Median quadratic offset-loss predictions from `autograd_lib.offset_losses`, with offsets 0/1/2 and fixed or optimized alpha; these are **not measured train/test loss changes**, and “test” refers to offset examples from the statistics batch |

### Curvature, noise and heuristic rates

The source emits `hess_trace`, `jac_trace`; KFAC Rayleigh quotients of the mean gradient `hess_curv`, `jac_curv`; and per-example directional quotients `hess_curv_grad`, `jac_curv_grad`, `sigma_curv_grad` with `_max` and `_median`. For example, `hess_curv = <g, (BB_H/n) g (AA/n)> / ||g||²` and `hess_curv_grad` averages that KFAC quotient over individual `G_i`. `sigma_curv_grad` substitutes the empirical-Fisher factors.

Other emitted names are `hess_noise`, `jac_noise`, `hess_noise_centered`, `jac_noise_centered`, `openai_gradient_noise`, `hess_noise_normalized`, `jac_noise_normalized`, `norms_hess`, `norms_jac`, `lyap_hess_max`, `lyap_hess_ave`, `lyap_jac_max`, `lyap_jac_ave`, `band_bottou`, `band_bottou_stoch`, `band_yaida`, `band_yaida_centered`, and `lr1`–`lr4`. These are algebraic moment/curvature proxies and heuristic rates, not optimizer settings or directly measured noise-driven loss increases. `norms_hess` accumulates square roots of directional quadratic forms, whereas `norms_jac` accumulates the quadratic forms themselves; their dimensions differ. The value called `openai_gradient_noise` uses this square-root accumulator and should not be relabeled a standard gradient-noise scale without rederivation.

Material source issues: `lyap_hess_ave` and `lyap_jac_ave` read unwritten `*_sum` fields, while accumulation writes `*_mean`; `band_bottou_stoch` reads unwritten `curv_ratio`. Permissive default dictionaries can silently provide zeros. `lr1`–`lr4` divide already-averaged directional curvatures by `n` again. Empty/zero-gradient cases can create undefined ratios. `skip_nans` filters infinities too, and some filtered sums retain the original `n` denominator. Treat these fields as unaudited legacy output, not targets for literal restoration.

### Spectra and histograms

Every diagnostic collection computes all of `hess_A`, `hess_B`, `fish_A`, `fish_B`, `jac_A`, and `jac_B` factor eigenspectra for every layer. It logs `{hess,fish,jac}{A,B}_erank`, where `erank = sum(eigenvalues)/max(eigenvalue)` is a trace-to-top-eigenvalue effective dimension, not entropy rank. It derives `hess_l2`, `jac_l2`, `fish_l2` from products of largest factor eigenvalues, and `jain1_sto`, `jain1_det`, `jain1_lr` from heuristic trace/spectral formulas.

Always-on histogram names inside the statistics block are `hist_hess_eig`, `hist_fish_eig`, `hist_jac_eig`, `hist_grad_norms`, `hist_grad_norms_hess`, `hist_curv_jac`, `hist_curv_hess`, and `hist_cosines`. **Both `hist_fish_eig` and `hist_jac_eig` mistakenly use Hessian eigenvalue products.** Product spectra can materialize one value per layer weight, up to five million entries for a single layer.

The output-space block always logs `mismatch/rho`, `mismatch/rho_cheap`, `mismatch/diagonalizability` and figures `mismatch/{sigma,hess,lyapunov,lyapunov_cheap}`. `log_spectra=1` additionally logs layer figures `{hess_A,hess_B,hess_AB,jac_A,jac_B,fish_A,fish_B,Lyap,Lyap_cheap}`, scalars `trace_ratio`, `dims`, `L_erank`, `L_cheap_erank`, `rho`, `rho_cheap`, and computed `jain2_*` values. The latter are assigned to `s` **after** its `log_scalars` call, so they are not actually emitted by that assignment path. Setting `log_spectra=0` therefore does not eliminate most spectral overhead.

## Proposed implementation contract

Canonical metric families below are suggestions; the runner may choose equivalent names, but must preserve their units and provenance. Use explicit `*_pct` for 0–100 accuracy and `*_fraction` for 0–1 quantities. Store scalar rows locally first and send one batched W&B row per logging boundary. W&B `_step` should not be the scientific x-axis; define each metric against the recorded optimizer-step axis. Do not enable `wandb.watch`, per-step histograms, or source/checkpoint auto-upload as a side effect.

| Family | Measurements | Proposed cadence and cost |
| --- | --- | --- |
| `train/stochastic_ce`, `optimizer/*`, `schedule/*` | Existing epoch mean stochastic loss, LR, momentum, shrinkage, branch drop probabilities, scheduled changes | Every epoch, reusing existing device reductions and host metadata |
| `validation/*`, `train_probe/*` | Dense CE, accuracy, sample count, predicted-class counts; CE/accuracy generalization gap only between explicitly labeled comparable splits | Existing evaluation boundaries; first epoch, every five epochs, and final is the present controlled-study cadence |
| `test/*` | CE, accuracy, confusion matrix, per-class support/error rates | Only at protocol-authorized endpoints; never add monitored test selection to a validation-controlled rerun |
| `time/*`, `memory/*` | Training, evaluation, telemetry, preparation, checkpoint I/O and run elapsed time; setup separately; peak allocated/reserved bytes | Coarse boundaries, with synchronization outside timed training; log sampled W&B transport/finish wall time separately |
| `branch/{j}/*` | Drop probability, realized kept/dropped counts, valid gradient observations, nominal affine MAC exposure | Every epoch from the masks already generated; do not sum stale skipped gradients |
| `parameter/layer_{i}/*` | Weight and bias L2/RMS, displacement from initialization, change since previous diagnostic snapshot, update-to-weight norm ratio | Sparse boundaries, separate weight/bias; reductions are memory bandwidth work over ~12M parameters |
| `training_sample/layer_{i}/*` | Last **active** sampled training gradient L2/RMS and validity/mask, momentum-buffer magnitude if useful | One explicitly identified training step at a diagnostic boundary; no per-step host synchronization |
| `probe/dense/layer_{i}/*` | Dense mean-CE weight/bias gradient L2; affine-input mean/RMS/zero and nonpositive fractions; preactivation-backprop mean/RMS/zero and nonpositive fractions | Fixed 128 training examples, initialization, epoch 1, every ten epochs, final; one ordinary dense forward/backward on the probe |
| `probe/dense/gradient_moments/*` | Per-example weight-gradient norm mean/RMS/median/max; exact empirical-Fisher weight-block trace; mean-gradient squared norm; covariance trace; gradient diversity | Same small probe, using outer-product identities without building per-example parameter-gradient tensors or covariance matrices |

The probe indices must come only from the fitting training split, selected and hashed before the rerun, using a dedicated RNG or deterministic indices. Keep the same probe across all recipes/seeds where scientifically appropriate, with its sample count and input normalization recorded. This makes trajectories comparable but does not turn that small fixed training subset into held-out generalization evidence.

Use model eval mode and ordinary FP32 scientific reductions for dense probes. Different training precision, dropout schedule, input normalization or logit activation must remain explicit configuration. Gradients of **mean** CE satisfy `B_mean = B_per_example/n`; multiply the backpropagated activation derivative by `n` before reporting legacy-scale per-example moments. Conversely the weight `.grad` from mean CE already has the correct scale for legacy `grad_fro`.

For each affine layer, `mean(||A_i||²||B_i||²)` is the exact weight-block empirical-Fisher trace on that probe. `||BᵀA/n||F²` is the squared mean gradient, and their difference is population-normalized covariance trace; the difference can be clipped only for diagnosed floating-point roundoff, with the convention recorded. Norms and quantiles require only per-example scalar arrays. Gradient diversity divides the first by the second; when the denominator is negligible, report null plus validity counts rather than an enormous misleading number. These are gradient second moments, **not Hessian curvature**, and need distinct plot titles.

### Parameter deltas and CUDA-graph safety

The actual parameter change includes momentum and any post-update multiplicative shrinkage. Do not substitute `−lr*gradient` and label it a measured update. A snapshot difference across ten epochs is `delta_since_snapshot` with start/end optimizer step and interval length; it is not an individual optimizer-step update or path length. If root wants true step updates, snapshot parameters before exactly one predeclared diagnostic step, then subtract afterward; this adds a ~48 MB FP32 snapshot for this model. Displacement from initialization is another separate quantity.

The optimized graph helper keeps persistent gradient storage. Skipped branches may retain **old** `.grad` values. Filter using the exact last active mask; log null/validity for skipped gradients, not zeros that imply a measured derivative. Shrinkage can still move skipped parameters, so their measured update is not necessarily zero. Momentum in skipped branches is frozen by the present controlled-study update rule.

Do not attach ordinary Python module hooks to captured graph replay and assume they run at every replay. Execute the probe separately through the existing eager model with `autograd.grad` targeting intermediates/parameters; avoid `.backward()` into or replacement of graph-owned `.grad` buffers. Restore model mode and preserve all optimizer, parameter, buffer and RNG state. Forward probes must not consume the training-order, stochastic-depth, or unit-dropout streams. Independent tests should confirm that instrumentation on/off leaves a short training trajectory and masks unchanged, and that zero gradients, all-skipped masks, and undefined ratios are handled explicitly.

## Curvature options and overhead controls

Default telemetry should omit full AA/BB matrices, repeated Hessian/Jacobian backpropagations, eigensolves, Lyapunov solvers, and millions-of-values spectra. The old source annotates roughly 600 ms for each Hessian/Jacobian pass and 27 ms per `compute_stats` layer call; these are author comments on historical hardware, **not measurements on the new A100 runner**. The asymptotic cost is enough to reject them as “free” instrumentation: factor construction is `O(n d²)`, dense factor eigensolves are cubic in widths up to 2500, and output-curvature/Jacobian passes can require up to ten output directions. Multiple large factor matrices, graph retention, repeated scalar transfers, histogram construction, and Matplotlib rendering amplify overhead.

If genuine curvature is required, use an explicitly optional coarse diagnostic at initialization and selected/final checkpoints, outside training timing: a fixed small probe and a verified Hessian-vector or GGN-vector product with a defined direction. Label signed directional true-Hessian curvature separately from nonnegative GGN curvature and from empirical-Fisher moments. One direction is not a top eigenvalue; a few power iterations are approximate and must disclose iteration count and convergence. Preserve probe/source hashes and count every added backward/vector product. Given the preceding audit's finite-difference precision failures, a tiny-epsilon loss-difference estimate should not be added as an inexpensive curvature substitute.

Proposed overhead target: telemetry measurement plus serialization should add no more than **5%** to the instrumented rerun's comparable training-and-existing-evaluation wall time, excluding remote network finalization; also report total end-to-end overhead including that finalization. This is a target to measure, not an asserted result. Root can qualify the probe on an existing authorized run. If the target is missed, deterministically reduce the probe cadence or size before experimental outcomes are used, record the decision, and keep all main loss/accuracy curves. Never silently change diagnostic precision based on an attractive or unattractive statistic. Collect lightweight JSON rows in memory, flush periodically, and render spectra/plots after training on CPU. Keep raw scalar histories even if W&B upload fails; training must not wait for every network request.

## Main plots for the rerun

1. Dense train-probe and validation CE/accuracy versus optimizer steps and isolated training seconds, with endpoint markers; show total elapsed time separately. Add official-test values only at declared endpoints. Include seed traces behind means and clearly labeled three-seed intervals.
2. Generalization gap and predicted-class counts/confusion at authorized endpoints, making the historical dead-class/ReLU failure visible instead of relying only on aggregate accuracy.
3. Layer-by-epoch heatmaps of dense-probe gradient L2, parameter L2, snapshot displacement/update ratio, activation zero fraction and backprop zero fraction. Keep sampled masked-training gradients in a separate plot with missing skipped observations.
4. SD probability/realized survival and nominal MAC exposure versus training step, alongside measured training throughput. This tests whether reduced branch work becomes real speed.
5. Probe per-example gradient RMS, covariance trace, and gradient diversity, with finite-count and sample-size annotations. Optional verified curvature belongs in its own chart with its approximation and overhead stated; it must not be conflated with these inexpensive gradient moments.

For meaningful historical overlays, use only the saved evaluation points and mark them as a different source/recipe/hardware regime. Historical `val_*` becomes labeled official test, not modern validation. Recovered old extrema are sparse observations, not a reconstructed dense curve. Parameter deltas, per-class diagnostics, and new fixed-probe measurements cannot be presented as if they had been measured in the old run.

## Implemented offline curvature extension

After the lightweight plan was written, the root runner selected a separate, bounded offline diagnostic rather than adding curvature to training. The implemented entry point is `telemetry.curvature.analyze_curvature(model, x, y, mode='exact', samples=1, seed=20260909, probe_id=..., eig_max_dimension=1536, include_directional=True)`. The planned checkpoint panel is epochs 0, 1, 5, 10, 20, 50, and 100 for seed 101 of residual dense, constant SD, and decreasing SD: seven snapshots per recipe, 21 total. The fixed 128-example fitting-training probe matches the recorder’s current first-128-example probe. The instrumented training cadence was reduced to ten epochs/probe128 after timing qualification; this is a measured-overhead adaptation, not selection using scientific outcomes. Resources and dispatch remain owned by the root runner; no runtime result is asserted here.

Exact mode uses all ten output-unit directions for the Jacobian Gram and ten square-root CE-output-Hessian directions for GGN. The observed-label Fisher is exact on the finite probe in either exact or sampled mode. Sampled mode is implemented for explicit alternatives: independent per-example Rademacher combinations averaged over the specified sample count, with estimator labels. It must not silently replace the pinned exact ten-direction panel after inspecting a statistic.

Per layer, the JSON stores the shared activation-factor spectrum and three families: `empirical_fisher`, `ce_ggn`, and `jacobian_gram`. Each contains weight/bias diagonal max, Frobenius norm, trace and average; normalized KFAC trace/Frobenius norm/top eigenvalue; effective and stable ranks; relative diagonal approximation error; and its backprop-factor eigenspectrum. Bias diagonals are separate and are null when a layer has no bias; weight block metrics exclude weight/bias cross terms. The largest smaller-Gram dimension is at most 1,280 for ten directions and 128 examples. Eigenvalue lists are factor spectra only: the implementation never materializes millions of Kronecker-product eigenvalues or a full parameter-square matrix. The actual largest eigenproblem dimension is logged. An eigenproblem exceeding the configured cap uses an explicitly labeled approximate top-spectrum fallback; the pinned N128/ten-output/1536-cap plan does not need that fallback.

Directional metrics use actual per-example CE weight gradients. They report each family's **KFAC** Rayleigh quotients in the mean and individual gradient directions; weighted-gradient norms; empirical covariance contraction retaining activation/backprop dependence; a separate factorized KFAC-noise contraction; and their ratios to the mean-gradient quadratic form. These are clearly defined replacements for the original curvature/noise intent, not aliases for its buggy or dimensionally inconsistent fields. Zero denominators produce null with validity counts. Approximate centered noise quantities may be negative and are not silently clipped.

The output-space mismatch block computes mean CE-logit Hessian H and empirical gradient second moment F, projects onto H's numerically retained support, and solves `H L + L H = 2F` there. It reports support cutoff/rank, discarded Fisher mass, solution residual, Lyapunov and preconditioned-Fisher spectra, and both ambient- and support-dimension rho conventions. The ratio of their effective ranks is given a descriptive name rather than claiming it proves “diagonalizability.” These are output-logit diagnostics, not a full-network stability theorem.

The offline wrapper loads a separate checkpoint model and raw inputs in **FP64**, so model derivatives, moments, and eigenspectra all use FP64; its input normalization still follows the original model factory. The reusable curvature function records the supplied model dtype and temporarily sets float32 matmul precision to `highest` for any float32 operations. CE output gradients and the Hessian factor use cancellation-resistant sums of other probabilities instead of `1−p` at high confidence, with the convention recorded. This is an explicit numerical implementation choice, not a retroactive modification of earlier frozen gradient probes. Parameters, `.grad` storage, optimizer state, model modes, and RNG states are preserved. Derivative and reduction dtypes, probe hashes, elapsed time, and matrix dimensions accompany the results.

Eleven CPU tests pass against independently constructed tiny parameter-space Fisher/GGN/Jacobian blocks and KFAC matrices, including directional/noise contractions, output support projection, sampled normalization, saturated-softmax behavior, state preservation, and the real crop-residual model class. GPU runtime and finite-probe representativeness remain to be measured by the authorized offline runner. Expensive legacy offset-regret predictions, heuristic learning-rate prescriptions, erroneous duplicate histograms, unnormalized Frobenius norms and unwritten accumulator fields remain omitted and named in the output metadata.


The implemented `telemetry/offline_curvature.py` wrapper accepts the `telemetry.run.run(spec, root, progress_commit)` dispatch contract. Its independent `curvature_source_sha256` mapping must pin `telemetry/curvature.py`, `telemetry/offline_curvature.py`, and `model_data.py`. It verifies the source run’s separately frozen training-source map, every snapshot file against `result.telemetry.snapshots`, original training/split hashes, and the recorder’s raw-image/label probe hash. Epoch 0 and epoch 100 also match the source initialization and trained-final parameter hashes. An optional `source_result_sha256` pins exact remote JSON bytes; local results containing additional dispatch metadata are not byte-identical substitutes. All SHA values and the distinction between externally pinned and merely recorded source-result digests are saved.

Every completed checkpoint writes `curvature-{epoch:03d}.json` and an updated aggregate `result.json`, followed by the supplied volume-commit callback. The default computational work limit is 420 seconds inside the parent’s 480-second invocation. It records the first snapshot’s runtime projection, and stops with an explicit incomplete `budget_censored` result if elapsed time plus 1.25 times the largest observed snapshot duration plus five seconds exceeds that limit. It does not switch to sampled directions, drop unfavorable checkpoints, or lower precision. All original snapshots remain untouched.

**Exact in this output means exact affine-block diagonal/trace and exact finite-probe factor spectra under the stated arithmetic and all output directions. It does not mean an exact full Fisher, GGN, Jacobian Gram, or parameter Hessian matrix.** KFAC still discards dependencies between activation and derivative factors and omits cross-layer structure. Recorder FP32 and offline FP64 probe CE/accuracy are shown side by side as descriptive precision comparisons, not required to be bitwise equal. Six synthetic filesystem-wrapper tests verify successful seven-snapshot completion, incremental persistence, duplicate-run refusal, source/checkpoint/probe tamper guards, and runtime censoring, in addition to the eleven curvature tests.
