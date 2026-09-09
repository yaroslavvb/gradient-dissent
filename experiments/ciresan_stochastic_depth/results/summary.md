# Ciresan-width MNIST stochastic-depth analysis

Generated 2026-09-09T22:58:38.128334+00:00.

Official-test analysis explicitly enabled with `--final`.

Frozen final manifest complete: **True**. Verification passed: **True**.

Only the original pilot manifest and resumed graph-tuning manifest enter validation analysis; the final manifest is additionally enabled by `--final`. Archived tuning, qualification and baseline optimization are excluded.

Values are seed means ± sample SD. Missing SD/CI is undefined, not zero. Primary: validation-CE-selected checkpoint. Secondary: final fixed-epoch checkpoint.

## Main cohort

Final manifest: 15 runs; seeds [101, 102, 103].

### Primary: validation-selected checkpoint

| Arm | Test accuracy (%) | Errors / test set | Test CE | Full-train CE | Validation CE | Validation − train CE | Full-train accuracy (%) |
|---|---|---|---|---|---|---|---|
| plain | 97.91 ± 0.33 (n=3) | 208.7 ± 33.2 (n=3) / [10000] | 0.0853 ± 0.0092 (n=3) | 0.0163 ± 0.0133 (n=3) | 0.0910 ± 0.0071 (n=3) | 0.0747 ± 0.0093 (n=3) | 99.51 ± 0.41 (n=3) |
| residual | 98.37 ± 0.07 (n=3) | 162.7 ± 6.7 (n=3) / [10000] | 0.0738 ± 0.0072 (n=3) | 0.0010 ± 0.0003 (n=3) | 0.0822 ± 0.0038 (n=3) | 0.0812 ± 0.0035 (n=3) | 99.98 ± 0.01 (n=3) |
| residual_unit_dropout | 98.33 ± 0.14 (n=3) | 167.0 ± 14.2 (n=3) / [10000] | 0.0714 ± 0.0042 (n=3) | 0.0047 ± 0.0021 (n=3) | 0.0785 ± 0.0013 (n=3) | 0.0739 ± 0.0024 (n=3) | 99.86 ± 0.08 (n=3) |
| sd_annealed | 98.00 ± 0.29 (n=3) | 200.3 ± 28.9 (n=3) / [10000] | 0.0725 ± 0.0035 (n=3) | 0.0202 ± 0.0172 (n=3) | 0.0787 ± 0.0044 (n=3) | 0.0585 ± 0.0168 (n=3) | 99.40 ± 0.50 (n=3) |
| sd_constant | 97.80 ± 0.18 (n=3) | 219.7 ± 17.9 (n=3) / [10000] | 0.0790 ± 0.0056 (n=3) | 0.0237 ± 0.0103 (n=3) | 0.0856 ± 0.0022 (n=3) | 0.0619 ± 0.0125 (n=3) | 99.24 ± 0.31 (n=3) |

### Secondary: last fixed-epoch checkpoint

| Arm | Test accuracy (%) | Errors / test set | Test CE | Full-train CE | Validation CE | Validation − train CE | Full-train accuracy (%) |
|---|---|---|---|---|---|---|---|
| plain | 98.57 ± 0.03 (n=3) | 142.7 ± 2.5 (n=3) / [10000] | 0.1116 ± 0.0064 (n=3) | 0.0000 ± 0.0000 (n=3) | 0.1292 ± 0.0062 (n=3) | 0.1292 ± 0.0062 (n=3) | 100.00 ± 0.00 (n=3) |
| residual | 98.40 ± 0.00 (n=3) | 160.0 ± 0.0 (n=3) / [10000] | 0.0972 ± 0.0056 (n=3) | 0.0000 ± 0.0000 (n=3) | 0.1146 ± 0.0015 (n=3) | 0.1146 ± 0.0015 (n=3) | 100.00 ± 0.00 (n=3) |
| residual_unit_dropout | 98.51 ± 0.12 (n=3) | 149.3 ± 11.5 (n=3) / [10000] | 0.1031 ± 0.0055 (n=3) | 0.0015 ± 0.0025 (n=3) | 0.1239 ± 0.0120 (n=3) | 0.1224 ± 0.0097 (n=3) | 99.97 ± 0.05 (n=3) |
| sd_annealed | 98.51 ± 0.05 (n=3) | 149.0 ± 4.6 (n=3) / [10000] | 0.1170 ± 0.0159 (n=3) | 0.0000 ± 0.0000 (n=3) | 0.1320 ± 0.0159 (n=3) | 0.1320 ± 0.0159 (n=3) | 100.00 ± 0.00 (n=3) |
| sd_constant | 98.44 ± 0.03 (n=3) | 155.7 ± 3.1 (n=3) / [10000] | 0.1336 ± 0.0225 (n=3) | 0.0000 ± 0.0000 (n=3) | 0.1449 ± 0.0191 (n=3) | 0.1449 ± 0.0191 (n=3) | 100.00 ± 0.00 (n=3) |

### Exposure and measured training time

Pooled runtime means below describe the assigned workers; they do not isolate a method speedup across different A100 variants. Hardware-specific measurements follow.

| Arm | Valid / failed | LR | Selected epoch | Pooled training seconds | Pooled total run seconds | First observed epoch ≥98% validation | Pooled training seconds to ≥98% | Reached / not reached |
|---|---|---|---|---|---|---|---|---|
| plain | 3 / 0 | [0.03] | 8.3 ± 2.9 (n=3) | 62.27 ± 0.34 (n=3) | 64.63 ± 0.36 (n=3) | 15.0 ± 5.0 (n=3) | 9.37 ± 3.10 (n=3) | 3 / 0 |
| residual | 3 / 0 | [0.01] | 10.0 ± 0.0 (n=3) | 66.38 ± 0.28 (n=3) | 69.04 ± 0.53 (n=3) | 10.0 ± 0.0 (n=3) | 6.69 ± 0.03 (n=3) | 3 / 0 |
| residual_unit_dropout | 3 / 0 | [0.01] | 13.3 ± 2.9 (n=3) | 69.03 ± 0.05 (n=3) | 70.39 ± 0.07 (n=3) | 13.3 ± 2.9 (n=3) | 9.23 ± 2.00 (n=3) | 3 / 0 |
| sd_annealed | 3 / 0 | [0.01] | 10.0 ± 5.0 (n=3) | 56.87 ± 0.18 (n=3) | 58.55 ± 0.54 (n=3) | 18.3 ± 2.9 (n=3) | 9.04 ± 1.42 (n=3) | 3 / 0 |
| sd_constant | 3 / 0 | [0.01] | 6.7 ± 2.9 (n=3) | 56.84 ± 0.18 (n=3) | 58.25 ± 0.31 (n=3) | 15.0 ± 0.0 (n=3) | 8.57 ± 0.02 (n=3) | 3 / 0 |

Time-to-98% summaries include achievers only; unreached runs are censored. Training time excludes loading, evaluation, checkpointing, graph setup and per-epoch preparation; it is not billed time or energy.

| Arm / GPU | Seeds | Training seconds | Total run seconds | Graph setup seconds | Epoch preparation seconds |
|---|---|---|---|---|---|
| plain / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 62.27 ± 0.34 (n=3) | 64.63 ± 0.36 (n=3) | 0.51 ± 0.03 (n=3) | 0.26 ± 0.02 (n=3) |
| residual / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 66.38 ± 0.28 (n=3) | 69.04 ± 0.53 (n=3) | 0.54 ± 0.03 (n=3) | 0.34 ± 0.09 (n=3) |
| residual_unit_dropout / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 69.03 ± 0.05 (n=3) | 70.39 ± 0.07 (n=3) | 0.03 ± 0.00 (n=3) | 0.19 ± 0.01 (n=3) |
| sd_annealed / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 56.87 ± 0.18 (n=3) | 58.55 ± 0.54 (n=3) | 0.13 ± 0.02 (n=3) | 0.26 ± 0.06 (n=3) |
| sd_constant / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 56.84 ± 0.18 (n=3) | 58.25 ± 0.31 (n=3) | 0.17 ± 0.06 (n=3) | 0.20 ± 0.00 (n=3) |

## Fidelity cohort

Final manifest: 9 runs; seeds [101, 102, 103].

### Primary: validation-selected checkpoint

| Arm | Test accuracy (%) | Errors / test set | Test CE | Full-train CE | Validation CE | Validation − train CE | Full-train accuracy (%) |
|---|---|---|---|---|---|---|---|
| source-plain | 91.34 ± 6.12 (n=3) | 866.3 ± 612.2 (n=3) / [10000] | 0.2231 ± 0.1450 (n=3) | 0.1651 ± 0.1429 (n=3) | 0.2219 ± 0.1347 (n=3) | 0.0567 ± 0.0101 (n=3) | 92.89 ± 6.20 (n=3) |
| source-residual | 28.56 ± 16.28 (n=3) | 7144.0 ± 1628.5 (n=3) / [10000] | 1.8998 ± 0.3504 (n=3) | 1.8930 ± 0.3592 (n=3) | 1.9050 ± 0.3464 (n=3) | 0.0120 ± 0.0212 (n=3) | 28.43 ± 16.20 (n=3) |
| source-sd | 9.80 ± 0.00 (n=3) | 9020.0 ± 0.0 (n=3) / [10000] | 2.3026 ± 0.0000 (n=3) | 2.3026 ± 0.0000 (n=3) | 2.3026 ± 0.0000 (n=3) | 0.0000 ± 0.0000 (n=3) | 9.80 ± 0.00 (n=3) |

### Secondary: last fixed-epoch checkpoint

| Arm | Test accuracy (%) | Errors / test set | Test CE | Full-train CE | Validation CE | Validation − train CE | Full-train accuracy (%) |
|---|---|---|---|---|---|---|---|
| source-plain | 91.26 ± 6.15 (n=3) | 874.3 ± 614.5 (n=3) / [10000] | 0.2276 ± 0.1456 (n=3) | 0.1655 ± 0.1433 (n=3) | 0.2263 ± 0.1370 (n=3) | 0.0608 ± 0.0067 (n=3) | 92.88 ± 6.20 (n=3) |
| source-residual | 28.56 ± 16.28 (n=3) | 7144.0 ± 1628.5 (n=3) / [10000] | 1.8998 ± 0.3504 (n=3) | 1.8930 ± 0.3592 (n=3) | 1.9050 ± 0.3464 (n=3) | 0.0120 ± 0.0212 (n=3) | 28.43 ± 16.20 (n=3) |
| source-sd | 9.80 ± 0.00 (n=3) | 9020.0 ± 0.0 (n=3) / [10000] | 2.3026 ± 0.0000 (n=3) | 2.3026 ± 0.0000 (n=3) | 2.3026 ± 0.0000 (n=3) | 0.0000 ± 0.0000 (n=3) | 9.80 ± 0.00 (n=3) |

### Exposure and measured training time

Pooled runtime means below describe the assigned workers; they do not isolate a method speedup across different A100 variants. Hardware-specific measurements follow.

| Arm | Valid / failed | LR | Selected epoch | Pooled training seconds | Pooled total run seconds | First observed epoch ≥98% validation | Pooled training seconds to ≥98% | Reached / not reached |
|---|---|---|---|---|---|---|---|---|
| source-plain | 3 / 0 | [0.001] | 68.3 ± 16.1 (n=3) | 68.14 ± 0.10 (n=3) | 70.14 ± 0.49 (n=3) | 15.0 ± — (n=1) | 10.24 ± — (n=1) | 1 / 2 |
| source-residual | 3 / 0 | [0.001] | 67.0 ± 57.2 (n=3) | 72.18 ± 0.06 (n=3) | 73.39 ± 0.15 (n=3) | — | — | 0 / 3 |
| source-sd | 3 / 0 | [0.001] | 1.0 ± 0.0 (n=3) | 63.00 ± 0.15 (n=3) | 64.46 ± 0.40 (n=3) | — | — | 0 / 3 |

Time-to-98% summaries include achievers only; unreached runs are censored. Training time excludes loading, evaluation, checkpointing, graph setup and per-epoch preparation; it is not billed time or energy.

| Arm / GPU | Seeds | Training seconds | Total run seconds | Graph setup seconds | Epoch preparation seconds |
|---|---|---|---|---|---|
| source-plain / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 68.14 ± 0.10 (n=3) | 70.14 ± 0.49 (n=3) | 0.03 ± 0.01 (n=3) | 0.28 ± 0.09 (n=3) |
| source-residual / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 72.18 ± 0.06 (n=3) | 73.39 ± 0.15 (n=3) | 0.02 ± 0.01 (n=3) | 0.20 ± 0.01 (n=3) |
| source-sd / NVIDIA A100-SXM4-40GB | [101, 102, 103] | 63.00 ± 0.15 (n=3) | 64.46 ± 0.40 (n=3) | 0.12 ± 0.02 (n=3) | 0.25 ± 0.07 (n=3) |

## Paired effects versus residual dense

All quality effects use same-cohort, same-seed pairs with verified initialization, data and executed epoch-order hashes. Accuracy is treatment minus control in percentage points; errors are treatment minus control in counts. Positive accuracy and negative error deltas favor the treatment. Brackets are 95% t intervals over paired seed differences. Time/speed columns additionally require matching GPU/runtime/source, so their n may be smaller or zero.

| Cohort / arm | Endpoint | Test accuracy Δ (pp) [95% CI] | Error count Δ [95% CI] | Matched training seconds Δ [95% CI] | Matched training speedup ratio [95% CI] |
|---|---|---|---|---|---|
| fidelity / source-plain | selected | 62.777 [7.347, 118.206], n=3 | -6277.7 [-11820.6, -734.7], n=3 | -4.04 [-4.40, -3.68], n=3 | 1.059 [1.054, 1.065], n=3 |
| fidelity / source-plain | final | 62.697 [7.225, 118.168], n=3 | -6269.7 [-11816.8, -722.5], n=3 | -4.04 [-4.40, -3.68], n=3 | 1.059 [1.054, 1.065], n=3 |
| fidelity / source-sd | selected | -18.760 [-59.213, 21.693], n=3 | 1876.0 [-2169.3, 5921.3], n=3 | -9.18 [-9.67, -8.70], n=3 | 1.146 [1.137, 1.154], n=3 |
| fidelity / source-sd | final | -18.760 [-59.213, 21.693], n=3 | 1876.0 [-2169.3, 5921.3], n=3 | -9.18 [-9.67, -8.70], n=3 | 1.146 [1.137, 1.154], n=3 |
| main / plain | selected | -0.460 [-1.443, 0.523], n=3 | 46.0 [-52.3, 144.3], n=3 | -4.11 [-5.19, -3.03], n=3 | 1.066 [1.048, 1.084], n=3 |
| main / plain | final | 0.173 [0.111, 0.236], n=3 | -17.3 [-23.6, -11.1], n=3 | -4.11 [-5.19, -3.03], n=3 | 1.066 [1.048, 1.084], n=3 |
| main / residual_unit_dropout | selected | -0.043 [-0.333, 0.247], n=3 | 4.3 [-24.7, 33.3], n=3 | 2.65 [1.85, 3.46], n=3 | 0.962 [0.950, 0.973], n=3 |
| main / residual_unit_dropout | final | 0.107 [-0.179, 0.392], n=3 | -10.7 [-39.2, 17.9], n=3 | 2.65 [1.85, 3.46], n=3 | 0.962 [0.950, 0.973], n=3 |
| main / sd_annealed | selected | -0.377 [-0.931, 0.177], n=3 | 37.7 [-17.7, 93.1], n=3 | -9.51 [-10.47, -8.54], n=3 | 1.167 [1.149, 1.185], n=3 |
| main / sd_annealed | final | 0.110 [-0.004, 0.224], n=3 | -11.0 [-22.4, 0.4], n=3 | -9.51 [-10.47, -8.54], n=3 | 1.167 [1.149, 1.185], n=3 |
| main / sd_constant | selected | -0.570 [-0.958, -0.182], n=3 | 57.0 [18.2, 95.8], n=3 | -9.54 [-10.68, -8.39], n=3 | 1.168 [1.146, 1.189], n=3 |
| main / sd_constant | final | 0.043 [-0.033, 0.119], n=3 | -4.3 [-11.9, 3.3], n=3 | -9.54 [-10.68, -8.39], n=3 | 1.168 [1.146, 1.189], n=3 |

Full-horizon training-time effects are repeated at both endpoints for context; they are the same measurements. Speedup is control time divided by treatment time. Matching within pairs does not standardize the GPU across all pairs; hardware-specific effects and every excluded timing pair, conditional time-to-target pair and error-disagreement count are in analysis.json.

## Pilots and LR candidates (validation only)

| Run | Stage | Recipe | LR | Status | Best validation CE | Accuracy at CE-selected epoch (%) | Selected epoch | Final train-probe CE / accuracy (%) | Training seconds |
|---|---|---|---|---|---|---|---|---|---|
| graph-tune-plain-lr0.01 | tune | plain | 0.01 | ok | 0.0933 | 97.41 | 5 | 0.0000 / 100.00 | 62.29 |
| graph-tune-plain-lr0.03 | tune | plain | 0.03 | ok | 0.0866 | 97.73 | 5 | 0.0000 / 100.00 | 62.45 |
| graph-tune-plain-lr0.1 | tune | plain | 0.1 | ok | 0.1074 | 97.58 | 5 | 0.0000 / 100.00 | 62.50 |
| graph-tune-residual-lr0.01 | tune | residual | 0.01 | ok | 0.0787 | 97.82 | 5 | 0.0000 / 100.00 | 66.58 |
| graph-tune-residual-lr0.03 | tune | residual | 0.03 | ok | 0.0907 | 97.58 | 5 | 0.0000 / 100.00 | 66.21 |
| graph-tune-residual-lr0.1 | tune | residual | 0.1 | ok | 0.1135 | 98.30 | 25 | 0.0000 / 100.00 | 66.81 |
| graph-tune-residual_unit_dropout-lr0.01 | tune | residual_unit_dropout | 0.01 | ok | 0.0855 | 98.11 | 15 | 0.0000 / 100.00 | 69.32 |
| graph-tune-residual_unit_dropout-lr0.03 | tune | residual_unit_dropout | 0.03 | ok | 0.0857 | 97.78 | 5 | 0.0000 / 100.00 | 68.79 |
| graph-tune-residual_unit_dropout-lr0.1 | tune | residual_unit_dropout | 0.1 | diverged | — | — | — | — / — | 14.58 |
| graph-tune-sd_annealed-lr0.01 | tune | sd_annealed | 0.01 | ok | 0.0805 | 98.08 | 15 | 0.0000 / 100.00 | 56.97 |
| graph-tune-sd_annealed-lr0.03 | tune | sd_annealed | 0.03 | ok | 0.0843 | 98.04 | 20 | 0.0000 / 100.00 | 56.90 |
| graph-tune-sd_annealed-lr0.1 | tune | sd_annealed | 0.1 | diverged | — | — | — | — / — | 0.48 |
| graph-tune-sd_constant-lr0.01 | tune | sd_constant | 0.01 | ok | 0.0792 | 98.03 | 10 | 0.0000 / 100.00 | 56.90 |
| graph-tune-sd_constant-lr0.03 | tune | sd_constant | 0.03 | ok | 0.0805 | 97.84 | 5 | 0.0001 / 100.00 | 56.93 |
| graph-tune-sd_constant-lr0.1 | tune | sd_constant | 0.1 | diverged | — | — | — | — / — | 6.84 |
| pilot-normalized-plain | pilot | plain | 0.03 | ok | 0.0866 | 97.73 | 5 | 0.0229 / 99.34 | 6.38 |
| pilot-normalized-residual | pilot | residual | 0.03 | ok | 0.0732 | 97.85 | 3 | 0.0213 / 99.27 | 12.57 |
| pilot-normalized-sd | pilot | sd_annealed | 0.03 | ok | 0.0906 | 97.58 | 5 | 0.0313 / 98.96 | 7.65 |
| pilot-source-linear | pilot | plain | 0.001 | ok | 0.0760 | 97.82 | 5 | 0.0088 / 99.86 | 9.91 |
| pilot-source-plain | pilot | plain | 0.001 | ok | 0.0754 | 97.88 | 5 | 0.0098 / 99.84 | 6.65 |
| pilot-source-residual | pilot | residual | 0.001 | ok | 2.3026 | 10.22 | 1 | 2.3026 / 9.82 | 13.43 |

Training-probe metrics above use the deterministic dense-inference probe, not the full training set or stochastic minibatch loss. Diverged candidates are attempted outcomes and are ineligible for LR selection; their partial trajectories do not count as completed training.

### Frozen LR selection

LR selection uses validation CE alone among completed, nondiverged candidates, with lower LR breaking exact ties; validation accuracy is descriptive. No selected LR is inferred if selection.json is absent.

- plain: 0.03
- residual: 0.01
- residual_unit_dropout: 0.01
- sd_annealed: 0.01
- sd_constant: 0.01

## Verification and failures

- fidelity seed 101: 3 runs, pairing passed=True.
- fidelity seed 102: 3 runs, pairing passed=True.
- fidelity seed 103: 3 runs, pairing passed=True.
- main seed 101: 5 runs, pairing passed=True.
- main seed 102: 5 runs, pairing passed=True.
- main seed 103: 5 runs, pairing passed=True.
Reported scientific divergence or job failure in an exploratory pilot/tune is retained but does not invalidate otherwise verified final results. Invalid records, missing final runs and divergent/failed final runs do block completeness.

- graph-tune-residual_unit_dropout-lr0.1: training_diverged
- graph-tune-sd_annealed-lr0.1: training_diverged
- graph-tune-sd_constant-lr0.1: training_diverged
- graph-tune-residual_unit_dropout-lr0.1: stopped at epoch 21; last valid validation observation at epoch 20: CE 0.1324, accuracy 97.68%. These are partial, ineligible observations.
- graph-tune-sd_annealed-lr0.1: stopped at epoch 1; last valid validation observation at epoch —: CE —, accuracy —%. These are partial, ineligible observations.
- graph-tune-sd_constant-lr0.1: stopped at epoch 12; last valid validation observation at epoch 10: CE 0.1206, accuracy 97.49%. These are partial, ineligible observations.

### Excluded files

The following filenames were inventoried without opening their contents. They are outside the controlled-study manifest scope (including archived, optimization, qualification and ancillary metadata files):

`billing-latest.json`, `budget-ledger.json`, `collapse-diagnostic.json`, `dataset-manifest.json`, `graph-qualification-v1.json`, `opt-compile-214-v1.json`, `opt-compile-214-v2.json`, `opt-compile-214-v3.json`, `opt-confirm-b256-bf16-s1.json`, `opt-confirm-b256-bf16-s101.json`, `opt-confirm-b256-bf16-s102.json`, `opt-confirm-b256-bf16-s103.json`, `opt-confirm-b256-bf16-s104.json`, `opt-confirm-b256-bf16-s105.json`, `opt-confirm-normalized-dropout-step-s101.json`, `opt-confirm-normalized-dropout-step-s102.json`, `opt-confirm-normalized-dropout-step-s103.json`, `opt-confirm-raw-linear-dropout-step-s101.json`, `opt-confirm-raw-linear-dropout-step-s102.json`, `opt-confirm-raw-linear-dropout-step-s103.json`, `opt-diagnostic-seed104-linear.json`, `opt-diagnostic-seed104-relu.json`, `opt-kernels-214-v1.json`, `opt-kernels-214-v2.json`, `opt-recipe-b1024-linear-bf16-s1.json`, `opt-recipe-b256-linear-s1.json`, `opt-recipe-b256-lower-lr-s1.json`, `opt-recipe-b256-stronger-shrink-s1.json`, `opt-recipe-b512-linear-bf16-s1.json`, `opt-recipe-b512-lower-lr-s1.json`, `opt-speed-eager-tf32-b64-s1.json`, `opt-speed-fused-bf16-b128-s1.json`, `opt-speed-fused-bf16-b256-s1.json`, `opt-speed-fused-bf16-b512-s1.json`, `opt-speed-fused-bf16-b64-s1.json`, `opt-speed-fused-tf32-b256-s1.json`, `opt-speed-fused-tf32-b64-s1.json`, `opt-speed-graph-tf32-b64-s1.json`, `opt-stable-linear-dropout-s1.json`, `opt-stable-linear-dropout-step-s1.json`, `opt-stable-linear-step-fp32-s1.json`, `opt-stable-linear-step-s1.json`, `opt-stable-normalized-dropout-step-s1.json`, `opt-stable-relu-step-s1.json`, `optimization-analysis.json`, `tune-plain-lr0.01.json`, `tune-plain-lr0.03.json`, `tune-plain-lr0.1.json`, `tune-residual-lr0.01.json`, `tune-residual-lr0.03.json`, `tune-residual-lr0.1.json`, `tune-sd_annealed-lr0.01.json`, `tune-sd_annealed-lr0.03.json`, `tune-sd_constant-lr0.01.json`, `tune-sd_constant-lr0.03.json`, `tune-sd_constant-lr0.1.json`

## Interpretation limits

- Validation-selected checkpoint is primary; last fixed-epoch checkpoint is secondary. This resumed study selects LRs/checkpoints using validation only. Its official test set was previously inspected during separate baseline optimization, so the overall investigation is exploratory rather than globally test-naive.
- Main and fidelity cohorts are separate; plain versus residual also changes architecture.
- Means and sample SD concern seeds. Paired 95% t intervals are conditional on one fixed data split, the same test examples, and validation-selected LRs; no multiple-comparison correction.
- Repeated seeds reuse the same test examples; they are not independent dataset replications. Confirmation seeds 101–103 were also used in the preceding baseline optimization, so they are not globally untouched initializations. CI including zero is not proof of equivalence. The resumed final manifest uses three seeds per arm, with df=2 for complete paired intervals.
- Training seconds include gather, graph execution, optimizer/loop diagnostics; exclude loading/evaluation/checkpointing, graph setup and per-epoch mask/order/scaling preparation. Separate graph/preparation timings and total run time are retained. None is billed runtime or energy.
- Assigned A100 variants differ. Per-arm pooled runtimes are descriptive; speed effects require the same reported hardware/runtime and source within a seed pair, with separate hardware strata. Matching model names still does not control clock, worker contention or thermal state.
- First observed validation accuracy >=98% is discretized by evaluation cadence. Unreached runs remain censored; achiever-only means may be selection-biased.
- Initialization, data/split, and executed epoch-order hashes verify paired inputs for the resumed runs; equal batch size and drop-last determine minibatch order. Mask streams are independent and intentionally recipe-specific. Older pilot runs predate the order digest.
- Outputs omit checkpoints, wrong-index arrays, host identifiers and arbitrary exception messages; scientific GPU/runtime identifiers and paired disagreement counts are retained.

Raw result filenames and SHA-256 digests, per-seed scientific measurements, selection checks, and paired error-disagreement counts are retained in analysis.json. No checkpoints or private metadata are copied.
