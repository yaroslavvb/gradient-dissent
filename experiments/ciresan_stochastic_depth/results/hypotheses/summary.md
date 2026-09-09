# Exact-mask layer-drop hypotheses: MNIST checkpoint audit

Generated 2026-09-09T23:26:27.981693+00:00. All 30 checkpoint states verified. The policy manifest was frozen before test arrays were accessed by this postprocessor.

Primary comparisons use the selected checkpoint, three seed pairs and six masks retaining exactly two of four body branches. Stem/head remain. Every mask has its own MAC cost. Earlier test/seed/validation reuse makes this an exploratory investigation.

## Hypothesis outcomes

- H1 / sd_constant: interval includes zero; directional claim unresolved; -0.054 [-0.111, 0.004]; n=3.
- H4 / sd_constant: direction supported by the unadjusted three-seed interval; -0.110 [-0.121, -0.099]; n=3.
- H1 / sd_annealed: direction supported by the unadjusted three-seed interval; -0.059 [-0.104, -0.014]; n=3.
- H4 / sd_annealed: direction supported by the unadjusted three-seed interval; -0.110 [-0.115, -0.105]; n=3.
- H2 / residual: interval includes zero; directional claim unresolved; 0.013 [-0.001, 0.026]; n=3.
- H3 / residual: direction supported by the unadjusted three-seed interval; 0.918 [0.764, 1.072]; n=3.
- H5 / residual: 0/3 selected-checkpoint routers meet the empirical test tolerance; mean nominal MAC saving 26.25%. This is not yet a measured-latency conclusion.
- H2 / sd_constant: interval includes zero; directional claim unresolved; 0.005 [-0.014, 0.024]; n=3.
- H3 / sd_constant: direction supported by the unadjusted three-seed interval; 0.999 [0.706, 1.292]; n=3.
- H5 / sd_constant: 2/3 selected-checkpoint routers meet the empirical test tolerance; mean nominal MAC saving 34.61%. This is not yet a measured-latency conclusion.
- H2 / sd_annealed: interval includes zero; directional claim unresolved; 0.003 [-0.001, 0.007]; n=3.
- H3 / sd_annealed: direction supported by the unadjusted three-seed interval; 0.947 [0.826, 1.068]; n=3.
- H5 / sd_annealed: 2/3 selected-checkpoint routers meet the empirical test tolerance; mean nominal MAC saving 28.09%. This is not yet a measured-latency conclusion.

## Robustness and interactions

Mean values over seeds; confidence intervals below concern seed differences. Absolute mask accuracy/CE accompany excess losses in analysis.json. A lower-quality dense reference must not win solely by changing little.

| Recipe / endpoint | Dense accuracy % | Two-retained accuracy % | Excess CE | Harmful flips % | Mean per-example high-order logit fraction |
|---|---:|---:|---:|---:|---:|
| plain / selected | 97.913 | 8.621 | 2.3348 | 89.532 | 0.7103 |
| plain / final | 98.573 | 8.826 | 2.3993 | 89.904 | 0.7197 |
| residual / selected | 98.373 | 97.536 | 0.0650 | 1.087 | 0.1331 |
| residual / final | 98.400 | 97.498 | -0.0019 | 1.156 | 0.1595 |
| sd_constant / selected | 97.803 | 97.387 | 0.0115 | 0.760 | 0.0231 |
| sd_constant / final | 98.443 | 98.335 | -0.0656 | 0.294 | 0.0371 |
| sd_annealed / selected | 97.997 | 97.665 | 0.0062 | 0.619 | 0.0233 |
| sd_annealed / final | 98.510 | 98.354 | -0.0512 | 0.283 | 0.0315 |
| residual_unit_dropout / selected | 98.330 | 97.443 | 0.1928 | 1.138 | 0.1572 |
| residual_unit_dropout / final | 98.507 | 97.788 | 0.0282 | 0.957 | 0.1869 |

| Treatment / endpoint vs residual | Excess CE difference [95% CI] | Harmful-flip difference (pp) [95% CI] | Interaction-fraction difference [95% CI] |
|---|---:|---:|---:|
| plain / selected | 2.2698 [2.2419, 2.2978]; n=3 | 88.446 [84.174, 92.717]; n=3 | 0.5772 [0.5656, 0.5887]; n=3 |
| sd_constant / selected | -0.0535 [-0.1109, 0.0039]; n=3 | -0.327 [-0.759, 0.106]; n=3 | -0.1100 [-0.1214, -0.0987]; n=3 |
| sd_annealed / selected | -0.0588 [-0.1036, -0.0140]; n=3 | -0.467 [-0.929, -0.006]; n=3 | -0.1098 [-0.1149, -0.1047]; n=3 |
| residual_unit_dropout / selected | 0.1278 [0.0728, 0.1828]; n=3 | 0.051 [-0.322, 0.425]; n=3 | 0.0241 [0.0132, 0.0351]; n=3 |
| plain / final | 2.4012 [2.3027, 2.4998]; n=3 | 88.749 [84.436, 93.062]; n=3 | 0.5602 [0.5280, 0.5923]; n=3 |
| sd_constant / final | -0.0637 [-0.0998, -0.0275]; n=3 | -0.861 [-1.121, -0.601]; n=3 | -0.1224 [-0.1380, -0.1067]; n=3 |
| sd_annealed / final | -0.0493 [-0.0827, -0.0159]; n=3 | -0.873 [-1.099, -0.647]; n=3 | -0.1281 [-0.1345, -0.1216]; n=3 |
| residual_unit_dropout / final | 0.0301 [0.0080, 0.0521]; n=3 | -0.199 [-0.545, 0.147]; n=3 | 0.0274 [0.0244, 0.0305]; n=3 |

Plain deletions are forced crop surgery without a trained bypass. Unit dropout is secondary. These are not equivalent interventions to residual branch deletion.

## Does digit 1 remain more resilient after margin adjustment?

Contrast is digit 1 minus the mean of digits 4 and 9, on full-correct examples. Calibration freezes common-support bins and weights; sparse or empty support stays undefined. All ten digit profiles, counts, retained coverage, bin-specific counts and the 1,000-permutation reference are in analysis.json.

| Recipe / endpoint | Raw resilience contrast (pp) [95% CI] | Adjusted contrast (pp) [95% CI] |
|---|---:|---:|
| plain / selected | -17.579 [-63.464, 28.306]; n=3 | -17.477 [-63.605, 28.651]; n=3 |
| plain / final | -9.613 [-42.167, 22.942]; n=3 | -9.219 [-41.939, 23.501]; n=3 |
| residual / selected | 0.999 [0.754, 1.245]; n=3 | 1.260 [-0.065, 2.584]; n=3 |
| residual / final | 1.076 [0.705, 1.447]; n=3 | 1.216 [0.765, 1.667]; n=3 |
| sd_constant / selected | 0.902 [-0.172, 1.975]; n=3 | 0.519 [-1.362, 2.401]; n=3 |
| sd_constant / final | -0.029 [-0.229, 0.170]; n=3 | -0.499 [-1.281, 0.283]; n=3 |
| sd_annealed / selected | 0.697 [0.336, 1.058]; n=3 | 0.302 [-0.065, 0.669]; n=3 |
| sd_annealed / final | 0.209 [0.139, 0.278]; n=3 | -0.119 [-0.185, -0.054]; n=3 |
| residual_unit_dropout / selected | 0.815 [-0.186, 1.816]; n=3 | 0.449 [-1.040, 1.939]; n=3 |
| residual_unit_dropout / final | 1.061 [-0.095, 2.217]; n=3 | 0.784 [-0.957, 2.524]; n=3 |

## Local-direction and interpolation probes

Each row describes one checkpoint on its fixed validation probe. Gradients, norms and angles are correlated with single-deletion damage. First-order score is 1−SSE/SSE(zero change), not mean-baseline R². Interior-gain RMS error is evaluated at real network states; binary reconstruction alone is not evidence of linearity.

| State | Probe n | Gradient/damage rho | Norm/damage rho | Angle/damage rho | First-order score | Interior logit multilinear error RMS |
|---|---:|---:|---:|---:|---:|---:|
| 101/residual-selected | 128 | 0.873 | -0.036 | 0.228 | 0.975 | 0.0412 |
| 101/residual-final | 128 | 0.001 | -0.046 | 0.218 | 0.966 | 0.0608 |
| 101/sd_constant-selected | 128 | 0.977 | -0.036 | 0.177 | 0.945 | 0.0336 |
| 101/sd_constant-final | 128 | -0.367 | -0.190 | 0.281 | 0.996 | 0.1021 |
| 101/sd_annealed-selected | 128 | 0.856 | -0.105 | 0.061 | 0.968 | 0.0775 |
| 101/sd_annealed-final | 128 | -0.404 | -0.159 | 0.172 | 0.923 | 0.1395 |
| 102/residual-selected | 128 | 0.815 | -0.046 | 0.201 | 0.974 | 0.0416 |
| 102/residual-final | 128 | 0.065 | 0.006 | 0.257 | 0.990 | 0.0615 |
| 102/sd_constant-selected | 128 | 0.970 | 0.095 | 0.354 | 0.926 | 0.0344 |
| 102/sd_constant-final | 128 | -0.558 | -0.139 | 0.320 | 0.990 | 0.1196 |
| 102/sd_annealed-selected | 128 | 0.952 | -0.035 | 0.037 | 0.941 | 0.0405 |
| 102/sd_annealed-final | 128 | -0.457 | -0.167 | 0.188 | 0.985 | 0.1096 |
| 103/residual-selected | 128 | 0.874 | -0.111 | 0.263 | 0.982 | 0.0411 |
| 103/residual-final | 128 | 0.040 | -0.086 | 0.241 | 0.976 | 0.0621 |
| 103/sd_constant-selected | 128 | 0.936 | -0.172 | 0.116 | 0.974 | 0.0430 |
| 103/sd_constant-final | 128 | -0.426 | -0.141 | 0.286 | 0.993 | 0.0986 |
| 103/sd_annealed-selected | 128 | 0.937 | 0.044 | 0.151 | 0.897 | 0.0566 |
| 103/sd_annealed-final | 128 | -0.088 | -0.100 | 0.238 | 0.993 | 0.0941 |

All branch-level errors, sign agreement, Gaussian-direction contrasts, interpolation paths and inverse-survival mixture/Jensen gaps are retained. No global Jacobian norm, contraction certificate or read/write circuit is inferred.

## Frozen image-only routing

49 pooled image features; depth-three multioutput tree; first 5,000 validation examples fit, remaining 5,000 calibrate. The 0.2pp extra-error allowance is empirical, not a confidence guarantee. The random policy shuffles exactly the routed mask multiset. Full calibration tables are in policy-manifest.json and analysis.json.

| State | Router accuracy % | Full accuracy % | Static accuracy % | Same-cost random accuracy % | Router MAC saving % | Static MAC saving % | Test tolerance met |
|---|---:|---:|---:|---:|---:|---:|---|
| 101/plain-selected | 97.560 | 97.560 | 97.560 | 97.560 | 0.00 | 0.00 | True |
| 101/plain-final | 98.550 | 98.550 | 98.550 | 98.550 | 0.00 | 0.00 | True |
| 101/residual-selected | 98.210 | 98.430 | 98.320 | 98.170 | 20.93 | 25.07 | False |
| 101/residual-final | 98.240 | 98.400 | 98.320 | 98.180 | 16.77 | 25.07 | True |
| 101/sd_constant-selected | 97.600 | 97.760 | 97.410 | 97.650 | 13.64 | 29.25 | True |
| 101/sd_constant-final | 98.350 | 98.470 | 98.360 | 98.410 | 21.03 | 54.33 | True |
| 101/sd_annealed-selected | 98.110 | 98.220 | 97.970 | 98.130 | 24.40 | 45.97 | True |
| 101/sd_annealed-final | 98.210 | 98.470 | 98.310 | 98.280 | 64.48 | 58.50 | False |
| 101/residual_unit_dropout-selected | 98.170 | 98.490 | 98.060 | 98.230 | 28.51 | 37.61 | False |
| 101/residual_unit_dropout-final | 98.240 | 98.510 | 98.040 | 98.250 | 31.38 | 41.79 | False |
| 102/plain-selected | 98.220 | 98.220 | 98.220 | 98.220 | 0.00 | 0.00 | True |
| 102/plain-final | 98.570 | 98.570 | 98.570 | 98.570 | 0.00 | 0.00 | True |
| 102/residual-selected | 97.960 | 98.300 | 98.110 | 97.940 | 27.81 | 25.07 | False |
| 102/residual-final | 98.400 | 98.400 | 98.160 | 98.400 | 0.00 | 25.07 | True |
| 102/sd_constant-selected | 96.940 | 97.650 | 97.170 | 96.900 | 65.79 | 54.33 | False |
| 102/sd_constant-final | 98.030 | 98.410 | 98.140 | 97.990 | 71.00 | 66.86 | False |
| 102/sd_annealed-selected | 97.490 | 97.670 | 97.580 | 97.540 | 25.76 | 25.07 | True |
| 102/sd_annealed-final | 98.300 | 98.560 | 98.170 | 98.290 | 42.28 | 66.86 | False |
| 102/residual_unit_dropout-selected | 98.060 | 98.280 | 98.190 | 98.000 | 27.43 | 29.25 | False |
| 102/residual_unit_dropout-final | 98.400 | 98.620 | 98.420 | 98.440 | 25.50 | 37.61 | False |
| 103/plain-selected | 97.960 | 97.960 | 97.960 | 97.960 | 0.00 | 0.00 | True |
| 103/plain-final | 98.600 | 98.600 | 98.600 | 98.600 | 0.00 | 0.00 | True |
| 103/residual-selected | 97.900 | 98.390 | 97.700 | 97.890 | 29.99 | 37.61 | False |
| 103/residual-final | 98.040 | 98.400 | 97.840 | 97.850 | 29.00 | 29.25 | False |
| 103/sd_constant-selected | 97.830 | 98.000 | 97.680 | 97.890 | 24.40 | 54.33 | True |
| 103/sd_constant-final | 98.310 | 98.450 | 98.310 | 98.230 | 44.64 | 54.33 | True |
| 103/sd_annealed-selected | 97.820 | 98.100 | 97.820 | 97.820 | 34.09 | 45.97 | False |
| 103/sd_annealed-final | 98.100 | 98.500 | 98.180 | 98.170 | 63.34 | 66.86 | False |
| 103/residual_unit_dropout-selected | 98.060 | 98.220 | 97.910 | 98.030 | 7.64 | 29.25 | True |
| 103/residual_unit_dropout-final | 98.080 | 98.390 | 98.100 | 97.910 | 33.20 | 37.61 | False |

A lower-cost point is not an equal-cost victory. Static/dynamic costs can differ; same-cost random comparison isolates assignment association conditional on these fitted routers. Real feature/routing/gather/scatter latency is measured separately. No neural parameters were retrained.

## Verification, raw evidence and limits

Policy manifest SHA-256: `6aff7b936f9ff89478536cbd9468727a1c63c7e3894da98ee5033e2999b0ad52`. Original dense test error indices and CE, compact-artifact hashes and executed source-specification agreement all verify.

Sources: [analysis.json](analysis.json), [frozen policies](policy-manifest.json), and the three seed audit folders listed in the JSON artifact ledger. No checkpoint is copied into this report.

- The test set, validation set and seed IDs were reused in earlier tasks. This is exploratory, conditional on fixed checkpoints and one dataset, not independent replication.
- Primary is selected checkpoint and uniform six two-retained masks; final endpoints, unit dropout and plain surgery are secondary or diagnostic.
- Three-seed paired t intervals have df=2. No multiple-comparison correction; examples, classes and masks are not additional trained-model replicates.
- Class adjustment uses calibration-frozen common support and test full-correct examples. Coarse margin bins leave residual confounding; zero support is unidentified, not evidence of equality.
- Primary logit interaction fraction averages per-example ratios; it differs from ratio of pooled energies. Loss/accuracy nonlinearities can introduce interactions without serial paths.
- Local derivatives, angular changes and finite differences are probes, not global contraction certificates; equality at binary corners does not establish interpolation linearity.
- Routing uses pooled raw-image features only. Dense logits/margins and true labels are analysis references, not free deployment inputs.
- A depth-three router and lambda are fitted/calibrated on previously used validation examples. Empirical 0.2pp tolerance is not a confidence guarantee. No test-based refitting.
- MAC savings omit routing/grouping overhead. Measured latency is a separate benchmark; no energy or invoice claim follows from this analysis.
