# What makes a layer droppable? An exact-mask MNIST test of the Fable hypotheses

**Completed exploratory checkpoint audit, 9 September 2026.** We evaluated every combination of four removable branches in 30 existing checkpoint states, then fitted and froze inexpensive image-based routing policies before opening their test outcomes. No neural networks were retrained for this audit.

The evidence supports a specific claim: stochastic-depth training can create functional redundancy in these tapered residual MLPs. At the primary, validation-selected checkpoint, decreasing SD reduced deletion-induced cross-entropy by **0.0588 nats** relative to residual-dense training, with a three-seed paired 95% interval of **[−0.1036, −0.0140]**. The higher-order fraction of centered-logit mask interactions fell from **13.31% to 2.33%**. Neither result establishes better full-model accuracy.

The broader story is less favorable. The apparent advantage of digit 1 over digits 4 and 9 became uncertain after margin adjustment. Direct gradient predictions ranked deletion damage well at selected checkpoints, but their ranking deteriorated at the final checkpoint and the finite-difference directional probes had substantial numerical error. A shallow input router saved nominal MACs, yet exceeded its test error tolerance in five of nine primary cases and ran slower than dense inference. These distinctions matter when translating an appealing geometric explanation into an actual inference policy.

## Reading the controls

**All boxes unchecked does not remove the whole network.** Only the four middle affine branches are skipped. The learned feature layer (stem) still maps 784 pixels to 2,500 ReLU features; crop bypasses retain the first 500, and the learned classifier maps them to ten digit scores. This remains a trained one-hidden-layer predictor, using **16.42% of the counted affine work**. Ten percent is the expectation for uniform random guessing, not for this remaining trained network.

**SD means stochastic depth: randomly skipping whole branches during training.** “Constant SD” uses fixed drop probabilities of **10%, 20%, 30%, 40%**, from the first to last branch. “Decreasing SD” starts at **20%, 40%, 60%, 80%** and reduces them linearly to zero by epoch 100. Its average number of active middle branches therefore **increases from two to four**. Both recipes drop later branches more often; “decreasing” refers to drop probability over training, not to the number of active layers. No recipe that increases dropout over time was tested here.

The method selector chooses already-trained weights. The checkboxes then choose a fixed **inference mask**, without retraining or random sampling. Retained branches use gain one (no rescaling). A validation-selected checkpoint can occur before the decreasing schedule reaches zero.

## What was tested

The [frozen protocol](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/layerdrop_hypotheses/PROTOCOL.md) specifies five tests: deletion resilience, class differences after margin adjustment, directional sensitivity, mask interactions, and adaptive routing. The starting checkpoints come from the [controlled stochastic-depth study](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/ciresan-stochastic-depth.md), with seeds 101–103. Residual dense, constant SD, and decreasing SD supply the central 18 checkpoint states: three recipes × three seeds × two endpoints. Residual plus unit dropout supplies six secondary states; six plain-MLP states supply a deliberately disruptive surgery diagnostic.

The **primary endpoint** is the checkpoint selected by validation cross-entropy. Epoch 100 is a separately reported **secondary endpoint**. These are repeated measurements of the same training runs, not six independent seeds. Each interval below uses three seed-level estimates or same-seed differences, Student's t with two degrees of freedom. Masks and test images are not counted as independent model replications. The intervals are descriptive, unadjusted for the many reported comparisons, and can be wide; an interval including zero does not establish equivalence.

The network has widths 784→2500→2000→1500→1000→500→10 and 11,972,510 parameters. Its four eligible middle transitions have a prefix-crop shortcut followed by ReLU. The learned stem and classifier remain compulsory. Cropping loses coordinates and widths change, so this architecture is not an invertible, equal-dimensional flow. The main models use normalized inputs and linear output logits; original raw-input/ReLU-output fidelity runs are outside this audit.

A four-bit integer identifies the retained branches, shallowest first: mask 15 keeps all four and mask 0 removes all four. Retained branches have gain **one at inference**. SD training used inverse-survival gains, so an unscaled deletion network is not exactly the corresponding conditional training network. The primary deletion panel is the six masks retaining exactly two branches: 3, 5, 6, 9, 10, and 12, equally weighted. Half the eligible branches does not mean half the network or half the work.

Nominal affine MACs are `1,965,000 + 5,000,000*b0 + 3,000,000*b1 + 1,500,000*b2 + 500,000*b3`, versus 11,965,000 for the full model. The primary panel averages 58.21% of full affine MACs. Biases, ReLU, memory traffic, routing, and indexing are outside that count.

The official MNIST test set and these seeds were already used in earlier experiments, including [test-targeted baseline optimization](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/ciresan-optimization.md). The original validation set also selected learning rates and checkpoints. The new policy split and frozen choices protect this particular routing comparison from direct test fitting; they cannot make the overall investigation an untouched confirmatory evaluation.

## H1: stochastic depth reduces deletion damage, but quality must be measured separately

For each mask, excess CE is its loss minus the same checkpoint's full-model loss. Negative values are retained. Harmful flips are full-correct predictions made incorrect by deletion, with all image/mask pairs as the denominator; repaired full-model mistakes are recorded separately. The following are three-seed means over the six two-branch masks. Accuracy columns are percentages; CE is nats per image.

| Recipe | Endpoint | Full accuracy % | Two-branch accuracy % | Full CE | Two-branch CE | Excess CE | Harmful flips % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Residual dense | selected | 98.373 | 97.536 | 0.0738 | 0.1387 | +0.0650 | 1.087 |
| Constant SD | selected | 97.803 | 97.387 | 0.0790 | 0.0905 | +0.0115 | 0.760 |
| Decreasing SD | selected | 97.997 | 97.665 | 0.0725 | 0.0787 | +0.0062 | 0.619 |
| Residual dense | final | 98.400 | 97.498 | 0.0972 | 0.0953 | -0.0019 | 1.156 |
| Constant SD | final | 98.443 | 98.335 | 0.1336 | 0.0681 | -0.0656 | 0.294 |
| Decreasing SD | final | 98.510 | 98.354 | 0.1170 | 0.0658 | -0.0512 | 0.283 |

The primary paired comparisons use SD minus residual dense. A smaller excess CE means less deletion damage, while absolute masked accuracy answers a different question.

| Recipe | Endpoint | Excess CE difference [95% CI] | Harmful-flip difference pp [95% CI] | Masked-accuracy difference pp [95% CI] |
| --- | --- | --- | --- | --- |
| Constant SD | selected | -0.0535 [-0.1109, +0.0039] | -0.327 [-0.759, +0.106] | -0.149 [-0.989, +0.691] |
| Decreasing SD | selected | -0.0588 [-0.1036, -0.0140] | -0.467 [-0.929, -0.006] | +0.129 [-0.723, +0.982] |
| Constant SD | final | -0.0637 [-0.0998, -0.0275] | -0.861 [-1.121, -0.601] | +0.837 [+0.602, +1.072] |
| Decreasing SD | final | -0.0493 [-0.0827, -0.0159] | -0.873 [-1.099, -0.647] | +0.856 [+0.603, +1.109] |

Decreasing SD supports the primary directional redundancy hypothesis on this panel. Constant SD has the same favorable direction, but its primary excess-CE interval includes zero. Neither selected-checkpoint SD comparison resolves an absolute masked-accuracy advantage: the starting full-model accuracy differs too. At the final checkpoint, both SD recipes retain markedly more accuracy after deletion. Their masked CE is also lower than their own full CE; this is a measured loss improvement, not by itself an identified calibration mechanism.

The secondary unit-dropout control does not reproduce the same resilience: at the selected checkpoint its accuracy falls from 98.330% to 97.443% and excess CE is +0.1928; at the final checkpoint, 98.507% to 97.788% and +0.0282. Ordinary unit dropout is not interchangeable with training the precise branch bypasses used at inference.

Plain-network surgery is catastrophic: selected accuracy falls from 97.913% to 8.621%, and final accuracy from 98.573% to 8.826%. Here a deleted affine/ReLU transition is replaced by a crop/ReLU that the plain model was never trained to use. This is evidence that the intervention is disruptive, not an architecture-only causal proof that plain networks cannot be compressed or re-encode information.

### Removing every eligible branch is a useful secondary diagnostic

With all four body branches removed, accuracy is 93.507%, 95.370%, and 96.323% for selected residual, constant SD, and decreasing SD respectively. At epoch 100 the corresponding values are **93.163%, 97.310%, and 97.460%**. Only 16.423% of nominal full MACs remain.

This is still a learned 2500-unit stem, successive prefix crops retaining 500 coordinates, and a learned classifier. It is not a zero-computation predictor or an early-exit head. These predeclared all-skipped measurements demonstrate substantial learned redundancy, but were not used to select a new deployment policy from test accuracy.

Branch importance is unequal. Removing the first eligible branch saves 5 million MACs: final residual accuracy falls from 98.400% to 97.640%, versus 98.443%→98.360% for constant SD and 98.510%→98.400% for decreasing SD. Later branches can cost much less accuracy. A uniform layer-count rule obscures both compute and functional differences.

## H2: “digit 1 is easier to drop layers on” weakens after confidence adjustment

The predeclared contrast is resilience of digit 1 minus the mean resilience of digits 4 and 9, among full-correct examples. Resilience is the fraction of the six primary masks that remain correct. Positive values favor digit 1.

True-class-margin deciles were defined using full-correct policy-fit examples. Calibration data then froze common-support bins containing at least 20 full-correct examples of each target digit, and common bin weights. This standardizes each digit to a shared coarse margin distribution; it does not fully match individual examples or establish a causal digit effect.

| Recipe | Endpoint | Raw contrast pp [95% CI] | Margin-standardized contrast pp [95% CI] |
| --- | --- | --- | --- |
| Residual dense | selected | +0.999 [+0.754, +1.245] | +1.260 [-0.065, +2.584] |
| Constant SD | selected | +0.902 [-0.172, +1.975] | +0.519 [-1.362, +2.401] |
| Decreasing SD | selected | +0.697 [+0.336, +1.058] | +0.302 [-0.065, +0.669] |
| Residual dense | final | +1.076 [+0.705, +1.447] | +1.216 [+0.765, +1.667] |
| Constant SD | final | -0.029 [-0.229, +0.170] | -0.499 [-1.281, +0.283] |
| Decreasing SD | final | +0.209 [+0.139, +0.278] | -0.119 [-0.185, -0.054] |

All three primary adjusted intervals include zero. The raw ordering therefore does not support a reliable intrinsic class rule at this precision. Adjustment does not always reduce the estimate: selected residual increases in magnitude but becomes less precise. At the secondary final decreasing-SD endpoint, the raw positive contrast reverses to **−0.119 pp [−0.185, −0.054]** after adjustment. The final residual contrast remains positive; there is no universal disappearance of class association either.

The support and permutation results below show why a single attractive class ranking is insufficient. Coverage is the fraction of **full-correct examples of digits 1, 4, and 9** retained in the frozen common bins. The JSON also reports a fraction using all ten digits' full-correct examples as denominator; that is a different quantity.

| Recipe | Seed | Adjusted contrast pp | Common-support coverage | One-sided permutation p |
| --- | --- | --- | --- | --- |
| Residual dense | 101 | +0.743 | 2535/3083 (82.2%) | 0.0569 |
| Residual dense | 102 | +1.808 | 2405/3079 (78.1%) | 0.0010 |
| Residual dense | 103 | +1.227 | 2209/3077 (71.8%) | 0.0030 |
| Constant SD | 101 | +0.913 | 2711/3060 (88.6%) | 0.0250 |
| Constant SD | 102 | +0.999 | 2592/3057 (84.8%) | 0.0070 |
| Constant SD | 103 | -0.354 | 2480/3079 (80.5%) | 0.7612 |
| Decreasing SD | 101 | +0.159 | 2846/3070 (92.7%) | 0.3966 |
| Decreasing SD | 102 | +0.454 | 2731/3062 (89.2%) | 0.1209 |
| Decreasing SD | 103 | +0.293 | 2655/3070 (86.5%) | 0.2687 |

The table’s one-sided permutation p-values concern the digit-1-minus-mean(4,9) contrast. The separately recorded all-ten-digit permutation statistic is between-digit variance. The 1,000 within-bin permutations keep each example's six-mask result together and use frozen seed 20260913. Their p-values describe conditional label exchangeability within coarse margin bins; they are not three-seed replication tests, corrected multiple-comparison probabilities, or proof that digit identity causes resilience. Selected residual shows a positive adjusted association in all seeds, constant SD changes sign in seed 103, and decreasing SD has weaker per-seed permutation evidence. Between-seed variation remains important even when one per-example permutation p-value is small.

Every digit's counts, raw and standardized rates, and exploratory ten-digit rankings are available in the analysis artifact. Those additional rankings were not used to turn the primary digit-1 hypothesis into a favorable post hoc class rule. Confidence and labels here are analysis covariates, not free features available to an inference router.

## H3: direction predicts damage at selected checkpoints; the final and numerical checks limit the claim

At a full block output `h`, deletion substitutes the same-width post-ReLU cropped bypass `p`. The actual intervention is `delta = p − h`, not simply the negative pre-ReLU branch output. On a fixed 128-example validation probe, the first-order prediction is the downstream CE gradient dotted with this delta. Norm and angle are alternative fixed proxies.

The table averages the four branch-wise Spearman correlations within a seed, then across three seeds. The SSE score is `1 − SSE(first-order prediction)/SSE(zero change)`, not the conventional R² relative to predicting mean damage.

| Recipe | Endpoint | Gradient rank correlation | Norm rank correlation | Angle rank correlation | First-order SSE score |
| --- | --- | --- | --- | --- | --- |
| Residual dense | selected | 0.854 | -0.064 | 0.231 | 0.977 |
| Constant SD | selected | 0.961 | -0.038 | 0.216 | 0.948 |
| Decreasing SD | selected | 0.915 | -0.032 | 0.083 | 0.935 |
| Residual dense | final | 0.035 | -0.042 | 0.239 | 0.977 |
| Constant SD | final | -0.451 | -0.157 | 0.296 | 0.993 |
| Decreasing SD | final | -0.316 | -0.142 | 0.200 | 0.967 |

At selected checkpoints, gradient predictions beat both proxies in all 36 central branch/seed cases. At final checkpoints their rank correlations collapse or become negative, despite high all-example SSE scores. A few large-damage, already misclassified examples can dominate squared error while many tiny changes have unreliable ranks. The selected full-correct-only mean block scores range about 0.47–0.89, substantially below the all-example scores. Thus the favorable selected result is not a universal layer-importance law.

The independent audit also identified a serious limitation in the Gaussian directional comparison. Changing finite-difference epsilon from 0.001 to 0.01 changes the estimated centered-logit response by 6–44% for selected deletion directions and 35–115% for selected Gaussian directions, measured as RMS disagreement relative to the epsilon-0.01 response; final Gaussian disagreement reaches about 120%. Even the last eligible branch, followed only by a linear classifier, has 6–44% deletion and 35–58% Gaussian disagreement at selected checkpoints. A numerical component is therefore demonstrated; ReLU boundary crossings cannot explain all of it. The relative contributions of cancellation and nonlinear crossings elsewhere are not isolated.

For final checkpoints, 102–119 of the 128 probe examples have dense CE below FP32 machine epsilon. An independently recorded post hoc threshold sensitivity improves some correlations, but does not replace the frozen all-example analysis. We regard the finite-difference Gaussian anisotropy comparison as **numerically unresolved**, and do not infer a global Jacobian bound, contraction, or a complete read/write circuit from it.

## H4: stochastic depth reduces exact mask interactions, without making the network globally linear

All 16 masks permit exact Walsh and Möbius expansions on four binary gates. For the primary statistic, logits are first centered across classes, then each example's fraction of nonconstant Walsh energy at degree two or higher is computed. The reported mean averages those per-example fractions; it is not the ratio of pooled energies. No test example had an undefined near-zero denominator.

| Endpoint | Residual dense | Constant SD | Decreasing SD |
| --- | --- | --- | --- |
| selected | 13.310% | 2.307% | 2.330% |
| final | 15.952% | 3.714% | 3.146% |

At selected checkpoints, constant SD minus residual is **−11.003 percentage points [−12.135, −9.871]**, and decreasing SD minus residual is **−10.979 [−11.492, −10.467]**. Mean absolute higher-order energy also falls, from 0.6799 to 0.0533 and 0.0315 in centered-logit squared units. Thus the normalized effect is not just an expanding denominator. Full quality, total response scale, and architecture still differ between training recipes; this association alone does not identify why SD worked.

Exact binary-corner reconstruction is an algebraic check that every four-variable function passes, not evidence of actual linear interpolation. We also ran the network at withheld fractional gains 0.25, 0.5, and 0.75 for individual branches and along the all-branch diagonal. At selected checkpoints the mean centered-logit discrepancy from the full corner multilinear interpolant is 0.041, 0.037, and 0.058 RMS for residual, constant SD, and decreasing SD. The corresponding degree-one truncation errors are much larger: 3.656, 1.136, and 0.700. These interpolation errors are descriptive in logit units; a small normalized Walsh interaction fraction does not make every anchored degree-one approximation accurate.

Gate scaling is essential. The separately measured survival mixture compares the gain-one output with `rho*F(1/rho) + (1−rho)*F(0)` at each checkpoint's training survival probability. Inverse-survival scaling preserves the conditional mean of the gated affine contribution, not generally the downstream logits or expected loss. Nonzero mean-logit bias and CE Jensen gaps occur in the nontrivial probes. At the final decreasing-SD checkpoint, rho equals one: its zero mixture bias is an algebraic identity, not evidence for successful nonlinear averaging. Do not mix these scaled training probes with the gain-one inference deletion results.

CE has its own nonlinear interactions, even when logits are additive; its coefficients are not literal counts of computational paths. Low higher-order energy likewise does not bound the size of first-order deletion effects. The evidence supports a measured reduction in this mask-response statistic, not a claim that dropout works if and only if the rest of the network is linear.

## H5: a calibrated tree saves nominal work, but does not establish a useful adaptive advantage

Each checkpoint received one multioutput CART tree: maximum depth three, minimum leaf size 200, fixed seed 20260909. Its 49 features are 7×7 averages of raw image pixels, requiring neither labels nor a full-model forward. Using the first 5,000 validation positions, it predicts each mask's error change relative to dense inference. Dense predicted extra error is fixed to zero.

The remaining 5,000 positions choose among a frozen grid of 11 accuracy/cost tradeoffs. Selection maximizes nominal MAC savings subject to at most ten net additional calibration errors, with deterministic tie rules and a full-model fallback. An independent static mask uses the same empirical tolerance. The random comparator shuffles the router's test mask assignments, preserving their exact realized MAC distribution while removing image-dependent assignment. This is one frozen shuffle, not an average over random policies.

The policy manifest was written and hashed before local test-outcome access. Audit jobs had already computed mask test arrays, but these were withheld from fitting and selection. The tolerance is an empirical 0.2-percentage-point rule, not a confidence-certified generalization bound. Repairs can offset harmful flips, so net error is not the fraction of originally correct examples harmed.

All nine primary quality/cost points follow. Static and routed policies can use different amounts of work; only the shuffled comparator exactly matches routed nominal cost. Test tolerance means no more than 20 net extra errors among the 10,000 test images.

| Recipe | Seed | Full acc % | Routed acc % | Static acc % (mask) | Shuffled acc % | Routed MAC saved % | Static MAC saved % | Router test tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Residual dense | 101 | 98.43 | 98.21 | 98.32 (13) | 98.17 | 20.93 | 25.07 | fail |
| Residual dense | 102 | 98.30 | 97.96 | 98.11 (13) | 97.94 | 27.81 | 25.07 | fail |
| Residual dense | 103 | 98.39 | 97.90 | 97.70 (9) | 97.89 | 29.99 | 37.61 | fail |
| Constant SD | 101 | 97.76 | 97.60 | 97.41 (5) | 97.65 | 13.64 | 29.25 | pass |
| Constant SD | 102 | 97.65 | 96.94 | 97.17 (10) | 96.90 | 65.79 | 54.33 | fail |
| Constant SD | 103 | 98.00 | 97.83 | 97.68 (10) | 97.89 | 24.40 | 54.33 | pass |
| Decreasing SD | 101 | 98.22 | 98.11 | 97.97 (6) | 98.13 | 24.40 | 45.97 | pass |
| Decreasing SD | 102 | 97.67 | 97.49 | 97.58 (13) | 97.54 | 25.76 | 25.07 | pass |
| Decreasing SD | 103 | 98.10 | 97.82 | 97.82 (6) | 97.82 | 34.09 | 45.97 | fail |

Cross-entropy and separate harm/repair counts expose quality changes hidden by net accuracy. CE was recorded, but was not a post hoc policy-selection constraint.

| Recipe | Seed | Full CE | Routed CE | Static CE | Net extra errors | Full-correct harmed | Full-wrong repaired |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Residual dense | 101 | 0.0669 | 0.0648 | 0.0559 | +22 | 42 | 20 |
| Residual dense | 102 | 0.0812 | 0.1367 | 0.0637 | +34 | 52 | 18 |
| Residual dense | 103 | 0.0733 | 0.0884 | 0.1500 | +49 | 63 | 14 |
| Constant SD | 101 | 0.0758 | 0.0826 | 0.0865 | +16 | 29 | 13 |
| Constant SD | 102 | 0.0758 | 0.1221 | 0.1100 | +71 | 115 | 44 |
| Constant SD | 103 | 0.0855 | 0.0820 | 0.0771 | +17 | 32 | 15 |
| Decreasing SD | 101 | 0.0754 | 0.0694 | 0.0659 | +11 | 33 | 22 |
| Decreasing SD | 102 | 0.0735 | 0.0822 | 0.0788 | +18 | 55 | 37 |
| Decreasing SD | 103 | 0.0686 | 0.0717 | 0.0699 | +28 | 52 | 24 |

Five of nine selected routers fail the test tolerance, despite meeting it on calibration. Only three of nine selected static masks meet it on test. Some reductions in nominal work are substantial, but the expected error constraint transfers imperfectly even within MNIST. There is no reason to rename the tolerance after observing failures.

Across seeds, the selected router-minus-shuffled accuracy differences are +0.023 pp [−0.015, +0.061] for residual, −0.023 [−0.160, +0.113] for constant SD, and −0.023 [−0.086, +0.039] for decreasing SD. The image-dependent assignments therefore do not establish an advantage over their own matched-cost shuffle. Router-minus-static accuracy intervals also all include zero, and those static comparisons are not matched in cost.

The final-checkpoint policies are secondary and do not rescue this claim:

| Recipe | Full acc % | Routed acc % | Static acc % | Shuffled acc % | Routed MAC saved % | Static MAC saved % | Router tolerance passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Residual dense | 98.400 | 98.227 | 98.107 | 98.143 | 15.26 | 26.47 | 2/3 |
| Constant SD | 98.443 | 98.230 | 98.270 | 98.210 | 45.56 | 58.50 | 2/3 |
| Decreasing SD | 98.510 | 98.203 | 98.220 | 98.247 | 56.70 | 64.08 | 0/3 |

The final decreasing-SD router saves the most nominal MACs on average but misses the test tolerance in all three seeds. More removable computation does not ensure that this particular coarse tree identifies the right examples or masks.

Actual A100 inference further changes the practical conclusion: the grouped router is slower than full inference despite its nominal MAC savings. The independently timed job includes image features, routing, gathering, affine calls, and scattering, and its predictions match the functional audit for all nine primary checkpoints. Static masks can run faster because they retain regular batches; their quality tolerance still often fails. The timing and cost ledger below distinguish these execution measurements from the affine-MAC estimates.

## What this says about the Fable explanation and useful dropout choices

The accompanying [mathematical audit](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/fable-layerdrop-audit.md) supplies explicit counterexamples and qualifications. A residual update can re-encode coordinates: `x + (A−I)x = Ax`, including a rotation. Inverse-survival scaling does not make nonlinear outputs unbiased. A small angle is not the same as a small increment; a large radial rescaling can have zero angle. Deleting an ODE-like step does not automatically approximate a larger step over the original horizon. Small higher-order interaction energy does not bound a potentially large first-order loss effect. Those are mathematical limitations of the supplied argument, independent of whether these MNIST results are favorable.

The practical evidence favors testing dropout that matches an explicitly trained, dimensionally valid bypass and deployment convention. Whole-branch stochastic depth fits that intervention here; ordinary unit dropout supplies a weaker control for branch deletion. Preserve the distinction between training-time inverse-survival gains and gain-one inference masks. Start with a validation-selected, cost-aware static mask panel, report absolute quality alongside damage, and measure actual execution. Class identity or a full-model margin can diagnose confounding but is not a free routing feature. This particular shallow router adds overhead without a demonstrated assignment advantage; a different router or accelerator would require its own frozen quality and timing test.

The five hypothesis outcomes are therefore mixed: decreasing SD supports the primary redundancy contrast; a universal digit-class ordering does not survive the full adjustment and endpoint evidence; directional gradients are useful at selected checkpoints but numerically limited at final ones; SD reduces exact centered-logit interactions without proving global linearity; and this cheap adaptive router does not deliver a practical speed/quality win.

## Verification, reproducibility, and limits

All 30 planned states and all 16 masks completed. The all-kept audit forward matched source logits bitwise on the 128-example direct probe, and original full validation accuracy/loss and full test accuracy/loss/error indices were verified. The older validation records contain no error-index lists. The independent audit checked 99 remote artifact hashes and sizes, 60 dense split references, and all 30 checkpoint states. Mechanistic probes used all 128 predeclared validation examples; none were reduced or selected for favorable correctness. Bulk-versus-probe predictions also agreed, while batch-shape logit differences were recorded. The compact per-example energy arrays reproduce the reported interaction fractions; the independent audit did not regenerate every transform from archived full logits.

The policy manifest SHA-256 is `6aff7b936f9ff89478536cbd9468727a1c63c7e3894da98ee5033e2999b0ad52`. The recorded analyzer source hash matches the implementation used to fit it. The initial evaluation refusal concerned local wrapper JSON versus exact remote source JSON; retrieving the pinned originals resolved that provenance mismatch without changing scientific tolerances, policies, or fits.

Three seeds, one reused dataset, one tapered MLP architecture, reused validation selection, multiple exploratory comparisons, a 128-example mechanism probe, and unresolved finite-difference precision all limit extrapolation. The policies optimize calibration net error, not class fairness or confidence-certified risk. MACs are not energy or latency. Results at all-skipped masks, final checkpoints, unit-dropout controls, and plain surgery remain separately labeled secondary evidence.

The [analysis JSON](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/analysis.json) contains every planned checkpoint, all mask/class/example summaries, both endpoints, seed-level values and intervals, calibration candidates, policy outcomes, and artifact hashes. The [independent check](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/independent-check.json) records numerical and provenance verification, including post hoc precision diagnostics. The [policy manifest](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/policy-manifest.json), [analysis implementation](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/layerdrop_hypotheses/analyze.py), and [audit implementation](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/layerdrop_hypotheses/audit.py) make the frozen choices inspectable. The [routing summary](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/routing-summary.json) records actual inference measurements.
