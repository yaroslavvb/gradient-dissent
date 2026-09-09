# Depth robustness across architectures

## Scientific interpretation

Useful pruned models do transfer beyond GPT in these experiments, but the conclusion depends on the architecture, deletion pattern and metric. For both image architectures, the predefined two-thirds-depth models have higher mean accuracy under ILD than under dense training. The predefined relative-CE endpoint tells a more qualified story: GPT clearly favors ILD; ConvNeXt has large observed gains with wide intervals; ViT's early-exit comparison reverses because pruning improves the dense model's CE more. All intervals discussed here use three paired training seeds and are unadjusted for multiple comparisons; accuracy and additional masks are secondary outcomes.

ConvNeXt supplies the strongest practical image-classification contrast. Retaining 12 of 18 residual blocks reduces dense accuracy from 51.34% to 14.19%. Constant ILD retains 51.01% from a 54.62% full model, and decreasing ILD retains 50.62% from 54.12%. Nevertheless, the primary CE-damage effects are −4.795 [−10.705, 1.116] and −4.815 [−10.913, 1.283] nats: all three seed differences favor ILD, but both intervals include zero. The secondary paired pruned-accuracy gains are +36.82 [13.55, 60.10] and +36.43 [9.60, 63.26] percentage points. These are substantial observed benefits, with considerable uncertainty about their magnitude. ILD pruning slightly improves CE while reducing accuracy by about 3.5 points, illustrating why the two metrics should remain separate. Learned LayerScale stage means reach roughly 0.04–0.16 from an initialization of 10⁻⁶; residual scales remaining near their initial values cannot explain this result, although gamma alone does not measure branch importance.

ViT exposes a limitation of interpreting relative pruning damage as absolute predictor quality. Dense CE falls from 3.079 to 2.244 when retaining the first eight blocks; constant ILD goes from 2.713 to 2.155, and decreasing ILD from 2.561 to 2.209. Thus ILD receives less CE improvement from pruning, producing primary effects of +0.278 [0.226, 0.330] and +0.483 [0.389, 0.578]. Its pruned models still have lower mean CE and higher accuracy: 50.57% and 48.77%, versus dense 46.68%. Paired pruned-CE intervals include zero; secondary accuracy gains are +3.89 [2.04, 5.75] and +2.09 [1.52, 2.67] points. Dense ViT's large CE improvement accompanies only a 0.36-point aggregate accuracy decline. This is consistent with confidence or generalization problems, but calibration, logit magnitudes and individual prediction agreement were not measured. Low final minibatch losses alongside much larger held-out losses support concern about generalization, rather than an explanation that the vision models never learned.

The deletion pattern changes the ViT conclusion. With the three predefined random masks that retain the first block and eight blocks in total, dense accuracy averages only 17.52–24.15%; constant ILD gives 44.90–46.75%, and decreasing ILD 35.37–42.23%. These masks also favor both ILD recipes on relative CE damage, with all six secondary unadjusted intervals below zero. Prefix exit and intermediate deletion therefore probe different behavior at the same retained depth. Conversely, deleting the first block—a deletion never sampled by ILD training—leaves ViT at roughly 1% classification accuracy and GPT at only 1–3% token accuracy, and reduces ConvNeXt accuracy to 24.93%/17.85% under ILD, versus 37.56% for dense. The recipe does not create robustness to arbitrary missing blocks.

The 124M-parameter GPT arm reproduces the pruning effect with a clear full-model tradeoff. Prefix-eight CE damage falls from 4.199 nats for dense training to 0.093 with constant ILD and 0.116 with decreasing ILD; the paired effects are −4.106 [−5.390, −2.823] and −4.083 [−5.364, −2.802]. However, full-model CE worsens from 3.512 to 3.645/3.736, and token accuracy falls by 1.47/2.42 points. Robust subnetworks came with lower full-model quality at this budget. Each GPT run saw only 0.843 token presentations per parameter, so these measurements do not settle the paper's much longer training regime.

Decreasing-dropout training ends with useful subnetworks, but a universal advantage for that schedule does not appear. Constant ILD has better mean full and primary-pruned CE in GPT and ConvNeXt. Decreasing ILD has ViT's best full CE, while constant ILD has better prefix-eight CE and accuracy; at the more aggressive four-block exit, the ViT accuracy ranking changes again. These schedule comparisons are descriptive. Equal average masking does not isolate temporal order, because maximum probabilities and noise distributions also differ. The final step is dense, with no prolonged dense continuation. Stochastic depth in CNNs, LayerDrop and growing-subnetwork methods already provide precedents; the contribution here is a controlled, limited comparison of this recipe across the specified settings.

The next decisive checks would repeat vision training in a regime with a smaller train–held-out gap, add validation-only calibration diagnostics, and include directly trained smaller dense controls. Eight of nine learning-rate winners lie at a grid boundary, only one tuning seed was used, and both vision models share CIFAR-100. Separate initialization audits permit only explicitly reproduced numerical variants for the vision models; their tiny differences do not establish identical trajectories or negligible final-score effects. These limits belong to this experiment, not to the paper's reported runs. Training executed every branch before masking, so this study measures robustness and predictive quality, not saved training work, latency or energy.

Scientific-input SHA-256: `5bad33d925dd3eb17e43351a69239f8a661df59c0a61a2916a1079dbd3c7d64f`. This binds the interpretation to the raw runs, evaluation manifest, core source and tuning manifest, excluding timestamps and billing.

## Primary comparison

Pruning lowers mean ViT CE for all recipes, most for dense training; ConvNeXt's primary CE intervals include zero.

Verified summary generated: 2026-09-09T20:32:30.649328+00:00.

27 completed final runs; three paired seeds per recipe and architecture; nine predefined masks. The primary effect is treatment-minus-dense in CE(two-thirds retained) minus CE(full). Negative means less pruning damage. The six primary intervals are paired 95% Student-t intervals with 2 degrees of freedom, unadjusted for multiple comparisons. They condition on the selected learning rates and fixed data split. An interval including zero does not establish equivalence.

| Family | Recipe | Paired primary effect [95% CI] | Interpretation |
|---|---|---|---|
| ViT · CIFAR-100 | Constant ILD | +0.2777 [+0.2259, +0.3296] | The interval is entirely above zero: a higher CE change on pruning than with dense training. |
| ViT · CIFAR-100 | Decreasing ILD | +0.4833 [+0.3887, +0.5780] | The interval is entirely above zero: a higher CE change on pruning than with dense training. |
| ConvNeXt · CIFAR-100 | Constant ILD | -4.7947 [-10.7050, +1.1156] | The interval includes zero: the direction is unresolved by these three seeds; this is not an equivalence result. |
| ConvNeXt · CIFAR-100 | Decreasing ILD | -4.8148 [-10.9129, +1.2833] | The interval includes zero: the direction is unresolved by these three seeds; this is not an equivalence result. |
| GPT · WikiText-103 | Constant ILD | -4.1062 [-5.3895, -2.8228] | The interval is entirely below zero: a lower CE change on pruning than with dense training. |
| GPT · WikiText-103 | Decreasing ILD | -4.0830 [-5.3643, -2.8016] | The interval is entirely below zero: a lower CE change on pruning than with dense training. |

## Absolute predictive quality

Each cell is the mean [95% Student-t interval] across the three final training seeds. CE is in nats per GPT-2 token for language and nats per image for vision. Accuracy is token top-1 or image-class top-1, in percent. These task-specific metrics should not be pooled across families. A smaller pruning penalty can coexist with worse predictions. The symmetric t intervals are untruncated: an accuracy bound outside 0–100% is not an observed accuracy.

| Family | Recipe | Full CE | Primary-pruned CE | Full accuracy (%) | Primary-pruned accuracy (%) |
|---|---|---|---|---|---|
| ViT · CIFAR-100 | Dense | 3.0794 [2.9752, 3.1837] | 2.2436 [2.2188, 2.2684] | 47.04 [46.27, 47.81] | 46.68 [45.85, 47.51] |
| ViT · CIFAR-100 | Constant ILD | 2.7132 [2.5168, 2.9096] | 2.1551 [2.0055, 2.3048] | 49.55 [47.41, 51.69] | 50.57 [48.76, 52.38] |
| ViT · CIFAR-100 | Decreasing ILD | 2.5614 [2.4737, 2.6490] | 2.2089 [2.1577, 2.2600] | 48.40 [48.09, 48.71] | 48.77 [48.51, 49.03] |
| ConvNeXt · CIFAR-100 | Dense | 3.4572 [3.2867, 3.6277] | 8.1557 [2.2906, 14.0209] | 51.34 [49.87, 52.81] | 14.19 [-10.77, 39.15] |
| ConvNeXt · CIFAR-100 | Constant ILD | 2.7445 [2.7009, 2.7881] | 2.6483 [2.3811, 2.9156] | 54.62 [52.92, 56.32] | 51.01 [46.02, 56.01] |
| ConvNeXt · CIFAR-100 | Decreasing ILD | 2.8914 [2.7301, 3.0526] | 2.7751 [2.5961, 2.9541] | 54.12 [52.22, 56.02] | 50.62 [48.71, 52.53] |
| GPT · WikiText-103 | Dense | 3.5118 [3.4986, 3.5250] | 7.7108 [6.4078, 9.0138] | 38.48 [38.44, 38.52] | 15.38 [9.65, 21.11] |
| GPT · WikiText-103 | Constant ILD | 3.6452 [3.5953, 3.6951] | 3.7380 [3.6979, 3.7782] | 37.01 [36.39, 37.63] | 36.31 [35.85, 36.77] |
| GPT · WikiText-103 | Decreasing ILD | 3.7364 [3.7113, 3.7616] | 3.8525 [3.8339, 3.8710] | 36.06 [35.77, 36.35] | 35.22 [35.13, 35.30] |

The GPT and ViT primary intervention retains the first 8 of 12 blocks and the unchanged final readout. ConvNeXt retains stage prefixes of 2/2/6/2 from stages of 3/3/9/3 blocks, with all downsampling transitions; this is stagewise thinning, not early exit. The same nine masks are used across recipes and seeds, without post-pruning adaptation. First-block deletion is a separate stress test outside the ILD training support.

## Model size and training exposure

Exposures below are per final run and are matched across recipes within each family. Counts are sampled presentations, not unique examples. Equivalent passes mean presentations divided by training-set size, with replacement; they are not shuffled epochs or guaranteed complete passes.

| Family | Parameters | Residual blocks | Steps × batch | Input | Training presentations | Exposure ratio | Fixed test panel |
|---|---|---|---|---|---|---|---|
| ViT · CIFAR-100 | 85,219,684 | 12 | 10,240 × 256 | 32×32 pixels | 2,621,440 images | 58.25 equivalent passes with replacement | 10,000 images |
| ConvNeXt · CIFAR-100 | 27,897,028 | 18 | 10,000 × 512 | 64×64 pixels | 5,120,000 images | 113.78 equivalent passes with replacement | 10,000 images |
| GPT · WikiText-103 | 124,439,808 | 12 | 3,200 × 32 | 1,024 tokens | 104,857,600 tokens | 0.843 tokens / parameter | 251,904 target tokens in 246 actual windows |

All models are randomly initialized. The GPT arm is a short-budget architectural baseline, far below the reviewed paper's principal 20-token-per-parameter regime; it is not full-scale GPT-2 pretraining. The recorded number of scored test windows and target tokens is shown above; a requested maximum is a cap, not a guarantee of that many valid document-contained spans. These test windows are not canonical full-corpus perplexity. The ViT uses mean patch pooling without a class token. ConvNeXt bilinearly resizes CIFAR images from 32×32 to 64×64; interpolation adds no new image information. The two image families share CIFAR-100 and its split, so they are not independent dataset replications. “Transfer” means replication of an effect, not reuse of trained weights.

## Initialization verification adjustment

GPT retains byte-identical same-seed initial states. Vision exceptions below are restricted to separately audited seeds and hashes; all other pairings must match exactly.

ConvNeXt: initialization for the audited seeds is same-seed and numerically close, not byte-identical. Independent CPU reconstruction from the frozen source compared all 27,897,028 parameters for 4 audited seeds (1000, 2000, 2001, 2002). The worst absolute difference was 7.45e-09; the worst relative L2 difference was 5.36e-09, within required bounds of 1e-07 and 1e-06, respectively. The two variants have identical post-initialization RNG states.

ViT: initialization for the audited seeds is same-seed and numerically close, not byte-identical. Independent CPU reconstruction from the frozen source compared all 85,219,684 parameters for 4 audited seeds (1000, 2000, 2001, 2002). The worst absolute difference was 7.45e-09; the worst relative L2 difference was 5.64e-09, within required bounds of 1e-07 and 1e-06, respectively. The two variants have identical post-initialization RNG states.

Accepting these exact, independently audited vision hashes is a posthoc verification adjustment made independently of test outcomes. It does not change the frozen executed training code, learning-rate selection, masks, final seeds, or test-metric requirements; unrecognized initialization hashes still fail. These bounds describe initial tensors, not a bound on later training divergence.

The ConvNeXt worker probes locate the first observed difference at CPU erfinv after identical uniform draws, consistent with math-library roundoff in the inverse-error-function transform. They do not establish the exact dispatch path. These were CPU probes on existing workers, without additional GPU training jobs.

[Initialization audit narrative](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/initialization-audit.md) · [initialization-numerical-audit.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/initialization-numerical-audit.json) · [initialization-variant-manifests.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/initialization-variant-manifests.json) · [initialization-worker-probes.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/initialization-worker-probes.json) · [vit-initialization-numerical-audit.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/vit-initialization-numerical-audit.json) · [vit-initialization-variant-manifests.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/vit-initialization-variant-manifests.json) · [vit-initialization-cpu-probes.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/vit-initialization-cpu-probes.json)

## Recipe selection

Each family/recipe selects a learning rate from an equal three-rate grid using full-depth validation CE and a separate tuning seed. There are 27 validation-only tuning runs, distinct from the 27 final runs. No test or pruned score chooses the learning rate. Final comparisons concern independently tuned recipes; they do not isolate dropout at an identical learning rate. Boundary winners are disclosed below.

| Family | Recipe | Candidate learning rates | Selected learning rate | Grid boundary? | Tuning seeds | Tuning horizon |
|---|---|---|---|---|---|---|
| ViT · CIFAR-100 | Dense | 0.0001, 0.0003, 0.001 | 0.0001 | yes | 1000 | matched to final |
| ViT · CIFAR-100 | Constant ILD | 0.0001, 0.0003, 0.001 | 0.0003 | no | 1000 | matched to final |
| ViT · CIFAR-100 | Decreasing ILD | 0.0001, 0.0003, 0.001 | 0.0001 | yes | 1000 | matched to final |
| ConvNeXt · CIFAR-100 | Dense | 0.0003, 0.001, 0.003 | 0.0003 | yes | 1000 | matched to final |
| ConvNeXt · CIFAR-100 | Constant ILD | 0.0003, 0.001, 0.003 | 0.0003 | yes | 1000 | matched to final |
| ConvNeXt · CIFAR-100 | Decreasing ILD | 0.0003, 0.001, 0.003 | 0.0003 | yes | 1000 | matched to final |
| GPT · WikiText-103 | Dense | 0.0003, 0.0006, 0.0012 | 0.0012 | yes | 1000 | matched to final |
| GPT · WikiText-103 | Constant ILD | 0.0003, 0.0006, 0.0012 | 0.0012 | yes | 1000 | matched to final |
| GPT · WikiText-103 | Decreasing ILD | 0.0003, 0.0006, 0.0012 | 0.0012 | yes | 1000 | matched to final |

## Candidate validation measurements

These are full-depth validation means over the listed tuning seeds, not final test results. The selected rate is fixed by lowest validation CE; accuracy is shown as a diagnostic and never changes the selection. Accuracy can improve while CE worsens, so the objective matters. Top-1 is token accuracy for GPT and image-class accuracy for vision, in percent.

| Family | Recipe | Candidate LR | Validation CE (nats) | Validation top-1 accuracy (%) | Selected by CE? |
|---|---|---|---|---|---|
| ViT · CIFAR-100 | Dense | 0.0001 | 3.0123 | 47.60 | yes — lowest CE |
| ViT · CIFAR-100 | Dense | 0.0003 | 3.1417 | 48.18 | no |
| ViT · CIFAR-100 | Dense | 0.001 | 3.3655 | 18.80 | no |
| ViT · CIFAR-100 | Constant ILD | 0.0001 | 2.6682 | 48.86 | no |
| ViT · CIFAR-100 | Constant ILD | 0.0003 | 2.6195 | 50.36 | yes — lowest CE |
| ViT · CIFAR-100 | Constant ILD | 0.001 | 3.3467 | 19.30 | no |
| ViT · CIFAR-100 | Decreasing ILD | 0.0001 | 2.5175 | 48.58 | yes — lowest CE |
| ViT · CIFAR-100 | Decreasing ILD | 0.0003 | 2.8148 | 48.68 | no |
| ViT · CIFAR-100 | Decreasing ILD | 0.001 | 3.2263 | 21.84 | no |
| ConvNeXt · CIFAR-100 | Dense | 0.0003 | 3.3723 | 53.22 | yes — lowest CE |
| ConvNeXt · CIFAR-100 | Dense | 0.001 | 3.5934 | 53.08 | no |
| ConvNeXt · CIFAR-100 | Dense | 0.003 | 3.9584 | 51.12 | no |
| ConvNeXt · CIFAR-100 | Constant ILD | 0.0003 | 2.6818 | 56.54 | yes — lowest CE |
| ConvNeXt · CIFAR-100 | Constant ILD | 0.001 | 2.9240 | 57.82 | no |
| ConvNeXt · CIFAR-100 | Constant ILD | 0.003 | 2.9580 | 58.96 | no |
| ConvNeXt · CIFAR-100 | Decreasing ILD | 0.0003 | 2.8196 | 56.64 | yes — lowest CE |
| ConvNeXt · CIFAR-100 | Decreasing ILD | 0.001 | 3.0644 | 57.40 | no |
| ConvNeXt · CIFAR-100 | Decreasing ILD | 0.003 | 3.1933 | 59.02 | no |
| GPT · WikiText-103 | Dense | 0.0003 | 3.7522 | 36.26 | no |
| GPT · WikiText-103 | Dense | 0.0006 | 3.5988 | 37.55 | no |
| GPT · WikiText-103 | Dense | 0.0012 | 3.5144 | 38.33 | yes — lowest CE |
| GPT · WikiText-103 | Constant ILD | 0.0003 | 3.9715 | 33.67 | no |
| GPT · WikiText-103 | Constant ILD | 0.0006 | 3.7138 | 36.19 | no |
| GPT · WikiText-103 | Constant ILD | 0.0012 | 3.6799 | 36.37 | yes — lowest CE |
| GPT · WikiText-103 | Decreasing ILD | 0.0003 | 4.0365 | 33.09 | no |
| GPT · WikiText-103 | Decreasing ILD | 0.0006 | 3.7531 | 35.90 | no |
| GPT · WikiText-103 | Decreasing ILD | 0.0012 | 3.7285 | 35.98 | yes — lowest CE |

Constant ILD uses depth-increasing omission probabilities with maximum 0.4 and the first block always retained. Decreasing ILD linearly anneals the maximum from 0.8 to zero, reaching exactly one zero-dropout terminal step, without an extended dense continuation. Both schedules imply a 20% expected masking rate for example-block residual contributions, averaged over depth and training steps; realized Bernoulli masks vary. All branches execute before masking, so this is masked contribution, not saved execution. The recipes also differ in maximum strength and variance; the comparison does not isolate temporal order alone.

## Cost and execution scope

Latest saved metered usage: **$27.81**, queried **2026-09-09 20:31 UTC**. This timestamped snapshot may lag; it is not a final invoice or settled charge.

The reservation-ledger snapshot included in the summary generated **2026-09-09T20:32:30.649328+00:00** records **$39.67** across **60 invocation reservations**. The internal reservation ceiling is **$42.00** and the user-authorized external-compute cap is **$50.00**. Reservations are conservative budget allocations, not spending or invoices. The summary-generation timestamp is not a billing-query timestamp.

[Reservation ledger](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/budget-ledger.json) · [Saved metered snapshot](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/billing-latest.json)

Training computes dense residual branches before applying masks. Expected omission is therefore not a measured saving in FLOPs, time, GPU memory, energy or dollars. No joules were measured. Recorded GPU memory and training time are execution measurements rather than a controlled efficiency comparison.

## Limits

- Few final training seeds; intervals are fragile and unadjusted across primary/secondary comparisons.
- Learning-rate selection uncertainty and heldout dataset uncertainty are not included in seed intervals; tuning uses its recorded independent seed count.
- ConvNeXt and ViT same-seed initial weights can differ by CPU host. Separate post hoc audits accept only two measured, numerically close hashes per architecture/seed with matching post-init RNG; identical training trajectories are not established. GPT initialization remains bit-identically paired.
- Compute-then-mask executes dense branches; these runs do not establish FLOP, latency, memory, energy, or dollar savings.
- Peak reserved GPU memory can include allocator cache from previous calls or prevalidation; it is not the model's required VRAM. Peak allocated memory describes live tensors in this implementation, including resident data.
- Two-thirds retained residual blocks does not imply two-thirds FLOPs; ConvNeXt stage transitions always remain.
- Fixed image resizing adds no observed information; CIFAR spatial geometry differs from ImageNet.
- Small ConvNeXt LayerScale and undertraining can make deletion appear harmless; inspect full learning and recorded gamma magnitudes.
- Language uses its recorded fixed document-contained context windows, not canonical full-corpus perplexity.
- This is training from scratch under a short declared compute budget, not a reproduction of the target paper's LLM training scale.

ConvNeXt's existing stochastic depth is relevant prior work; applying dropout to a CNN is not itself new. Its residual contribution is γ × f(h): LayerScale magnitude alone cannot establish branch importance or irrelevance. This larger study includes neither separately trained smaller dense models nor alternating-dropout training; earlier toy controls do not supply those missing A100-scale comparisons.

[Code and raw measurements](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer) · [Verified statistics](data.json) · [All paired comparisons](paired.csv)
