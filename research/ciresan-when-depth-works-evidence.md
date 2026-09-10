# When stochastic depth helps: evidence for a practical recipe

**Evidence synthesis, 9 September 2026.** In these MNIST experiments, the strongest use for stochastic depth is learning a network whose residual branches can later be removed with limited damage. Improving full-network generalization, reducing training time, and accelerating inference are separate claims. Only some succeed. There is no dependable rule that digit 1 intrinsically needs less depth than digits 4 and 9.

This synthesis combines the [original controlled training study](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/ciresan-stochastic-depth.md), its [exact-mask checkpoint audit](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/mnist-layerdrop-hypotheses.md), and the later [offline curvature study](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/curvature-summary.md). Training and deletion results below use the original controlled runs; curvature uses the telemetry rerun. They must not be treated as additional independent replications of the same endpoint.

## What “works” means

The central comparison is residual dense versus constant or decreasing stochastic depth (SD), with three seeds. The primary checkpoint was chosen by validation cross-entropy (CE); epoch 100 is secondary. The following intervals are paired, three-seed 95% t intervals, conditional on the selected recipe and checkpoint rule.

| Goal | Measured result | Practical interpretation |
| --- | --- | --- |
| Improve primary full-model accuracy | Constant SD minus residual: −0.570 percentage points [−0.958, −0.182]; decreasing SD: −0.377 [−0.931, +0.177] | No primary accuracy benefit; constant SD is worse on this endpoint. |
| Improve final full-model accuracy | Constant +0.043 pp [−0.033, +0.119]; decreasing +0.110 [−0.004, +0.224] | Small favorable means remain uncertain and are secondary. |
| Resist removal of two of four branches | Decreasing SD lowers deletion-induced CE by 0.0588 nats [0.0140, 0.1036] relative to residual at the primary checkpoint | Evidence for redundancy, even though full-model accuracy did not improve. |
| Train for 100 epochs faster | Residual 66.38 s; constant 56.84 s; decreasing 56.87 s | Actual branch skipping saves about 14.3–14.4% of this training clock. |
| Reach 98% validation accuracy faster | First scheduled observation: residual 6.69 s; constant 8.57 s; decreasing 9.04 s | Cheaper epochs did not produce faster time to this quality target. Evaluations were five epochs apart. |

These are measured A100-SXM4-40GB timings, not energy measurements or cold job latency. The [training study](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/ciresan-stochastic-depth.md) records total run time separately. A smaller generalization gap also does not suffice: less fitting can shrink the gap while absolute validation or test quality deteriorates.

The primary deletion panel averages all six masks retaining two eligible branches. Selected full→masked accuracy is 98.373%→97.536% for residual, 97.803%→97.387% for constant SD, and 97.997%→97.665% for decreasing SD. Thus reduced damage does not establish a higher absolute masked accuracy at this endpoint. At epoch 100, masked accuracy is 97.498%, 98.335%, and 98.354%; SD-minus-residual differences are +0.837 pp [+0.602, +1.072] and +0.856 [+0.603, +1.109]. Both SD models' masked CE is lower than their own full CE at epoch 100. Negative excess CE is a valid outcome, not a value to clamp to zero. [Exact-mask results](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/mnist-layerdrop-hypotheses.md).

## Which layers, and how much computation?

This is a tapered MLP, 784→2500→2000→1500→1000→500→10. Four middle affine branches have trained prefix-crop shortcuts. Those shortcuts discard coordinates; they are dimensionally valid bypasses, not identity maps on the original wider state. The stem and classifier always remain. Ordinary unit dropout does not train the same intervention: its selected two-branch excess CE is +0.1928, versus +0.0115/+0.0062 for constant/decreasing SD. Forcing previously untrained crop bypasses into the plain MLP destroys accuracy; that surgery is not a proof that plain networks cannot be compressed. [Architecture and controls](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/mnist-layerdrop-hypotheses.md).

Individual branches differ in both value and cost. The table gives descriptive epoch-100 mean accuracy across three seeds, using the predeclared masks. It is not a deployment choice selected on test performance.

| Inference intervention | Affine MACs removed | Residual accuracy % | Constant SD % | Decreasing SD % |
| --- | ---: | ---: | ---: | ---: |
| Full network | 0 | 98.400 | 98.443 | 98.510 |
| Remove eligible branch 1 | 5,000,000 (41.79%) | 97.640 | 98.360 | 98.400 |
| Remove eligible branch 2 | 3,000,000 (25.07%) | 98.193 | 98.417 | 98.430 |
| Remove eligible branch 3 | 1,500,000 (12.54%) | 98.273 | 98.457 | 98.467 |
| Remove eligible branch 4 | 500,000 (4.18%) | 98.300 | 98.460 | 98.510 |
| Remove all four | 10,000,000 (83.58%) | 93.163 | 97.310 | 97.460 |

Full cost is 11,965,000 affine MACs. Removing every eligible branch still leaves 1,965,000 MACs: a learned 2500-unit stem, prefix crops retaining 500 coordinates, and the classifier. This is substantial remaining computation, not a zero-depth predictor. Conversely, the six two-branch masks average 58.21% of full affine MACs. “Half the branches” is not “half the work.” The [analysis JSON](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/analysis.json) contains all sixteen masks and both endpoints.

## Which digits tolerate deletion?

The predeclared contrast asks whether digit 1 is more resilient than the average of digits 4 and 9. Resilience is the fraction of the six two-branch masks remaining correct, conditional on the full model being correct. Positive numbers favor digit 1. Matching uses shared, validation-defined bins of true-class logit margin; this controls one coarse measure of confidence, not every image characteristic.

| Recipe and endpoint | Raw contrast, pp [95% CI] | Margin-standardized contrast, pp [95% CI] |
| --- | --- | --- |
| Residual, selected | +0.999 [+0.754, +1.245] | +1.260 [−0.065, +2.584] |
| Constant SD, selected | +0.902 [−0.172, +1.975] | +0.519 [−1.362, +2.401] |
| Decreasing SD, selected | +0.697 [+0.336, +1.058] | +0.302 [−0.065, +0.669] |
| Residual, final | +1.076 [+0.705, +1.447] | +1.216 [+0.765, +1.667] |
| Constant SD, final | −0.029 [−0.229, +0.170] | −0.499 [−1.281, +0.283] |
| Decreasing SD, final | +0.209 [+0.139, +0.278] | −0.119 [−0.185, −0.054] |

All three **primary adjusted intervals include zero**. This does not prove equal resilience, but it prevents a confident universal digit-1 rule. At the secondary decreasing-SD endpoint, adjustment reverses the sign with an interval below zero. Residual's final adjusted association remains positive: adjustment does not universally eliminate class differences either.

Common bins cover 71.8–92.7% of full-correct target-digit examples at selected checkpoints; the excluded images limit generalization. Within-bin permutation tests give another perspective, not extra seed replications: decreasing SD's selected one-sided p-values are 0.121–0.397; constant SD changes adjusted sign in one seed. Three seeds, coarse matching and many exploratory comparisons constrain the claim. The [class audit](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/mnist-layerdrop-hypotheses.md) provides support counts, permutation definitions and all ten digits' exploratory rankings. Neither the true label nor a full-model margin is a free routing feature at inference.

## Saved arithmetic becomes speed only with suitable execution

A validation-fitted depth-three tree using 49 pooled-pixel features was inexpensive as a model, yet expensive as a GPU execution policy. Across nine primary residual/SD checkpoints, median whole-test inference time was 4.828 ms dense, 3.449 ms for a static mask and 13.589 ms routed. Per-model routing was 2.07–3.25× slower than dense; static inference was 1.19–1.93× faster. Timing included routing, grouping, gathers and scattering, using GPU-resident inputs and batch 2048. It excluded initial loading. [Measured routing results](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/routing-summary.json).

Quality still mattered: only 4/9 routers and 3/9 static policies met the 0.2-percentage-point extra-error tolerance on test, despite passing the empirical calibration criterion. The router's advantage over shuffled assignments with the identical realized MAC distribution was unresolved in every primary recipe. Static and routed policies had different costs, so their accuracy comparison is not cost matched. This supports starting with a static mask panel and measuring actual hardware latency; it does not establish that every static policy is acceptable or every adaptive router must fail. [Policy selection and outcomes](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/mnist-layerdrop-hypotheses.md).

## What transfers from existing work—and what does not

Huang et al.'s **Deep Networks with Stochastic Depth** (2016) established whole-minibatch residual bypass training in deep convolutional networks, with full-depth testing. Its motivation and architecture already extend beyond language models. The original convention uses unscaled surviving branches during training and survival-weighted branches at test; our experiments use inverse-survival training gains and gain-one inference. Those conventions must be stated explicitly when reproducing or pruning a model. [Original paper, §3](https://arxiv.org/html/1603.09382v3).

Fan, Grave and Joulin's **LayerDrop** explicitly joins two goals: regularization during training and extracting shallower Transformer subnetworks without finetuning. Our separation of full-model quality from deletion resilience follows that distinction. Neither its results nor this MNIST study imply arbitrary layers can be removed from arbitrary trained networks. [LayerDrop](https://arxiv.org/abs/1909.11556).

Modern vision code also distinguishes regularization from execution saving. ConvNeXt uses depth-increasing DropPath, but its official block computes the convolution/MLP branch before applying DropPath to its output. Consequently that implementation's masked output is not evidence that the branch's forward computation was avoided. This is an inference directly from the [official ConvNeXt implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py). Our specialized graph runner actually skips the affine branch; a library named “stochastic depth” need not provide the same speed benefit.

## A defensible practical recipe

1. **Establish a stable dense baseline and the objective first.** Measure full accuracy/CE, time to a validation target and fixed-budget performance separately. Adding SD did not rescue the historical raw-input/ReLU-head failures.
2. **Train the bypass that deployment will use.** Handle changes in width or resolution explicitly, preserve required stem/head operations, and document training and inference gains. Branch dropout and unit dropout are different interventions.
3. **Tune drop strength and schedule on validation.** The tested constant rates were [0.1, 0.2, 0.3, 0.4]; the decreasing schedule started at twice these and reached zero only at the final epoch. Equal mean omission does not isolate temporal ordering: peak probability and variance also change. These values are tested settings, not universal optima.
4. **Choose masks using quality and cost together.** Keep a fixed panel including full depth; measure absolute masked quality as well as damage relative to full. Use validation for selection, then test once on data outside the selection process. A digit stereotype or a low branch count is not an adequate policy.
5. **Benchmark the intended execution path.** Static branch skipping is the promising deployment route in this audit. Adaptive routing needs both an assignment advantage at matched cost and enough saved execution to pay for grouping and routing.

Mechanistic measurements refine this advice without proving a universal explanation. SD lowers selected higher-order centered-logit mask interactions from 13.31% to about 2.3%, consistent with more independently usable branches. Selected-checkpoint gradients predict deletion damage well, but final-checkpoint ranks and finite-difference probes have precision problems. Separately, the FP64 curvature rerun shows lower loss curvature alongside much larger raw logit sensitivity than at initialization. Its 128-example training probe is saturated, one seed only, and KFAC diagonal errors reach 83–583% at epoch 100. Thus “flatter,” “more linear,” and “better generalization” cannot substitute for the actual quality/deletion/timing measurements. [Mask and mathematical audit](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/fable-layerdrop-audit.md), [curvature evidence](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/curvature-summary.md).

The whole investigation remains exploratory: one reused MNIST test set, three reused confirmation seeds, a particular tapered architecture, validation-dependent model selection and unadjusted intervals. The practical result is a tested direction—train removable branches and assess them explicitly—not a theorem that SD improves every model or that simple digits require fewer layers.
