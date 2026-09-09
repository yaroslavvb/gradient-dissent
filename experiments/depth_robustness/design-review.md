# Independent design review: depth robustness toy replication

Prepared 2026-09-09 before inspecting new training outcomes. This document contains recommendations for the experimental protocol; the executed manifest and final report determine what was actually run. It is not a preregistration or a claim that every recommended ablation was performed.

## Scope and estimands

The proposed six-block, width-64 causal transformer is a useful advance over the repository's earlier residual-MLP toy. It tests whether layer-dropout pretraining reduces the damage caused by deleting transformer blocks. It cannot quantitatively reproduce the paper's 271M–8.2B models, CompleteP transfer, accelerator savings, or speculative decoding.

Use three distinct quantities for each run and each retained depth k:

- Full-model test cross-entropy, CE(full), to expose any cost to the intact model.
- Submodel cross-entropy, CE(k), to show the deployed submodel's absolute quality.
- Degradation, D(k) = CE(k) − CE(full), to isolate robustness relative to that run's own intact model. exp(D(k)) is the submodel/full-model perplexity ratio.

The recommended primary endpoint is four-of-six prefix deletion robustness, D(4). Estimate the paired difference D_dropout(4) − D_dense(4) across final training seeds; negative values indicate less degradation. Report raw CE(4) and full CE alongside it. Small degradation alone does not make a weak model useful. Treat other depths and masks as descriptive secondary endpoints, or explicitly account for multiple comparisons if reporting formal significance.

## Training comparisons

1. Same six-block architecture, dense versus constant increasing-with-depth dropout (maximum .4) versus increasing-with-depth/time-decreasing dropout (initial maximum .8). Both dropout recipes have exactly .2 mean omission under discrete endpoints l/(L−1) and 1−t/(T−1). For L=6 the first block is never dropped. Log expected and realized per-layer survival.
2. Match tokens and optimizer steps for the primary comparison. Tune learning rate with identical candidate grids, training durations, validation windows, and tuning seeds for all recipes. Do not use test loss or pruned test performance to choose the rate. Select the rate on full-depth validation loss if that is the intended deployment quality target, and state that this does not optimize pruning performance directly.
3. Use separate final seeds. Five is a useful small-budget minimum, but intervals remain wide and only represent variation over initialization/data-order/mask randomness on these tasks. Pair each seed's initial weights and data stream across recipes; mask randomness needs its own generator so it does not change subsequent data draws.
4. A dense four-block model is the most interpretable smaller-architecture control for the primary four-block exit. Tune it equally. Its full loss measures whether pruning the large model competes with training the desired depth directly. A same-token control has fewer block computations; a matched expected active-block budget control can train it longer, with its LR schedule ending at its own endpoint. These answer different questions.
5. If testing matched expected active-block budgets, use completed runs with separately terminal LR schedules. For T dense-six steps, mean-.2 dropout needs 1.25T steps and dense-four needs 1.5T. Record integer rounding and exact endpoint averages. Label this modeled block work rather than actual FLOPs, elapsed compute, or dollars when training computes every sequence before masking.

Do not add controls at the expense of training enough to make the task meaningful. An initial development run can determine a practical duration, but its seed and losses must be separated from final evaluation. If all learning-rate winners lie on the search boundary, extend the grid equally or explicitly report the unresolved tuning limitation.

## Data and evaluation

For Tiny Shakespeare, pin the download revision, record its SHA-256, and split the character stream into disjoint contiguous train/validation/test segments before constructing windows. Fixed evaluation windows should avoid overlapping scored targets where practical and must be identical for every run and pruning mask. Avoid windows crossing split boundaries. Character-level perplexity is not comparable with the paper's token-level perplexity.

A second, fresh-data synthetic task can distinguish finite-corpus regularization from structural robustness. A delayed/repeated-copy task should report accuracy and CE specifically on the positions requiring recall, because predictable markers and easy input positions can dilute the result. Fixed-length copying can be learned by positional shortcuts; call it a toy sequence task rather than a demonstration of general reasoning. Use separate data RNG seeds for validation and test, and disclose the task distribution.

Evaluate without adapters, auxiliary exit losses, fine-tuning, or mask selection on the test set. A prefix of k blocks uses original blocks 0 through k−1, followed by the same final normalization and readout. It does not rescale the remaining residuals to compensate for removed layers. The layer order and positional encoding remain unchanged.

For noncontiguous deletion, specify the exact original block indices. At six layers it is feasible to enumerate all subsets: the total across retained depths 2–6 is 57. Since ILD never drops block 0, first report the 31 subsets at depths 2–6 that preserve block 0, then treat unrestricted deletion as an explicit out-of-training-distribution stress test. Do not report a best test-selected subset as generic pruning robustness. If runtime requires a sampled panel, publish the masks, select them independently of outcomes, and share the same panel across methods. An average over masks is an average over submodel losses, not the loss of an ensemble of predictions.

Show per-seed points and intervals across training seeds. Use a paired Student-t interval for paired differences, with degrees of freedom n−1; neither tokens, evaluation batches, nor pruning masks are independent training replicates. State that these intervals do not capture dataset/architecture uncertainty. If showing a mask distribution, distinguish its spread from uncertainty across training seeds.

## Correctness checks worth running

- **No-drop equivalence:** probabilities identically zero in the dropout training path reproduce the dense output and gradients with identical parameters.
- **Shared per-sequence masks:** one Bernoulli draw per sequence and block, broadcast across token and channel dimensions and shared by attention and FFN. Independent block draws; no per-token or per-sublayer draws hidden in the implementation.
- **Scaling:** each retained attention and FFN residual uses 1/(1−p), as in Eq. 6, and evaluation uses scale one. There should be no additional whole-block inverse-survival scaling. Do not claim equality of expected train activations and eval activations; the earlier repository counterexample applies.
- **All-kept/all-dropped semantics:** forcing every multiplier to zero returns the input residual unchanged; forcing all masks to one at p=0 gives the dense block. All-kept at p>0 is intentionally scaled and is not dense-equivalent.
- **Pruning equivalence:** depth-six or all-blocks-selected evaluation matches ordinary evaluation; a k-block prefix matches an explicit forward pass through those original blocks plus the unchanged final normalization/readout.
- **Causality:** changing future input tokens cannot change earlier logits in evaluation.
- **Schedule/accounting:** verify depthwise first/last probabilities and temporal first/final endpoints; mean omission is exactly .2 for both proposed recipes when L,T>1. Count realized active sequence-blocks separately from all actually executed blocks.
- **Optimizer semantics:** compute-then-mask creates explicit zero gradients for fully dropped blocks, allowing AdamW momentum and decay. If the implementation truly skips all rows of a block, explicitly decide whether to materialize zero gradients or leave them absent; these give different optimizer trajectories.
- **Dataset boundaries and targets:** next-token labels are shifted by one; no scored test target appears in training windows; evaluation order and mode do not draw training masks or change training RNG state.
- **Reproducibility:** record source hash, environment, device, precision, thread count, seeds, data hash, parameter count, selected hyperparameters, wall time, and actual cloud charges. At least one short repeatability check is useful, but do not assert deterministic MPS execution without testing it.

## What can and cannot be inferred about the paper

The paper's Section 8.1.1 explicitly defines static early exit as keeping the embedding, a prefix of original transformer blocks, and the unembedding. Section 8.1.2 separates this from intermediate-layer skipping and reports that alternating layer dropout can be better for the latter. Therefore an ILD-only experiment can reproduce an ILD depth-robustness mechanism, but cannot establish which training mask distribution is best for arbitrary deletion. [Don't Drop Dropout, Sections 8.1.1–8.1.2](https://arxiv.org/abs/2609.05275).

LayerDrop had already demonstrated pruning transformer subnetworks without fine-tuning. A positive toy result supports this broader family of structured-dropout effects; it is not unique evidence for the 2026 recipe. [Fan et al., LayerDrop](https://arxiv.org/abs/1909.11556).

LayerSkip combines increasing depthwise dropout with a shared early-exit loss. Omitting that auxiliary objective here makes this a cleaner check of dropout alone, but it is not an apples-to-apples LayerSkip reproduction. [Elhoushi et al., LayerSkip](https://aclanthology.org/2024.acl-long.681/).

A decreasing schedule ending at zero can retain robustness learned earlier. A single zero-dropout endpoint step does not show robustness survives a substantial dense continuation; testing that stronger claim requires an explicit extra dense tail with a fixed length. A constant schedule outperforming a decreasing schedule on this toy would limit universality, not contradict a reported ranking on different datasets and model scales.

No result in these experiments demonstrates inference latency, energy efficiency, self-speculative acceptance, or lossless decoding. Lower submodel CE is relevant to draft quality but is not a substitute for timing the full verification procedure.
