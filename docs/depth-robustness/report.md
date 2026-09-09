# Depth robustness: a separate experimental report

9 September 2026. Dropout reduces the damage from early exit on both toys.

## Main findings

Two local tasks, five final seeds per treatment, 90 tuning runs and 45 final runs. Exact enumeration produced 2315 model/subset evaluations. External experiment spend was **$0** against a $50 ceiling. No Modal job or paid service was launched.

Increasing layer dropout (ILD) raises dropout probability with layer depth. Constant and decreasing refer to its schedule over training time.

The primary endpoint is `D4 = CE(prefix4) - CE(full)`, compared with the dense six-block model using paired seeds. A negative difference means less damage from removing the final two blocks. Raw CE accompanies it because robustness alone can reward a poor starting model.

## Tiny Shakespeare · causal transformer

| Recipe | Full CE | Prefix4 CE | Excess CE | Paired excess-CE difference vs dense6 [95% CI] |
|---|---:|---:|---:|---|
| Dense · 6 blocks | 2.1123 | 2.1621 | 0.0498 | Reference |
| Constant ILD | 2.1123 | 2.1156 | 0.0033 | -0.0465 [-0.0619, -0.0311] |
| Decreasing ILD | 2.1189 | 2.1254 | 0.0065 | -0.0433 [-0.0587, -0.0279] |
| Alternating dropout | 2.1103 | 2.1383 | 0.0280 | -0.0218 [-0.0445, +0.0009] |
| Dense · 4 blocks | 2.1030 | 2.1030 | 0.0000 | Architecture control |

## 8×8 digits · residual MLP

| Recipe | Full CE | Prefix4 CE | Excess CE | Paired excess-CE difference vs dense6 [95% CI] |
|---|---:|---:|---:|---|
| Dense · 6 blocks | 0.0800 | 0.1563 | 0.0763 | Reference |
| Constant ILD | 0.0784 | 0.0888 | 0.0104 | -0.0659 [-0.0844, -0.0474] |
| Decreasing ILD | 0.0800 | 0.0886 | 0.0086 | -0.0677 [-0.0858, -0.0496] |
| Dense · 3 blocks | 0.0828 | — | — | Architecture control |

## Interpretation

Layer dropout reduces early-exit damage on both tasks. In the language model, the full-depth paired CE intervals for all dropout methods include zero; these data do not resolve an intact-model quality advantage. Constant ILD has the lowest mean pruned CE at several depths, while decreasing ILD retains useful robustness when its dropout schedule reaches zero. This does not establish persistence after a long additional dense tail.

Separately training the desired smaller model is an essential control. The dense four-block transformer has mean full CE 2.1030, versus 2.1156/2.1254 for constant/decreasing ILD four-block prefixes. Dense 3 on digits has CE 0.0828, better than either ILD prefix-3 (0.1187/0.1087). These mean comparisons are not formal superiority claims. Small-model zero degradation at its own full depth is a definition, not evidence of better robustness.

Which blocks are removed matters. All 63 nonempty subsets are evaluated per six-block model. Summaries distinguish prefixes, every subset, and subsets preserving the first block. ILD never drops block 1 in training; deleting it is outside the training-mask distribution. Alternating dropout additionally always retains blocks 3 and 5. Exact mask averages are averages of individual-model losses, not ensemble prediction loss. No mask was selected as a deployment winner on test data.

## Language-model protocol

Tiny Shakespeare, immutable char-rnn revision 370cbcd448eb7daf32f21a6be560b70e0b33c4e3. Disjoint contiguous 80/10/10 split; training-derived vocabulary. Validation uses 64 and test 128 evenly spaced nonoverlapping context 64 windows (8,192 test target characters). Each run trains 800 steps with batch 16 for 819,200 character targets, using random training windows with replacement.

Architecture: six pre-LayerNorm causal transformer blocks, hidden width 64, four heads, GELU FFN 256, learned positions, untied embedding/head; 312,448 parameters. AdamW, matrix-only weight decay 0.01, 20-step warmup, cosine cooldown to 10%, gradient clipping 1. Controls: dense6, constant ILD max 0.4, decreasing ILD max 0.8, alternating max 0.4, and dense 4. All dropout treatments have 20% expected omission. Attention and FFN share one per-sequence mask, with inverse-survival scaling on each residual branch as in paper Eq. 6. Compute-then-mask execution does not save actual branch computation.

Each method receives five learning rates × two tuning seeds; full-depth validation CE alone selects the rate. All select 0.01, an interior grid value. Five final seeds (100–104) are disjoint from tuning seeds (10,11). An initial partial three-rate sweep was extended equally based on boundary validation results before final evaluation. The partial data and engineering pilot are retained.

## Digits protocol

scikit-learn 8×8 digits (1,797 images), fixed stratified 60/20/20 row split: 1,078 train, 359 validation, 360 test. Train-only feature normalization. This is not MNIST and not UCI's writer-disjoint protocol; the sklearn data were copied from the original UCI test set.

Six residual pre-LayerNorm MLP blocks of width 64 (FFN 128/GELU), 105,162 parameters, final 10-class head; dense 3 control. Matched 600 steps, batch 128 (76,800 training presentations). AdamW weight decay 0.01, cosine cooldown, clipping 1. Constant/decreasing ILD each omit 20% of example-block work in expectation. Training computes before masking. Five rates × two tuning seeds select 0.0003 for every method, followed by five evaluation seeds (2701–2705).

The initial three-rate digits experiment completed before its validation-triggered grid extension. Its 44 runs, including test outputs, remain archived; the final grid has 60 runs. Selection used validation CE, not test/pruned performance. This is disclosed development, not a claim of pristine test preregistration. Digits effect size is LR-sensitive: original-grid improvements were about 0.0164/0.0142 nats, versus 0.0659/0.0677 in the final grid.

## Inference and uncertainty

Retain original block order and the same trained final normalization/readout. No auxiliary losses, adapters, rescaling, fine-tuning, or calibration. Prefix4 is the primary six-block endpoint. For arbitrary subsets, average masks within seed, then construct 95% Student-t intervals across five seeds. Intervals are not multiplicity-adjusted and omit dataset/split uncertainty. Character CE is not comparable numerically with LLM-token CE.

## Verification and spending

Verified disjoint data splits, schedule means/endpoints, all-kept/full equivalence, pruning/zero-residual equivalence, transformer causality, no-drop forward/gradient equivalence, complete subset counts, source hashes, independent primary-CI calculations, and saved-checkpoint readback. See each experiment's verification artifacts.

All compute was local CPU on Apple M5 Max, float32. The language sweep used up to four isolated one-thread workers; digits used one thread. External spend $0; no paid services. Electricity/hardware amortization are not estimated. Run timings are provenance, not a performance comparison.

## Limits

These are qualitative mechanism reproductions, not the paper's 271M–8.2B experiments. No CompleteP, ALiBi, squared ReLU, paper tokenizer, modern token stream, long-budget training, energy instrumentation, inference-speed timing, generation study, or speculative decoding. Full-model quality and smaller dense models remain essential controls. A local positive result does not imply a universal optimum or efficiency gain.

## Reproduce and sources

- [Transformer code and raw measurements](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/depth_lm).
- [Digits code and raw measurements](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/depth_digits).
- [Interactive report](https://yaroslavvb.github.io/gradient-dissent/depth-robustness/).
- [Reviewed paper PDF](https://arxiv.org/pdf/2609.05275v1), especially §8.1.
- [LayerDrop](https://arxiv.org/abs/1909.11556), ICLR 2020.
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/), ACL 2024.
- [Tiny Shakespeare immutable revision](https://github.com/karpathy/char-rnn/tree/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare).
- [scikit-learn load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- [Original UCI digit dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
