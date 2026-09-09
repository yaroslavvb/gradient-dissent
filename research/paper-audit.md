# Critical technical review: Don't Drop Dropout

**Paper:** Mostafa Elhoushi et al., *Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference*, arXiv:2609.05275v1, 4 September 2026. [PDF](https://www.alphaxiv.org/pdf/2609.05275), [paper page](https://www.alphaxiv.org/abs/2609.05275). Reviewed 9 September 2026. Page references below are printed PDF pages; all 27 pages, including the appendix, were read. Key equations and tables were also checked in rendered pages.

**Assessment:** A useful empirical recipe study with credible evidence for better training/inference tradeoffs in its tested model family. The most valuable result is that structured depth noise can be annealed away while leaving useful pruning robustness. The evidence does **not** establish a general prescription for frontier LLMs, an across-the-board accuracy improvement, or 25% end-to-end training acceleration. Several mathematical and reporting statements need correction. The empirical results can remain useful despite those problems.

This audit separates **reported measurements**, **exact consequences of the printed equations**, and **hypotheses requiring new experiments**. Local toy experiments elsewhere in this repository are mechanism checks, not reproductions of the authors' 2400+ proprietary-scale runs.

## What the recipe actually is

Each training sequence receives a random subset of transformer blocks. Attention and FFN in a block share the same Bernoulli mask. Within a retained block, the paper's Eq. 6 scales both residual branches by inverse survival probability, `1/rho`, where `rho = 1-p`. A CompleteP depth multiplier is applied in addition. The preferred dropout rate increases linearly with layer index and decreases linearly with training progress:

`p(l,t) = p_max * l/(L-1) * (1-t/(T-1))`.

Thus expected active depth starts at `(1-p_max/2)L` and reaches `L`. Average omitted nonembedding block FLOPs equal `p_max/4` under equal-cost blocks, constant work per step, and actual conditional execution. In particular, `p_max=0.99` means **24.75% average block FLOPs omitted**, not 99% of total training omitted. The last block initially survives only 1% of sequences, but average initial model depth is still 50.5% of full depth. [Paper §§4-7, Eqs. 6, 8-9, pp. 4-10; Table A.2, p. 24.]

There are two distinct goals. One is to preserve full-depth model quality using fewer active block computations. The other is to train a full model whose submodels remain useful. The best schedule for one need not be best for the other: the paper itself shows alternating dropout is better for some intermediate skipping patterns, while increasing dropout across depth is better for early exit.

## What the measurements support

| Result | Reported evidence | Defensible interpretation |
|---|---|---|
| Decreasing dropout beats constant/increasing dropout | Table 3, p. 10: matched nominal average sparsity across 271M, 503M, 906M | Strong within this model family and search grid; the differences between schedules are much larger than several claimed differences from dense training. |
| 1.8B full-depth quality | Table 5, p. 15: loss 1.849 dense vs 1.836 dropout, 15% block FLOPs saved | A promising improvement of 0.013 cross-entropy, or 0.70% relative CE; statistical certainty is unavailable without seed variation. |
| 3.9B full-depth quality | Table 5: loss 1.732 dense vs 1.745 dropout, 20% block FLOPs saved | An explicit quality/compute tradeoff: +0.013 CE, +0.75% relative CE, about +1.31% perplexity. It is not an equal-token improvement. |
| 8.2B largest run | Table 5: dropout only, loss 1.663, 24.75% nominal block FLOPs saved | Demonstrates a large run can train successfully. There is no matched dense 8.2B result in the supplied PDF, so preserved baseline quality at the headline saving is unmeasured. |
| Depth robustness | Table 5, Fig. 5, Figs. A.2-A.3 | Large, persuasive reduction in degradation relative to pruning dense-trained models. This remains a quality tradeoff relative to the intact dropout-trained model. |
| Self-speculative acceleration | Tables 4-5, pp. 14-15; XSUM explicitly specified for Table 4, draft length 5 | Useful reported speedups after searching layer subsets. Not a generic guarantee across workloads, batch sizes, or hardware. |
| Transfer beyond 20 TPP | Fig. 8, p. 15 | The displayed sweep is a 503M model over 2-64 TPP. This is modest evidence for duration transfer, not a fitted trillion-token scaling law. |

The low-dropout 906M Table 3 result is particularly small: displayed losses 1.953 dense and 1.951 dropout, with a printed relative delta of -0.06% based on unrounded values. At 503M, both displayed losses are 2.110 and the printed delta is **+0.03%**, so the §7.2 assertion that both 503M and 906M beat dense at 5% savings contradicts the table for 503M. Do not turn these results into a strong statistical claim without repeated seeds.

Downstream performance is mixed. The model called 1.8B in Table 5 is called 1.9B in Table A.3. Assuming these rows are intended to match, the dropout model loses on **9 of 11** listed tasks, ties 2, and wins none, despite its improved validation CE. This does not establish significant degradation on each task, because uncertainty estimates are absent, but it contradicts a casual inference that better CE necessarily improved capabilities. For 3.9B the task comparison is 7 wins and 4 losses, including OBQA 0.356 to 0.324. The 8.2B task row again lacks a dense control. [Table A.3, p. 25.]

## Connections to earlier work and the actual contribution

| Primary source | What it already established | What this paper adds or must distinguish |
|---|---|---|
| Huang et al., [Stochastic Depth](https://arxiv.org/abs/1603.09382), ECCV 2016 | Training randomly shortened residual networks and evaluating the deep model; training-time savings and improved generalization in vision. | Modern causal LM recipe and scale exploration. Random block omission and depth-dependent survival are inherited ideas. |
| Fan et al., [LayerDrop](https://arxiv.org/html/1909.11556), ICLR 2020 | Structured transformer dropout enables shallow subnetworks; §6 compares sublayer and whole-layer dropping. §3.2.2 uses alternating layers as an inference pruning rule. | A larger causal-LM granularity study. Claims of being the first to compare whole layers and sublayers need narrowing. Its alternating *training* distribution differs from LayerDrop's alternating *inference* rule. |
| Zhang & He, [Progressive Layer Dropping](https://proceedings.neurips.cc/paper/2020/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html), NeurIPS 2020 | Transformer training acceleration from increasingly dropping more layers during training; published abstract reports 25% FLOP and 24% wall-clock savings on BERT. | Evidence that the opposite temporal direction works better for this causal-LM setting. Training savings without accuracy collapse were already known. |
| Panigrahi et al., [Efficient Stagewise Pretraining via Progressive Subnetworks / RaPTr](https://arxiv.org/html/2402.05913v2), 2024 | Random subnetworks grow progressively to the full model; depth and width variants; explicit critique of increasing dropout; BERT and UL2-1.6B experiments; theory for learning complexity and stage transitions. | This is the closest omitted precursor. The present paper studies a continuous depth/time schedule, per-sequence granularity, parameterization, and inference elasticity in causal LMs. The general growing-subnetwork principle is not new. |
| Elhoushi et al., [LayerSkip](https://aclanthology.org/2024.acl-long.681/), ACL 2024 | Depth-increasing layer dropout plus shared early-exit loss; self-speculation by drafting early and verifying with remaining layers; experiments in Llama models. | Removes the need for auxiliary early-exit loss during the main pretraining recipe and measures additional granularity/schedule choices. The inference motivation and broad mechanism are continuous with LayerSkip. |
| Zhang et al., [Draft & Verify](https://aclanthology.org/2024.acl-long.607/), ACL 2024 | Layer-subset drafting and full-model verification, without further neural training; measured gains on Llama-2. | Dropout pretraining can improve the space of useful draft subsets. It is not the invention of lossless self-speculation. |
| Dey et al., [CompleteP](https://arxiv.org/abs/2505.01618), NeurIPS 2025 | Width/depth parameterization for hyperparameter transfer and complete feature learning; residual branch scaling with inverse depth. | Treats retained depth as another parameterization axis and empirically favors inverse survival scaling. That is a plausible extension, not a complete theorem about arbitrary dropout. |
| Bergsma et al., [Power Lines](https://arxiv.org/html/2505.13738v2), NeurIPS 2025 | AdamW weight-averaging timescale `tau=B/(eta*lambda*D)` and empirical data/batch scaling. | Supplies much of the optimizer-transfer machinery; Table A.2 attempts to combine these rules with dropout but has a data-exponent inconsistency discussed below. |
| Hillier et al., [STLM Engineering Report: Dropout](https://arxiv.org/html/2409.05423v1), 2024 | Decreasing activation-dropout schedules on small LMs; its §2 applies masks to embedding and attention outputs. | Schedule motivation transfers, but activation dropout is a different intervention from skipping blocks. |
| Liu et al., [Drop Dropout on Single-Epoch Language Model Pretraining](https://aclanthology.org/2025.findings-acl.111/), ACL Findings 2025 | Activation dropout on attention/MLP outputs worsens several single-epoch LM evaluations, including early dropout. | The new result does not refute this finding: structured layer omission changes computation and inductive bias differently. Both findings can hold simultaneously. |

The strongest fair contribution is **an empirical reconciliation and practical combination**: choose per-sequence whole-block masks, a depth-increasing distribution, a time-decreasing schedule, and suitable residual/optimizer scaling, then measure both full-model and submodel quality. It is valuable that a familiar regularizer has a use even when avoiding classic overfitting is not the goal.

## Critical issue 1: the expectation argument fails for the preferred block implementation

**Confidence: high; exact algebra from the printed equations. Scope: the justification, not a claim that the reported experiments are false.**

At the end of §5 (p. 5), the paper says `r_eval=1` ensures equality of evaluation activations and expected training activations. There is a valid *local* statement for Eq. 2 when the whole residual branch `f(h)` is held fixed:

`E_M[h + (M/rho) f(h) | h] = h + f(h)`.

But the preferred transformer configuration is different: Eq. 6 scales attention and FFN separately and §6.1 sets their masks equal. Consider one scalar transformer-shaped block with fixed input `x`, linear attention `a*x`, linear FFN `b*z`, and shared `M ~ Bernoulli(rho)`:

```
z_train = x + (M/rho)*a*x
y_train = z_train + (M/rho)*b*z_train
        = [1 + (M/rho)*(a+b) + (M/rho)^2*a*b]*x
E[y_train] = [1+a+b+a*b/rho]*x

y_eval = [1+a+b+a*b]*x
E[y_train]-y_eval = a*b*(1-rho)/rho*x
```

This fails even with linear sublayers and a deterministic block input. For `a=b=x=1`, `rho=0.5`, evaluation gives 4 and the training expectation gives 5. The reason is that the FFN input depends on the same mask multiplying its output; one cannot replace each mask independently by its expectation.

Independent attention/FFN masks recover equality in this *linear* example. Another locally unbiased construction is to evaluate the dense composite block and scale its aggregate residual once. Neither implies global equality in an arbitrary nonlinear deep network: generally `E[f(H)] != f(E[H])`. Moreover, matching logits in expectation would still not match expected cross-entropy.

The actual implementation may differ from the simplified equations; the supplied PDF does not resolve that ambiguity. A useful author clarification is precisely where inverse survival scaling is applied. This distinction can change the comparison between whole-layer and sublayer dropout, since the two then differ in both mask correlation and effective within-block interaction strength.

## Critical issue 2: hyperparameter transfer is empirical and incomplete

**Confidence: high for the limited evidence; medium for extrapolated failure mechanisms.**

The coordinate checks in Fig. 2 and Appendix B measure activations for ten steps of a 40-layer model. The plotted y-axis is mean absolute activation, while Appendix B describes a Frobenius norm. Stable activation magnitude alone does not verify the stated maximal residual stream update condition, gradient noise, update direction, or long-run optimal learning rate. Fig. 3 provides useful evidence for a few learning-rate/batch/AdamW-timescale sweeps, but §11 explicitly acknowledges weaker transfer at high dropout and that transfer was mainly tested with constant schedules.

For a single inverted Bernoulli multiplier `Z=M/rho`, `E[Z]=1`, `E[Z^2]=1/rho`, and `Var(Z)=(1-rho)/rho`. Thus `rho=0.01` preserves the first moment while creating multiplier variance 99 and a retained multiplier 100. Mean activation stability is compatible with rare large updates. Per-sequence sampling and large batches can soften this, but do not turn it into an exact effective-depth equivalence. This is a reason to inspect update norms, quantiles, gradient clipping, and optimizer-state dynamics; it is not proof that the large run was unstable.

A second issue is the printed weight-decay transfer rule. Table A.2 says:

```
B = B_base * m_D^0.4
eta_hidden = eta_base / m_d
D = D_base * m_D
tau = tau_base * (TPP/TPP_base)^(-0.5)
```

Substituting into the Power Lines definition `lambda=B/(eta*tau*D)` gives:

`lambda_hidden = B_base*m_d / (eta_base*tau*D_base*m_D^0.6)`.

Table A.2 instead prints `m_D^0.4` in the denominator. The resulting ratio is `m_D^0.2`; at `m_D=16`, the printed rule gives approximately 1.74 times the weight decay implied by the other rows. This appears to be a table error or an undocumented change to the scaling rule. It cannot establish what the code actually ran. [Paper Table A.2, p. 24; Power Lines §2.2 Eqs. 2 and 4.]

## Critical issue 3: FLOP savings are narrower than training acceleration

**Confidence: high.**

Footnote 11 (p. 8) explicitly defines subsequent FLOPs to mean **nonembedding FLOPs**. Equations 8-9 assume skipping a proportion of identical-cost block executions saves that proportion of block work. They omit unchanged vocabulary projection, softmax, embeddings, optimizer work, routing/gather/scatter overhead, and communication. The first-page graphic nevertheless says up to 25% faster training; there is no accompanying end-to-end training timing table in the supplied PDF.

If `q` is the fraction of dense compute in unchanged components and `s` is the block-compute saving, then ideal total-compute saving is `(1-q)s`, before added implementation overhead. For illustration, with unchanged work 20% and block saving 25%, total compute falls 20% and ideal compute-bound speedup is `1/0.8=1.25x`. This is illustrative arithmetic, not a measurement of the authors' system.

Per-sequence dropout is especially sensitive to implementation. Masking the output after computing every sequence saves no block arithmetic. Gathering active sequences can save arithmetic but reduce matrix dimensions and utilization. The authors discuss compute-bound conditions in §6.2; their statement that equal FLOPs should yield similar speedup is a conditional expectation, not a benchmark. Hardware-specific performance on CS-3 need not transfer to a GPU cluster.

There is also a small exact-budget issue for alternating dropout: all three small models have odd layer counts, 13, 17, and 23. The formula is `p_mean=floor(L/2)*p_max/L`, not exactly `p_max/2`. With `p_max=0.2`, nominal 10% sparsity is actually 9.23%, 9.41%, and 9.57%. Table 2 groups these with exact 10% settings. This can slightly favor ALD by giving it more compute; it probably cannot explain every schedule effect, but matters for very small differences.

## Critical issue 4: matched-cost curves are not a tuned compute frontier

**Confidence: high that the control is not demonstrated; medium that it changes the conclusion.**

Figure 9 compares loss at equal cumulative block FLOPs along runs trained to the same token endpoint. A dropout run has seen more tokens and reached a later fraction of its scheduled training at a fixed FLOP count. That is a real operational advantage, but a dense checkpoint from a longer run may not have received an appropriate terminal learning-rate cooldown for the smaller comparison budget.

The clean experiment trains a dense control with the same **total** FLOP budget, a schedule ending at that budget, separately tuned hyperparameters, and repeated seeds. It should compare that control with (a) the dropout run and (b) alternative dense model sizes trained to completion. The supplied PDF does not show that complete reoptimization. Accordingly, Fig. 9 supports useful trajectory efficiency, not a proved global Pareto frontier.

Likewise, 20 tokens per nominal parameter is a conventional dense-model reference point, not proof that stochastic-depth models remain compute-optimal there. The optimum allocation between stored parameters, active depth, and tokens can change. [Paper §§3, 7, 9-10; Fig. 9; compare Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).]

## Critical issue 5: inference claims conflate different comparisons

**Confidence: high.**

Static skipping is approximate inference. In Table 5 the 3.9B dropout model's full CE is 1.745. Early exit at 75% depth produces CE 2.143, about **48.9% higher perplexity**; alternating-layer skipping produces 2.129, about **46.8% higher perplexity**. These are much better than pruning a dense-trained model, but not negligible differences from the intact dropout model. Even the 8.2B 75%-depth exit increases perplexity about 12.1% from 1.663 to 1.777.

Correct speculative verification can preserve the chosen **target model's** distribution (or greedy output under the applicable algorithm), as established by the cited speculative-decoding literature. It does not restore the quality of an independently trained dense baseline. If dropout changed target quality, lossless verification preserves that changed target. [Paper §8.2.2; [Draft & Verify](https://aclanthology.org/2024.acl-long.607/).]

Section 8.2.2 searches draft layer subsets using Bayesian optimization, genetic algorithms, hill climbing, and simulated annealing, then selects the highest speedup. The supplied PDF does not give enough detail about inference hardware, batch size, precision, timing methodology, search budget, or separation of search/evaluation examples to independently reproduce Tables 4-5. A held-out evaluation after identical search budgets is needed; selection bias is a possibility, not an established error.

Key takeaway 4 and the conclusion describe the 1.55x figure as zero-shot speedup, although it arises from the paper's own **post-training/search-based self-speculative** category. It can be weight-frozen without being an unconditional, zero-configuration property. The §10 claim that dropout is a prerequisite for useful self-speculation is too strong: Draft & Verify already measured gains without such training. Also, a speedup of 1.02x is a small improvement, not the regression described in §10.

Finding 6 says average dropout predicts both early exit and layer-skipping robustness. Average dropout is a useful scalar exposure measure, but the paper's ALD/ILD comparison shows it is not sufficient: the location of missing layers and the inference pattern matter. A practical predictor should include per-layer exposure and the inference mask, not just the mean.

## Critical issue 6: the scaling extrapolation is not established

**Confidence: high.**

Section 9 discusses predicting trillion-token behavior, but provides no fitted loss surface, exponent estimates, prediction interval, or held-out large-budget scaling test. Figure 8 displays only the 503M family across 2-64 TPP, while the largest reported runs remain at 20 TPP. Many architecture and deployment axes remain untested: MoE, alternative normalization/positional schemes, longer contexts, heavily overtrained models, and inference robustness after SFT or continued training. The authors acknowledge several of these limitations.

The 8.2B model proves feasibility at `p_max=0.99`, not monotone improvement of tolerated dropout with scale. Model width, depth, total parameters, dataset size, and chosen maximum dropout all change together in Table 5. Separate factorial sweeps are required to isolate what governs the tolerance.

Reproducibility is limited by the supplied PDF: §3 calls the data a diverse natural-language/code corpus without enough mixture details or a public artifact reference; Appendix A gives architecture and symbolic transfer rules rather than a complete numeric recipe, seeds, optimizer settings, context lengths, and validation protocol. There is no consolidated manifest for the claimed 2400+ runs. This review does not assert that these artifacts exist nowhere; it states that they are not supplied with enough specificity in the reviewed PDF.

## Additional reporting checks

These are secondary to the scientific issues above.

- **Table 1 training-loss percentages do not match their displayed values.** For 271M, 2.421 and 2.419 relative to 2.293 imply +5.58% and +5.49%, not +4.72% and +3.95%. For 503M, 2.260 and 2.178 relative to 2.140 imply +5.61% and +1.78%, not +3.94% and +2.60%. These discrepancies exceed rounding. The validation percentages are broadly consistent. [p. 7.]
- **Scope numbers disagree:** abstract reaches 8.2B/160B tokens; introduction says 3.9B/116B. The names 503M/504M and 1.8B/1.9B also vary. These may be stale text or parameter-count conventions but require explicit reconciliation.
- **Stochastic Depth notation:** §5 says the original paper used evaluation scaling `p` while locally defining `p` as dropout probability. The original uses survival probability; converted to this paper's notation the factor should be `rho=1-p`.
- **Schedule endpoints:** §7 uses `T-1` so the final dropout is exactly zero, whereas the glossary uses `T`. This is usually a small numerical difference but is relevant to an executable recipe.
- **Broad superiority language:** key takeaway 2 says layer dropout always improves early exit, despite ALD failing to improve early exit in the accompanying discussion. §8.2.1 calls depth-aware pretraining a prerequisite for maximizing adapter efficacy; the experiments show an advantage in tested settings, not a necessity theorem.

## Highest-value next experiments

1. **Resolve Eq. 6 scaling placement.** Compare shared-mask/per-sublayer scaling, independent masks, and shared-mask/aggregate-block scaling. Match expected block work and optimize each recipe equally. Log evaluation/Monte Carlo mean discrepancies and full/pruned quality.
2. **Retune dense fixed-budget controls.** End both LR schedules at the same actual compute budget and repeat across seeds. Compare a smaller dense model as well as the same nominal architecture.
3. **Measure systems costs.** Separate active block FLOPs, total arithmetic, tokens/s, accelerator-hours, optimizer/communication overhead, and end-to-end time to target loss. Benchmark actual skipped execution against dense masking.
4. **Probe noise and updates.** At high `p_max`, record per-layer survival counts, gradient/update norm quantiles, clipping events, optimizer moments, and weight decay under skipped steps. Test whether per-sequence masks help through independent averaging.
5. **Test retention and deployment tradeoffs.** Measure early exit and noncontiguous skipping after a dense tail, SFT, and long continued pretraining. Evaluate searched draft subsets on held-out domains, batch sizes, and generation lengths.
6. **Add missing scaling controls.** Train a matched dense 8.2B model, vary depth and width independently at fixed parameter/compute budgets, and test high TPP with a predeclared predicted loss gap.

## Questions the authors could answer concretely

- Does the implementation scale each attention/FFN residual separately as Eq. 6 states, or the full transformer-block residual once? If separate, what is the intended meaning of the activation-expectation equality?
- What numeric hyperparameters were used, and is the `m_D^0.4` weight-decay denominator in Table A.2 intentional?
- Can the authors provide seeds, confidence intervals, exact losses, and a run manifest, including the unavailable dense 8.2B comparison?
- Do the efficiency figures include vocabulary projection, optimizer work, communication, and time spent searching draft subnetworks?
- How do completed, separately tuned dense runs perform at the dropout runs' compute budgets?
- What is the explanation for the 1.8B/1.9B downstream losses despite lower pretraining CE?
- How does the method compare with RaPTr, and why is the earlier LayerDrop layer/sublayer comparison absent from the novelty discussion?

The work is worth following up because its large pruning-robustness gains and consistent schedule ablation have practical value. The strongest current conclusion is that **well-configured stochastic depth can be a useful training/inference tradeoff in this family of causal LMs**. Its larger claims need stronger controls, reproducibility details, and corrected mathematical exposition.
