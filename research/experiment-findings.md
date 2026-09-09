# Local experimental findings

Source: [Don't Drop Dropout, arXiv:2609.05275v1](https://arxiv.org/abs/2609.05275), especially Eqs. 2, 4, 6–9 and Sections 5–7. All results below were run locally; they are not values copied from the paper. The reproducible script, raw data, environment, and limitations are in [`experiments/`](../experiments/README.md). No LLM-scale replication or energy measurement was attempted.

## 1. Mean preservation is conditional; the shared mask creates a stronger counterexample

For a single residual update, with a fixed incoming activation, inverted scaling correctly gives

\[
\mathbb E_M[x + M f(x)/\rho \mid x] = x + f(x),\qquad \rho=1-p.
\]

That statement does not imply equality between a deterministic evaluation network and the mean of a composed stochastic network. Nonlinear transformations do not generally commute with expectation. More strikingly, the paper recommends sharing the same mask between attention and FFN (Eq. 6, Section 6.1). Even **affine** attention and FFN then give a discrepancy within a single transformer-style block:

\[
z=x+(M/\rho)ax,\quad y=z+(M/\rho)bz.
\]

Since \(M^2=M\),

\[
\mathbb E[y_{train}]=(1+a+b+ab/\rho)x,\qquad y_{eval}=(1+a+b+ab)x.
\]

The bias is \(abp x/\rho\). Our exact enumeration with \(x=1,a=.5,b=.4,p=.4\) gives expected training output **2.233333** versus evaluation output **2.100000**. The isolated first branch still has mean exactly **1.500000**, matching its evaluation output. Thus this is not a failure of the usual isolated-branch dropout identity.

With independent masks, the affine example becomes unbiased. Replace the second branch by \(f_2(z)=bz^2\), however, and independent masks produce mean **2.466667** versus deterministic **2.400000**. The exact bias is \(ba^2x^2p/\rho\).

**Interpretation:** the unqualified assertion at the end of Section 5 that evaluation features equal expected training features needs a conditional, isolated-residual-branch qualification. The recommended shared mask makes the distinction particularly consequential. This exact counterexample does not invalidate the empirical observation that inverted scaling helps hyperparameter transfer or model quality.

## 2. An unbiased branch prediction does not imply an unbiased dense-objective gradient

For the one-branch scalar model \(y=x+(M/\rho)ax\) and half-MSE target \(t\),

\[
\mathbb E\left[\tfrac12(y-t)^2\right]
=\tfrac12((1+a)x-t)^2+\tfrac12\frac p\rho (ax)^2.
\]

The expected gradient therefore includes \((p/\rho)ax^2\), even though the mean prediction is exactly unbiased. At \(x=1,a=.5,t=.7,p=.4\), the added penalty is **.083333**, and the expected gradient with respect to \(a\) is **1.133333** versus the dense gradient **.800000**. For the shared two-sublayer example above, the corresponding gradients are **3.925926** and **1.960000**.

This is an exact dropout-induced objective change, not merely zero-mean noise added to the dense gradient. In more general models, a Taylor expansion expresses analogous effects through the loss Hessian and feature-noise covariance. Connecting the paper to gradient-noise or curvature discussions should retain both changed objective/mean-gradient terms and covariance terms. The paper does not claim a general equivalence of objectives; this experiment identifies a common overinterpretation of its scaling argument.

## 3. Skipped work, elapsed time, and optimizer updates are separate questions

The local CPU benchmark retains exactly half of the sequence-block or batch-block work. It uses six residual GELU MLP blocks, float32, one thread, precomputed balanced masks, four warmups, and nine randomized-order repetitions of sixteen executions. These are forward-plus-backward medians; the optimizer is excluded.

| Execution | Small tensor [8,16,32] | Speedup | Medium tensor [32,32,64] | Speedup |
|---|---:|---:|---:|---:|
| Dense | .3663 ms | 1.000× | 2.5431 ms | 1.000× |
| Per-sequence, compute then mask | .4280 ms | .856× | 2.7204 ms | .935× |
| Per-sequence, gather active sequences | .3810 ms | .962× | 1.6455 ms | 1.545× |
| Per-batch, true whole-block skip | .2241 ms | 1.635× | 1.3219 ms | 1.924× |

The dimensions are [batch, sequence length, hidden width]. Each MLP expands hidden width by two. Forward-only and batch-gather controls are included in the raw results. Forward and input/parameter gradients match to floating-point tolerance between implementations using the same masks, including all-dropped and all-kept layers. The batch and sequence mask patterns are different models, so they are not asserted to have identical outputs.

**Interpretation:** compute-then-mask does all block matmuls and adds overhead. Actual sequence gathering becomes useful for the larger local shape but fails to deliver a speedup for the small shape. Batch skipping is cheaper here, while the paper's accuracy results favor sequence masks. This is a concrete quality/implementation tradeoff and supports the paper's own distinction between masking and actual sparse execution. It neither measures nor disputes Cerebras speedups. No inference from these measurements to joules is justified.

There is also an optimizer boundary detail. After one identical scalar AdamW update with lr=.1 and weight decay=.1, the parameter is **.890000**. On a dropped step, supplying an explicit zero gradient moves it to **.814094**, because momentum/weight decay still act. Supplying `grad=None` leaves it at **.890000** under the tested PyTorch AdamW semantics. Consequently, identical forward outputs and mathematical zero gradients do not guarantee identical training trajectories for a true-skip implementation. A replication needs an explicit policy for dropped-layer optimizer state and weight decay. This is an implementation requirement, not evidence about the unpublished details of the paper's training system.

## 4. A controlled regression toy separates optimization, budget, and depth elasticity

The task uses fresh iid Gaussian 8D inputs with a fixed nonlinear analytic teacher. A trainable tanh embedding feeds six residual 24-dimensional tanh-linear blocks and a scalar readout. A five-block dense model provides a smaller-architecture control. This is an online, noiseless toy problem, not an overfit finite training dataset and not a transformer.

All three dropout treatments have **20% mean dropout**. Constant ILD uses maximum .4; increasing/decreasing schedules use maximum .8. The schedules exactly follow the paper's discrete endpoints, and the code verifies that ILD+DTS mean dropout equals \(p_{max}/4\). Each configuration and budget regime gets five learning rates × two tuning seeds. All choose the interior rate .03. Final results use six separate evaluation seeds and a fixed 2,048-example test set. Entries below are mean MSE [95% Student-t interval across seeds].

| Configuration | Matched 240 steps | Matched expected active sequence-block budget |
|---|---:|---:|
| Dense, six blocks | .01925 [.01500,.02350] | .01925 [.01500,.02350] |
| Constant ILD | .02382 [.01970,.02794] | .01583 [.01283,.01883] |
| Decreasing ILD | .02674 [.02010,.03339] | .01563 [.01303,.01823] |
| Increasing ILD | .02708 [.02244,.03172] | .01862 [.01657,.02066] |
| Dense, five blocks | .02174 [.01036,.03311] | .01486 [.00779,.02193] |

At matched steps, all dropout treatments are worse than dense6; the paired 95% difference intervals exclude zero. At matched expected active work, dropout gets 300 steps and dense5 gets 288, compared with dense6's 240. All then have mean losses below dense6, but **every paired interval against dense6 includes zero**. For decreasing ILD, the difference is −.003618 with interval [−.007292,+.0000556]. It would be misleading to call this a decisive superiority result. Dense5 has the lowest mean and substantial uncertainty; excluding that control would overstate the case for layer dropout.

The training code computes all rows before applying masks. Its active sequence-block budget is a **conceptual model of ideal skipped work**, not measured training FLOPs or wall-clock compute. It excludes the embedding, head, optimizer, and routing overheads. The separate execution benchmark above tests the runtime consequences of real skipping.

The inference result is stronger within this toy. At the matched active budget, retaining only the first four blocks gives MSE **.19204** for dense6, **.02172** for constant ILD, **.02168** for decreasing ILD, and **.02351** for increasing ILD. The dense5 control at depth four has **.06414**. These numbers support a depth-robustness benefit in this task. Constant and decreasing schedules give almost identical depth-four means, so this experiment provides little reason to prefer the decreasing schedule for elasticity.

**Interpretation:** schedule rankings depend on the training regime and budget. Our toy does not establish a universal decreasing-schedule optimum, and it does not falsify the paper's empirical schedule comparisons on its own language-model family. It illustrates why “same steps,” “same active work,” “same elapsed time,” and “best smaller dense architecture” answer different questions. It also independently demonstrates the plausible depth-elasticity mechanism.

## Limits and next experiments

The six-seed intervals characterize initialization, data-stream, and mask variation on one fixed teacher and test set; they do not capture variation over datasets or architectures. The LR grid is finite, only LR is independently retuned, and intervals are not adjusted for multiple comparisons. Neither dropout schedules nor CompleteP can be declared generally optimal from these experiments. Balanced benchmark masks intentionally remove random active-count fluctuations and use precomputed masks; real routing and distributed load imbalance add costs.

A stronger replication would use a small decoder-only transformer with the paper's shared attention/FFN mask, equal searches over LR/batch/weight decay, both zero-versus-absent-gradient optimizer policies, shared token streams, a tuned smaller dense baseline, repeated seeds, and actual end-to-end throughput at fixed validation loss. For the gyroscope connection, directly measure dense-versus-mask expected gradients, gradient covariance, residual magnitudes, and top curvature directions through the dropout schedule. That would distinguish annealed regularization, gradient-noise averaging, and residual-stream stabilization instead of treating them as the same mechanism.
