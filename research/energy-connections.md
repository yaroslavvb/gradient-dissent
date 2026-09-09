# Layer dropout through Sutro's energy and locality lens

Research note, 9 September 2026. This note connects the supplied paper, **Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference** (arXiv:2609.05275v1, 4 September 2026), to the energy/data-movement mission in the linked Google document and the public [Sutro #30 meeting notes](https://yaroslavvb.github.io/gyroscope/meetings/2026-09-07-061a993a3e32a40a.html). Page references below are the printed pages of the [paper PDF](https://www.alphaxiv.org/pdf/2609.05275). The meeting describes a proposed MNIST challenge; it is not a frozen benchmark specification. The proposed experiments below are research proposals, not reported results or official submissions.

## The useful connection

Layer dropout is a useful, relatively conservative baseline for Sutro: make the existing learning algorithm do less work, then ask whether the work removed was physically expensive. It is not a replacement for backpropagation, and it does not establish that reducing FLOPs reduces joules proportionally. Its strongest connection to the group's agenda is precisely the tension between **statistically good sparsity** and **physically cheap sparsity**.

The paper itself identifies this tension in §6.2, p. 7. Independently dropping a layer for individual sequences improves loss in its experiments, whereas dropping a layer for an entire batch can avoid loading that layer's weights. The authors argue that similar FLOP savings should yield similar speedups *when training is compute bound*, and suggest per-device or per-microbatch masks as an unexplored middle ground. Sutro can test that conditional claim under an objective in which location and transport explicitly matter.

The interesting research question is therefore: **At equal accuracy, which granularity and schedule of omitted computation minimizes data movement, peak live state, and total energy?** The paper optimizes some of the statistical dimensions; a Sutro experiment can add the physical dimensions.

## Five deductions worth testing

### 1. Half the active FLOPs can mean almost no reduction in weight loads

Let a layer be independently dropped with probability `p` for each of `B` examples. If its weights must be loaded whenever at least one example uses the layer, then

`P(layer used by batch) = 1 − p^B`.

This is an exact probability calculation under independent masks, not an energy measurement. With `p = 0.5` and `B = 32`, half the example-layer computations disappear in expectation, but the layer is used with probability `0.999999999767`. In contrast, a single mask shared by the entire batch uses the layer with probability `0.5`.

| Drop probability | Batch size | Expected active example fraction | Probability weights are needed |
|---|---:|---:|---:|
| 0.2 | 32 | 0.8 | effectively 1 |
| 0.5 | 32 | 0.5 | 0.999999999767 |
| 0.9 | 32 | 0.1 | 0.965663 |
| 0.99 | 32 | 0.01 | 0.275020 |

The distinction is particularly relevant when weight traffic dominates. With unchanged weight bytes and fewer examples processed per layer, arithmetic intensity against those bytes decreases. A model that was compute bound before dropout can move toward a bandwidth bottleneck afterward. That is not a contradiction of the paper; it limits the regime in which its compute-bound argument applies.

There are important qualifiers. The formula concerns whether weights are needed, not how many physical transfers occur. Cached or permanently local weights may require no expensive reload. Conversely, an optimizer may still touch an unused layer for momentum or decoupled weight decay. A faithful implementation must state what happens when a layer has no gradient; silently skipping its optimizer step can change the algorithm. Training framework semantics, physical placement, and optimizer traffic belong in the experiment manifest.

### 2. Activation savings and model-state savings are different resources

An actually bypassed function does not need its internal forward intermediates for that example's backward pass. This can reduce saved activations. Multiplying a computed output by zero does not achieve that benefit: the costly function has already executed. The paper explicitly distinguishes these implementations in §4, p. 4.

The model's weights remain allocated, and adaptive optimizer state generally remains allocated too. Thus a dropout network with `P` parameters still has an `O(P)` parameter-state floor even if its active computation is much smaller. A batch-wide zero mask may avoid touching state on a particular step; it does not by itself release that state for the rest of training. Per-sequence gathering can also allocate packed copies, index maps, and full-batch residual buffers, so saved-activation bytes should be instrumented rather than inferred from the dropout rate.

This is a direct link to the meeting's area objective. If area is represented by peak occupied scratch cells, a schedule that becomes dense near the end can still reach approximately the dense peak activation demand. Lower *average* active depth does not imply proportionally smaller *peak* area. A useful schedule comparison should therefore display both cumulative movement and maximum live state over training.

The distinction also motivates an optimizer control. [Adafactor](https://arxiv.org/abs/1804.04235) reduces second-moment storage for matrices through row/column statistics. That attacks persistent optimizer state, unlike layer dropout's omission of selected forward/backward computations. Its relevance is complementary, not evidence that combining them will preserve the paper's accuracy findings.

### 3. A gather is a physical algorithm, not a free indexing operation

The paper's computationally efficient formulation selects only active sequences. On hardware, that may require gathering those sequences into a compact batch, evaluating the block, and scattering updates back. The same logical mask can have different transport costs under different layouts.

A minimal accounting identity is

`E_sparse − E_dense = −E_omitted_work + E_mask + E_pack/scatter + ΔE_other`.

`ΔE_other` includes changes in memory hierarchy behavior, synchronization, utilization, and the optimizer. This identity provides a crossover criterion: sparse execution helps only when the omitted work saves more energy than the added overhead. It does not assign hypothetical joules to a CPU timing measurement.

For a grid model, separately accumulate `Σ(bytes transferred × prescribed distance)`. Count the forward gather/scatter and the inverse movement of gradients. State whether selected examples remain packed between successive layers or return to a canonical layout each time. Keeping them packed can avoid some copies but may cause later masks to require a new permutation; the mapping itself must be represented and costed.

**Proposed extension:** share masks within physically nearby groups of examples, and sweep group size. If there are `G` independent groups, the probability a shared layer is used somewhere is `1 − p^G`. Larger groups create more opportunities to omit an entire block of execution, but reduce mask diversity. Local weight replicas could improve transport while increasing occupied area. Neither grouped masks nor replication is automatically beneficial; this is a natural Pareto problem.

### 4. Skipping graph layers need not shorten physical distance

Two inference paths may execute the same number of layers but have different layouts. A static prefix exit omits a suffix; scattered layer skipping may keep communicating across the full physical depth of a pipeline. Bypassing arithmetic at a spatially assigned layer does not teleport the activation to the next active layer.

By contrast, a compiler that statically removes and repacks unused layers may reduce both model state and communication distance. Compare at least two deployment assumptions: **resident elastic model with bypasses**, and **specialized compact model produced for a fixed skip pattern**. They serve different use cases. For a single execution engine that streams weights, omitted weight loads may be more important than the distances between logical layers.

The paper measures static early exit, intermediate skipping, and self-speculation (§8, pp. 10–14). Those results support depth robustness in its models. They do not identify which placement of an elastic model minimizes byte-distance. A visual demonstration should show logical edges and physical positions separately.

### 5. The optimum dropout schedule depends on the lifetime workload

The paper trains with layer dropout and then normally evaluates the full model, with optional reduced-depth execution or self-speculative decoding. Its decreasing dropout schedule improves base-model loss, while its inference results show a different tradeoff across training schedules (Table 4, p. 14). Therefore “best schedule” needs a workload objective.

Use a lifecycle expression such as

`E_total(N) = E_fit + E_adaptation/search + N × E_prediction`.

Here `N` counts predictions or tokens using a common unit. A choice that costs more to prepare can win after enough deployment use. For two alternatives, the crossover is `(E_fit,A − E_fit,B)/(E_prediction,B − E_prediction,A)` when A has higher preparation cost and lower per-prediction cost. If either numerator or denominator has the opposite sign, interpret the comparison directly rather than reporting a spurious positive crossover.

The MNIST meeting proposes one end-to-end transductive computation from labeled training data and unlabeled test data to test labels. In that workload there is one concrete total cost; training and test inference cannot be amortized over a hypothetical infinite deployment. The boundary for reusable code, pretrained weights, development-time search, and instance-specific adaptation must come from the challenge rules. Report those costs separately until the boundary is frozen.

Static classification also has no direct analogue of the paper's lossless autoregressive self-speculative verification. An early-exit MNIST classifier can be useful, but its error/energy curve must be measured. Do not import a 1.55× text-decoding speedup into that task.

## Connections to existing work

### Checkpointing: more arithmetic can improve the physical objective

[Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016)](https://arxiv.org/abs/1604.06174) trade recomputation for reduced activation storage, with an `O(sqrt(L))` checkpointing construction for an `L`-layer chain and roughly one additional forward pass under the paper's assumptions. This gives an essential control: dropout reduces executed computations; checkpointing can increase computation while reducing expensive storage/transport. A FLOP-only score can rank these in the opposite order from an energy score. Recompute stochastic masks consistently during checkpoint replay; drawing a new mask changes the function whose gradient is being computed.

### Reversible networks: preserve backpropagation, change what must be stored

[Gomez et al., *The Reversible Residual Network* (2017)](https://arxiv.org/abs/1707.04585) reconstruct layer activations during backpropagation using a reversible architecture; storage for the reversible blocks does not grow with their depth. This is relevant to the group's concern about activation traffic, but reversibility does not eliminate parameter state, recomputation, nonreversible boundary operations, or numerical issues. The appropriate question is whether a reversible architecture shifts the same accuracy/energy/area frontier further than stochastic depth does.

### Local learning: a more direct challenge to long-distance credit assignment

[Mostafa, Ramesh and Cauwenberghs (2017)](https://arxiv.org/abs/1711.06756) train layers with local errors from auxiliary random classifiers and explicitly discuss reducing custom-hardware communication. [Nøkland and Eidnes (2019)](https://arxiv.org/abs/1901.06656) develop supervised local losses for image learning. These methods target the dependence on errors arriving from later layers, whereas dropout still backpropagates through the active subnetwork. Include a local-loss baseline for MNIST, accounting for auxiliary classifiers, label transport, and any accuracy cost. “Local error” is not synonymous with zero communication or proven lower joules.

### Mixture-of-Depths: an informative contrast, not an interchangeable baseline

[Raposo et al., *Mixture-of-Depths* (2024)](https://arxiv.org/html/2404.02258v1) use learned top-`k` token routing with a fixed capacity. This creates known tensor sizes but adds routing and token selection. Unlike whole-sequence dropout, it also changes which tokens can participate in attention. Its stochastic-routing control performs poorly (§4.1); that does not refute the new paper because granularity, attention context, schedule, and tuning differ.

MoD's sequence-wide top-`k` decision is noncausal, requiring special handling at autoregressive inference (§3.5). A transductive MNIST workload can legally inspect the whole unlabeled test set if the frozen rules permit it. This creates an interesting opportunity for globally budgeted image routing without that particular causality problem. Its sorting, buffering, and transport costs still count. The proposal is to compare random, confidence-based, and learned allocation under equal total physical budgets, not to assume learned routing wins.

## The meeting's claims also need an audit

The motivation for charging communication is sound; the illustrative numbers and statements should not become unexamined physical laws.

* **Dally reference and calibration.** The cited work is [Dally, *On the Model of Computation: Point* (2022), pp. 30–32](https://doi.org/10.1145/3548783); [primary article PDF](https://www.cs.mun.ca/~harold/Courses/Old/CS3600.W25/Diary/3548783.pdf). It charges communication according to location and explicitly discusses recomputing neural activations. On p. 30, transporting eight bytes one millimeter costs 1.9 pJ and 400 ps: `0.2375 fJ/(byte·µm)` and `0.4 ps/µm`. The meeting's `1 fJ/(byte·µm)` and `0.5 ps/µm` are respectively 4.21× and 1.25× these values. Additional overhead could explain a different calibration, but “same energy” requires clarification. Preserve official constants if adopted; label this comparison rather than silently changing them.
* **“Hundred fetches per add” is not a general backpropagation identity.** In a dense layer, activations are reused across many multiply-adds. Traffic per arithmetic operation depends on width, batch size, tiling, placement, checkpointing, and fusion; depth alone does not fix that ratio. The correct object is an implementation's access trace or a defensible lower bound.
* **The 8×8 matrix example lacks a measurement boundary.** Conventional dense multiplication uses about `2×8^3 = 1,024` FLOPs. A claimed 0.1 J corresponds to about 98 microjoules per FLOP. That can be an end-to-end experiment including startup or a repeated workload, but it is not a credible intrinsic arithmetic calibration without duration, repetition count, precision, and baseline-power handling. The neighboring 8K example has roughly `10^12` FLOPs, so the two statements need separate provenance rather than a guessed correction.
* **Sharing a process node does not validate the machine model.** [NVIDIA's A100 whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) confirms 7 nm fabrication, but its actual cache hierarchy, tensor cores, interconnect, and execution overhead still matter. Agreement on a few matrix multiplications would be a sanity check, not evidence that arbitrary sparse gathers or learning algorithms are calibrated.
* **Read/write asymmetry depends on the model.** An address request and returned payload can give reads a round-trip latency, but their byte sizes, address availability, acknowledgment rules, and write-data transport must be specified. Equal read/write energy does not follow from round-trip versus one-way latency alone.
* **Transduction is a different information setting.** Access to the entire unlabeled test set can weakly enlarge the feasible algorithm class; it does not guarantee a strict accuracy improvement on every task. Batched online inference need not permit test-set adaptation. Benchmark both only if separate tracks are intended.
* **“Almost nobody” studies joules is rhetoric.** Memory-efficient training, communication-avoiding computation, and energy-oriented accelerators are established research areas. [EIE (Han et al., 2016)](https://arxiv.org/abs/1602.01528), for example, explicitly co-designs compressed networks and inference hardware around memory cost. Sutro's distinctive opportunity is a transparent, searchable end-to-end objective and compiler/hardware co-design, not an absence of predecessors.

## Proposed MNIST experiment protocol

1. **Pin the problem before claiming a benchmark result.** Record the released specification version, exact dataset bytes and splits, permitted precision, input/output placement, allowed processors, distance metric, tape traversal rules, and scoring boundary. The September 7 notes explicitly leave I/O/multicore choices and resolution rungs open. Until a release is available, name the implementation an exploratory surrogate and publish its assumptions beside results.

2. **Start with comparable predictors.** Use a small residual MLP or CNN that can actually bypass whole residual blocks. Compare tuned dense training; per-example dropout; batch-shared dropout; and spatial-group masks. Match architecture, initialization, data order, and optimizer family, while tuning learning rate for each configuration as the paper motivates. Include a smaller dense network at comparable executed work: “same architecture, fewer FLOPs” does not establish the optimal architecture at that budget. Add checkpointing, a reversible architecture, and a local-loss baseline only as separately specified comparisons.

3. **Separate the statistical experiment from the systems experiment.** First establish accuracy and loss under equal examples seen and under equal executed arithmetic budgets. Then use identical saved masks/inputs to compare dense-zero masking, actual bypass, gather/scatter, and grouped layouts. Check forward values and gradients against the mathematically equivalent masked implementation, including empty-active-set behavior. Treat any deliberate optimizer change as its own ablation.

4. **Record three kinds of cost without mixing them.** Record operation counts and byte-distance under the declared model; wall time and peak allocation on the local implementation; and energy only if measured by a documented instrument or estimated by explicitly stated coefficients. A Mac CPU timer is not a GPU power meter. Include masks, index arrays, packing buffers, gradients, optimizer state, intermediate predictions, test-set adaptation, and output transport within the declared boundary.

5. **Test the strongest locality hypothesis.** At fixed expected active depth, sweep mask-sharing group size and batch size. Evaluate both contiguous placement and a declared alternative layout; compare static prefix exits with scattered skips before and after specialization/repacking. Log unique weights touched, persistent bytes, saved activation bytes, total transferred bytes, byte-distance, and peak scratch occupancy. This makes a failure of FLOPs-to-energy proportionality observable and explains it.

6. **Use an accuracy frontier, not one cherry-picked rate.** Hold test labels out of configuration selection. Use validation splits and multiple paired seeds; publish run-level measurements and dispersion. Show accuracy against energy, time, peak area proxy, and scorer time, and mark nondominated configurations. Never compare one method at an easier target without displaying the difference. Where a frozen specification defines an accuracy threshold, report cost to reach that threshold with the stated success criterion.

7. **Price the scorer.** Implement exact event tracing at the smallest rung and compare it with aggregated analytical accounting on the same executions. Publish agreement checks and scorer runtime. If costs are approximate at larger scales, label the approximation and validate its error. The meeting's time-to-score objective makes the compiler's ability to summarize repeated structured operations a research result in its own right.

### Falsifiable hypotheses and useful outcomes

* Per-example masks may preserve the best accuracy at equal active FLOPs but lose to grouped masks at equal byte-distance.
* A decreasing dropout schedule may reduce cumulative work while leaving peak scratch close to dense training.
* Static compaction may outperform a resident elastic model on transport and area even when the logical active path is identical.
* Checkpointing may outperform dropout on peak activation storage despite using more FLOPs.
* A small dense model may dominate some elastic models once persistent parameter state and mask handling are charged.
* These hypotheses can all fail. A failure is useful if the trace identifies which cost component was small or amortized; none should be presented as an already observed MNIST result.

The paper supplies a strong question for Sutro: whether training a model to tolerate missing computation also makes it tolerate **locally organized** missing computation. Answering that requires an algorithm, layout, compiler, and cost model together—the co-design objective in the user's research agenda.
