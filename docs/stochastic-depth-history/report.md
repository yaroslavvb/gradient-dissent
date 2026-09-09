# Stochastic depth after 2016

Stochastic depth became an established training ingredient, especially in computer vision. Its descendants also pursued a second goal: making one network useful at several depths. **The main contribution of *Don’t Drop Dropout* is a modern causal-language-model recipe and evaluation of that tradeoff. Most of the underlying mechanism has substantial prior art.** The evidence below distinguishes a controlled improvement from adoption inside a successful system.

Three developments explain the history. Vision libraries made the method easy to use under the name **DropPath**. Language-model researchers combined layer omission with normalization, schedules and inference objectives. Efficient execution remained a separate engineering problem: a zero residual update does not itself mean a skipped computation. These conclusions are supported by the implementations and experiments linked throughout this report.

## The original result and the mechanism

Huang, Sun, Liu, Sedra and Weinberger’s ECCV 2016 paper randomly bypasses residual blocks, using a common mask for a minibatch in its main experiments. Survival decreases with depth. Training uses an unamplified retained residual; inference keeps all blocks and multiplies each residual by its survival probability. The introduction also mentions per-sample sampling, so that possibility was not absent from the original idea. [Original paper, §§1, 3](https://arxiv.org/pdf/1603.09382v3).

Its strongest clean examples were CIFAR-10: a 110-layer model improved from **6.41% to 5.25%** error; a 1202-layer model improved from **6.67% to 4.91%**. The ImageNet result was mixed: at 120 epochs, stochastic depth reached **21.98%** error versus **21.78%** for the dense control, while consuming less training computation. Thus even the original results supported a conditional benefit, not universal accuracy improvement. [Original paper, Table 1 and §4](https://arxiv.org/pdf/1603.09382v3#page=7).

Use **q** for drop probability and **ρ = 1 − q** for survival; papers sometimes call opposite quantities “p.” The schematic residual is a function F of the current activation h:

| Convention | Training residual update | Inference residual update |
|---|---|---|
| Original stochastic depth | h + M·F(h) | h + ρ·F(h) |
| Common modern DropPath | h + (M/ρ)·F(h) | h + F(h) |

The first row suppresses the original ResNet’s post-addition ReLU for readability. The second describes the default scaling in modern implementations. Both leave the identity stream alone. These are different conventions for a fixed set of weights; translating the convention does not establish identical optimization trajectories. [Original equations](https://arxiv.org/pdf/1603.09382v3#page=5); [Torchvision implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html).

Torchvision exposes both `batch` and `row` sampling. Its `row` mode broadcasts a single mask over all non-batch dimensions and divides a retained residual by survival. The function accepts an already-computed tensor, so calling it on `F(h)` does not avoid evaluating F. **Regularization and conditional execution are separate properties.** [Torchvision API](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.stochastic_depth.html) and [source](https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html).

The widely used `timm` library explicitly names its per-sample residual masking **DropPath / stochastic depth**. Its source also explains why the `drop_connect` name used in some EfficientNet implementations is misleading: classic DropConnect drops weights, whereas this operation drops residual paths. Merely reusing one DropPath module twice does not share the sampled mask; the mask must be explicitly reused to couple attention and FFN. [timm implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py).

[Explore the paper’s per-sequence masks, scaling and training schedule in the animation →](https://yaroslavvb.github.io/gradient-dissent/dropout-animation/)

## Three branches of the research line

| Branch | What changes | What counts as success |
|---|---|---|
| Regularization | Random residual or path omission during training; usually full inference afterward | Better held-out quality under a controlled training comparison |
| Training efficiency | Bypassed branches really avoid forward/backward work | Measured time or energy at a stated quality and training budget |
| Flexible inference | A fixed or input-dependent subset is used at deployment | Quality–latency tradeoffs, or verified speculative decoding against a specified target |

This is an analytical classification, not three mutually exclusive methods. A system can pursue more than one. It prevents a common mistake: evidence for regularization cannot be silently promoted into evidence for faster execution or arbitrary pruning robustness.

### Early extensions beyond a plain ResNet

**FractalNet (2016 preprint; ICLR 2017)** used interacting paths of different lengths without identity residual connections. Its DropPath training supported extraction of useful fixed-depth subnetworks and an anytime prediction tradeoff. That establishes an early related route from path dropout to usable submodels, although its graph and masking scheme differ from residual-block stochastic depth. [Larsson, Maire and Shakhnarovich](https://arxiv.org/abs/1605.07648).

**NASNet (2017 preprint; CVPR 2018)** made scheduled path dropping part of a successful architecture-search recipe. Zoph and colleagues found fixed DropPath less effective and increased path-drop probability during training. The resulting NASNet system reached 82.7% ImageNet top-1 accuracy, but that headline combines architecture search, augmentation and other choices; it is not the isolated contribution of dropout. Its paths are edges inside a cell, not necessarily whole residual blocks. [NASNet, §4 and Appendix A.5](https://arxiv.org/pdf/1707.07012).

**BlockDrop (2017 preprint; CVPR 2018)** moved to input-dependent inference. Wu and colleagues trained a policy to select residual blocks and jointly fine-tuned the backbone. They reported an average 20% inference speedup for ResNet-101 at the same 76.4% ImageNet top-1 accuracy. This is a measured conditional-computation result in the reported setup. It does not imply that random omission alone produces such a router, or that an arbitrary subset preserves accuracy. [BlockDrop](https://arxiv.org/abs/1711.08393).

## Successful use in vision and speech

The strongest answer to “who used it successfully?” includes Google’s EfficientNet work, FAIR/Meta’s DeiT–CaiT and DINOv2 work, Microsoft’s Swin, and the FAIR/Berkeley ConvNeXt collaboration. They supply different kinds of evidence:

| Work | Concrete evidence | What the evidence supports |
|---|---|---|
| **EfficientNet**, Tan and Le, ICML 2019 | Its training recipe uses stochastic depth under the “drop connect” name. [§5.2](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf) | Adoption inside a strong convolutional system. Its scaling gains are not an isolated dropout ablation. |
| **DeiT**, Touvron et al., ICML 2021 | Table 7’s reference recipe gets 81.8% ImageNet top-1; removing stochastic depth with other settings fixed fails, reaching 3.4%. [Table 7](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf) | Strong recipe dependence. The authors caution that ablated models may need different hyperparameters; this is not a generic 78-point benefit. |
| **CaiT**, Touvron et al., ICCV 2021 | After LayerScale restores trainability in a 36-block model, adapting stochastic depth raises top-1 from 80.5% to 83.0%. [Tables 1, 8](https://arxiv.org/pdf/2103.17239) | A substantial rate-tuning benefit, conditional on preceding architecture changes. Not a no-drop versus drop comparison. |
| **Swin**, Liu et al., ICCV 2021 | Recipes specify stochastic depth for classification, object detection and segmentation. [Appendix A.2](https://arxiv.org/pdf/2103.14030) | Adoption across vision tasks, not proof of an isolated causal gain in each task. |
| **ConvNeXt**, Liu et al., CVPR 2022 | DropPath is explicitly implemented; published rates vary with model and dataset. Tiny/Small ImageNet-22K pretraining uses zero. [Table 5](https://arxiv.org/pdf/2201.03545) | Direct reuse by an original stochastic-depth author, with evidence that the appropriate strength depends on the data regime. |
| **DINOv2**, Oquab et al., 2023/2024 | High-rate stochastic depth is implemented to omit residual computation. [§5](https://arxiv.org/html/2304.07193v2) | A concrete systems adaptation for self-supervised vision, with mixed joint-ablation results detailed below. |

### A close scheduling precedent: early stochastic depth in 2023

Liu and colleagues’ **Dropout Reduces Underfitting** directly tests stochastic depth, not only ordinary activation dropout. In its basic ViT-T recipe, three-seed ImageNet top-1 means are **73.9% without stochastic depth, 72.6% with standard stochastic depth, and 74.4% with early stochastic depth**. Early omission decays to zero over 50 epochs, then remains off. Its strength is separately selected, so this does not isolate timing at equal dropout exposure. [Table 2 and implementation details](https://arxiv.org/pdf/2303.01500).

The same work finds a different preferred direction for ViT-B: **77.0% without, 81.6% with tuned standard, and 82.3% with late stochastic depth**, starting after 50 epochs. Late scheduling did not significantly improve ConvNeXt-B or Swin-B. This is a particularly relevant antecedent to the newer paper’s decreasing schedule, and evidence against treating that direction as universal. [Table 4 and Appendix B](https://arxiv.org/pdf/2303.01500).

### DINOv2 separates implementation success from metric success

DINOv2’s ablation jointly adds LayerScale and stochastic depth: k-nearest-neighbor accuracy changes **74.5 → 75.4**, while linear-probe accuracy changes **83.2 → 82.0**. It neither isolates stochastic depth nor improves both readouts. The distillation phase also removes stochastic depth. Its broader speed/memory improvements combine several engineering changes, so they should not be attributed to this one ingredient. [DINOv2, Table 1 and §5](https://arxiv.org/html/2304.07193v2).

At high drop rates, the official implementation chooses a fixed-size random subset of examples, evaluates the residual only for them, and rescales by batch size divided by retained count. Attention and FFN draw subsets separately. This avoids work, but differs from independent Bernoulli choices and the newer paper’s shared whole-block mask. At low positive rates it instead uses ordinary compute-then-mask DropPath. [Official DINOv2 block](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/block.py).

### Speech is another substantial application

Baevski, Zhou, Mohamed and Auli’s **wav2vec 2.0** (NeurIPS 2020) uses layer-drop rates of 0.05 for BASE and 0.2 for LARGE in its Librispeech pretraining recipes. This is direct adoption in self-supervised speech representation learning and subsequent speech recognition. However, the much larger LV-60k pretraining configuration uses **no layer drop**. The method is therefore part of important successful speech recipes, not an explanation for every headline wav2vec result. [Original paper, §4.2](https://papers.nips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf).

Presence in a codebase is insufficient evidence of use. For example, the official Vim-Tiny pretraining command explicitly sets `--drop-path 0.0`. It would be misleading to count that recipe as a positive stochastic-depth result merely because the model supports the operation. [Official training command](https://github.com/hustvl/Vim/blob/main/vim/scripts/pt-vim-t.sh).

## Language models and measured acceleration

**LayerDrop — Fan, Grave and Joulin, ICLR 2020.** This made inference at reduced depth an explicit transformer-training objective. A 16-layer, 247M-parameter WikiText-103 model improved test perplexity from 18.7 to 18.3. The paper’s alternating-layer rule is for inference pruning; it is not the default training distribution. Removing contiguous first or last portions could perform poorly. Its §6 comparison found little difference between whole-layer and sublayer dropout in the tested setting. Those results already establish useful depth flexibility, while limiting the interpretation that every subnetwork is equally good. [Paper, §§3–6](https://arxiv.org/html/1909.11556).

The associated fairseq iterator draws layer decisions shared across a batch, does not add inverse-survival scaling, and includes all layers at evaluation. That is a concrete difference from both modern per-sample DropPath and the preferred recipe in the new paper. [Official implementation](https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_drop.py).

**Progressive Layer Dropping — Zhang and He, Microsoft, NeurIPS 2020.** PLD combines pre-normalization, shared block gates and a schedule that increases omission over training. Its equal-sample BERT runtime was 38.45 → 29.22 hours on 64 V100 GPUs. More ambitious speedup comparisons also change optimization and the amount of data processed, so they should not be read as pure savings from skipping. [Paper, Tables 1–2](https://proceedings.neurips.cc/paper_files/paper/2020/file/a1140a3d0df1c81e24ae954d935e8926-Paper.pdf).

DeepSpeed integrated PLD and published a training tutorial. This is stronger evidence of practical uptake than a citation count: a supported workflow existed for BERT pretraining and GLUE fine-tuning. It does not establish how often the feature is used in production, or that its original speedup transfers to current hardware. [DeepSpeed announcement, October 2020](https://www.deepspeed.ai/2020/10/28/progressive-layer-dropping-news.html); [tutorial](https://www.deepspeed.ai/tutorials/progressive_layer_dropping/).

**RaPTr — Panigrahi and colleagues, 2024 preprint; ICLR 2025.** RaPTr progressively increases the depth of random subnetworks rather than increasing omission. Its BERT model uses a 6–8–10–12 depth schedule, reporting validation loss 1.75 versus 1.76. Speedups estimated from measured stage throughput are 1.26× for BERT and 1.19× for UL2; its strong configuration includes an initial full-model warmup. First and last layers stay present in those experiments. The basic algorithm and an appendix scaling variant must be distinguished. This is close prior art for progressively denser training, with profiling of distributed execution. [Paper, §4 and Appendices F–J](https://arxiv.org/html/2402.05913v2); [ICLR 2025 record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b21ae5a5df83632324b61b595ab653b9-Abstract-Conference.html).

**LayerSkip — Elhoushi and colleagues, ACL 2024.** This combines per-sequence layer dropout, shared early-exit losses and a common output head. Its reported self-speculative decoding speedups include up to 2.16× on summarization and 1.82× on coding, in specified Llama experiments. These gains include drafting, verification and cache reuse. The paper’s residuals lack the newer inverse-survival factor, and its scratch-training schedule increases omission; continuation uses a constant temporal factor. Unverified early exits can change predictions. Correct speculative verification preserves the selected full target, not the quality of a separate dense-trained model. [LayerSkip, §§4–6](https://aclanthology.org/2024.acl-long.681.pdf).

This lineage is already more than “try a CNN trick on GPT.” It studies normalization, mask placement, curricula, training systems and decoding objectives. The newer recipe should be compared with those advances collectively.

There is relevant negative evidence too. Liu, Bauer and Manning’s **Drop Dropout on Single-Epoch Language Model Pretraining** (Findings ACL 2025) reports benefits from removing ordinary attention/MLP activation dropout, including comparisons with early-only dropout. That intervention is different from skipping an entire block. It is evidence about the cost of adding noise in a particular pretraining regime, not a refutation of every form of stochastic depth. [Paper](https://aclanthology.org/2025.findings-acl.111.pdf).

## The original authors after 2016

The five authors followed several related research directions. A shared author is evidence of a professional connection; it does not establish that every later method is a stochastic-depth derivative.

### Gao Huang: efficient architectures and adaptive computation

Huang received his Tsinghua PhD in 2015 and was a Cornell postdoc with Weinberger during the stochastic-depth work. His CV dates the postdoc to 2015–2018. He is now an associate professor in Tsinghua’s Department of Automation, leading the LEAP Lab. [Current homepage](https://gaohuang-net.github.io/); [CV](https://gaohuang-net.github.io/CV_Gao_Huang.pdf).

His immediate continuation included **DenseNet**, **Multi-Scale Dense Networks** and **CondenseNet**: respectively, dense feature reuse, anytime/budgeted prediction, and efficient learned group convolutions. More recently his listed work includes efficient attention, foundation-model architectures and **ViT³**, a CVPR 2026 test-time-training vision model. These extend an efficiency and adaptive-computation agenda, but are not one unchanged dropout algorithm. [DenseNet](https://arxiv.org/abs/1608.06993); [MSDNet](https://arxiv.org/abs/1703.09844); [CondenseNet](https://arxiv.org/abs/1711.09224); [ViT³](https://arxiv.org/abs/2512.01643).

### Yu Sun: calibration and test-time training

The correct Yu Sun is the author behind **yueatsprograms**, linked by the original code repository. His public homepage identifies him as a Stanford postdoc and NVIDIA researcher. He completed a Berkeley PhD in 2023, coadvised by Alexei Efros and Moritz Hardt, with a dissertation on test-time training. [Original repository](https://github.com/yueatsprograms/Stochastic_Depth); [homepage](https://yueatsprograms.github.io/); [Berkeley dissertation record](https://www2.eecs.berkeley.edu/Pubs/Dissertations/Faculty/mhardt.html); [Efros alumni](https://people.eecs.berkeley.edu/~efros/).

He coauthored **On Calibration of Modern Neural Networks** with Weinberger in 2017, then led **Test-Time Training with Self-Supervision** in 2020, also with Zhuang Liu. Later work includes **TTT layers** as expressive recurrent states and **End-to-End Test-Time Training for Long Context**. This changes how models learn from test context or store recurrent state; it is conceptually distinct from varying the number of active layers. [Calibration](https://proceedings.mlr.press/v70/guo17a.html); [TTT 2020](https://proceedings.mlr.press/v119/sun20b.html); [TTT layers, 2024](https://arxiv.org/abs/2407.04620); [long-context TTT, 2025](https://arxiv.org/abs/2512.23675).

### Zhuang Liu: direct reuse, pruning and dropout schedules

Liu studied at Tsinghua, completed his Berkeley PhD in 2022, worked at Meta FAIR in New York, and joined Princeton as an assistant professor on **July 1, 2025**. The university’s current record provides that exact start date. [Princeton profile](https://www.cs.princeton.edu/people/profile/zhuangl).

His later contributions include **DenseNet, Network Slimming, Rethinking the Value of Network Pruning, ConvNeXt, ConvNeXt V2,** and **Wanda**. ConvNeXt is a direct implementation connection: the official model applies DropPath to its residual branches. His **Dropout Reduces Underfitting** paper supplies an even closer conceptual connection through early stochastic depth followed by dropout-free training, discussed above. [Author’s publication page](https://www.cs.princeton.edu/~zhuangl/); [ConvNeXt code](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py); [early-dropout paper](https://proceedings.mlr.press/v202/liu23aq.html).

The pruning work is particularly relevant to interpreting reduced-depth models: directly training the smaller target architecture is an important baseline. A good subnetwork extracted from a large network does not automatically beat a carefully trained small network. [Rethinking the Value of Network Pruning, ICLR 2019](https://arxiv.org/abs/1810.05270).

### Daniel (Dan) Sedra: a more limited public follow-up

Sedra’s historical Cornell page identifies him as a master’s student advised by Weinberger, following a Carnegie Mellon computer-science degree in 2014. His Cornell thesis is *Training Paradigms for Deep Residual Networks* (2016). These are historical education records, not evidence that he is currently at Cornell. [Student-era homepage](https://www.cs.cornell.edu/~dms422/); [university thesis record/PDF](https://ecommons.cornell.edu/server/api/core/bitstreams/39357dbf-365b-4deb-b39e-f5e80f940c27/content).

Weinberger’s public lab data lists him as a former master’s student with **Pinterest** as the destination. That supports “listed by his former lab at Pinterest”; it does not establish his present title or a complete employment chronology. Public primary evidence does not support a detailed account of his subsequent stochastic-depth research. This gap should not be interpreted as evidence that he left machine learning. [Official lab data, Daniel Sedra entry](https://www.cs.cornell.edu/~kilian/coauthor_graph_data.js).

### Kilian Q. Weinberger: the broader resource-efficient-learning agenda

Weinberger remains a Cornell computer-science professor; his homepage also lists an ASAPP affiliation. His education was at Oxford and the University of Pennsylvania, where his PhD advisor was Lawrence Saul. His research has long combined learning algorithms with resource constraints; he served as ICML president during 2023–2025. [Cornell homepage](https://www.cs.cornell.edu/~kilian/).

After stochastic depth, his collaborations included **DenseNet, MSDNet, CondenseNet, Snapshot Ensembles, calibration**, and **Simplifying Graph Convolutional Networks**. His 2021 first-person retrospective explicitly links network compression, redundancy and DenseNet. This is useful evidence of the group’s intellectual progression, while its interpretation remains the author’s account. [Snapshot Ensembles](https://arxiv.org/abs/1704.00109); [SGC](https://proceedings.mlr.press/v97/wu19e.html); [Weinberger’s retrospective](https://www.asapp.com/blog/from-network-compression-to-densenets).

**The clearest continuity is specific:** Huang, Liu and Weinberger continued together into DenseNet; Sun and Weinberger into calibration; Sun and Liu into test-time training; Liu directly reused stochastic depth in later vision work. Dense feature concatenation, inference-time adaptation and residual DropPath should remain distinct mechanisms.

## What the 2026 paper adds

The useful comparison is with the accumulated literature, not only the 2016 baseline. *Don’t Drop Dropout* favors per-sequence, whole-transformer-block masks; inverse-survival scaling on both residual branches; a depth-increasing drop rate that decreases to zero over training; and CompleteP-based parameterization and transfer rules. It evaluates causal LMs up to 8.2B parameters and measures full-model quality and reduced-depth inference. That combined empirical recipe is its main value. [Paper, §§5–10](https://arxiv.org/pdf/2609.05275v1).

| Ingredient | Earlier evidence | What remains to establish in the newer setting |
|---|---|---|
| Randomly bypass a residual block | Stochastic depth; now a standard library operation | The appropriate probability and budget for a causal LM |
| Per-example masks | Modern per-sample DropPath; LayerSkip per-sequence dropout | Whether finer batch granularity helps enough to offset execution costs |
| Shared attention/FFN gate and inverse-survival scaling | Progressive Layer Dropping, 2020 | Comparative performance under the newer parameterization and recipe |
| Useful smaller subnetworks | FractalNet, LayerDrop, LayerSkip | Which deployment masks remain useful at larger causal-LM scales |
| Progressively grow active depth | RaPTr; early stochastic depth in 2023 vision experiments | Whether a smooth decreasing-dropout schedule improves the relevant frontier |
| Residual scaling with depth | Scaled-residual analysis by Hayou and Ayed; CompleteP | Reliable hyperparameter transfer with stochastic effective depth |

These rows identify antecedents, not equivalent replications. Architecture, objective, normalization, schedule and hardware can make an established mechanism work differently. The language and vision sections give the primary evidence behind each comparison.

### A specific related-work error

Section 6.1 of *Don’t Drop Dropout* classifies Zhang and He’s 2020 Progressive Layer Dropping as independently dropping attention and FFN. The older paper’s **Eq. 1, §4.2 and Algorithm 1** instead use one gate for the pair and scale each retained residual by inverse survival. Its prose states that both functions activate or skip together. This is direct prior art for the shared-gate construction. [New paper, §6.1](https://arxiv.org/pdf/2609.05275v1#page=6); [PLD, pp. 5–6](https://proceedings.neurips.cc/paper_files/paper/2020/file/a1140a3d0df1c81e24ae954d935e8926-Paper.pdf#page=5).

The correction narrows the novelty of the mechanism. It does not negate the new experiments: a broader comparison can still be useful when a mechanism is old. LayerDrop also already compared whole-layer and sublayer dropping in §6, which weakens an unqualified priority claim for that comparison. [LayerDrop](https://arxiv.org/pdf/1909.11556).

### What an explanation still has to account for

The same gate appearing twice matters mathematically. In a scalar block with input 1 and identity attention/FFN, a shared gate with survival ½ gives training outputs 1 or 9, averaging 5. Dense inference gives 4. This exact counterexample follows by applying the two residual updates in order. It does not show that the recipe fails; it shows that inverse-survival scaling alone cannot justify equality of the full block’s expected output. [Derivation and reproducible checks](https://yaroslavvb.github.io/gradient-dissent/#counterexample).

This issue also illustrates why “it is just dropout” is an incomplete implementation specification. Mask correlation, where scaling is applied, and whether the second branch receives a mask-dependent input all change the stochastic computation.

## Why the idea persisted, and why results differ

The best-supported interpretation is that stochastic depth is a flexible tool for changing the training objective and the paths through which learning occurs. Its persistent success does not require every original motivation to remain equally important in every later architecture.

Hayou and Ayed’s NeurIPS 2021 analysis made this distinction more precise. Under their stated residual-network assumptions, they relate stochastic depth to an explicit regularizer and, at large depth, data-dependent noise. Their initialization analysis finds that dropping blocks can reduce gradient growth without eliminating its exponential character; residual scaling remains necessary in the studied regime. Their budget-dependent survival-rate analysis also gives no reason to expect one probability profile to be universally optimal. These are results for their models and assumptions, not a theorem about modern transformer pretraining. [Regularization in ResNet with Stochastic Depth](https://proceedings.neurips.cc/paper/2021/file/82ba9d6eee3f026be339bb287651c3d8-Paper.pdf).

The apparent conflict between increasing and decreasing dropout schedules is informative. Increasing omission asks a network to tolerate progressively stronger disruption. Decreasing omission increasingly trains the complete network near the endpoint. Which is preferable can depend on whether the objective is full-depth final quality, submodel robustness, or total execution cost. NASNet, PLD, RaPTr and the newer causal-LM experiments occupy different points in that design space; their results do not establish a single universal curriculum.

There is also an implementation explanation for the method’s changing reputation. A cheap multiply applied after a residual is computed is easy to integrate into dense GPU kernels, and may improve generalization. Actually saving work needs a conditional batch execution, or a gather/compute/scatter implementation over retained examples. Small effective batches, routing and communication can change the outcome. The method can therefore be a successful regularizer even where its original speed argument does not carry through.

For the depth-robustness experiments in this repository, the most important question is **which missing blocks the training distribution prepared the model to tolerate**. A model trained never to omit its first block has no direct dropout exposure to that intervention. Prefix removal, scattered removal and deletion of an important transition block are different tests. Our A100 results already show family- and mask-dependent transfer; they should be read as an extension of this literature rather than the first evidence outside GPT. [Independent experimental report](https://yaroslavvb.github.io/gradient-dissent/a100-transfer/).

## Implications for further experiments

Three comparisons would be especially informative. These are proposed tests, not findings of the cited papers:

1. **Hold the model family fixed and cross mask granularity with time schedule.** Compare a shared attention/FFN gate with separate residual gates, and constant, early/decreasing and late/increasing omission. Match the integrated expected residual work before claiming one curriculum is better. Otherwise changing both timing and total dropout can confound the answer.
2. **Separate quality from realized execution savings.** Compare compute-then-mask, independent per-sequence bypass and fixed-size subset execution. Record full-model quality, omitted-block quality, actual latency and energy. DINOv2’s subset construction is a useful implementation baseline, but its correlated batch decisions should be disclosed.
3. **Measure the intended inference pattern and include trained-small controls.** Tune a fixed prefix, a scattered subset and any dynamic router on development data, then evaluate once on held-out examples. Compare against smaller dense models trained for the same deployment budget. Check calibration alongside accuracy and cross-entropy when those metrics disagree; that is a proposed diagnostic, not an established explanation of our current results.

For the energy-focused research agenda, grouped masks may trade statistical diversity for more efficient execution and fewer weight transfers. That is a hypothesis to test on the target hardware, not a conclusion supplied by the original stochastic-depth result. The historical literature makes the experiment more specific: determine which mask distribution and schedule improve the measured quality–cost frontier, rather than asking whether layer dropout is useful in general.

## Sources and scope

This report covers the five authors of the 2016 paper and selected influential descendants through 9 September 2026. It is a curated primary-source history, not a complete citation census or a survey of undisclosed production systems. Live institutional pages support the stated professional affiliations; Sedra’s current title and detailed industry chronology remain unresolved. His thesis metadata is supported by the university-indexed record even where direct PDF access is restricted.

Code links refer to the official branches inspected for this report; they are not claims that an implementation remained unchanged since the associated paper. All numerical results above are reported literature results or explicitly linked earlier experiments. No new models were trained for this history.

### Foundational work and implementations

- Huang, G.; Sun, Y.; Liu, Z.; Sedra, D.; Weinberger, K. Q. **Deep Networks with Stochastic Depth.** ECCV 2016; arXiv v3, 28 July 2016. [PDF](https://arxiv.org/pdf/1603.09382v3). Core mechanism, scaling and original comparisons.
- Torchvision maintainers. **stochastic_depth API and implementation.** Live documentation, inspected September 2026. [API](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.stochastic_depth.html); [source](https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html).
- Wightman, R., and timm contributors. **DropPath implementation.** Official source inspected September 2026. [Source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py).
- Larsson, G.; Maire, M.; Shakhnarovich, G. **FractalNet: Ultra-Deep Neural Networks without Residuals.** 2016 preprint; ICLR 2017. [Record](https://arxiv.org/abs/1605.07648).
- Zoph, B.; Vasudevan, V.; Shlens, J.; Le, Q. V. **Learning Transferable Architectures for Scalable Image Recognition.** 2017 preprint; CVPR 2018. [PDF](https://arxiv.org/pdf/1707.07012).
- Wu, Z., et al. **BlockDrop: Dynamic Inference Paths in Residual Networks.** 2017 preprint; CVPR 2018. [Record](https://arxiv.org/abs/1711.08393).
- Hayou, S.; Ayed, F. **Regularization in ResNet with Stochastic Depth.** NeurIPS 2021. [PDF](https://proceedings.neurips.cc/paper/2021/file/82ba9d6eee3f026be339bb287651c3d8-Paper.pdf).

### Applications and newer recipes

- Tan, M.; Le, Q. V. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML 2019. [PDF](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf).
- Touvron, H., et al. **Training data-efficient image transformers & distillation through attention.** ICML 2021. [PDF](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf).
- Touvron, H., et al. **Going deeper with Image Transformers.** ICCV 2021. [PDF](https://arxiv.org/pdf/2103.17239).
- Liu, Z., et al. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** ICCV 2021. [PDF](https://arxiv.org/pdf/2103.14030).
- Liu, Z., et al. **A ConvNet for the 2020s.** CVPR 2022. [PDF](https://arxiv.org/pdf/2201.03545); [official code](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py).
- Liu, Z.; Xu, Z.; Jin, J.; Shen, Z.; Darrell, T. **Dropout Reduces Underfitting.** ICML 2023. [Proceedings](https://proceedings.mlr.press/v202/liu23aq.html); [PDF](https://arxiv.org/pdf/2303.01500).
- Oquab, M., et al. **DINOv2: Learning Robust Visual Features without Supervision.** 2023 preprint; reviewed v2, February 2024. [Paper](https://arxiv.org/html/2304.07193v2); [official block implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/block.py).
- Baevski, A.; Zhou, H.; Mohamed, A.; Auli, M. **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.** NeurIPS 2020. [PDF](https://papers.nips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf).
- Fan, A.; Grave, E.; Joulin, A. **Reducing Transformer Depth on Demand with Structured Dropout.** 2019 preprint; ICLR 2020. [Paper](https://arxiv.org/html/1909.11556); [fairseq implementation](https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/layer_drop.py).
- Zhang, M.; He, Y. **Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping.** NeurIPS 2020. [PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/a1140a3d0df1c81e24ae954d935e8926-Paper.pdf); [DeepSpeed release, 28 October 2020](https://www.deepspeed.ai/2020/10/28/progressive-layer-dropping-news.html).
- Panigrahi, A., et al. **Efficient Stagewise Pretraining via Progressive Subnetworks.** 2024 preprint; ICLR 2025. [Paper, v2](https://arxiv.org/html/2402.05913v2); [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b21ae5a5df83632324b61b595ab653b9-Abstract-Conference.html).
- Elhoushi, M., et al. **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.** ACL 2024. [PDF](https://aclanthology.org/2024.acl-long.681.pdf).
- Liu, H.; Bauer, J.; Manning, C. D. **Drop Dropout on Single-Epoch Language Model Pretraining.** Findings ACL 2025. [PDF](https://aclanthology.org/2025.findings-acl.111.pdf).
- Elhoushi, M., et al. **Don’t Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference.** 2026; reviewed arXiv v1. [PDF](https://arxiv.org/pdf/2609.05275v1).
- Dey, N., et al. **Don’t be lazy: CompleteP enables compute-efficient deep transformers.** NeurIPS 2025. [Record](https://arxiv.org/abs/2505.01618). Parameterization antecedent; see the earlier technical review for the detailed comparison.

- Vim contributors. **Vim-Tiny pretraining command.** Official source inspected September 2026. [Script](https://github.com/hustvl/Vim/blob/main/vim/scripts/pt-vim-t.sh). Explicit zero DropPath rate.

### Author backgrounds and connected work

- **Gao Huang:** [professional homepage](https://gaohuang-net.github.io/) and [CV](https://gaohuang-net.github.io/CV_Gao_Huang.pdf). Tsinghua/Cornell chronology and current role.
- **Yu Sun:** [professional homepage](https://yueatsprograms.github.io/), [Stanford directory](https://www.cs.stanford.edu/people/yu-sun), [Berkeley dissertation list](https://www2.eecs.berkeley.edu/Pubs/Dissertations/Faculty/mhardt.html), [Efros alumni](https://people.eecs.berkeley.edu/~efros/). Identity, affiliations and doctoral background.
- **Zhuang Liu:** [Princeton faculty profile](https://www.cs.princeton.edu/people/profile/zhuangl), [dated university announcement, 25 March 2026](https://www.cs.princeton.edu/news/zhuang-liu-joins-faculty-bringing-expertise-deep-learning-and-computer-vision), [publication list](https://www.cs.princeton.edu/~zhuangl/). Education, FAIR and Princeton appointment.
- **Daniel Sedra:** [historical Cornell homepage](https://www.cs.cornell.edu/~dms422/), [2016 thesis](https://ecommons.cornell.edu/server/api/core/bitstreams/39357dbf-365b-4deb-b39e-f5e80f940c27/content), [official lab data](https://www.cs.cornell.edu/~kilian/coauthor_graph_data.js). Education and limited later-affiliation evidence.
- **Kilian Q. Weinberger:** [Cornell homepage](https://www.cs.cornell.edu/~kilian/) and [From network compression to DenseNets, 7 April 2021](https://www.asapp.com/blog/from-network-compression-to-densenets). Professional background and first-person historical account.
- Huang, G., et al. **Densely Connected Convolutional Networks.** CVPR 2017. [Record](https://arxiv.org/abs/1608.06993).
- Huang, G., et al. **Multi-Scale Dense Networks for Resource Efficient Image Classification.** ICLR 2018. [Record](https://arxiv.org/abs/1703.09844).
- Huang, G., et al. **CondenseNet: An Efficient DenseNet using Learned Group Convolutions.** CVPR 2018. [Record](https://arxiv.org/abs/1711.09224).
- Huang, G., et al. **Snapshot Ensembles: Train 1, Get M for Free.** ICLR 2017. [Record](https://arxiv.org/abs/1704.00109).
- Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. **On Calibration of Modern Neural Networks.** ICML 2017. [Proceedings](https://proceedings.mlr.press/v70/guo17a.html).
- Sun, Y., et al. **Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.** ICML 2020. [Proceedings](https://proceedings.mlr.press/v119/sun20b.html).
- Sun, Y., et al. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** 2024 preprint. [Record](https://arxiv.org/abs/2407.04620).
- Tandon, A., et al. **End-to-End Test-Time Training for Long Context.** December 2025 preprint. [Record](https://arxiv.org/abs/2512.23675).
- Han, D., et al. **ViT³: Unlocking Test-Time Training in Vision.** 2025 preprint; CVPR 2026. [Record](https://arxiv.org/abs/2512.01643).
- Liu, Z., et al. **Rethinking the Value of Network Pruning.** ICLR 2019. [Record](https://arxiv.org/abs/1810.05270).
- Wu, F., et al. **Simplifying Graph Convolutional Networks.** ICML 2019. [Proceedings](https://proceedings.mlr.press/v97/wu19e.html).
