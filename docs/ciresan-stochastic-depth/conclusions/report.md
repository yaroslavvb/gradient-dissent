# What it takes to train Ciresan-style MNIST networks with stochastic depth

**The practical answer:** a 12-million-parameter fully connected MNIST network trains successfully with stochastic depth using a small architectural change and ordinary SGD. The robust configuration we tested combines normalized pixels, linear classifier logits, four dimensionally valid residual bypasses, and a learning rate selected for that residual model. Its clearest demonstrated benefit is that trained branches can subsequently be removed with much less damage. Better full-model accuracy and faster convergence to a quality target were not established.

“Ciresan-style” here means the supplied tapered fully connected architecture, 784→2500→2000→1500→1000→500→10, with 11,972,510 parameters. This is substantial matrix computation on A100s, but remains MNIST with six affine layers. It is not a reproduction of Ciresan’s augmentation-heavy record-setting system, and it does not establish that the same recipe will stabilize a much deeper vision or language model.

The [main training plots](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/telemetry/) expose 838 scalar fields from 18 verified reruns. The [mask and digit explorer](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/) contains all 16 branch masks and individual examples. The run registry below connects these results to source records and 18 verified W&B runs in [gradient-dissent](https://wandb.ai/yaroslavvb/gradient-dissent). The [curated W&B report](https://wandb.ai/yaroslavvb/gradient-dissent/reports/Ciresan-MNIST:-measured-training-curves-·-2026-09-09--VmlldzoxNzkwNDAzNA==) provides eight focused learning-curve, gradient, and baseline timing charts. These runs import the completed experiments: their recorded training clocks describe the original optimization, while W&B runtime describes the later upload.

## How difficult was it to get training working?

There were two different obstacles: a fragile inherited model configuration, and efficient implementation of real conditional execution.

The raw-input, ReLU-output historical configuration was an unreliable starting point. In matched 50,000-example fidelity runs, plain-network final test accuracies were 98.28%, 86.87%, and 88.62%. Adding crop residual bypasses without stochastic depth gave 9.80%, 36.83%, and 39.05%. Adding constant stochastic depth gave 9.80% for all three seeds. Thus failure already appeared in the residual control; it cannot all be attributed to dropout. Two residual runs eventually escaped near-chance behavior, so “permanently dead network” would overstate the observation. [Full fidelity results](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/ciresan-stochastic-depth.md#historical-fidelity-seed-sensitivity-and-late-escapes).

The adapted configuration—pixels divided by 255, a linear output head, crop residual blocks, and LR 0.01—successfully trained both stochastic-depth recipes at all three confirmation seeds. All six SD runs reached 100% dense fitting accuracy by epoch 100. We did not need a learned bypass projection: the architecture only narrows, so a fixed prefix crop supplies the correctly shaped shortcut.

This is a **tested sufficient package, not a proof of the smallest necessary package**. Input scale, head, LR and shrinkage differ between the historical and adapted experiments. We have not isolated each change in a full matched factorial SD study. A separate raw-input, linear-head plain baseline also succeeded, so normalization is not universally necessary. The output-head diagnostics provide stronger single-change evidence than the bundled historical/main comparison.

Learning rate mattered. Each main arm received the same grid {0.01, 0.03, 0.1} at tuning seed 1. Constant SD at LR 0.1 diverged at epoch 12, decreasing SD at epoch 1, and residual unit dropout at epoch 21. LR 0.01 was selected for the residual control and every residual regularizer; LR 0.03 was selected for plain. Both SD schedules also completed tuning at LR 0.03; 0.01 was selected, not uniquely necessary. A rate that works for the plain network should therefore not be carried over unquestioned after adding residual paths and inverse-survival scaling. [Frozen validation selection](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/selection.json).

## A minimal recipe that actually worked

The following reproduces the model and optimization choices of the successful constant-SD arm. It is a starting configuration with evidence, not a globally optimal hyperparameter prescription.

| Component | Successful setting | Why it is present / evidence strength |
|---|---|---|
| Inputs and head | Raw MNIST pixels scaled by 1/255; linear ten-class logits; cross-entropy | Stable tested package. Separate head diagnostics identify an output-ReLU failure; normalization was not proved individually necessary. |
| Bypass | Around each of the four middle affine layers, retain the first `out_features` coordinates of the input | A valid path must remain when a branch is absent. The crop has zero learned parameters and is lossy, not an identity on the wider vector. |
| Compulsory layers | Keep the first affine/ReLU stem and the final classifier | These supply the representation and readout for every sampled subnetwork in this architecture. |
| Drop probabilities | Shallow to deep: 0.1, 0.2, 0.3, 0.4 | A directly tested constant schedule. More aggressive probabilities or different ordering need their own check. |
| Training gates | Fresh independent Bernoulli decision for each body branch, shared across the minibatch; retained gain `1/(1-p)` | The implemented convention preserves the expected affine contribution before ReLU. It does not make final logits or loss unbiased. |
| Dense evaluation | Keep all branches at gain one | This is the evaluated deployment convention. Explicit deletion experiments use a separate gain-one masked forward. |
| Optimizer | SGD, LR 0.01, momentum 0.9, batch 64; no parameter shrinkage in this arm | Selected on validation from the common three-rate grid. A skipped branch receives no gradient/momentum update on that minibatch. |
| Training and selection | 50k fitting examples / 10k validation; 100 epochs; evaluate at epoch 1 and every 5; choose minimum validation CE | Makes the experiment repeatable. Also report epoch 100, because accuracy and CE prefer different checkpoints. |

The body computation is:

```text
training: h_next = ReLU(crop(h) + kept * Linear(h) / survival)
evaluation: h_next = ReLU(crop(h) + Linear(h))
```

When a branch is skipped, our implementation executes only `ReLU(crop(h))`; it does not compute `Linear(h)` first. The multiply-by-zero formula describes the function, not the efficient control flow.

The repository already implements this model:

```python
import torch
from model_data import CiresanMLP

model = CiresanMLP(
    recipe="sd_constant", pmax=0.4,
    input_scale=1/255, output_relu=False,
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Use the [frozen main manifest](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/telemetry/main-manifest.json) and [runner](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/telemetry/controlled_train.py) for the complete experiment. The small snippet describes the recipe; it omits data splitting, epoch shuffling and the optimized execution loop. In an ordinary eager implementation, clearing gradients with `set_to_none=True` before a conditionally executed forward gives skipped parameters no optimizer gradient. Our graph implementation explicitly preserves the same skipped-branch momentum behavior. These semantics should be tested when changing optimizers or adding shrinkage.

The second successful schedule starts at probabilities [0.2, 0.4, 0.6, 0.8] and decreases linearly to zero at epoch 100. It has the same expected integrated omitted work as the constant schedule. It changes peak strength and variance as well as timing; it is not an isolated test of schedule order. It also has no extended dense fine-tuning tail. Constant SD is the simpler demonstrated starting point; superiority of either schedule depends on the chosen outcome.

## Making it fast is a separate requirement

A branch computed and then multiplied by zero can regularize, but cannot save that branch’s matrix multiplication. Genuine skipping, regular minibatches, and correct optimizer state handling are the requirements for the execution savings measured here.

Our A100 path keeps MNIST resident on the device, uses FP32 parameters with TF32 matrix multiplication, fuses parameter updates, and captures the full update in CUDA graphs. Four eligible branches give only 16 possible topologies, which we capture and select between. Gates are prepared without reading a GPU scalar back to Python at each decision. These are performance choices; CUDA graphs are not a requirement for the statistical idea to work. Pre-capturing every mask also does not scale to dozens of optional layers.

The instrumented reruns took approximately 66.22 seconds of training for residual dense and 56.85–56.86 seconds for the SD arms over 100 epochs: about 14% less time at fixed exposure. But the earlier matched comparison first observed 98% validation accuracy after about 6.69 seconds for residual dense, versus 8.57 and 9.04 seconds for constant and decreasing SD. **Cheaper epochs did not give faster convergence to that quality target.** Evaluation cadence limits the precision of those first-crossing times. [Controlled timing report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/).

The separate 5.29–6.05-second result to 98.63% official-test accuracy belongs to an optimized **unit-dropout plain baseline**, using 60k fitting images, batch 256 BF16 and a different LR/shrinkage schedule. It is not a stochastic-depth result or a matched comparison with the 50k SD arms. Its elapsed time to first target, including setup and telemetry, was 10.45–14.87 seconds; full invocations took 10.82–15.37 seconds. The target was selected/monitored on the test set. [Baseline details](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/optimization/).

## When did stochastic depth work?

It is useful to keep four success criteria separate.

| Intended benefit | What these experiments support |
|---|---|
| Train a large fully connected model successfully | Yes. Both adapted SD recipes train at all three seeds and fit the training set completely. |
| Improve full-model generalization | Not established. Neither improves primary mean test accuracy, and both have higher epoch-100 test CE than residual dense. |
| Make computation removable | Yes, with endpoint qualifications. SD substantially reduces damage from the explicitly trained branch-bypass intervention. |
| Deliver faster inference | Fixed reduced networks can be faster. Our input-dependent tree router was slower and did not establish a quality/cost advantage. |

The primary checkpoint is selected by validation CE; the final checkpoint is epoch 100. Full-model test accuracy is:

| Method | Validation-selected accuracy | Epoch-100 accuracy | Epoch-100 test CE |
|---|---:|---:|---:|
| Residual dense | 98.373% | 98.400% | 0.0972 |
| Constant SD | 97.803% | 98.443% | 0.1336 |
| Decreasing SD | 97.997% | 98.510% | 0.1170 |

All are three-seed means. Constant SD’s primary accuracy difference from residual is −0.570 percentage points, 95% paired interval [−0.958, −0.182]; decreasing SD’s is −0.377 [−0.931, +0.177]. Their small positive final-accuracy differences both have intervals including zero. This is not strong evidence for an accuracy regularizer, even though the masked-network results are encouraging. [Complete quality tables](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/training-findings.md).

At selected checkpoints, keeping two of the four eligible branches reduces accuracy to 97.536% for residual, 97.387% for constant SD, and 97.665% for decreasing SD, averaged over all six two-branch masks. More informative for deletion tolerance is the increase in CE over the same model’s full output: +0.0650, +0.0115 and +0.0062 nats. Decreasing SD improves this deletion-damage measure relative to residual by 0.0588 nats, with a paired interval excluding zero; constant SD’s analogous interval includes zero. Absolute masked quality and damage relative to each model’s own baseline are different outcomes.

At epoch 100 the contrast is stronger: two retained branches give 97.498% for residual, 98.335% for constant SD and 98.354% for decreasing SD. Even removing all four eligible branches leaves 97.310% / 97.460% for the two SD recipes. That remaining network still runs a learned stem, successive crops and the classifier. This demonstrates how shallow a useful MNIST computation can be after this training; it is not evidence that arbitrary deep computations can be omitted.

Which layer is removed also matters. The first eligible branch costs five million MACs; the last costs half a million. At the final checkpoint, removing just the first branch changes residual accuracy from 98.400% to 97.640%, constant SD from 98.443% to 98.360%, and decreasing SD from 98.510% to 98.400%. Layer count alone hides both the computational benefit and the error cost. [All masks and branch costs](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/).

There is a useful mechanistic observation: at selected checkpoints the mean per-example fraction of nonconstant centered-logit mask-response energy in higher-order interactions falls from 13.31% in residual dense to about 2.3% in both SD arms. The branches’ effects combine more simply under this measured intervention. That does not make the network globally linear or bound the size of a single branch’s effect.

## Are particular digits more amenable to dropping layers?

There are visible digit differences, but we did not establish a dependable “skip more layers for this digit” rule.

The predeclared class test compared digit 1 with the average of digits 4 and 9, among examples that the full network classified correctly. Resilience was the fraction of the six two-branch masks that kept an example correct. We then standardized the groups to a shared distribution of full-model confidence margins, using bins and weights fixed on validation data.

For decreasing SD at selected checkpoints, digit 1’s raw advantage was **+0.697 percentage points [0.336, 1.058]**. After margin standardization it was **+0.302 [−0.065, +0.669]**. Every central method’s primary adjusted interval includes zero. At the final decreasing-SD checkpoint, the raw +0.209-point advantage instead becomes **−0.119 [−0.185, −0.054]** after adjustment. The result is sensitive to confidence and checkpoint, not a stable intrinsic ordering of digit classes. Coarse margin adjustment is descriptive; it does not prove that confidence causally explains every class difference.

A separate head diagnostic is especially instructive. In one raw-input baseline at seed 104, no test sixes were correctly classified after five epochs with the ReLU classifier head: 0/958. Removing only the output ReLU in that diagnostic yielded 945/958 correct sixes, while overall accuracy rose from 88.69% to 97.91%. This was not a stochastic-depth experiment. It shows how a class-specific training pathology can masquerade as an intrinsically difficult digit. The output-head activation/gradient path should be checked before assigning a semantic explanation. [Head diagnostic evidence](https://github.com/yaroslavvb/gradient-dissent/blob/main/research/ciresan-optimization.md).

Our practical routing attempt used only 49 cheap image features, not true labels or a free full-model confidence score. A shallow tree selected masks under a calibration error tolerance. Only four of nine primary routers kept that tolerance on test; none established an assignment advantage over a matched-cost shuffled policy. Actual grouped routing was 2.07–3.25× slower than dense A100 inference. Static masks were 1.19–1.93× faster, but their quality tolerance also often failed. Fewer nominal MACs and appealing class patterns did not suffice for a useful adaptive system. [Routing results](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/hypotheses/routing-summary.json).

## What the extra statistics changed about the explanation

The added diagnostics help distinguish hypotheses; they do not supply a universal scalar that predicts whether dropout will work.

On the fixed 128-example training probe, late gradient “diversity” often approaches 128 because one example dominates the remaining tiny gradients. In the FP64 offline pass, final SD norm participation is close to one example. A large diversity/noise ratio therefore should not automatically be interpreted as many useful independent gradients or a recommended batch size.

Similarly, the residual control’s exact weight CE-GGN trace falls from 87.04 to 0.4235 while its logit-Jacobian Gram trace grows from 976 to roughly 170,621. Confidence can flatten the loss even while raw logit sensitivity grows. Final SD CE-GGN traces are 19.2× and 31.2× below residual, whereas the Jacobian traces are only 1.53× and 2.08× lower. These are one-seed observations on a reused training probe.

KFAC’s accurately computed factor spectra do not make the approximation accurate: final GGN diagonal relative-L1 errors range from 83% to 583% across these models/layers. Fisher, GGN and Jacobian statistics must remain separately named. The restored logger corrects historical normalization and labeling bugs. [Curvature report and spectra](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/curvature-summary.md).

## How this connects to successful uses elsewhere

The idea already worked outside language models in Huang et al.’s [Deep Networks with Stochastic Depth](https://arxiv.org/html/1603.09382v3), which trained deep convolutional residual networks with minibatch-wide branch omissions. Its original scaling convention differs from ours: surviving training branches are unscaled and testing weights them by survival; our code uses inverse-survival training gains and gain-one testing. Specify that convention when transferring a recipe.

[LayerDrop](https://arxiv.org/abs/1909.11556) explicitly targets both regularization and extracting shallower Transformer subnetworks. That distinction matches the strongest result here: a removable network can be valuable even when its full-depth accuracy is not better.

Modern vision architectures also retain this form of regularization. In the [official ConvNeXt implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py), the branch is computed before DropPath masks its output. The inference from that code is that this DropPath call supplies regularization without itself avoiding the branch’s forward work. Successful use of stochastic depth elsewhere therefore does not guarantee the execution savings of our specialized skipping implementation.

## The practical decision

For this architecture, I would start from the validated residual-dense configuration, add the constant [0.1, 0.2, 0.3, 0.4] branch schedule, and check both dense validation quality and the exact fixed-mask deployment modes of interest. That is a compact evidence-backed route to successful SD training. The more elaborate decreasing schedule is a subsequent comparison, rather than a prerequisite.

If the objective is full-model accuracy or quickest convergence, keep the residual-dense and plain/unit-dropout alternatives in the comparison: these results do not justify choosing SD by default. If the objective is removable computation, the SD results are substantially more promising. Prefer a measured fixed reduced network before investing in per-example routing, and validate its absolute quality under the intended gain/scaling convention.

A theorem-level or empirically universal minimum is still missing. The most informative follow-up would isolate the head/input/LR package in matched SD runs, test a larger fresh probe for the gradient claims, and repeat the digit-resilience analysis on new seeds and a second dataset. An extended dense fine-tuning tail, different bypasses and alternative routers are plausible experiments, not demonstrated requirements or improvements.

## Experiment map and limits

| Phase | Scope | What it establishes |
|---|---|---|
| Historical audit and fidelity controls | Historical W&B metadata; nine final source-style plain/residual/SD runs | The inherited raw/ReLU configuration is seed sensitive and can fail before useful SD training. |
| Controlled training | Six pilots, 15 LR-grid attempts including three divergences, 15 main final runs | A sufficient stable package, matched residual comparisons, checkpoint dependence and measured fixed-horizon savings. |
| Baseline optimization | Separate kernel/recipe trials and test-targeted confirmations | Fast ordinary/unit-dropout training; not the causal SD comparison. |
| Layer-removal audit | Five methods × three seeds × selected/final states × all 16 masks | Deletion damage, digit/confidence contrasts, interactions and routing limits. |
| Instrumented reruns | The same 15 main runs (five recipes × three seeds) plus three optimized baselines | Exact recorded learning-curve reproduction, 401 telemetry rows and measured diagnostic overhead. These are repeats, not new independent seeds. |
| Offline curvature | Three central methods, seed 101, seven snapshots each | Precisely defined finite-probe Fisher/GGN/Jacobian and factor statistics; no additional optimization steps. |

The study reuses MNIST test examples, validation choices and seeds across phases. Three-seed intervals are conditional, descriptive and uncorrected for multiple comparisons. Test-targeted baseline optimization is explicitly separate from validation-selected SD comparisons. One 128-example probe cannot establish population geometry, and a six-affine MNIST network cannot establish behavior on arbitrary deep tasks.

The last measured MNIST compute total was about **$5.00 of the authorized $30**, including approximately $0.93 for telemetry/curvature reruns. All experiment GPUs were stopped. This synthesis and W&B import preparation launch no new GPU training. Full frozen manifests, attempts, failures, sources, hashes, code and data remain in the [experiment repository](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth).
