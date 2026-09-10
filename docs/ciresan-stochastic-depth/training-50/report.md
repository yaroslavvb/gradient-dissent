# Train with branches missing half the time. Test with every branch.

## What the experiment says

**Yes: this residual MLP can drop each of its four middle branches on half the training minibatches and use all branches at testing.** At the primary, validation-loss-selected checkpoint, making all four branches droppable changes accuracy from **97.990% to 98.043%**, a paired **+0.053 percentage points** (95% interval −0.197 to +0.303). At the fixed 100-epoch endpoint, accuracy changes from **98.440% to 98.510%**, **+0.070 points** (−0.288 to +0.428). These intervals include zero: this small experiment establishes neither an accuracy improvement nor a tight guarantee of no harm.

The practical benefit is cheaper training at fixed exposure: **66.34 → 41.22 seconds for 100 epochs**, about **37.9% less measured training time**. This excludes setup, evaluation and telemetry. Every test prediction still computes the same full network, so this is not an inference speedup or a time-to-target-accuracy result.

The clearest cost appears in probability quality. At epoch 100, test cross-entropy rises from **0.09255 to 0.13723 nats**, a paired increase of **0.04468** (95% interval **0.02642 to 0.06295**). All three seeds worsen. Cross-entropy scores the probability assigned to the correct digit, rather than just whether it ranks first; more confident mistakes can therefore coexist with similar accuracy. This is evidence of worse log-loss, not a separate calibration measurement. At the primary checkpoint, CE is essentially unchanged. Both no-dropout and all-four-dropout models fit the full training set at 100% by epoch 100, so this intervention does not prevent interpolation.

## How much does making a branch droppable harm individual examples?

For all four branches, the epoch-100 comparison creates **34.3 new mistakes** and repairs **41.3 old mistakes per 10,000 images**, on average across seeds. That is **0.343% newly wrong**, **0.413% newly correct**, and a net gain of only seven correct predictions. About **0.349% of previously correct predictions** become wrong. At the primary checkpoint, the corresponding means are 69.7 harmed and 75.0 repaired. Thus similar aggregate accuracy does not mean the same digits remain correct.

The viewer compares identical images between seed-matched training runs. Select “Newly wrong” to see concrete harms, or “Newly correct” for repairs. Select “Previous step in chosen order” to isolate one additional droppable branch; its population paragraph uses all 10,000 examples, while that mode’s illustrated images come only from the fixed reference panel. Changing a checkbox selects a separately trained model. It never removes a test-time layer.

## Is there a natural order?

The validation rule chose **4 → 1 → 3 → 2**, counting branches from shallow to deep. Its primary test-accuracy path is **97.990 → 97.950 → 97.967 → 98.040 → 98.043%**. The corresponding epoch-100 path is **98.440 → 98.470 → 98.537 → 98.573 → 98.510%**. Neither path is monotonic. Shallow-first and deep-first paths, plus every other ordering, are available in the explorer.

This is **a fragile preference, not a universal layer ranking**. The frozen order ranks 4th, 10th and 2nd when its validation criterion is computed separately for seeds 201, 202 and 203; no individual seed selects it. At the primary endpoint, every branch has both positive and negative mean effects depending on which other branches are already droppable. All four context-averaged branch-effect intervals include zero.

There is one useful hypothesis for a follow-up: at epoch 100, making **branch 1** droppable has positive mean effects in all eight contexts, averaging **+0.090 points** (95% interval −0.007 to +0.186). It also skips the largest affine transform and therefore saves more computation than dropping a later branch. Its primary-endpoint effects change sign across contexts, so the evidence does not justify recommending it as an accuracy-optimal first choice. The order experiment also does not test gradually enabling dropout during one training run.

## Do particular digits tolerate it better?

The per-digit table exposes real differences, but they depend strongly on the endpoint. With all four branches droppable, digit **3** changes by **+2.112 points** at the primary checkpoint and **−0.264 points** at epoch 100. Digit **9** changes by **−0.727 → +0.429 points**. Those sign changes undermine a simple story that one digit intrinsically needs less depth.

At epoch 100, digits **0, 1 and 6** improve in all three seeds, while **8** worsens in all three. Most corresponding intervals still include zero. Digit 1’s +0.264-point interval excludes zero before adjustment, but it is one of many class/subset comparisons; treat it as an exploratory lead. Every prediction here uses full depth, so these observations describe sensitivity to the training procedure, not intrinsic requirements for inference depth.

## Minimum changes demonstrated here

The successful recipe has **a compatible bypass around every optional affine branch**, **gain 2 when a branch survives 50% training dropout**, and **gain 1 with every branch present at evaluation**. The bypass is needed because adjacent Ciresan widths differ; simply deleting an affine layer would not preserve the tensor shape. Training really omits the branch’s forward/backward computation and its parameter update when skipped.

No dropout warm-up, special learning-rate schedule, normalization layer, augmentation or per-subset tuning was required for these 100-epoch runs. This demonstrates one working construction, not that every ingredient is necessary or that an untouched feedforward Ciresan network behaves the same way. The next useful tests would separate branch-1 compute savings from its accuracy effect, and address the final log-loss penalty with validation-only tuning. They were not performed in this cohort.

## Why the primary and final accuracies differ

The primary rule minimizes validation **cross-entropy**, not validation error. For the no-dropout runs it selected epochs 15, 5 and 5; for all-four-dropout it selected epochs 25, 10 and 5. Continuing to epoch 100 improves the number of correct test labels while worsening their probability score. Both endpoints are retained because choosing whichever one looks favorable would obscure this distinction.

## Experimental design

This experiment asks how much full-network prediction quality changes when we make particular layers optional during training. Each eligible middle branch is skipped independently on half the minibatches in expectation. All branches are present at evaluation. This is a new training experiment, separate from the earlier explorer that removed branches from already trained networks at inference.

We trained all 16 subsets of four eligible residual branches under three new seeds (201–203), for 48 new runs. Each seed shares the same initial parameters, training-example order and four underlying random gate draws per step across every subset. The baseline has the same residual architecture and no dropped branches. SGD uses learning rate 0.01, momentum 0.9, batch 64, and 100 epochs, with no augmentation, ordinary unit dropout or weight decay. This isolates the eligibility intervention under a fixed optimization recipe; it does not optimize each subset's learning rate separately.

The model is the normalized-input, linear-output Ciresan-width MLP: 784→2500→2000→1500→1000→500→10. The four middle branches have prefix-crop residual bypasses. The first feature layer and final classifier always run. During training, a retained eligible branch contributes twice its affine output; a skipped branch contributes zero, leaving the bypass. At testing, every branch contributes once. This matches the average branch contribution before ReLU, not the nonlinear network's exact mean prediction.

The primary endpoint is the checkpoint with minimum validation cross-entropy at the fixed evaluation cadence. Epoch 100 is a separate secondary endpoint. Both are shown. Every reported test evaluation uses the same official 10,000 images, in the same order. The dataset was used in earlier project experiments, so fresh seeds do not turn it into a new untouched test set.

## Reading the graphs and individual predictions

The horizontal axis counts branches eligible for 50% dropout **during training**. Zero means no training dropout; four means every middle branch is eligible. It never means zero to four layers present at inference. Every plotted point evaluates all six learned affine layers.

At each count, the green line averages all subsets of that size within a seed, then averages the three seed results. Gray points show every subset's three-seed mean. The orange line follows a chosen order. Its intervals describe variation across three training seeds, not across the 10,000 images or the many subsets. We preserve numerical intervals without clipping; the full-scale graphic may crop interval lines at the displayed axis boundary. Use the table for their complete values. Symmetric t intervals can extend below zero for rare-event rates; that is a limitation of this small-sample approximation, not a negative number of mistakes.

A digit is **harmed** when the no-dropout baseline gets it right and its seed-matched dropout-trained model gets it wrong. A **repair** is the reverse. Both rates divide by all test inputs; accuracy change equals repairs minus harms. Small net accuracy differences can conceal many changed predictions. Per-class comparisons use all examples of that true digit, rather than treating the illustrated gallery as a sample.

The example viewer shows the same image, seed and endpoint in two separately trained models. Its ten bars show their softmax probabilities. The fixed reference panel contains the first ten original test indices per digit. Additional first harmed/repaired examples are explicitly selected to illustrate outcomes and do not estimate their frequency. Changing a checkbox selects another completed training run; it does not alter a single model during inference. The optional previous-step comparison uses only the fixed 100-image panel shared by every model, and recomputes harm/repair relative to the previous prefix of the chosen order. The population summaries continue to compare with the no-dropout baseline.

## What an order can—and cannot—tell us

There are 24 orders for adding four eligible branches. Each order traces five independently trained configurations: none, one eligible branch, two, three, and all four. It is not a curriculum that enables dropout gradually within a training run.

One order is chosen using validation error, averaged over the three intermediate subsets and three seeds at the validation-CE-selected checkpoints. Ties use validation CE, then lexicographic branch order. This decision is frozen before test-outcome analysis and reused for the final-epoch view. All 24 paths remain inspectable, but their test results do not retrospectively choose our reported order.

To assess whether a branch is consistently a good next choice, we also compare all 32 directed additions of one branch to an existing subset. Each branch has eight contexts. A small average effect with a wide context range does not establish a universal ordering. All differences and intervals remain conditional on this architecture, learning rate, training horizon and three seeds.


# Half-drop training: exhaustive four-branch eligibility results

Complete 48-run panel: 16 independently trained eligibility subsets × seeds 201–203. Every test evaluation keeps all six affines at gain one. Eligibility means a 50% minibatch drop chance during training, not deleting a layer during inference.

Validation-CE-selected checkpoints are primary; epoch 100 is secondary. Values are seed means with 95% t intervals (n=3, df=2), without multiple-comparison correction.

Validation-selected static prefix order: 4 → 1 → 3 → 2. This order was frozen before test analysis and reused for both endpoints. It is not a temporal training schedule.

| Endpoint | Eligible branches | Mean full-test accuracy % [CI] | Accuracy delta vs none pp [CI] | Mean CE [CI] | Subset-mean accuracy range % |
| --- | ---: | --- | --- | --- | --- |
| selected | 0 | 97.990 [96.936, 99.044] | 0.000 [0.000, 0.000] | 0.0748 [0.0673, 0.0824] | 97.990–97.990 |
| selected | 1 | 97.857 [97.704, 98.009] | -0.133 [-1.053, 0.786] | 0.0739 [0.0622, 0.0856] | 97.710–98.040 |
| selected | 2 | 97.933 [97.627, 98.239] | -0.057 [-0.880, 0.765] | 0.0741 [0.0651, 0.0830] | 97.710–98.103 |
| selected | 3 | 97.857 [97.751, 97.962] | -0.133 [-1.095, 0.828] | 0.0730 [0.0656, 0.0804] | 97.777–98.040 |
| selected | 4 | 98.043 [97.179, 98.908] | 0.053 [-0.197, 0.303] | 0.0748 [0.0645, 0.0851] | 98.043–98.043 |
| final | 0 | 98.440 [98.171, 98.709] | 0.000 [0.000, 0.000] | 0.0926 [0.0842, 0.1009] | 98.440–98.440 |
| final | 1 | 98.491 [98.422, 98.560] | 0.051 [-0.155, 0.256] | 0.1031 [0.0990, 0.1071] | 98.470–98.540 |
| final | 2 | 98.496 [98.412, 98.580] | 0.056 [-0.225, 0.337] | 0.1132 [0.1079, 0.1184] | 98.393–98.577 |
| final | 3 | 98.533 [98.323, 98.744] | 0.093 [-0.225, 0.411] | 0.1232 [0.1173, 0.1291] | 98.457–98.573 |
| final | 4 | 98.510 [98.420, 98.600] | 0.070 [-0.288, 0.428] | 0.1372 [0.1270, 0.1475] | 98.510–98.510 |

Each k averages subsets within each seed before forming an interval. The range describes the means of different subsets and is not a confidence interval.

| Endpoint | Eligibility mask (shallow→deep) | Accuracy % [CI] | CE [CI] | Harm % of all images [CI] | Repair % of all images [CI] | Training seconds [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| selected | 0000 | 97.990 [96.936, 99.044] | 0.0748 [0.0673, 0.0824] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 66.34 [65.74, 66.94] |
| selected | 1000 | 97.727 [97.363, 98.090] | 0.0745 [0.0645, 0.0844] | 0.873 [0.191, 1.556] | 0.610 [-0.105, 1.325] | 54.88 [54.55, 55.21] |
| selected | 0100 | 98.040 [97.013, 99.067] | 0.0732 [0.0640, 0.0823] | 0.490 [0.017, 0.963] | 0.540 [-0.058, 1.138] | 59.61 [59.12, 60.10] |
| selected | 1100 | 97.710 [97.277, 98.143] | 0.0742 [0.0590, 0.0895] | 0.927 [0.241, 1.612] | 0.647 [-0.168, 1.462] | 47.65 [47.36, 47.93] |
| selected | 0010 | 97.710 [97.194, 98.226] | 0.0736 [0.0534, 0.0938] | 0.810 [-0.180, 1.800] | 0.530 [-0.051, 1.111] | 61.96 [61.57, 62.36] |
| selected | 1010 | 98.103 [97.082, 99.124] | 0.0732 [0.0702, 0.0762] | 0.560 [-0.151, 1.271] | 0.673 [-0.262, 1.608] | 50.37 [50.15, 50.59] |
| selected | 0110 | 98.053 [97.259, 98.848] | 0.0756 [0.0698, 0.0814] | 0.607 [0.080, 1.134] | 0.670 [-0.239, 1.579] | 55.42 [54.96, 55.88] |
| selected | 1110 | 97.827 [97.244, 98.409] | 0.0721 [0.0561, 0.0881] | 0.947 [0.223, 1.671] | 0.783 [-0.127, 1.694] | 43.48 [43.31, 43.66] |
| selected | 0001 | 97.950 [97.519, 98.381] | 0.0743 [0.0652, 0.0834] | 0.617 [0.537, 0.697] | 0.577 [-0.055, 1.208] | 63.99 [63.59, 64.40] |
| selected | 1001 | 97.967 [97.571, 98.363] | 0.0719 [0.0581, 0.0857] | 0.687 [0.550, 0.823] | 0.663 [-0.111, 1.438] | 52.58 [52.57, 52.59] |
| selected | 0101 | 97.860 [96.989, 98.731] | 0.0747 [0.0570, 0.0924] | 0.857 [-0.031, 1.744] | 0.727 [-0.255, 1.708] | 57.48 [57.05, 57.91] |
| selected | 1101 | 97.777 [97.377, 98.176] | 0.0732 [0.0652, 0.0813] | 0.883 [0.307, 1.459] | 0.670 [-0.108, 1.448] | 45.42 [45.16, 45.68] |
| selected | 0011 | 97.903 [97.138, 98.669] | 0.0747 [0.0656, 0.0838] | 0.637 [0.491, 0.782] | 0.550 [0.078, 1.022] | 59.83 [59.39, 60.27] |
| selected | 1011 | 98.040 [97.354, 98.726] | 0.0704 [0.0661, 0.0747] | 0.630 [0.269, 0.991] | 0.680 [-0.058, 1.418] | 48.10 [48.04, 48.16] |
| selected | 0111 | 97.783 [97.148, 98.419] | 0.0763 [0.0676, 0.0850] | 0.803 [0.421, 1.185] | 0.597 [-0.047, 1.241] | 53.12 [52.80, 53.44] |
| selected | 1111 | 98.043 [97.179, 98.908] | 0.0748 [0.0645, 0.0851] | 0.697 [0.143, 1.250] | 0.750 [-0.011, 1.511] | 41.22 [41.04, 41.40] |
| final | 0000 | 98.440 [98.171, 98.709] | 0.0926 [0.0842, 0.1009] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 66.34 [65.74, 66.94] |
| final | 1000 | 98.540 [98.490, 98.590] | 0.0976 [0.0922, 0.1031] | 0.220 [0.170, 0.270] | 0.320 [0.083, 0.557] | 54.88 [54.55, 55.21] |
| final | 0100 | 98.483 [98.314, 98.652] | 0.1045 [0.0978, 0.1111] | 0.210 [0.144, 0.276] | 0.253 [0.065, 0.441] | 59.61 [59.12, 60.10] |
| final | 1100 | 98.530 [98.336, 98.724] | 0.1102 [0.1004, 0.1200] | 0.237 [0.112, 0.362] | 0.327 [0.167, 0.486] | 47.65 [47.36, 47.93] |
| final | 0010 | 98.470 [98.356, 98.584] | 0.1043 [0.1001, 0.1084] | 0.200 [0.150, 0.250] | 0.230 [0.051, 0.409] | 61.96 [61.57, 62.36] |
| final | 1010 | 98.577 [98.376, 98.777] | 0.1066 [0.0959, 0.1173] | 0.217 [0.116, 0.317] | 0.353 [-0.051, 0.758] | 50.37 [50.15, 50.59] |
| final | 0110 | 98.497 [98.417, 98.577] | 0.1150 [0.0994, 0.1307] | 0.233 [0.059, 0.408] | 0.290 [0.031, 0.549] | 55.42 [54.96, 55.88] |
| final | 1110 | 98.540 [98.366, 98.714] | 0.1167 [0.1082, 0.1252] | 0.297 [0.209, 0.384] | 0.397 [0.109, 0.685] | 43.48 [43.31, 43.66] |
| final | 0001 | 98.470 [98.380, 98.560] | 0.1059 [0.1017, 0.1100] | 0.177 [0.125, 0.228] | 0.207 [0.057, 0.356] | 63.99 [63.59, 64.40] |
| final | 1001 | 98.537 [98.371, 98.702] | 0.1100 [0.0992, 0.1207] | 0.237 [0.091, 0.382] | 0.333 [0.056, 0.610] | 52.58 [52.57, 52.59] |
| final | 0101 | 98.443 [98.229, 98.658] | 0.1188 [0.0985, 0.1391] | 0.237 [0.136, 0.337] | 0.240 [0.085, 0.395] | 57.48 [57.05, 57.91] |
| final | 1101 | 98.563 [98.247, 98.880] | 0.1198 [0.1005, 0.1392] | 0.260 [0.194, 0.326] | 0.383 [0.153, 0.614] | 45.42 [45.16, 45.68] |
| final | 0011 | 98.393 [98.257, 98.530] | 0.1184 [0.1093, 0.1274] | 0.273 [0.151, 0.396] | 0.227 [-0.022, 0.475] | 59.83 [59.39, 60.27] |
| final | 1011 | 98.573 [98.337, 98.809] | 0.1269 [0.1198, 0.1340] | 0.223 [0.049, 0.398] | 0.357 [0.091, 0.622] | 48.10 [48.04, 48.16] |
| final | 0111 | 98.457 [98.222, 98.691] | 0.1294 [0.1180, 0.1409] | 0.280 [0.166, 0.394] | 0.297 [-0.034, 0.627] | 53.12 [52.80, 53.44] |
| final | 1111 | 98.510 [98.420, 98.600] | 0.1372 [0.1270, 0.1475] | 0.343 [0.178, 0.509] | 0.413 [0.220, 0.606] | 41.22 [41.04, 41.40] |

Harm and repair compare each run with mask 0 under the same seed and endpoint. Accuracy change equals repair minus harm. Conditional harm and all ten class-specific metrics, all 32 directed subset edges per endpoint, the 24 path scores, and every seed value are preserved in analysis.json. Paired harm/repair flags for every official test image are retained in paired-errors.npz.

Training time refers to the complete 100 epochs even when a selected checkpoint is displayed. Source hardware is recorded per run; these values are not energy, job latency or invoice cost.

Gallery files separate the fixed 100 label-selected original images from outcome-selected first two harmed/repaired examples per digit. Empty categories remain empty. No new test result selects a preferred subset or changes the frozen order.

Limits: fixed learning rate 0.01, one architecture and reused MNIST test set; three new study seeds do not restore an untouched dataset. Any apparent best subset is conditional on this recipe and does not establish an optimally tuned or test-independent winner.


[W&B comparison dashboard](https://wandb.ai/yaroslavvb/gradient-dissent/reports/MNIST:-50%-training-dropout,-full-depth-testing-·-48-runs--VmlldzoxNzkwNDI4NQ==) · [Download the accuracy graph (SVG)](accuracy-vs-drop-count.svg) · [PNG](accuracy-vs-drop-count.png)


## Individual W&B runs

All 48 runs were uploaded after training and read back to verify their scientific histories. Plot against measured `epoch` or `training_seconds`; W&B importer runtime is not GPU training time.

| Droppable branches | Seed | Run |
|---|---:|---|
| none | 201 | [halfdrop-m00-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-d52b0247dc74fa61b8ce) |
| none | 202 | [halfdrop-m00-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-9b1d0b53abdbf4c24594) |
| none | 203 | [halfdrop-m00-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-d40ee1503e929b5f668e) |
| 1 | 201 | [halfdrop-m01-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-358bc482ffb70f2e9256) |
| 1 | 202 | [halfdrop-m01-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-dcb519e9276dcc349100) |
| 1 | 203 | [halfdrop-m01-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-107924b556b11c9a61c5) |
| 2 | 201 | [halfdrop-m02-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-d684a52e30329d49cbe8) |
| 2 | 202 | [halfdrop-m02-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-32eeb4ad30984d30df20) |
| 2 | 203 | [halfdrop-m02-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-6128e5b56814129942fa) |
| 1, 2 | 201 | [halfdrop-m03-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-0f1458d831d1f0683874) |
| 1, 2 | 202 | [halfdrop-m03-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-28f1a88354002ec60b07) |
| 1, 2 | 203 | [halfdrop-m03-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-8059bd4c95188ab069cd) |
| 3 | 201 | [halfdrop-m04-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-ff96e15f5dbc747c513a) |
| 3 | 202 | [halfdrop-m04-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-6fae195dcc124f7b7974) |
| 3 | 203 | [halfdrop-m04-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-1a98fa2fc4f580b4b97a) |
| 1, 3 | 201 | [halfdrop-m05-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-a6af7066e15541e4f816) |
| 1, 3 | 202 | [halfdrop-m05-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-19c5126d94aa807f2e4e) |
| 1, 3 | 203 | [halfdrop-m05-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-c0767028095a08d8d6af) |
| 2, 3 | 201 | [halfdrop-m06-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-bceb22a8d880cfd5da1e) |
| 2, 3 | 202 | [halfdrop-m06-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-f164d5f23fe4899da09a) |
| 2, 3 | 203 | [halfdrop-m06-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-020e3862af04144541ac) |
| 1, 2, 3 | 201 | [halfdrop-m07-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-32aabbe5f27fab5d8891) |
| 1, 2, 3 | 202 | [halfdrop-m07-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-efc703a2c2b8f4df2329) |
| 1, 2, 3 | 203 | [halfdrop-m07-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-33677e8dd7e7b7fefb8c) |
| 4 | 201 | [halfdrop-m08-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-4a5ba683e5bbfca7f76e) |
| 4 | 202 | [halfdrop-m08-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-b9ea81bc938977015503) |
| 4 | 203 | [halfdrop-m08-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-494a7bffbfe85c01206a) |
| 1, 4 | 201 | [halfdrop-m09-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-473188ddf0e111af7e19) |
| 1, 4 | 202 | [halfdrop-m09-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-f42926e9210004bf53d0) |
| 1, 4 | 203 | [halfdrop-m09-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-7df05042911c8cb49853) |
| 2, 4 | 201 | [halfdrop-m10-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-0f0554713f7e5a893e88) |
| 2, 4 | 202 | [halfdrop-m10-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-1e0f4d08cc258c4b86e1) |
| 2, 4 | 203 | [halfdrop-m10-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-1f40c8833989e4b068a7) |
| 1, 2, 4 | 201 | [halfdrop-m11-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-ac77ca41c430f63042b6) |
| 1, 2, 4 | 202 | [halfdrop-m11-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-90b5885abe56c63eb2f4) |
| 1, 2, 4 | 203 | [halfdrop-m11-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-f90b3a07df91b6dbcf83) |
| 3, 4 | 201 | [halfdrop-m12-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-a54a0709832c00e7177e) |
| 3, 4 | 202 | [halfdrop-m12-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-d43a91ec2d5c09b90ae1) |
| 3, 4 | 203 | [halfdrop-m12-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-71c445f8e8c69b75f34d) |
| 1, 3, 4 | 201 | [halfdrop-m13-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-13e04235d5a1226dfdae) |
| 1, 3, 4 | 202 | [halfdrop-m13-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-6feb92514bfe3141ab81) |
| 1, 3, 4 | 203 | [halfdrop-m13-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-fcb5150cede745d6cabe) |
| 2, 3, 4 | 201 | [halfdrop-m14-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-cd7a061e5e41abbf2d9f) |
| 2, 3, 4 | 202 | [halfdrop-m14-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-26706fbdd039dbc98161) |
| 2, 3, 4 | 203 | [halfdrop-m14-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-197dc282e80ad80160f8) |
| 1, 2, 3, 4 | 201 | [halfdrop-m15-s201-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-cd8240449cf4f6ab75ce) |
| 1, 2, 3, 4 | 202 | [halfdrop-m15-s202-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-0307b5edabecb6ff3019) |
| 1, 2, 3, 4 | 203 | [halfdrop-m15-s203-v1](https://wandb.ai/yaroslavvb/gradient-dissent/runs/halfdrop-2a594f6c406115d4322c) |


## Execution and reproducibility

The main cohort completed **48/48 runs**, with no failed or replaced seeds. Each ran 100 epochs on an **A100 40 GB**, using the existing TF32 CUDA-graph implementation, at most four workers in parallel. Qualification used one additional GPU call containing two two-epoch pilots; it did not access the official test set.

The interval from first main submission to last returned result was **840.94 seconds (14.02 minutes)**. Summed across overlapping calls, training consumed **2,584.34 seconds**, complete trainer clocks **3,058.25 seconds**, and dispatch-to-return clocks **3,267.83 seconds**. These sums are not elapsed study wall time. Detailed telemetry consumed **53.56 seconds summed across the cohort** (about 2.1% of summed training time); no W&B network calls ran in the training loop. The existing scalar training, validation, per-class, parameter, gradient and update diagnostics were retained. This cohort did not add expensive curvature measurements.

The environment meter rose from **$5.08899 to $7.16521**, an observed increment of **$2.07621 including qualification**. This is a provider meter difference, not a settled itemized cohort invoice; billing can lag. Opening and mid-study app snapshots found no other active apps. The conservative shared reservation ledger remains **$19.95334**, below the user’s **$30** shared cap, with no reservations released. Both owned Modal apps and all apps returned by the closing inventory are **stopped with zero tasks**.

The six GPU graph qualification tests passed. CPU trainer, launcher, analysis, upload and transport tests passed. Independent checks verified all **96 prediction archives**, all **48 authoritative server results**, paired initialization/data-order/gate-draw hashes, and **544 independently recomputed statistical summaries**. The JavaScript integration checks cover both endpoints, all 24 orders, all six lazy galleries and 2,304 previous-step comparison cases. The PNG figure was rendered and inspected.

All **48 W&B runs** were uploaded to `yaroslavvb/gradient-dissent` after training. Their histories, configurations, endpoint summaries and finished states were read back before publishing links. The eight-panel W&B report shows accuracy versus eligibility count, training time, validation loss and the original learning curves. Its axis/filter configuration was also read back.

The repository contains the frozen protocol, executed source hashes, exact run manifest, qualification records, validation-only order selection, scalar histories, paired prediction arrays, figures, transport checks and budget receipts. Weight files remain on the owner’s Modal volume with hashes and retrieval instructions in the [code README](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/halfdrop/README.md); they are not public checkpoint downloads. The report can be regenerated from the committed records without launching GPU jobs.


[Full analysis](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/halfdrop/analysis.json) · [Validation-order freeze](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/halfdrop/order-manifest.json) · [Protocol](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/halfdrop/PROTOCOL.md) · [Closing billing and shutdown receipt](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/halfdrop/budget-closing.json)
