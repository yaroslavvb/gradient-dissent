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
