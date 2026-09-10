## Six-layer inference follow-up: the learned input representation matters

The extended explorer adds two switches: the learned feature layer and the classifier. It enumerates **64 settings on the same 30 frozen checkpoint states**, using all 10,000 official test images. No network was trained or checkpoint selected again. Every original four-branch mask is reproduced exactly. This remains exploratory analysis of an already reused test set.

Six mask bits read **stem → branch 1 → branch 2 → branch 3 → branch 4 → classifier**. `111111` keeps everything; `100001` keeps only learned features and the classifier; `000001` keeps only the classifier; `000000` abstains. These are displayed bit strings, not conventional most-significant-bit-first binary notation.

Dropping a layer must specify what replaces it. Removing the stem inserts normalized pixels followed by zeros to reach 2,500 coordinates. Removing a middle branch retains the first coordinates needed by the next layer, followed by ReLU. Removing the classifier emits no digit and exits before any affine computation. Only middle branches were eligible for stochastic depth during training: endpoint deletion tests an untrained intervention.

### Measured classification outcomes

Each entry below is a mean over three training seeds. All four columns retain the classifier, emit a prediction for every image, and therefore have 100% coverage. “Stem removed” retains all four body branches. “Classifier only” passes padded/cropped pixels directly to the saved classifier; it is not a newly trained linear MNIST model.

| Training method | Checkpoint | All 6 layers | Stem removed | All 4 body branches removed | Classifier only |
|---|---|---:|---:|---:|---:|
| Residual, no dropout | Selected | 98.373% | 12.373% | 93.507% | 14.583% |
| SD, fixed drop rates | Selected | 97.803% | 12.163% | 95.370% | 14.470% |
| SD, drop rates decay to zero | Selected | 97.997% | 12.277% | 96.323% | 13.903% |
| Residual + unit dropout | Selected | 98.330% | 11.137% | 91.173% | 15.387% |
| Plain MLP | Selected | 97.913% | 10.937% | 12.373% | 16.020% |
| Residual, no dropout | Epoch 100 | 98.400% | 11.970% | 93.163% | 14.553% |
| SD, fixed drop rates | Epoch 100 | 98.443% | 11.743% | 97.310% | 14.600% |
| SD, drop rates decay to zero | Epoch 100 | 98.510% | 11.103% | 97.460% | 13.680% |
| Residual + unit dropout | Epoch 100 | 98.507% | 11.293% | 89.007% | 16.000% |
| Plain MLP | Epoch 100 | 98.573% | 10.007% | 10.983% | 16.553% |

**Tolerance to missing middle computation does not imply tolerance to missing learned input features.** In the selected SD checkpoints, dropping all four middle branches retains 95–96% accuracy, while removing the stem alone leaves about 12%. The zero-padded replacement changes the representation drastically. This establishes dependence on the learned stem under this replacement, not a theorem that every possible alternative stem must fail.

“Classifier only” spans **9.48–20.97%** across the 30 individual checkpoint states. It still processes image-dependent coordinates with learned classifier weights and biases; it is not uniform random guessing. Removing more layers need not lower accuracy monotonically: for some checkpoints, the classifier-only intervention is more accurate than keeping all four branches after removing the stem. Neither intervention preserves useful MNIST quality.

### What zero means when all layers are dropped

With no classifier there are no ten class scores to compare. The policy **abstains**. We report three distinct quantities:

| Metric | Definition | All layers dropped |
|---|---|---:|
| Correct-output rate | Correct predictions / all test inputs | 0% |
| Prediction coverage | Inputs receiving any digit prediction / all inputs | 0% |
| Classification accuracy | Correct predictions / emitted predictions | Undefined (0/0) |

All 32 settings with the classifier absent implement this same abstention policy. They add no classification measurements and should not be read as 32 independent failures. Cross-entropy is also undefined. A hypothetical network forced to guess a digit after removing all computation would instead have a guess-dependent accuracy; uniform random guessing averages 10%. Forcing the displayed classification accuracy to zero would misdescribe that experiment.

### Random dropout during inference

Choose the layers eligible for dropout and set a shared probability `p`. Each eligible layer is dropped independently for each input. Ineligible layers stay present. A mask with `d` dropped and `k` kept eligible layers has probability `p^d (1−p)^k`; masks dropping an ineligible layer have probability zero. The page sums the measured outcomes of all masks with these weights. Its plots are **exact expected single-draw scores**, not a Monte Carlo estimate and not an ensemble formed by averaging logits or probabilities. “Draw one random mask” shows one example outcome; changing that draw does not change the aggregate scores.

If the classifier is eligible, coverage is exactly `1−p`. Correct-output rate is coverage times accuracy conditional on a prediction. At `p=1`, coverage and correct outputs are zero and conditional accuracy is undefined. If only the four middle branches are eligible, coverage remains 100% even at `p=1`: the trained shallow predictor remains. Checking no eligible layers makes the policy identical to full inference at every slider position.

Surviving branches use gain one. We did not evaluate inverse-survival rescaling at inference. The computed curve is therefore a deletion policy on saved weights, not a replay of the scaled stochastic-depth training network. Expected MAC counts are arithmetic counts only: this follow-up did not measure the overhead or latency of a per-image random-mask implementation.

### Verification and reproducibility

The three successful A100 jobs evaluate 32 classifier-present masks per checkpoint. Every full-mask forward matches the saved model's ordinary forward bitwise. The 16 embedded original masks reproduce all **4.8 million** archived test predictions exactly; labels, dense error indices, checkpoint hashes and dataset hashes also agree. Mean cross-entropy agreement uses the original absolute tolerance of 0.00003. The 1,408 previously selected gallery entries remain unchanged; no new examples were selected based on this follow-up.

The extended JSON contains 1,920 mask records, including the explicit abstention settings, plus per-digit metrics and frozen-gallery predictions. NPZ sidecars preserve all per-image predictions and losses; abstention uses prediction −1 and undefined loss. These masks and images do not increase the number of independent training seeds.

The three successful jobs ran in parallel on **A100-SXM4-40GB** GPUs and took approximately **22–34 seconds each** inside the evaluator. An earlier wrapper-import failure was fixed before the evaluator ran, and both attempts' reservations and records are retained. The closing provider meter increased by **$0.08524** for this follow-up, to **$5.08899** in the shared MNIST environment. Both apps were stopped with zero tasks. The conservative shared reservation bound, including the failed attempt, is **$10.42982**, below the original $30 cap and the $24 stop guard. Metering is a provider snapshot, not a final invoice. [Closing cost and shutdown receipt](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/full-depth/budget-closing.json).

[Six-layer protocol](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/layerdrop_hypotheses/FULL_DEPTH_PROTOCOL.md) · [Evaluator](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/layerdrop_hypotheses/full_depth.py) · [Frozen specifications and raw results](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/ciresan_stochastic_depth/results/full-depth) · [Extended explorer data](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/hypotheses/extended-data.json)
