# MNIST browser lab

[Open the runnable lab](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/browser-lab/) · [Controlled A100 results](../training-50/) · [Source and checks](https://github.com/yaroslavvb/gradient-dissent/tree/main/scripts/browser-lab)

This page **trains real networks on your device**. Its graphs start empty and fill with measured results. It uses TensorFlow.js 4.22.0 in a dedicated worker, trying WebGPU, then WebGL, then CPU. Each backend must successfully run the selected network's forward pass, gradients, and momentum updates before it is accepted. An explicitly selected backend does not fall back. A failure later in training is reported rather than silently changing backend or restarting the experiment.

WebGPU training remains experimental in this TensorFlow.js release. Qualification is a practical operation check, not a guarantee against later device loss or numerical differences. CPU fallback rejects the full 12-million-parameter preset; choose Tiny or Small instead. HTTPS or localhost, Web Workers, Web Crypto, and gzip `DecompressionStream` support are required.

## Use the experiment runner

1. Choose **Small Ciresan residual MLP** or **Tiny** for a quick start. Select the training branches and dropout probability (50% by default).
2. Choose a run plan. **Selected branches** trains one model. **Compare with baseline** trains a no-dropout model and your selection. **Add branches in order** trains five fresh models with zero through four droppable branches. All 24 branch orders are available.
3. Click **Run**. Watch real minibatch gates, validation accuracy, cross-entropy, and elapsed training time. **Stop** cancels pending dataset downloads or stops training/evaluation at a cooperative boundary. A running GPU kernel or backend initialization cannot be forcibly interrupted; closing/reloading the tab releases its worker.
4. For sweeps, select **Test accuracy / droppable branches** in the graph. The horizontal axis is how many branches are *eligible* for dropout, not how many were simultaneously absent. Every point evaluates its trained model with all branches present. Lines connect only nested masks with otherwise matched settings and unique branch counts.
5. Inspect held-out images, predicted probabilities, and per-digit accuracy. **Newly wrong** means correct under a matched no-dropout baseline and wrong under the chosen model; **repaired** means the reverse. Gray probability bars show the baseline. The comparison uses the same seed, architecture, data sizes, training budget, learning rate, batch size, dropout rate, and backend. It is descriptive; repeated seeds are needed for uncertainty estimates.
6. In **Try removing branches at inference**, uncheck branches and click **Evaluate this mask**. This uses the latest completed model's selected or final checkpoint. The image gallery and per-digit results change, while the main table and graph keep full-network test results. Selecting another run or checkpoint clears this preview. Earlier runs retain predictions but not model weights.
7. **Export results** downloads JSON containing settings, backend qualification, epoch history, timings, dataset/initialization hashes, test predictions, probabilities, confusion matrices, and the current inference preview. Weights are not exported. Results live in memory and disappear on reload. There is no login, W&B upload, server-side training, or cloud compute charge.

## Model and dropout semantics

These are **fully connected residual MLPs**, adapted from the Ciresan-width experiment; they are not convolutional ResNets. All have one input feature layer, four optional middle branches, and one classifier.

| Preset | Widths (input → hidden → output) | Parameters |
| --- | --- | ---: |
| Tiny | 784 → 128 → 96 → 64 → 48 → 32 → 10 | 124,090 |
| Small | 784 → 256 → 192 → 128 → 96 → 64 → 10 | 294,250 |
| Equal width | 784 → 128 → 128 → 128 → 128 → 128 → 10 | 167,818 |
| Wide | 784 → 512 → 384 → 256 → 192 → 128 → 10 | 772,810 |
| Full Ciresan widths | 784 → 2500 → 2000 → 1500 → 1000 → 500 → 10 | 11,972,510 |

The feature layer computes `ReLU(W₀x + b₀)`. A middle branch of output width `d` computes:

```
bypass = h[:, :d]
training: h_next = ReLU(bypass + gate / (1 − p) * (Wh + b))
testing:  h_next = ReLU(bypass + (Wh + b))
```

`gate` is independently sampled once per selected branch per minibatch, with keep probability `1 − p`. An unselected branch has `p = 0`. A skipped branch executes only `ReLU(bypass)`; it does not calculate its learned transform. Its weights **and momentum** are frozen for that step. Active parameters use ordinary momentum SGD: `v = 0.9v + gradient`, `weight -= learning_rate * v`. There is no augmentation, ordinary unit dropout, weight decay, or learning-rate schedule.

At 50% dropout, an active training branch contributes twice its learned output; at standard testing, every branch contributes once. This preserves the expected residual contribution before the nonlinearity, **not** the exact expected output or accuracy of the nonlinear network. The classifier remains linear before softmax cross-entropy.

Removing all four optional branches at inference still leaves the trained feature layer, crop bypasses, and classifier. This can classify digits and need not approach 10% accuracy. Even a constant classifier can obtain about 10% on balanced MNIST; removing computation does not imply zero accuracy. This lab does not present input/output deletion as stochastic depth.

## Dataset, selection, and timing

The default configuration uses 4,000 training images (options: 1,000 or 10,000), 2,000 validation images, and a balanced 2,000-image official test subset. All 10,000 official test images are optional. The default test subset weights the ten digits equally; the full test set retains natural class frequencies. [Dataset provenance, selection, binary format, and CC BY-SA 3.0 attribution](dataset/README.md) are separate from application code. Bundled default gzip data totals 2,296,866 bytes; JavaScript libraries download separately.

Training and validation subsets preserve the disjoint 50,000/10,000 parent split from the A100 experiment. Train/validation data are loaded before training; the training worker reads test images only after the fixed epoch budget and checkpoint selection. The initial preview displays only training images. Displayed test images must match the result's exact count and SHA256 before predictions are drawn.

The selected checkpoint is the **earliest minimum validation cross-entropy**, evaluated after each completed epoch. The final checkpoint is also reported. Stopped or failed runs have no test endpoints. Repeatedly exploring test outcomes can overfit the test set; this is an exploratory lab.

Every configuration in a plan uses fresh seed-matched weights, the same epoch shuffles, and the same underlying four gate draws per minibatch, even when a branch is ineligible. Separate random generators prevent branch selection from changing image order. Exports include a SHA256 of initialization and compact FNV-style digests of order/gate draws; the latter are pairing checks, not cryptographic attestations. GPU arithmetic is not guaranteed identical across devices/backends.

The training clock includes minibatch preparation, synchronization, worker yielding, and remaining shape-specific shader compilation. Validation and test evaluation have a separate clock. Complete-run time includes dataset loading, initialization, training, checkpoint operations, and evaluations. Backend qualification occurs before the run and is reported separately. These timings **do not reproduce A100 performance**.

## Reproduce and verify

From the repository root, with Node 22 or later and Python 3:

```sh
npm ci --ignore-scripts
npm run build:browser
npm run check:browser
python3 -m http.server 8770 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8770/ciresan-stochastic-depth/browser-lab/`.

The numerical test compares all 16 active masks, logits, cross-entropy, every present gradient, and five momentum steps against an independent PyTorch CPU/float32 reference. It also checks gain-one inference, seeded initialization, gate draws, repeated-update memory usage, and partial-allocation cleanup. Offline jsdom tests cover the interface, prediction/data identity, baseline comparisons, inference separation, Stop, and JSON exports. Dataset tests exercise decompression, hash checks, cancellation, and retry.

Optional real GPU checks on a Metal-capable Mac:

```sh
npm run check:browser:gpu
```

These execute the production engine and worker under **Node with Dawn Metal WebGPU**, including real MNIST learning, paired initialization/order, full-depth inference parity, masked inference, tensor cleanup, Stop, and restart. They are **not browser UI tests or browser speed measurements**. WebGL is qualified on the user's device at runtime; no claim of a separately completed WebGL training benchmark is made. Test receipts are [committed with the checks](https://github.com/yaroslavvb/gradient-dissent/tree/main/scripts/browser-lab); `manifest.json` lists deployed asset hashes and corresponding source/check hashes.

To regenerate the independent reference:

```sh
uv run python scripts/browser-lab/generate-reference.py
npm run check:browser
npm run check:browser:gpu
npm run build:browser
```

To rebuild data, supply the four original MNIST gzip IDX files (available from the [CVDF mirror](https://github.com/cvdfoundation/mnist)) and the existing experiment archive:

```sh
uv run python scripts/browser-lab/build_dataset.py \
  --repo . --cache /path/to/mnist-cache --output /tmp/browser-mnist
```

The producer verifies cached files and the parent experiment's archive hashes before selecting images. Copy the output's four `*.bin.gz` files, `manifest.json`, and `preview.json` into this page's `dataset/` directory, then run the checks and build. Raw `.bin` files need not be deployed.

## Source map and libraries

- `core.js`: deterministic gates/shuffles, experiment plans, metrics, and configuration matching.
- `tf-engine.js`: residual network, explicit stable-index momentum, snapshots, evaluation, and backend qualification.
- `dataset.js`: checksum-verified assets, decompression, caching, and cancellable downloads.
- `worker.js`: training orchestration and checkpoint/test protocol.
- `app.js`, `index.html`, `style.css`: controls, graphs, dataset gallery, and prediction comparisons.
- [Vendor provenance and licenses](vendor/README.md).
- [TensorFlow.js platform/backends](https://www.tensorflow.org/js/guide/platform_environment) and [pinned WebGPU implementation](https://github.com/tensorflow/tfjs/tree/tfjs-v4.22.0/tfjs-backend-webgpu).
