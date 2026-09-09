# Historical Ciresan-style MNIST experiments

Read-only audit on 2026-09-09 of [yaroslavvb/train_ciresan](https://wandb.ai/yaroslavvb/train_ciresan) and the supplied [train_ciresan_new.py](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py). No training was launched, no credentials were needed, and no W&B state was modified. Public GraphQL queries returned the project data.

## What can serve as a historical baseline?

The most useful recent reference is **ts4k9n55**, a finished February 2024 run: **98.55% last logged official-test accuracy, cross-entropy 0.051922**, at logged epoch 96. W&B reports 394.55 seconds through the last history row; the final evaluation itself occurred at 392.67 seconds. Its best logged accuracy was 98.63% at epoch 94. The configuration records full MNIST, batch 64, SGD learning rate 0.001, momentum 0.9, per-step shrinkage 0.00002, no activation dropout, and curvature collection disabled. This is a useful scale and configuration reference, **not an exact-source reproduction guarantee or an independently selected test estimate**.

The largest peak in the ten selected complete histories is 98.75% in dz4nupwx, but this was selected from 4,574 repeated test evaluations during a diagnostic run ending at 487 logged epochs. Its last accuracy was 98.49%. Do not use that peak as a fixed-budget target, and do not describe it as the project-wide maximum: histories for all 327 runs were not scanned.

## Inventory and comparability

- Retrieved all 327 run summaries, matching the project run count. Runs span 2019-09-01 through 2024-02-08; 290 have numeric last test accuracy.
- States: 14 finished, 227 killed, 70 failed, 16 crashed. These labels do not by themselves distinguish intentional diagnostic stopping from implementation failure.
- Of 264 runs with scientific configs, all record learning rate 0.001, seed 1, bias enabled, ReLU enabled, and activation dropout disabled. Momentum is 0.9 in 261 and 0 in three. The other 63 runs have no retained scientific config.
- Full training-set size 60,000 is recorded in 215 runs; others use smaller subsets. For subsets below 10,000, the source also truncates the official test split to that size. Do not pool all runs into one comparable leaderboard.
- Curvature statistics are enabled in 251 of the 264 configured runs; disabled in 13. Most runtime differences consequently reflect substantially different instrumentation, evaluation frequency, stopping time, and possibly hardware, rather than training speed.
- These records are not an independent-seed study and contain no configured stochastic-depth treatment. A historical activation-dropout or stochastic-depth effect cannot be estimated from this inventory.

## Selected historical outcomes

Percent accuracy is already in 0–100 units in W&B. CE is mean cross-entropy per example. “Last” means last logged evaluation; it does not guarantee evaluation of the final updated parameters. Times below are W&B wall-clock runtime including evaluation, logging and any curvature diagnostics, not isolated GPU training time.

| Run / date / state | Configured stats | Last test accuracy / CE | Best logged test accuracy (epoch) | Last logged epoch | Run runtime |
|---|---|---|---|---|---|
| [ts4k9n55](https://wandb.ai/yaroslavvb/train_ciresan/runs/ts4k9n55) · 2024-02-07 · finished | off | 98.55% / 0.051922 | 98.63% (94) | 96 | 394.55 s (6.58 min) |
| [dkh42jiw](https://wandb.ai/yaroslavvb/train_ciresan/runs/dkh42jiw) · 2024-02-07 · killed | off | 98.44% / 0.054262 | 98.50% (70) | 78 | 324.02 s (5.40 min) |
| [s0og2713](https://wandb.ai/yaroslavvb/train_ciresan/runs/s0og2713) · 2019-10-18 · killed | on | 98.54% / 0.053760 | 98.67% (112) | 121 | 5602.10 s (93.37 min) |
| [dz4nupwx](https://wandb.ai/yaroslavvb/train_ciresan/runs/dz4nupwx) · 2019-10-11 · crashed | on | 98.49% / 0.056156 | 98.75% (137) | 487 | 88827.13 s (1480.45 min) |
| [0ovwv4x6](https://wandb.ai/yaroslavvb/train_ciresan/runs/0ovwv4x6) · 2020-01-16 · finished | on | 98.40% / 0.065561 | 98.46% (10) | 10 | 1725.93 s (28.77 min) |
| [wjpw622x](https://wandb.ai/yaroslavvb/train_ciresan/runs/wjpw622x) · 2019-09-21 · killed | on | 98.44% / 0.055080 | 98.44% (9) | 9 | 2844.21 s (47.40 min) |
| [425pu650](https://wandb.ai/yaroslavvb/train_ciresan/runs/425pu650) · 2019-10-10 · killed | on | 98.35% / 0.057221 | 98.48% (26) | 34 | 2517.72 s (41.96 min) |

All rows in this table use configured training size 60,000, training batch 64, learning rate 0.001 and momentum 0.9. All except 0ovwv4x6 record weight_decay=0.00002; 0ovwv4x6 records zero. This still does not establish identical executed code.

- ts4k9n55 first logs at least 98% at epoch 5, 23.83 seconds (98.05%, CE 0.066827). A modest reproduction can test learning behavior without matching its much longer final exposure.
- ts4k9n55 minimum logged CE equals its final CE; its maximum accuracy is at a different checkpoint. s0og2713 minimum logged CE is 0.051188 at epoch 49, while maximum accuracy occurs at epoch 112. Accuracy and CE checkpoint selection are distinct objectives.
- 425pu650 is explicitly linked from the supplied script near its curvature-statistics code. Its final test accuracy is 98.35%, peak 98.48%; that linkage is stronger contextual evidence than name similarity but still does not provide the complete executed source.
- The 2020 finished run 0ovwv4x6 has a declared finite stats_steps=100, train_steps=100 and no weight decay: last accuracy 98.40% at epoch 10. Its 28.8-minute runtime is dominated by a different diagnostics regime, so it is not a speed reference.

## Source behavior that must be explicit in a revival

The public source snapshot matches the supplied local reference byte for byte. The helper source below was independently fetched from public GitHub and its SHA-256 is saved in source-audit.json. These are claims about that source snapshot, not a blanket claim that every historical run executed it.

1. **Architecture and initialization.** The script constructs dimensions `[784, 2500, 2000, 1500, 1000, 500, 10]`: six affine layers, five hidden layers, 11,972,510 trainable weights and biases. There are no residual connections or normalization layers. `SimpleFullyConnected2` uses ordinary PyTorch `nn.Linear` initialization. A straightforward stochastic-depth identity bypass is dimensionally impossible between its unequal hidden widths; any projection, cropping, padding or residual redesign is a substantive experimental intervention and needs its own matched dense control. [Script lines 124–128](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py#L124), [helper](https://github.com/yaroslavvb/stuff/blob/master/autotune/util.py#L1651).

2. **Output activation.** `SimpleFullyConnected2(..., last_layer_linear=False)` appends ReLU after the output affine layer when `nonlin=1`; the script does not override that default. Cross-entropy therefore receives nonnegative rectified logits. Replacing this with linear logits is a sensible separately identified correction, not a faithful reproduction detail. It also changes which output preactivations receive gradients. [Helper lines 1654–1677](https://github.com/yaroslavvb/stuff/blob/master/autotune/util.py#L1654).

3. **Input scale and augmentation.** For original 28×28 MNIST, TinyMNIST casts raw pixels to float without dividing by 255, and returns tensors directly; there is no random crop, affine distortion, elastic augmentation, standardization, or transform callback. Its alternate resized-image path does divide by 255, so changing width changes both geometry and scale. Merely switching to torchvision ToTensor changes the original training problem. This script should be called a Ciresan-style MLP, not a reproduction of every augmentation and training detail of a Ciresan paper. [Helper lines 1386–1438](https://github.com/yaroslavvb/stuff/blob/master/autotune/util.py#L1386).

4. **Shrinkage is not standard weight decay.** The SGD optimizer is constructed without a weight_decay argument. After each optimizer update, all parameters, including biases, are multiplied by `1-weight_decay`. With lr=0.001 and weight_decay=0.00002, the plain multiplicative shrinkage magnitude corresponds to an lr-scaled coefficient of 0.02, but is still not equivalent to coupled SGD L2 with momentum or to all details of AdamW. Preserve this operation or declare its replacement; do not pass 0.00002 directly as SGD weight_decay and call it identical. [Script lines 133 and 610–613](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py#L133).

5. **Budget, seed and device flags.** `seed_random(1)` is called before and after parsing; the CLI seed has no effect. `epochs` is parsed but training is controlled by stats_steps×train_steps. The script uses train_batch_size, not batch_size; evaluation uses stats_batch_size, not test_batch_size. `no_cuda` does not govern the helper global device. Several other parsed flags such as swa and save_model are not implemented in this main loop. The default one million outer loops is unsuitable for a finite benchmark. [Script argument and main-loop section](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py#L61).

6. **Test selection and stopping.** The source names official-test metrics `val_accuracy`/`val_loss`; it has no separate model-selection validation split. Evaluation precedes every training chunk, including the first, and there is no final post-loop evaluation or checkpoint save. Thus the last reported evaluation can lag completed training. Epoch is integer token_count//60000 regardless of configured subset size. W&B `_step` is not consistently an optimizer step across old and new logging integrations. [Script lines 148–180 and 596–617](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py#L148).

7. **Randomness and diagnostic overhead.** A shuffled statistics DataLoader draws from the same global randomness machinery as shuffled training; collecting or omitting diagnostics can change the subsequent data order. The source evaluates full training and test sets repeatedly and may compute expensive Hessian/Jacobian spectra. Removing diagnostics is appropriate for a training study, but logged speed from the original project cannot be compared without accounting for that change. For paired experiments, explicitly separate the data-order and stochastic-depth RNG streams.

8. **Diagnostic bugs are distinct from training behavior.** In the curvature block, fish/jac eigenvalue histogram calls reuse Hessian eigenvalues; some consumed fields (e.g. lyap_hess_sum/lyap_jac_sum and curv_ratio) have no corresponding writes in this file, and permissive default dictionaries can silently supply zeros. These affect diagnostic interpretations rather than the ordinary SGD update. A revival should not preserve these statistics as trustworthy evidence without an independent audit. [Script lines 416–421 and 515–518](https://github.com/yaroslavvb/stuff/blob/master/autotune/train_ciresan_new.py#L416).

## Provenance and limitations

The 2024 reference run records commit `c59482dc38dc7f0e57ba143fde52dc9611de05a1`. That identifier did not resolve through the public yaroslavvb/stuff commit API during this audit (HTTP 422). The configuration includes `uniform` and `disable_hess` flags absent from the supplied `train_ciresan_new.py`, so the exact executed variant is unverified. Selected old runs record no commit. Public run files did not expose wandb-metadata.json, and the available output logs did not identify a train_ciresan program basename. Hardware, dirty source modifications, and exact executed architecture cannot be recovered from the selected records alone.

Selected requirements files are useful environmental evidence: ts4k9n55 records torch 2.1.0+cu121, torchvision 0.16.0+cu121, NumPy 1.23.5 and W&B 0.16.3; the 2019 examples record torch 1.1.0 / torchvision 0.2.2, and 0ovwv4x6 torch 1.3.1 / torchvision 0.4.2. These package declarations do not identify a GPU model or prove an exact environment recreation.

## Saved evidence

- `run-metadata.json`: whitelisted scientific config and final scalar summary for all 327 runs; exact retrieval timestamp and project count. No credentials, machine paths, hostnames, email, full log files or environment dumps.
- `selected-histories.json`: ten selected runs, whole-evaluation-history retrieval count checks, first/last evaluations, extrema, and first evaluation at least 98%; full raw histories deliberately omitted. For each, returned count matched W&B number-valued val_accuracy count. This supports extrema over logged evaluations, not over unlogged checkpoints.
- `execution-metadata.json`: selected package versions and explicit gaps in execution provenance.
- `source-audit.json`: public source URLs, content hashes and line counts.

For the new study, freeze the training-only validation split, finite update count, seed set, data order, input scale, output activation, shrinkage rule, bypass mapping, probability schedule and full/pruned test endpoints before evaluating treatments. Use the historical results to check plausibility, not to select a favorable reported test checkpoint.
