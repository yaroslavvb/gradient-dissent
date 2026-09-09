# A100 depth robustness transfer experiment

Verified **27 completed final runs** against the locked evaluation manifest. All reported statistics below derive from the saved raw results.

The primary endpoint is treatment minus dense in **CE after the predefined two-thirds-depth intervention minus the same model's full-depth CE**. Negative values indicate less loss increase after pruning. Confidence intervals are paired 95% Student-t intervals across training seeds, unadjusted for multiple comparisons.

| Family | Treatment | Seeds | Primary effect (CE) | Paired 95% interval | Individual paired effects |
|---|---|---:|---:|---|---|
| ConvNeXt / CIFAR-100 | constant_ild | 3 | -4.79470 | [-10.70497, 1.11557] | -6.04091, -6.29193, -2.05126 |
| ConvNeXt / CIFAR-100 | decreasing_ild | 3 | -4.81480 | [-10.91289, 1.28330] | -6.22505, -6.23911, -1.98023 |
| GPT / WikiText-103 | constant_ild | 3 | -4.10619 | [-5.38954, -2.82284] | -4.66890, -3.99632, -3.65334 |
| GPT / WikiText-103 | decreasing_ild | 3 | -4.08296 | [-5.36433, -2.80160] | -4.64541, -3.97147, -3.63201 |
| ViT / CIFAR-100 | constant_ild | 3 | 0.27775 | [0.22595, 0.32955] | 0.28048, 0.25567, 0.29710 |
| ViT / CIFAR-100 | decreasing_ild | 3 | 0.48334 | [0.38873, 0.57796] | 0.52285, 0.44686, 0.48033 |

## Full quality and the retained model

CE is nats per target token for language and nats per image for vision. Accuracy is token top-1 for language and class top-1 for vision. Their raw scales should not be averaged across families.

| Family | Treatment | Full CE | Primary CE | CE increase | Full accuracy | Primary accuracy | Accuracy change |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt / CIFAR-100 | dense | 3.45720 | 8.15574 | 4.69854 | 51.34% | 14.19% | -37.147 pp |
| ConvNeXt / CIFAR-100 | constant_ild | 2.74450 | 2.64834 | -0.09616 | 54.62% | 51.01% | -3.607 pp |
| ConvNeXt / CIFAR-100 | decreasing_ild | 2.89137 | 2.77511 | -0.11626 | 54.12% | 50.62% | -3.497 pp |
| GPT / WikiText-103 | dense | 3.51178 | 7.71080 | 4.19902 | 38.48% | 15.38% | -23.102 pp |
| GPT / WikiText-103 | constant_ild | 3.64521 | 3.73804 | 0.09283 | 37.01% | 36.31% | -0.704 pp |
| GPT / WikiText-103 | decreasing_ild | 3.73641 | 3.85247 | 0.11606 | 36.06% | 35.22% | -0.846 pp |
| ViT / CIFAR-100 | dense | 3.07945 | 2.24360 | -0.83585 | 47.04% | 46.68% | -0.360 pp |
| ViT / CIFAR-100 | constant_ild | 2.71323 | 2.15513 | -0.55810 | 49.55% | 50.57% | +1.020 pp |
| ViT / CIFAR-100 | decreasing_ild | 2.56135 | 2.20885 | -0.35250 | 48.40% | 48.77% | +0.370 pp |

Full and primary CE/accuracy intervals, secondary mask curves, and all seed values are in [summary.json](summary.json). Paired differences in raw quality and accuracy are in [paired.csv](paired.csv). A narrow pruning-damage difference alone does not establish a full-quality improvement or equivalence.

## Training exposure, memory and learning

| Family | Parameters | Steps × batch | Exposure | Input | Primary retained blocks |
|---|---:|---|---|---|---|
| ConvNeXt / CIFAR-100 | 27,897,028 | 10000 × 512 | 5,120,000 image presentations; 113.78 equivalent passes | 64×64 | 12/18 |
| GPT / WikiText-103 | 124,439,808 | 3200 × 32 | 104,857,600 tokens; 0.843 tokens/parameter | context 1024 | 8/12 |
| ViT / CIFAR-100 | 85,219,684 | 10240 × 256 | 2,621,440 image presentations; 58.25 equivalent passes | 32×32 | 8/12 |

| Family | Recipe | Initial validation CE | Final validation CE | Last minibatch CE | Peak allocated GiB | Peak reserved GiB | Mean train seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt / CIFAR-100 | dense | 4.7516 | 3.3976 | 0.0016 | 3.51 | 4.00 | 432.4 |
| ConvNeXt / CIFAR-100 | constant_ild | 4.7516 | 2.6791 | 0.0829 | 3.51 | 4.00 | 477.2 |
| ConvNeXt / CIFAR-100 | decreasing_ild | 4.7516 | 2.8585 | 0.0053 | 3.51 | 4.00 | 443.9 |
| GPT / WikiText-103 | dense | 10.9857 | 3.5305 | 3.5672 | 30.35 | 35.84 | 941.6 |
| GPT / WikiText-103 | constant_ild | 10.9857 | 3.6656 | 3.7609 | 30.35 | 35.84 | 939.9 |
| GPT / WikiText-103 | decreasing_ild | 10.9857 | 3.7557 | 3.8030 | 30.36 | 35.84 | 943.2 |
| ViT / CIFAR-100 | dense | 4.7318 | 3.0664 | 0.0351 | 6.66 | 6.96 | 960.4 |
| ViT / CIFAR-100 | constant_ild | 4.7318 | 2.6745 | 0.1567 | 6.66 | 6.96 | 976.0 |
| ViT / CIFAR-100 | decreasing_ild | 4.7318 | 2.5329 | 0.2497 | 6.66 | 16.59 | 976.1 |

Image sampling is with replacement; equivalent passes do not mean shuffled epochs. Memory and training time are provenance measurements for these runs, not a controlled efficiency comparison. Reserved memory can include allocator cache inherited from earlier calls or prevalidation; it is not a minimum VRAM requirement. Last-minibatch loss is noisy and is not full-training-set loss. Compute-then-mask evaluates all training branches. Every model keeps its original final normalization/head, and inference applies no inverse-survival scaling or classifier adaptation.

## Learning-rate selection

| Family | Recipe | Tuning seeds | Grid: LR → CE / accuracy | Selected LR | Grid boundary? | Full training horizon? |
|---|---|---|---|---:|---|---|
| ConvNeXt / CIFAR-100 | dense | 1000 | 0.0003 → 3.3723 / 53.22%; 0.001 → 3.5934 / 53.08%; 0.003 → 3.9584 / 51.12% | 0.0003 | yes | yes |
| ConvNeXt / CIFAR-100 | constant_ild | 1000 | 0.0003 → 2.6818 / 56.54%; 0.001 → 2.9240 / 57.82%; 0.003 → 2.9580 / 58.96% | 0.0003 | yes | yes |
| ConvNeXt / CIFAR-100 | decreasing_ild | 1000 | 0.0003 → 2.8196 / 56.64%; 0.001 → 3.0644 / 57.40%; 0.003 → 3.1933 / 59.02% | 0.0003 | yes | yes |
| GPT / WikiText-103 | dense | 1000 | 0.0003 → 3.7522 / 36.26%; 0.0006 → 3.5988 / 37.55%; 0.0012 → 3.5144 / 38.33% | 0.0012 | yes | yes |
| GPT / WikiText-103 | constant_ild | 1000 | 0.0003 → 3.9715 / 33.67%; 0.0006 → 3.7138 / 36.19%; 0.0012 → 3.6799 / 36.37% | 0.0012 | yes | yes |
| GPT / WikiText-103 | decreasing_ild | 1000 | 0.0003 → 4.0365 / 33.09%; 0.0006 → 3.7531 / 35.90%; 0.0012 → 3.7285 / 35.98% | 0.0012 | yes | yes |
| ViT / CIFAR-100 | dense | 1000 | 0.0001 → 3.0123 / 47.60%; 0.0003 → 3.1417 / 48.18%; 0.001 → 3.3655 / 18.80% | 0.0001 | yes | yes |
| ViT / CIFAR-100 | constant_ild | 1000 | 0.0001 → 2.6682 / 48.86%; 0.0003 → 2.6195 / 50.36%; 0.001 → 3.3467 / 19.30% | 0.0003 | no | yes |
| ViT / CIFAR-100 | decreasing_ild | 1000 | 0.0001 → 2.5175 / 48.58%; 0.0003 → 2.8148 / 48.68%; 0.001 → 3.2263 / 21.84% | 0.0001 | yes | yes |

Learning rates minimize terminal full-depth validation CE with equal grids per recipe. Accuracy is reported alongside CE to expose disagreements between the objectives; it does not affect selection. Selection uses the recorded tuning seeds, separate from final seeds. A one-seed search and boundary winners add uncertainty not represented by the final seed intervals.

## ConvNeXt LayerScale diagnostic

Small residual scales can create trivial pruning robustness in an undertrained network. The table reports means of absolute gamma within each stage, then across seeds; maxima are the per-run stage maxima averaged across seeds. This diagnostic complements full-model learning and does not by itself establish useful learned residual computation.

| Recipe | Stage (zero-based) | Initial mean absolute gamma | Final mean absolute gamma | Mean of per-seed stage maxima |
|---|---:|---:|---:|---:|
| dense | 0 | 1e-06 | 0.0844834 | 0.260047 |
| dense | 1 | 1e-06 | 0.0402407 | 0.166386 |
| dense | 2 | 1e-06 | 0.0847616 | 0.265381 |
| dense | 3 | 1e-06 | 0.143664 | 0.281691 |
| constant_ild | 0 | 1e-06 | 0.0973317 | 0.298273 |
| constant_ild | 1 | 1e-06 | 0.0775562 | 0.265521 |
| constant_ild | 2 | 1e-06 | 0.120371 | 0.237378 |
| constant_ild | 3 | 1e-06 | 0.158996 | 0.25658 |
| decreasing_ild | 0 | 1e-06 | 0.104354 | 0.304274 |
| decreasing_ild | 1 | 1e-06 | 0.09491 | 0.275536 |
| decreasing_ild | 2 | 1e-06 | 0.126396 | 0.240835 |
| decreasing_ild | 3 | 1e-06 | 0.146579 | 0.217821 |

## Budget and audit

The authorization cap is **$50.00**. The ledger contains 60 invocation reservations totaling **$39.6661**. Reservations are conservative allocations, **not invoices or measured spend**.

The latest saved Modal metered-usage snapshot is **$27.8092**, queried 2026-09-09T20:31:56.025095+00:00. It may lag and is not a final invoice.

[Budget ledger](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/budget-ledger.json) · [Locked evaluation manifest](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/evaluation-manifest.json) · [Analysis source](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/analyze.py)

Verification requires every locked final run, bit-identical GPT initialization, separate exact allowlists and numerical audits for ConvNeXt and ViT initialization variants, identical paired data streams, executed data/source hashes matching the saved artifacts, completed steps, unique fixed mask panels, identical target counts, and passed GPU full-mask/default equivalence. Raw-file SHA-256 values are included in summary.json. Failed attempts and earlier pilots remain in the result/ledger audit trail and are not statistical replicates.

## Initialization audit qualification

A separate **post hoc verification adjustment** for **ConvNeXt**, based on untrained CPU initializations and independent of test outcomes, permits two exact recorded hashes for each of seeds 1000, 2000, 2001, 2002. Every tensor was compared across the two variants: maximum absolute difference **7.450580597e-09**, maximum relative L2 difference **5.362363754e-09**, with identical post-initialization RNG states. The analyzer recomputes these bounds from per-tensor statistics and checks pinned evidence SHA-256 values, the frozen source, PyTorch version, shapes, dtypes, and the full parameter count. Unknown hashes fail verification.

[initialization-numerical-audit.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/initialization-numerical-audit.json) · [initialization-variant-manifests.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/initialization-variant-manifests.json) · [initialization-worker-probes.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/initialization-worker-probes.json)

A separate **post hoc verification adjustment** for **ViT**, based on untrained CPU initializations and independent of test outcomes, permits two exact recorded hashes for each of seeds 1000, 2000, 2001, 2002. Every tensor was compared across the two variants: maximum absolute difference **7.450580597e-09**, maximum relative L2 difference **5.635263729e-09**, with identical post-initialization RNG states. The analyzer recomputes these bounds from per-tensor statistics and checks pinned evidence SHA-256 values, the frozen source, PyTorch version, shapes, dtypes, and the full parameter count. Unknown hashes fail verification.

[vit-initialization-numerical-audit.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/vit-initialization-numerical-audit.json) · [vit-initialization-variant-manifests.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/vit-initialization-variant-manifests.json) · [vit-initialization-cpu-probes.json](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/results/vit-initialization-cpu-probes.json)

ConvNeXt and ViT are seed-paired with separately audited, numerically close initializations; weights are not bit-identical across all runs. **GPT retains bit-identical paired initialization.** These audits do not establish identical subsequent training trajectories or quantify downstream host effects. The LR grid, CE-only selection, evaluation seeds, training code and primary endpoint were unchanged. ViT numerical tolerances were fixed before its full tensor comparison.

## Limits of the result

- Few final training seeds; intervals are fragile and unadjusted across primary/secondary comparisons.
- Learning-rate selection uncertainty and heldout dataset uncertainty are not included in seed intervals; tuning uses its recorded independent seed count.
- ConvNeXt and ViT same-seed initial weights can differ by CPU host. Separate post hoc audits accept only two measured, numerically close hashes per architecture/seed with matching post-init RNG; identical training trajectories are not established. GPT initialization remains bit-identically paired.
- Compute-then-mask executes dense branches; these runs do not establish FLOP, latency, memory, energy, or dollar savings.
- Peak reserved GPU memory can include allocator cache from previous calls or prevalidation; it is not the model's required VRAM. Peak allocated memory describes live tensors in this implementation, including resident data.
- Two-thirds retained residual blocks does not imply two-thirds FLOPs; ConvNeXt stage transitions always remain.
- Fixed image resizing adds no observed information; CIFAR spatial geometry differs from ImageNet.
- Small ConvNeXt LayerScale and undertraining can make deletion appear harmless; inspect full learning and recorded gamma magnitudes.
- Language uses its recorded fixed document-contained context windows, not canonical full-corpus perplexity.
- This is training from scratch under a short declared compute budget, not a reproduction of the target paper's LLM training scale.

[Vision implementation/provenance](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/vision-notes.md) · [Language implementation/provenance](https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer/language-notes.md)
