# Depth robustness beyond GPT: A100 transfer experiments

This extends the [local toy study](https://yaroslavvb.github.io/gradient-dissent/depth-robustness/) to three larger models, trained from scratch on Modal A100 GPUs. It tests the **depth-robustness mechanism** in [Don't Drop Dropout](https://arxiv.org/pdf/2609.05275v1), especially whether it transfers to image classifiers.

## Frozen design

| Family | Parameters | Data / input | Training per run | Pruning intervention |
|---|---:|---|---|---|
| nanoGPT-style GPT | 124,439,808 | WikiText103, GPT-2 tokenizer, context 1,024 | 3,200 steps × 32 sequences; 104,857,600 targets | Keep original blocks 1–8 of 12 |
| Vision transformer | 85,219,684 | CIFAR-100, native 32×32 images, 4×4 patches | 10,240 steps × 256 images; 58.25 equivalent passes over 45,000 training images | Keep original blocks 1–8 of 12 |
| ConvNeXt-Tiny | 27,897,028 | CIFAR-100, deterministic 32→64 bilinear resize after augmentation | 10,000 steps × 512 images; 113.78 equivalent passes | Retain 2,2,6,2 residual blocks from stages of 3,3,9,3 |

Image sampling uses replacement, so equivalent passes are not shuffled epochs. Upsampling adds no new image information. ConvNeXt's stem and downsampling transitions always remain. The vision transformer uses mean pooling of normalized patch tokens, without a class token. This is not the original paper's architecture, optimizer transfer, data mixture, or full LLM pretraining budget.

Each family compares dense training, constant increasing layer dropout (ILD, maximum 0.4), and decreasing ILD (initial maximum 0.8, ending at zero). Probability grows linearly with block index. Both dropout treatments omit 20% of example-block work in expectation. A shared per-sequence mask separately scales attention and MLP residual branches in transformers; ConvNeXt has one residual branch. Training computes branches before masking: no sparse execution or energy saving is claimed.

Every family/recipe receives three full-duration learning rates and one tuning seed (1000), selected only by final full-depth validation cross-entropy. Three distinct final seeds (2000–2002) pair initialization seeds and data streams, with independent augmentation/mask generators. GPT and ViT initializations match byte for byte. ConvNeXt has two worker-dependent roundoff variants, independently bounded at a maximum absolute parameter difference of 7.45×10⁻⁹; see the initialization audit below. There are 27 tuning and 27 final runs. The complete design, source hashes and runtime limits are in [protocol.json](protocol.json); the exact tuning calls are in [tuning-manifest.json](tuning-manifest.json). The final evaluation manifest is generated only after the complete tuning grid finishes. Boundary winners are disclosed; test results do not trigger grid extensions.

The primary endpoint is `CE(two-thirds retained) − CE(full)`, paired against dense training. Full-model and pruned CE/accuracy accompany this difference to expose quality tradeoffs. GPT/ViT use early exit; ConvNeXt uses stagewise thinning. Effect directions can be compared across families, but raw CE magnitudes from token and image prediction are not interchangeable. Nine fixed, published interventions are evaluated per final model, not every possible layer subset. No head retraining, fine-tuning, calibration, learned router, auxiliary exit loss, or test-selected mask is used.

## Budget and hardware

The authorization ceiling remains **$50**. The launcher reserves maximum execution time plus startup/idle allowances before submitting each batch. It uses explicit CPU/RAM limits, at most nine simultaneous GPU calls, no region premium or nonpreemptible premium, and no automatic user-code retries. A separate monitor cancels work if this dedicated environment's metered usage reaches $42. Billing snapshots can lag; the reservation ceiling is also $42, leaving an $8 buffer. Infrastructure restarts can occur independently of user-code retries; persistent run claims reject repeated work after a durable claim.

The main sweep requests Modal's `A100` option, which can supply a 40 GB A100 or a no-surcharge 80 GB upgrade. Published pricing checked on 9 September 2026: $0.000583/GPU-second, plus at most two CPU cores and 8 GiB RAM for a total resource rate of $0.00062696/second. Earlier explicit 80 GB pilots were reserved at their higher rate. See [Modal GPU behavior](https://modal.com/docs/guide/gpu), [pricing](https://modal.com/pricing), and [resource limits](https://modal.com/docs/guide/resources).

The initial GPT pilot used approximately **30.4 GiB allocated / 35.8 GiB reserved during training** at batch 32 and context 1,024. It completed on both 40 GB and 80 GB A100s. These memory readings describe this implementation and batch size; smaller batches or checkpointing can change hardware requirements. The vision models use less memory. Runtime measurements are planning inputs, not a controlled hardware speed comparison.

Modal may reuse containers between calls. Resetting peak-memory counters does not empty PyTorch's allocator cache, so reserved memory can include an earlier call or prevalidation pass. It is not a minimum-VRAM requirement. The observed GPT workload exceeds 24 GiB in **live allocated tensors**, independently of that cache distinction. Vision allocations include the resident raw image splits.

[results/budget-ledger.json](results/budget-ledger.json) contains conservative allocations, not invoices. [results/billing-latest.json](results/billing-latest.json) is a timestamped, potentially lagging metered-usage snapshot. Credits are not assumed when budgeting. Source/data preparation, failed result transfers and exploratory pilots remain part of the cost history.

## Reproducibility and execution

Use an authenticated Modal CLI 1.5.3. The remote image pins Python 3.11, PyTorch 2.8.0, NumPy 2.2.6, PyArrow 21.0.0 and tiktoken 0.11.0. The training code is fixed across tuning and final evaluation.

```sh
# Create this isolated environment once, if it does not already exist.
modal environment create gradient-dissent-a100

# Preparation uses CPU only; archives and token files stay in the named volume.
modal run -e gradient-dissent-a100 experiments/a100_transfer/modal_app.py --stage prep

# The launcher refuses duplicate run IDs or reservations above the ceiling.
modal run -e gradient-dissent-a100 experiments/a100_transfer/modal_app.py \
  --stage tune --manifest experiments/a100_transfer/tuning-manifest.json

# Wait for the complete tuning grid, select validation winners, run the three
# final seeds, then independently verify all raw results and compute intervals.
python3 experiments/a100_transfer/continue_after_tuning.py
```

The committed ledger records this experiment. Repeating an already executed sweep requires a separate, explicitly budgeted run namespace/output history; the supplied code intentionally refuses to overwrite claimed runs or silently repay for them. Do not discard existing spending records to bypass the cap. Keep all data splits, model settings, step counts, tuning rules and seeds when comparing a reproduction.

Local numerical checks can be repeated in a PyTorch environment:

```sh
experiments/.venv/bin/python experiments/a100_transfer/language.py --selfcheck
experiments/.venv/bin/python experiments/a100_transfer/vision.py
python3 experiments/a100_transfer/analyze.py --check-only
```

The last command requires every declared final run. It fails on missing results, mismatched data streams, unaudited initialization hashes, source or data mismatches, invalid masks, or a selected LR that is not the validation winner. The only exception to byte-identical initialization is the explicit, independently validated ConvNeXt hash allowlist described below. It uses Student-t intervals across training seeds (df=2 for three seeds), without treating masks or examples as independent trained-model replicates. Intervals do not include uncertainty from the data split, tuning seed, architecture, or multiple comparisons.

Raw checkpoints remain in the experiment's Modal volume; the public repository includes source, hashes, protocols, per-run measurements, and reproducible analysis. Raw datasets, token binaries and model weights are not Git blobs.

## Data and implementation provenance

- [language-notes.md](language-notes.md) documents the pinned nanoGPT reference, its MIT license, the GPT implementation and exact CPU parity checks. WikiText103 uses immutable official split files with verified SHA-256; GPT-2 tokens are stored as uint16. Windows remain inside articles. Held-out scoring uses fixed nonoverlapping windows and omits short articles/tails, so it is not the canonical full-corpus WikiText perplexity.
- [vision-notes.md](vision-notes.md) documents original model sources, ConvNeXt's MIT notice, augmentation and masks. The official CIFAR-100 training set is stratified into 45,000 train / 5,000 validation images; all 10,000 official test images remain held out. Normalization is a fixed affine mapping, not fitted on test data.
- Initial CIFAR transfer from the original server was slow. [parallel_cifar.py](parallel_cifar.py) retrieved byte ranges from a faster mirror and verified the **entire official archive checksum** before use. The original CPU download was stopped only after complete data and metadata were committed. URL/range provenance is in the preparation manifest.
- The initial GPU pilots trained and saved successfully, but their first client return failed because a PyTorch version-string subclass required PyTorch during unpickling. Their results were recovered directly from the committed volume; no training was repeated to recover them. Subsequent calls return plain JSON.

## Interpretation limits

A positive pruning result in image models would extend this recipe's observed usefulness to non-causal architectures. It would not establish a new universal property: stochastic depth and ConvNeXt's use of it already provide relevant precedent. Constant and decreasing schedules match average omission but differ in their temporal distributions and maximum probabilities. This comparison does not isolate only the final dense phase.

ConvNeXt initializes residual LayerScale at 10⁻⁶. Small pruning damage can therefore reflect weak residual use. The analysis reports learned LayerScale magnitudes alongside absolute classifier quality. Always-retained transitions also make ConvNeXt thinning a different intervention from deleting transformer suffixes.

The GPT model is much larger than the previous toy, but approximately 105M training tokens are still a short pretraining budget for 124M parameters. The vision experiments use one small-image dataset. Three final seeds and one tuning seed support a limited transfer study, not a claim of convergence, optimality, broad generalization or measured efficiency.

## Initialization audit adjustment

An independent audit found that identical ConvNeXt seeds produced two initialization hashes across worker CPUs. Full reconstruction reproduced both observed hashes for all four declared seeds, with equal post-initialization RNG states. Comparing every parameter found a worst absolute difference of 7.45×10⁻⁹; the first divergent primitive was CPU inverse-erf after identical uniform samples. Thus the study retains seed pairing with numerically close CNN initialization, not a claim of byte-identical weights or trajectories.

This is a **post hoc verification adjustment**, made using only untrained CPU states and no test outcomes. All executed training sources, seeds, schedules, learning rates, and pruning masks remain unchanged. Unknown hashes still fail the audit. See [initialization-audit.md](initialization-audit.md), the [full numerical evidence](results/initialization-numerical-audit.json), and the [exact diagnostic commands](diagnostics/README.md). No additional GPU calls were launched for the diagnosis; it ran within existing workers and their resource caps.
