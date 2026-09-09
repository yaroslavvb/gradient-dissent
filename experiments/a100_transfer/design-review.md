# Independent design review: A100 transfer experiments

Prepared 2026-09-09. This is a design recommendation, not an executed-results manifest. Final numerical budgets, step counts and hyperparameters must come from the frozen execution protocol. No paid jobs were launched in preparing this review.

## What would constitute useful transfer evidence

Use three architecture families if measured throughput permits:

| Family | Proposed task and architecture | Primary pruning intervention |
|---|---|---|
| Causal transformer | WikiText-103 raw, GPT-2 tokenizer, nanoGPT-style 12 layers × width 768 × 12 heads, about 124M parameters | Keep original blocks 1–8 of 12; unchanged final normalization and language-model head |
| Vision transformer | CIFAR-100 at native 32×32, patch size 4, 12 layers × width 768 × 12 heads, about 86M parameters | Keep original blocks 1–8 of 12; unchanged class-token normalization and classifier |
| Convolutional residual network | CIFAR-100, ConvNeXt-Tiny, stages with 3/3/9/3 residual blocks, about 28M parameters | Keep stagewise prefixes of 2/2/6/2 residual blocks; retain every stem/downsampling transition and final classifier |

These are three forms of residual computation, not three independently sampled model populations. The two image experiments share their dataset and split. A positive ConvNeXt result extends the measured effect beyond attention, while a ViT result separates causal language modeling from bidirectional visual attention. Stochastic depth already exists in vision architectures; this is a recipe/robustness transfer check, not a claim that dropping vision blocks is new.

For every family, compare dense, constant ILD with maximum probability .4, and time-decreasing ILD with initial maximum .8. The spatial index runs across all residual blocks in original order. Both ILD treatments have .2 expected mean omission with the exact discrete temporal endpoints. Disable all other native dropout/DropPath paths unless explicitly included and held identical across treatments.

Train from scratch. Do not initialize only one family from pretrained weights and present the result as a matched training comparison. Report the actual parameter count from the instantiated architecture: changing the vision classifier, patch projection, positional embeddings, context length, or tied LM vocabulary changes counts from standard model-card totals.

The nanoGPT source provides the 12×768 GPT configuration, tied token embedding/readout, causal attention and two residual branches. Its published full GPT-2 training configuration has a vastly longer training budget than this experiment can afford. Calling this "nanoGPT-style" is more accurate than claiming reproduction of that completed recipe. [nanoGPT model source](https://github.com/karpathy/nanoGPT/blob/master/model.py), [GPT-2 training configuration](https://github.com/karpathy/nanoGPT/blob/master/config/train_gpt2.py).

The proposed ViT is a CIFAR-sized-input adaptation of the base transformer, not the stock 224-pixel ViT-B/16 evaluation setup. Torchvision's standard model has approximately 86.6M parameters. [Torchvision ViT-B/16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html).

## Endpoint and uncertainty

For seed s and the predefined primary mask m, calculate

`D_s(m) = CE_s(m) - CE_s(full)`

and the paired dropout effect

`E_s(m) = D_s,dropout(m) - D_s,dense(m)`.

Negative effects indicate less damage from pruning. Always display full CE, pruned CE, and the degradation alongside the paired difference; otherwise a poorly trained or nearly constant model can look perfectly robust. For images also display top-1 accuracy and its change in percentage points. For language modeling report nats per token and, if useful, the pruned/full perplexity ratio `exp(D_s)`.

The primary masks retain two-thirds of residual blocks in every family, but the convolutional intervention is **stagewise thinning**, not a global early exit. It preserves the computation that changes feature dimensions. Two-thirds retained blocks is not two-thirds retained FLOPs, parameters, latency, or useful capacity. Compare the direction of the effects across families; do not average nats per wordpiece and nats per image into one transfer score.

Use three new final training seeds, paired across the three recipes within each family. The same seed must give identical initial parameters and training example order for same-architecture arms. Keep separate RNG streams for initialization, data order, augmentation, and layer masks. Three-seed 95% Student-t intervals use **df=2 and critical value 4.30265273**, not the five-seed value 2.776 used in the earlier local report. Show all three paired differences; such small samples produce fragile intervals. Six primary comparisons across three families are still multiple comparisons. Label intervals unadjusted, or make any adjusted analysis explicit.

There is no statistical requirement that every individual dataset/family achieve a nonzero interval. A weak result, or failure of one family to learn, should be reported rather than removed from the study. In the latter case it does not establish absence of a robustness effect in a properly trained model.

## Budget: use measured speed to lock completed schedules

The user's existing $50 experiment ceiling remains binding. Plan at most $40 of gross resource usage and keep at least $10 outside the planned run reservations. Count retries, failed runs, model compilation, data preparation, evaluation, and idle allocated-container time. Free credits should not be treated as extra experimental authorization.

Official pricing checked 2026-09-09 lists A100 80GB at $.000694/second, physical CPU at $.0000131/core/second, and RAM at $.00000222/GiB/second. With at most two physical cores and 8 GiB host RAM, the summed rate is $.00073796/second = **$2.656656/hour**. Thirteen aggregate container-hours would be about **$34.54**, before separately charged resources. A100 40GB is cheaper, but this plan honors the selected 80GB device. Avoid optional region and nonpreemptible premiums. [Modal pricing](https://modal.com/pricing), [GPU selection](https://modal.com/docs/guide/gpu).

CPU/RAM requests alone are not billing caps: Modal bills the higher of requested and actual use. Explicit limits, for example `cpu=(2,2)` and `memory=(8192,8192)`, make the quoted bound meaningful, at the risk of OOM if preprocessing needs more RAM. If larger limits are necessary, raise the reservation rate before dispatch. An environment compute budget does not cover every possible workspace charge. [Resource requests, limits and billing](https://modal.com/docs/guide/resources), [Modal budgets](https://modal.com/docs/guide/budgets).

One workable allocation, subject to a no-test throughput pilot:

| Phase | Aggregate GPU-container envelope |
|---|---:|
| GPT training, including LR search and final seeds | 5 hours |
| ViT training, including LR search and final seeds | 4 hours |
| ConvNeXt training, including LR search and final seeds | 2 hours |
| Pilot, validation, pruning evaluation, startup and checkpoint overhead | 2 hours |
| Total planned | 13 hours, approximately $34.54 at the capped rate |

This leaves about $5.46 inside the $40 working allocation for separately charged overhead, and $10 outside it. These are resource reservations, not promised invoices or verified runtimes. Concurrency reduces elapsed waiting; it does not reduce aggregate cost.

For each family, first run a short engineering pilot using a seed excluded from tuning and final evaluation. Measure forward/backward/optimizer step time after warmup, peak GPU memory, and full-depth validation throughput. Exercise both dense and masked computation, including the aggressive initial DTS regime. Do not score test data. Include compilation and initialization costs separately if enabled.

Use the pilot to choose one fixed step count per family **before** LR tuning and test scoring. For the recommended full-duration three-rate design below, there are 18 training runs per family. The table implies nominal per-run training budgets of 1000 seconds for GPT, 800 seconds for ViT, and 400 seconds for ConvNeXt. Divide each by a conservatively inflated observed step time, subtract startup/evaluation overhead, and round down to a convenient terminal step count or whole epoch. Use the same completed warmup/cooldown schedule and data budget for every recipe in the family. A time cap is a failure guard, not the intended training stopping rule.

If the three-family plan gives only a handful of vision epochs, do not equate model size with useful scale transfer. Aim for tens of image epochs, preferably roughly 50 or more if actual speed allows; this is a design target, not a correctness threshold. Log training and validation learning curves, and compare full-model predictions with the uniform-class baseline. A nearly untrained ViT's low pruning damage is uninformative. Options, decided before final testing, are to reallocate time toward ViT, reduce the equally applied LR grid, or explicitly mark the family as a short-budget exploratory test. Do not selectively lengthen only the arm with disappointing pruning results.

For GPT, report total training tokens and tokens per parameter. Tens of millions of tokens on a 124M model are a short-budget mechanism test, substantially under the target paper's training scale, even if every planned step and LR cooldown is completed. It is important that the run is complete under its declared schedule; it is not important to label it converged.

## Learning-rate strategy

Preferred within the measured envelopes: three rates × one independent tuning seed × three recipes = nine complete tuning runs per family, then three independent final seeds × three recipes = nine complete evaluation runs. Select LR independently per recipe using only terminal full-depth validation CE. This is 54 training runs across all three families; it is not 54 statistically independent final replicates.

Reasonable initial grids are `[2e-4, 6e-4, 1.8e-3]` for the GPT arm and `[1e-4, 3e-4, 1e-3]` for the image arms, with family-specific AdamW defaults and a fixed effective batch. These are proposals requiring the frozen implementation's optimizer/batch context; they are not universal optimal rates. Hold weight decay, augmentation, clipping and warmup rules fixed within each family and disclose that only LR is searched.

If throughput requires economizing, two rates per recipe on one full-duration tuning seed is preferable to quietly optimizing dense only or changing rates using test outcomes. A common canonical rate is also a legitimate controlled sensitivity experiment, but it supports a fixed-recipe comparison rather than superiority over tuned baselines. The earlier digits study showed that a validation-driven LR change can substantially change apparent pruning effect size, so avoiding all LR checks would weaken interpretation.

Do not use short-horizon validation winners as if they were full-horizon tuning. If a shorter proxy search is unavoidable, state the proxy budget, keep it equal for all recipes, and lock the selections before final tests. If a selected rate lands on a grid boundary, either expand the grid equally within the reserved budget before final testing or disclose the unresolved boundary. No automatic test-driven retries, seed replacement, checkpoint selection, or post hoc best-mask headline.

## Mask and implementation checks

- GPT and ViT have two residual branches per block. Sample one Bernoulli mask per example/block and share it between attention and FFN; apply inverse survival to each branch separately. Set ordinary activation dropout to the same value, preferably zero, in all arms.
- ConvNeXt has one composite residual branch per identity-shape block. Apply the mask once to that branch. Its stock implementation already has row-wise stochastic depth and a depthwise schedule; disable or replace that path rather than multiplying two independently active masks. Its small learned LayerScale is a real architectural feature: preserve and report it unless deliberately studying a modified model. [Torchvision ConvNeXt source](https://docs.pytorch.org/vision/main/_modules/torchvision/models/convnext.html).
- Never prune ConvNeXt stem/downsampling projections as though they were residual blocks. Index masks over the 18 residual blocks only. Check every chosen mask's output shape and classifier compatibility.
- Inference retains original order, positional embeddings/class token, final normalization and head. Every retained branch has inference scale one; there is no retained-depth renormalization, auxiliary exit loss, head fitting, or calibration.
- Verify no-drop forward/gradient equivalence, full-mask equality, forced dropped-block identity, pruning versus explicit zeroed residuals, causal masking for GPT, finite gradients, exact schedule averages/endpoints, and saved checkpoint readback.
- Use compute-then-mask for semantic simplicity unless true sparse execution is explicitly benchmarked. It changes active-work accounting but does not save the branch matmuls. All-dropped branch parameters still need the declared AdamW zero-versus-absent-gradient policy.
- If activation checkpointing is used, sample masks outside the recomputed closure or otherwise prove masks are reused. Resampling masks during backward recomputation changes the trained function.

## Evaluation masks and memory

At 12 blocks, exhaustive evaluation has 4095 nonempty masks; at 18 it has 262143. Do not carry the earlier six-block exhaustive protocol into these models by default. The primary masks and a small published, outcome-independent secondary panel are enough.

Recommended secondary panel: full model; a short retained-depth curve; one predetermined spread/alternating mask at the primary retained count; and 8–16 unique masks preserving the first block, sampled once and shared across recipes and seeds. Keep any unrestricted first-block deletion panel separate because ILD never trains that deletion. ConvNeXt secondary masks must retain mandatory transitions, and stagewise mask cardinalities should be explicit. Average individual-submodel losses within seed before estimating seed uncertainty. It is not prediction ensembling.

Use identical held-out examples for each mask. Full official image test scoring is affordable if pilot timings confirm it. For language, a published fixed nonoverlapping token-window set is acceptable; state its size and do not compare that partial/context-reset loss with canonical WikiText perplexity. Validate with the same context convention used for test.

BF16 mixed precision, fused AdamW and efficient causal attention are reasonable on A100. Memory usage must include logits, not just parameters: a GPT2-sized vocabulary with 16384 tokens per microbatch produces about 824M logit values before loss temporaries. Start with modest microbatches and gradient accumulation, measuring actual peak usage. Keep the effective batch identical across recipes and ensure every microbatch gets independent masks. Matching examples, optimizer steps and schedules is the primary fairness condition; runtime is provenance until a separately controlled systems comparison exists.

## Data provenance and split isolation

Use WikiText-103's official train/validation/test partitions, pin the dataset revision and tokenizer version, and hash token files. Raw rows include text structure and empty lines; publish how they are joined and whether document/EOT boundaries are inserted. Never let packed windows span dataset splits. A GPU worker should consume prepared tokens instead of repeating download/tokenization per training seed. [Salesforce WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext).

CIFAR-100 contains 500 training and 100 test images per class. Reserve a stratified 5000-example validation set from the official 50000 training examples, leaving 45000 training examples; retain the official 10000 test examples. Fit any data-derived normalization on training rows only, and give paired image recipes identical augmentation streams. Avoid pretrained normalization/weights being silently mixed into a from-scratch claim. [Original CIFAR dataset documentation](https://www.cs.toronto.edu/~kriz/cifar.html).

Freeze dataset hashes, model configurations, seeds, training duration, LR selection rule, primary masks, secondary masks and evaluation windows before the first final test evaluation. Preserve development failures and reservations in the audit trail. Report measured results even when they disagree across architecture families; the disagreement is part of what transfer testing is meant to discover.
