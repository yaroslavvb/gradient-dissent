# Six-affine inference intervention: protocol draft

This extends the existing frozen-checkpoint four-branch audit. It performs no
training, checkpoint selection, LR tuning or inference fine-tuning. The parent
runner must freeze this protocol, source hashes, recipe/state/seed panel and
gallery example indices before dispatch. It owns cloud authorization and budget.
No previous audit files or manifests are modified.

## Cohort and ordering

By default, use all five original main recipes, seeds 101–103, both the selected
and final checkpoints: 30 states. Any smaller panel must be explicitly listed
in the dispatch manifest. Checkpoints are the original
`/work/runs/graph-eval-{recipe}-s{seed}/{best,final}.pt`, not later telemetry
replays. Selected remains the validation-selected checkpoint; final is epoch
100. All comparisons here are exploratory follow-up interventions on an already
reused official test set, not an additional independent replication.

Bits 0 through 5, displayed shallow to deep, mean stem, body1, body2, body3,
body4 and head. Integer 63 keeps all six. Existing body mask m embeds as
`33 + 2*m`, preserving both endpoints. Evaluate all 64 masks without selecting
only favorable masks, using all 10,000 official MNIST test images in original
order. Preserve the previously chosen gallery indices; do not select a new
gallery based on these new outcomes.

## Fixed semantics

The normalized-input, linear-output main model has widths
784→2500→2000→1500→1000→500→10. All retained branches have inference gain one;
ordinary dropout is disabled and no training gates are sampled.

* Present stem: original learned affine then ReLU. Absent stem: divide flattened
  pixels by 255, append zeros to reach width 2500, then ReLU. This is an arbitrary,
  untrained embedding intervention and has no learned alternative stem.
* Present body: original residual crop-plus-affine/ReLU operator, or the original
  affine/ReLU for a plain checkpoint. Absent body: fixed prefix crop followed by
  ReLU. For a plain checkpoint this remains untrained crop surgery.
* Absent head: **abstain**, short-circuiting before every affine computation.
  Do not reinterpret hidden coordinates as class logits. Thus all 32 head-absent
  masks are equivalent abstention policies regardless of other selected bits.
  There is no mathematical requirement that a model with arbitrary endpoint
  surgery should perform at 10% chance; abstention is not a random classifier.

Report coverage (fraction receiving any prediction), correct-output rate
(correct predictions divided by all inputs), and accuracy conditional on an
emitted prediction separately. Head-absent coverage and correct-output rate are
zero; accuracy and CE are undefined (JSON null). Head-present coverage is one,
so accuracy equals correct-output rate. This prevents conflating abstention
with incorrect digit classification. Per-class metrics use the same definitions.
Harm is the fraction of all inputs that the dense model gets right but the
intervention does not correctly emit; it therefore includes abstentions.

## Cost convention

Affine MACs by layer are 1,960,000; 5,000,000; 3,000,000; 1,500,000; 500,000;
5,000, totaling 11,965,000. Head-present masks sum exactly the selected affine
costs. Head-absent masks execute zero affines because the policy short-circuits;
retain nominal selected MACs separately for UI transparency. Padding, crop,
ReLU, bias, indexing and memory work are excluded. Zero affine MACs is not a
measured zero-latency claim. No runtime or energy savings are inferred from this
functional audit. Full stem computation is counted whenever it is present,
even if subsequent crops discard some coordinates.

## Verification and artifacts

Use FP32 tensors, `torch.set_float32_matmul_precision('high')`, no autocast,
inference batch size 2048, and the same batch boundaries as the old audit.
Every all-kept batch must match `model()` logits bitwise. All sixteen embedded
old masks must match archived predictions exactly on every test example,
including label order. Compare each embedded mean CE with the archived mean
using the original audit's 3e-5 tolerance; record the maximum per-example CE
difference without turning that measurement into a separate tolerance claim.
Also require exact dense accuracy and error IDs against the original training
result, checkpoint/source JSON hash agreement with the archived audit, and
dataset file SHA-256 agreement. Any failure stops the run; no exceptions or
automatic retries are introduced here.

Each state gets a JSON summary with all64 masks, per-class metrics, frozen-gallery
predictions and provenance, plus an NPZ sidecar: `pred[64,10000]` int8 (`-1` for
abstention), `ce[64,10000]` float64 (NaN only for undefined head-absent rows),
`labels[10000]`, `mask_ids[64]`, and `coverage[64]`. The head-present32 rows alone
require model inference. JSON must never contain nonstandard NaN/Infinity.
The returned job result also embeds every complete state summary in `models`,
so publishing the tables and frozen gallery does not require a second cloud
invocation. NPZ matrices remain separate reproducibility artifacts.
Preserve incomplete and failed runs as such; publish no invented results.

## Interpretation limits

Body-only deletion with both endpoints present is the existing trained-bypass
study. Stem removal is a different, untrained representation stress test. Head
removal is an explicit abstention policy, not evidence about classification
robustness. Do not average head-absent cases into classification accuracy or CE,
or present their zero correct-output rate as a collapse of a classifier. All64
controls may be displayed together only with these distinctions visible.
