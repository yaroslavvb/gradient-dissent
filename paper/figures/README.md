# Manuscript figure assets

All three figures use only `experiments/a100_transfer/results/summary.json`, the verified A100 study aggregate. No training or paid service was invoked.
PNG width is exactly 6.5 inches at 300 dpi (1,950 pixels); all text is at least
9 points at that physical width. White background, DejaVu Sans, shared palette:
dense `#546478`, constant ILD `#216aab`, decreasing ILD `#bb5425`.

## Files and suggested captions

1. **`fig1_full_to_pruned_ce.png`** (6.5 × 3.15 inches). Full-depth and retained
   model CE, means over three training seeds. Each panel has its own vertical
   scale and target units. Retained means prefix 8/12 blocks for GPT and ViT,
   and stagewise 12/18 blocks for ConvNeXt, with mandatory downsampling retained.
   Numeric table rows follow the legend order: dense, constant ILD, decreasing
   ILD. The original final normalization and classifier are reused. CSV contains
   18 task × recipe × mask cells, exact means and individual seed CE values.

2. **`fig2_vit_accuracy_secondary.png`** (6.5 × 3.1 inches). Secondary ViT test
   accuracy across six predefined structural interventions. Open dots are the
   three individual training seeds; squares show means and are offset to avoid
   hiding seed dots. All interventions use the same 10,000 CIFAR-100 test images.
   The random masks retain eight blocks including the first; deleting only the
   first block retains eleven. Random-mask numbers are identifiers, not depths
   or rankings. CSV contains 54 seed-level observations and corresponding means.

3. **`fig3_primary_paired_effects.png`** (6.5 × 3.35 inches). Within each seed,
   define ΔCE = CE(retained) − CE(full). Plotted effects are ΔCE(ILD) − ΔCE(dense).
   Squares show paired means, dots show the three individual paired effects,
   and bars show unadjusted 95% Student-t intervals with df = 2. Horizontal
   scales differ across panels; target units are labeled. Negative values mean
   lower signed CE change, not necessarily a superior full model; positive ViT
   effects do not alone imply worse absolute pruned prediction. CSV contains
   18 individual paired effects with their corresponding means and CI bounds.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python make_figures.py --summary ../../experiments/a100_transfer/results/summary.json --out .
```

`figure-provenance.json` records source and script SHA-256 values, runtime
versions, units, palette, and plotting conventions. `environment-freeze.txt`
captures the full installed plotting environment. All PNGs were visually
inspected at final resolution after correcting the first figure's legend
spacing. Figures 1 and 2 were then compacted for manuscript placement by removing
redundant main headings and extra vertical space, preserving all data and the
minimum 9-point text size. No manuscript, site, experiment, or PDF files were
modified.
