# Gradient dissent

An in-depth, critical review of **Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference** (arXiv:2609.05275v1), with exact counterexamples, locally executed toy experiments, and connections to Sutro's energy-efficient learning agenda.

- [Interactive report and 16-chapter slide view](https://yaroslavvb.github.io/gradient-dissent/)
- [How layer dropout works](https://yaroslavvb.github.io/gradient-dissent/dropout-animation/): animated per-sequence paths, shared attention/FFN masks, inverse-survival scaling, depth/time scheduling, and fixed inference modes.
- [Extended review](https://yaroslavvb.github.io/gradient-dissent/review.html)
- [Separate depth-robustness experiment report](https://yaroslavvb.github.io/gradient-dissent/depth-robustness/): a tiny causal transformer and 8×8 digit classifier, five seeds per treatment, exact layer-subset enumeration, and $0 cloud spend.
- [Larger A100 transfer experiment report](https://yaroslavvb.github.io/gradient-dissent/a100-transfer/): 124M-parameter nanoGPT-style language model, 85M-parameter vision transformer, and 28M-parameter ConvNeXt on WikiText-103 and CIFAR-100. Three paired final seeds, nine fixed pruning interventions, and a $50 experiment ceiling.
- [Session transcript](https://yaroslavvb.github.io/gradient-dissent/transcripts/research-session/): searchable user/assistant conversation through the export request, with Markdown and JSON downloads; viewer reused from animated-groups-fable.
- [Independent 13-page research paper](https://yaroslavvb.github.io/gradient-dissent/significance/depth-robustness-audit.pdf): analytical audit, local controls, A100 transfer results, uncertainty and reproducibility; [LaTeX and figure sources](paper/README.md).
- [Original paper PDF](https://arxiv.org/pdf/2609.05275v1)
- [Experiments and reproduction instructions](experiments/README.md)

- [High-level significance report](https://yaroslavvb.github.io/gradient-dissent/significance/): what the paper and independent experiments establish, with an interactive explanation of metric and mask dependence.

## Findings

The paper offers a useful empirical recipe, with strong depth-robustness results. Its unqualified mean-preservation argument fails under the preferred shared-mask equations. The largest 8.2B row lacks a dense same-size control, and nonembedding FLOPs savings are not measured end-to-end time or energy savings. Progressive random subnetworks also have an important predecessor in RaPTr.

Local exact enumeration confirms the shared-mask counterexample. A CPU execution benchmark shows actual gathering helps at one tensor size but not another. A 100-run tuning sweep followed by 60 evaluation runs finds mixed full-model quality results and substantial early-exit robustness. These are mechanism checks, **not** an LLM-scale reproduction, energy measurement, or official MNIST challenge entry. All data and limitations are included.

## Repository

- `docs/`: static GitHub Pages report, interactive diagrams, extended review, and measured data.
- `research/paper-audit.md`: full paper audit and literature connections.
- `research/experiment-findings.md`: exact derivations, measured results, and limitations.
- `research/energy-connections.md`: data movement analysis, critical meeting-note audit, and proposed MNIST protocol.
- `experiments/`: scripts, pinned dependencies, raw CSV/JSON, uncertainty estimates, and verification records.
- `scripts/`: paper fetch, static report build, math rendering, and report checks.

## Build and view

```bash
npm ci
python3 -m pip install 'Markdown==3.10.2'
npm run build
npm run check
npm run serve
```

Open `http://localhost:8765`. The interactive edition needs HTTP to fetch its local results JSON. The published report has no runtime CDN, tracking, or backend dependency. Use **Present** to enter slide mode; arrow keys navigate, Escape returns to the report. Charts have associated explanations and data tables. Motion starts only on request.

`npm run build` regenerates `docs/review.html` from the three research Markdown files and renders equations into local HTML/MathML. `docs/index.html`, `docs/style.css`, and `docs/app.js` are authored static sources. GitHub Pages publishes the committed `docs/` directory from `main`.

The separate depth report is built from saved measurements with `python3 -m pip install -r requirements-depth-report.txt`, then `npm run build:depth` and `npm run check:depth`. Its source template is `scripts/depth-report-template.html`; the builder regenerates HTML, Markdown, plot data, and standalone scientific figures without retraining. See the [transformer](experiments/depth_lm/README.md) and [digits](experiments/depth_digits/README.md) protocols to repeat training. Final sweeps contain 90 tuning and 45 evaluation runs; earlier boundary-search runs are archived and disclosed separately.

The larger GPU study has its own [frozen protocol, execution instructions, and cost ledger](experiments/a100_transfer/README.md). Run `python3 experiments/a100_transfer/analyze.py` to audit all saved tuning and final results, then `npm run build:a100` and `npm run check:a100` to rebuild its report without training or paid calls. The A100 study evaluates a fixed panel of masks rather than every layer subset.

## Provenance

Review date: 9 September 2026. The full 27-page PDF, including appendices, was read and key equations/tables were visually checked. Source PDFs and raw meeting/document text stay in ignored `research/sources/`; use `scripts/fetch-paper.sh` to retrieve the public paper. The linked Google document was read only for research context, and its raw contents are not republished.

Reported paper values, independent derivations, synthetic simulations, and measured toy results are labeled separately. The repository includes original plots and analysis, not redistributed paper figures. Vendored KaTeX assets retain their MIT license. Reproducing timing will yield different numbers on different machines; the saved run includes exact environment metadata.

The significance overview uses the verified saved A100 summary: `npm run build:significance` and `npm run check:significance`. It performs no training.

The standalone dropout animation is authored in `docs/dropout-animation/`. Run `npm run check:dropout` to check the exact schedule, sampling/scaling, inference settings, and interactive controls. Its four sequences and twelve blocks are schematic; it performs no training or paid calls.

Build the independent paper from saved results with `python3 scripts/build_paper.py` (TeX Live 2026). See `paper/README.md` for figure regeneration and rendering.
