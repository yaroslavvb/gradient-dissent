# Depth Robustness Across Architectures: An Audit of Layer Dropout

Independent research note, 9 September 2026. The paper synthesizes the original-paper audit, local toy controls and completed A100 transfer study. No new training was run to create it.

- [High-level significance page](https://yaroslavvb.github.io/gradient-dissent/significance/)
- [Paper PDF](https://yaroslavvb.github.io/gradient-dissent/significance/depth-robustness-audit.pdf)
- [Immutable experimental snapshot](https://github.com/yaroslavvb/gradient-dissent/tree/28f9db3bdc2e8910a105c7e40944b521c8bf0c1f)

## Build

From the repository root, with Python 3 and TeX Live 2026 (`pdflatex`, `bibtex`):

```sh
python3 scripts/build_paper.py
```

The build regenerates numerical A100 table rows from the verified summary, uses committed figure PNGs, compiles the manuscript and rejects unresolved references or overfull boxes. It writes the local deliverable to `output/pdf/depth-robustness-audit.pdf` and the identical public copy to `docs/significance/depth-robustness-audit.pdf`. It performs no network or training calls. The LaTeX intermediate directory is ignored under `tmp/pdfs/build/`.

To regenerate the scientific figures in a separate environment:

```sh
python3 -m venv /tmp/gradient-dissent-paper-plot
/tmp/gradient-dissent-paper-plot/bin/pip install -r paper/figures/requirements.txt
/tmp/gradient-dissent-paper-plot/bin/python paper/figures/make_figures.py \
  --summary experiments/a100_transfer/results/summary.json --out paper/figures
```

The first and second figures contain means; the second also shows each seed. The primary-effect figure contains paired seed differences and df=2 confidence intervals. Panel scales differ where labeled. Figure CSVs provide the underlying plotted values; `table-data.csv` supplies all 108 seed-level values for the full/primary-pruned table. All numbers derive from the saved study, and the figure provenance records the input digest.

Local table values come from `experiments/depth_lm/results/result.json` and `experiments/depth_digits/result.json`; their n=5 confidence intervals and smaller-model qualifications are given in the manuscript. Sources and methods remain distinct from the original authors' experiments.

## Rendering and verification

```sh
pdftoppm -r 120 -png output/pdf/depth-robustness-audit.pdf tmp/pdfs/page
pdfinfo output/pdf/depth-robustness-audit.pdf
```

Every page of the final release is inspected for text, figure and table legibility. Bibliographic links and numerical claims were independently audited. The single-column Letter layout, Latin Modern body type, sans-serif headings, running title, numbered equations and scientific captions follow the general visual organization of the original report; no original branding or figures are reused.
