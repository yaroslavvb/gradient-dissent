#!/usr/bin/env python3
"""Build the source-linked history report, without network calls or training."""
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/stochastic-depth-history'
OUT.mkdir(parents=True, exist_ok=True)
source = (ROOT / 'research/stochastic-depth-history.md').read_text()
md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'], extension_configs={'toc': {'toc_depth': '2-2'}})
body = md.convert(source)
body = body.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A source-grounded history of stochastic depth: original authors, successful applications, language-model descendants, and the contribution of Don't Drop Dropout."><title>Stochastic depth after 2016 · Gradient dissent</title><link rel="stylesheet" href="style.css"></head><body><a class="skip" href="#report">Skip to report</a><header><a class="brand" href="../">GRADIENT DISSENT</a><nav aria-label="Related reports"><a href="../ciresan-stochastic-depth/">MNIST experiments</a><a href="../dropout-animation/">Dropout animation</a><a href="../significance/">Experimental significance</a></nav></header><div class="layout"><aside><nav aria-label="Report contents">''' + md.toc + '''</nav><div class="downloads"><a href="report.md" download>Download Markdown</a><a href="https://github.com/yaroslavvb/gradient-dissent/tree/main/research">Research sources ↗</a></div></aside><main id="report">''' + body + '''</main></div></body></html>'''
(OUT / 'index.html').write_text(page)
(OUT / 'report.md').write_text(source)
print(f'Built stochastic-depth history: {len(source.split()):,} Markdown words.')
