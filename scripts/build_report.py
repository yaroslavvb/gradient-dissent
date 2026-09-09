#!/usr/bin/env python3
"""Build the extended static review from the reviewed Markdown sources."""
from pathlib import Path
import re, html, json
import markdown

ROOT = Path(__file__).resolve().parents[1]
parts = [
    ('audit', 'Technical review', 'research/paper-audit.md'),
    ('experiments', 'Local experiments', 'research/experiment-findings.md'),
    ('connections', 'Energy and Sutro', 'research/energy-connections.md'),
]
rendered=[]
for prefix,title,filename in parts:
    text=(ROOT/filename).read_text()
    text=text.replace("the user's research agenda", "Sutro's research agenda")
    text=text.replace('](../experiments/README.md)', '](https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments)')
    equations=[]
    def save_math(match):
        kind='display' if match.group(1)=='[' else 'inline'
        equations.append((kind,match.group(2)))
        return f' MATHPLACEHOLDER{len(equations)-1}END '
    text=re.sub(r'\\([\[(])(.*?)\\[\])]',save_math,text,flags=re.S)
    md=markdown.Markdown(extensions=['tables','fenced_code','toc'],extension_configs={'toc':{'slugify':lambda value,separator:prefix+'-'+re.sub(r'[^a-z0-9]+','-',value.lower()).strip('-')}})
    body=md.convert(text)
    for i,(kind,eq) in enumerate(equations):
        body=body.replace(f'MATHPLACEHOLDER{i}END',f'<span class="tex-{kind}" data-tex="{html.escape(eq,quote=True)}"></span>')
    body=body.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
    body=re.sub(r'<(/?)h([1-5])',lambda m:'<'+m[1]+'h'+str(int(m[2])+1),body)
    rendered.append(f'<section id="{prefix}" class="extended-section"><div class="eyebrow">{title}</div>{body}</section>')
source_html=(ROOT/'docs/index.html').read_text()
refs=re.search(r'<ol class="references">.*?</ol>',source_html,re.S).group()
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Extended technical review, exact mathematical counterexamples, multi-seed toy experiments, and connections to energy-efficient learning."><title>Gradient dissent — extended review</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="vendor/katex/katex.min.css"><style>.extended{max-width:1100px;margin:auto;padding:70px 40px}.extended h1{font-size:65px;max-width:900px}.extended h2{font-size:38px;margin-top:25px}.extended h3{font:28px/1.25 var(--serif);margin-top:45px}.extended h4{font-size:18px;margin-top:30px}.extended p,.extended li{max-width:900px}.extended li{margin-bottom:10px}.extended-section{padding:60px 0;border-bottom:1px solid var(--line)}.extended pre{background:var(--soft);font:13px/1.65 ui-monospace,monospace;padding:20px;overflow:auto}.extended code{font-family:ui-monospace,monospace;font-size:.87em}.tex-display{display:block;overflow:auto;padding:16px 0}.katex-display{margin:0!important}.extended-nav{display:flex;gap:22px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:25px;font-size:14px}@media(max-width:700px){.extended{padding:35px 22px}.extended h1{font-size:46px}.extended h2{font-size:32px}.extended h3{font-size:25px}}</style></head><body><header><a class="brand" href="index.html">GRADIENT DISSENT</a><nav><a href="index.html">Interactive report ↗</a><a href="https://github.com/yaroslavvb/gradient-dissent">Code &amp; data ↗</a></nav></header><main class="extended"><h1>Gradient dissent.</h1><p class="lead">Extended critical review, experiments, and connections to energy-efficient learning.</p><nav class="extended-nav" aria-label="Report parts"><a href="#audit">Technical audit</a><a href="#experiments">Local experiments</a><a href="#connections">Energy and Sutro</a><a href="#references">References</a><a href="index.html">Interactive edition ↗</a></nav>'''+''.join(rendered)+f'<section id="references" class="extended-section"><h2>Core references</h2>{refs}<p>Additional primary sources are linked beside the corresponding arguments above. The private document is cited for context; its raw text is not included.</p></section></main></body></html>'
(ROOT/'docs/review.html').write_text(page)
print(f'Built extended review: {len(page):,} characters, {len(parts)} source documents.')
