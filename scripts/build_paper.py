#!/usr/bin/env python3
"""Build the research note from saved tables and figures, without training."""
import os, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'; BUILD=ROOT/'tmp/pdfs/build'
BUILD.mkdir(parents=True,exist_ok=True)
subprocess.run(['python3',str(PAPER/'build_tables.py')],check=True,cwd=ROOT)
env=os.environ.copy();env['SOURCE_DATE_EPOCH']='1788991200';env['FORCE_SOURCE_DATE']='1';env['TZ']='UTC'
for tool in ['pdflatex','bibtex']:
 if not shutil.which(tool):raise SystemExit(f'{tool} required (TeX Live 2026 used for this release).')
for step in range(4):
 if step==1:
  benv=env.copy();benv['BIBINPUTS']=str(PAPER)+os.pathsep
  cmd=['bibtex','manuscript'];cwd=BUILD
 else:
  benv=env;cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'manuscript.tex'];cwd=PAPER
 proc=subprocess.run(cmd,cwd=cwd,env=benv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (BUILD/f'pass-{step}.stdout').write_text(proc.stdout)
 if proc.returncode:raise SystemExit(proc.stdout[-4000:])
log=(BUILD/'manuscript.log').read_text()
# Additional reference pass keeps cross-references stable after bibliography layout.
proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'manuscript.tex'],cwd=PAPER,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
log=(BUILD/'manuscript.log').read_text()
for bad in ['Overfull \\hbox','Overfull \\vbox','undefined references','Citation `','Label(s) may have changed']:
 if bad in log:raise SystemExit('Resolve LaTeX warning before release: '+bad)
final=ROOT/'output/pdf/depth-robustness-audit.pdf';final.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(BUILD/'manuscript.pdf',final)
shutil.copyfile(final,ROOT/'docs/significance/depth-robustness-audit.pdf')
shutil.copyfile(PAPER/'table-data.csv',ROOT/'docs/significance/table-data.csv')
print('Built',final)
