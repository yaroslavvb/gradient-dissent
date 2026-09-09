#!/usr/bin/env python3
"""Fetch the immutable corpus revision used in the depth-robustness experiment."""
from pathlib import Path
import hashlib,urllib.request,json
root=Path(__file__).resolve().parent
url='https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt'
with urllib.request.urlopen(url,timeout=60) as r:data=r.read()
manifest=json.loads((root/'results/dataset.json').read_text())
assert hashlib.sha256(data).hexdigest()==manifest['sha256'],'Corpus differs from the executed experiment'
(root/'data').mkdir(exist_ok=True)
(root/'data/input.txt').write_bytes(data)
print('Verified immutable corpus:',manifest['sha256'])
