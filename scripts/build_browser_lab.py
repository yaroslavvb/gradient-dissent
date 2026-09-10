#!/usr/bin/env python3
"""Verify browser MNIST assets, vendor pinned bundles, and write a deployment manifest."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'docs/ciresan-stochastic-depth/browser-lab'
CHECKS = ROOT / 'scripts/browser-lab'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    for package, bundle in [('@tensorflow/tfjs', 'tf.min.js'),
                            ('@tensorflow/tfjs-backend-webgpu', 'tf-backend-webgpu.min.js')]:
        source = ROOT / 'node_modules' / package
        assert json.loads((source / 'package.json').read_text())['version'] == '4.22.0'
        shutil.copyfile(source / 'dist' / bundle, PAGE / 'vendor' / bundle)
    assert 'Apache License' in (PAGE / 'vendor/LICENSE-tensorflow.txt').read_text()
    metadata = json.loads((PAGE / 'dataset/manifest.json').read_text())
    assert metadata['producer']['script_sha256'] == sha((CHECKS / 'build_dataset.py').read_bytes())
    reference = json.loads((CHECKS / 'reference-fixture.json').read_text())
    assert reference['provenance']['generator_sha256'] == sha((CHECKS / 'generate-reference.py').read_bytes())
    splits = {}
    assertions = []
    for name, record in {**metadata['splits'], **metadata['optional_splits']}.items():
        compressed = (PAGE / 'dataset' / record['gzip_file']).read_bytes()
        assert len(compressed) == record['gzip_bytes'] and sha(compressed) == record['gzip_sha256']
        raw = gzip.decompress(compressed)
        assert len(raw) == record['bytes'] and sha(raw) == record['sha256']
        assert raw[:8] == b'GDMNIST1'
        n, rows, cols, po, lo, io = struct.unpack_from('<6I', raw, 8)
        assert (n, rows, cols, po, lo, io) == (record['count'], 28, 28, 32, 32+n*784, (32+n*785+3)//4*4)
        assert len(raw) == io + n*4
        pixels, labels = raw[po:lo], raw[lo:lo+n]
        ids = struct.unpack_from(f'<{n}I', raw, io)
        assert sha(raw[io:]) == record['original_indices_sha256_uint32_le']
        assert len(set(ids)) == n and max(ids) < (60000 if name in ('train', 'validation') else 10000)
        assert all(y < 10 for y in labels)
        assert [labels.count(y) for y in range(10)] == record['class_counts']
        if name != 'test_full':
            assert all(y == i % 10 for i, y in enumerate(labels))
        else:
            assert ids == tuple(range(10000))
        splits[name] = (pixels, labels, ids)
        assertions.append(f'{name}: compressed/raw hashes, sizes, schema, unique bounded indices, class counts and ordering')
    assert not set(splits['train'][2]) & set(splits['validation'][2])
    assertions.append('training/validation original indices disjoint')
    preview_bytes = (PAGE / 'dataset/preview.json').read_bytes()
    assert sha(preview_bytes) == metadata['preview']['sha256']
    for example in json.loads(preview_bytes)['examples']:
        i = example['browser_index']
        assert base64.b64decode(example['pixels']) == splits['train'][0][784*i:784*(i+1)]
        assert example['label'] == splits['train'][1][i] and example['original_index'] == splits['train'][2][i]
    assertions.append('all 20 preview images and labels match exact training assets')
    for name in ['core.js', 'dataset.js', 'tf-engine.js', 'worker.js', 'app.js']:
        subprocess.run(['node', '--check', str(PAGE / name)], check=True)
    for required in ['index.html', 'style.css', 'README.md', 'vendor/README.md', 'dataset/README.md']:
        assert (PAGE / required).is_file()
    receipts = sorted(CHECKS.glob('test-*.json'))
    assert len(receipts) == 5
    for receipt in receipts:
        assert json.loads(receipt.read_text())['status'] == 'passed', receipt
    (CHECKS / 'verification.json').write_text(json.dumps({
        'pass': True, 'checks': assertions, 'dataset_manifest_sha256': sha((PAGE / 'dataset/manifest.json').read_bytes()),
        'default_gzip_bytes': metadata['default_total_gzip_bytes'],
    }, indent=2) + '\n')
    sources = [ROOT / 'package.json', ROOT / 'package-lock.json', Path(__file__).resolve()] + sorted(p for p in CHECKS.iterdir() if p.is_file())
    document = {
        'schema': 'gradient-dissent-browser-lab-assets-v1',
        'runtime': 'TensorFlow.js 4.22.0; qualified WebGPU/WebGL/CPU in a dedicated worker',
        'verification_note': 'See scripts/browser-lab test receipts. Native Dawn Metal checks are not browser UI tests or browser benchmarks.',
        'outputs': {str(p.relative_to(PAGE)): sha(p.read_bytes()) for p in sorted(PAGE.rglob('*')) if p.is_file() and p != PAGE / 'manifest.json'},
        'sources': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sources},
    }
    (PAGE / 'manifest.json').write_text(json.dumps(document, indent=2) + '\n')
    print(f'Browser lab verified: {len(document["outputs"])} deployed files, {len(document["sources"])} source/check files; MNIST subsets and library hashes match.')


if __name__ == '__main__':
    main()
