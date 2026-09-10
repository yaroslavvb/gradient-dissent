#!/usr/bin/env python3
"""Build versioned MNIST browser assets from verified existing experiment data.

This script writes only beneath --output. Training and validation are selected
and validated before official test files are decoded.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def packed_indices(values):
    return np.asarray(values, dtype='<i8').tobytes()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_idx(cache, name, metadata, images, expected):
    path = cache / name
    blob = path.read_bytes()
    record = metadata['files'][name]
    assert len(blob) == record['bytes']
    assert digest(blob) == record['sha256']
    assert hashlib.md5(blob).hexdigest() == record['md5']
    raw = gzip.decompress(blob)
    if images:
        assert struct.unpack('>IIII', raw[:16]) == (2051, expected, 28, 28)
        assert len(raw) == 16 + expected * 784
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(expected, 28, 28)
    assert struct.unpack('>II', raw[:8]) == (2049, expected)
    assert len(raw) == 8 + expected
    result = np.frombuffer(raw, dtype=np.uint8, offset=8)
    assert np.all(result < 10)
    return result


def balanced_indices(pool, labels, per_class):
    # Preserve the existing seeded parent-pool order within each class, then
    # interleave classes 0..9. Every prefix of a multiple of 10 is balanced.
    columns = [pool[labels[pool] == digit][:per_class] for digit in range(10)]
    assert all(len(column) == per_class for column in columns)
    return np.stack(columns, axis=1).reshape(-1)


def write_split(output, split, images, labels, indices, collection, selection, optional=False):
    count = len(indices)
    x = np.ascontiguousarray(images[indices], dtype=np.uint8)
    y = np.ascontiguousarray(labels[indices], dtype=np.uint8)
    pixel_offset = 32
    label_offset = pixel_offset + count * 784
    index_offset = (label_offset + count + 3) // 4 * 4
    header = b'GDMNIST1' + struct.pack('<IIIIII', count, 28, 28,
                                            pixel_offset, label_offset, index_offset)
    raw = header + x.tobytes() + y.tobytes()
    raw += b'\x00' * (index_offset - len(raw))
    raw += np.asarray(indices, dtype='<u4').tobytes()
    assert len(raw) == index_offset + 4 * count
    raw_name, gz_name = split + '.bin', split + '.bin.gz'
    (output / raw_name).write_bytes(raw)
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    (output / gz_name).write_bytes(compressed)
    return {
        'id': split, 'count': count, 'optional': optional,
        'source_collection': collection, 'selection': selection,
        'class_counts': np.bincount(y, minlength=10).tolist(),
        'file': raw_name, 'bytes': len(raw), 'sha256': digest(raw),
        'gzip_file': gz_name, 'gzip_bytes': len(compressed), 'gzip_sha256': digest(compressed),
        'pixel_offset': pixel_offset, 'label_offset': label_offset,
        'original_index_offset': index_offset,
        'original_indices_sha256_uint32_le': digest(np.asarray(indices, dtype='<u4').tobytes()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--cache', type=Path, default=Path('/tmp/ciresan-hypothesis-data'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/gradient-dissent-browser-assets'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive_rel = Path('experiments/ciresan_stochastic_depth/results/hypotheses/hypothesis-audit-s101-v1/data.npz')
    record_path = args.repo / archive_rel.parent / 'result.json'
    record = json.loads(record_path.read_text())
    meta = record['dataset']
    archive = args.repo / archive_rel
    assert digest(archive.read_bytes()) == record['artifacts']['data.npz']['sha256']
    x = read_idx(args.cache, 'train-images-idx3-ubyte.gz', meta, True, 60000)
    y = read_idx(args.cache, 'train-labels-idx1-ubyte.gz', meta, False, 60000)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(meta['split_seed'])).numpy()
    train_pool, val_pool = order[:50000], order[-10000:]
    assert digest(packed_indices(train_pool)) == meta['train_indices_sha256']
    assert digest(packed_indices(val_pool)) == meta['val_indices_sha256']
    train_idx = balanced_indices(train_pool, y, 1000)
    val_idx = balanced_indices(val_pool, y, 200)
    assert len(set(train_idx.tolist())) == 10000
    assert len(set(val_idx.tolist())) == 2000
    assert not set(train_idx.tolist()) & set(val_pool.tolist())
    assert not set(val_idx.tolist()) & set(train_pool.tolist())
    train_record = write_split(args.output, 'train', x, y, train_idx, 'official-training-60000',
        'First 1000 examples per digit within the original seeded 50000 training pool; interleaved by digit 0..9.')
    val_record = write_split(args.output, 'validation', x, y, val_idx, 'official-training-60000',
        'First 200 examples per digit within the original seeded 10000 validation pool; interleaved by digit 0..9.')
    preview = {
        'dataset': 'MNIST', 'split': 'train', 'shape': [28, 28],
        'encoding': 'base64 uint8, row-major 784 bytes, black=0 white=255',
        'selection': 'First 20 browser training examples: two of each digit, no outcome selection.',
        'examples': [{'browser_index': i, 'original_index': int(index), 'label': int(y[index]),
                      'pixels': base64.b64encode(x[index].tobytes()).decode('ascii')}
                     for i, index in enumerate(train_idx[:20])],
    }
    write_json(args.output / 'preview.json', preview)
    # Test access happens only after train/validation selection and checks above.
    test_x = read_idx(args.cache, 't10k-images-idx3-ubyte.gz', meta, True, 10000)
    test_y = read_idx(args.cache, 't10k-labels-idx1-ubyte.gz', meta, False, 10000)
    with np.load(archive, allow_pickle=False) as old:
        assert np.array_equal(old['test_images'], test_x)
        assert np.array_equal(old['test_labels'], test_y)
        # The archived audit panel permuted the validation pool and stores that
        # permutation explicitly in val_positions.
        assert np.array_equal(np.sort(old['val_positions']), np.arange(10000))
        assert np.array_equal(old['val_labels'], y[val_pool][old['val_positions']])
    test_idx = balanced_indices(np.arange(10000), test_y, 200)
    test_record = write_split(args.output, 'test', test_x, test_y, test_idx, 'official-test-10000',
        'First 200 examples per digit in the original official test order; interleaved by digit 0..9. Never used to select training or validation.')
    full_test = write_split(args.output, 'test-full', test_x, test_y, np.arange(10000), 'official-test-10000',
        'All 10000 official test images, in their original order; natural class frequencies.', optional=True)
    manifest = {
        'schema_version': 1, 'dataset': 'MNIST', 'asset_version': 'browser-v1',
        'description': 'Small stratified browser subset; not the full 50000-example A100 experiment.',
        'input': {'shape': [28, 28], 'pixels': 'uint8', 'range': [0, 255],
                  'recommended_normalization': 'Convert to float32 and divide by 255. No fitted preprocessing.'},
        'format': {
            'magic_ascii': 'GDMNIST1', 'header_bytes': 32, 'endianness': 'little',
            'header': ['0..7: ASCII magic GDMNIST1', '8: uint32 count', '12: uint32 rows=28',
                       '16: uint32 columns=28', '20: uint32 pixel offset=32',
                       '24: uint32 label offset', '28: uint32 original-index offset'],
            'body': ['count*784 uint8 pixels, images row-major', 'count uint8 labels in 0..9',
                     '0..3 zero padding bytes to 4-byte alignment', 'count uint32 original indices'],
            'index_semantics': 'Zero-based index within source_collection; train and validation refer to official-training-60000, while test and test-full refer to official-test-10000.',
            'compression': 'The .bin.gz file is the exact .bin payload wrapped in deterministic gzip; browsers may use DecompressionStream("gzip"). SHA256 values are given for both forms.',
        },
        'splits': {'train': train_record, 'validation': val_record, 'test': test_record},
        'optional_splits': {'test_full': full_test},
        'default_total_bytes': sum(r['bytes'] for r in [train_record, val_record, test_record]),
        'default_total_gzip_bytes': sum(r['gzip_bytes'] for r in [train_record, val_record, test_record]),
        'split_provenance': {
            'parent_split': 'torch.randperm(60000), first 50000 for train, final 10000 for validation',
            'split_seed': meta['split_seed'], 'train_pool_indices_sha256_int64_le': meta['train_indices_sha256'],
            'validation_pool_indices_sha256_int64_le': meta['val_indices_sha256'],
            'balanced_prefixes': 'Every split is interleaved 0..9, so any prefix with size divisible by 10 is exactly class balanced (except optional test-full). Shuffle training minibatches before each epoch.',
            'separation_verified': True,
            'test_selection_policy': 'Official test data was decoded only after train and validation selections were complete. Test labels were used only to construct the declared balanced test subset.',
            'archive_data_path': str(archive_rel), 'archive_data_sha256': digest(archive.read_bytes()),
            'archive_record_path': str(archive_rel.parent / 'result.json'), 'archive_record_sha256': digest(record_path.read_bytes()),
            'cached_source_directory': str(args.cache), 'files': meta['files'],
            'test_archive_equal_to_verified_idx': True,
        },
        'preview': {'file': 'preview.json', 'count': 20, 'bytes': (args.output / 'preview.json').stat().st_size,
                    'sha256': digest((args.output / 'preview.json').read_bytes())},
        'attribution': {
            'title': 'The MNIST Database of Handwritten Digits',
            'authors': ['Yann LeCun', 'Corinna Cortes', 'Christopher J.C. Burges'],
            'homepage': 'https://yann.lecun.org/exdb/mnist/index.html',
            'license': 'CC BY-SA 3.0', 'license_url': 'https://creativecommons.org/licenses/by-sa/3.0/',
            'license_documentation': 'https://keras.io/api/datasets/mnist/',
            'licensed_mirror_description': 'https://github.com/cvdfoundation/mnist',
            'modification': 'Selected balanced subsets from the existing experiment split and repacked unchanged grayscale images into browser binary files.',
            'reference': 'LeCun, Bottou, Bengio, and Haffner (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE 86(11):2278–2324. DOI:10.1109/5.726791.',
        },
        'evaluation_note': 'The default test subset is class balanced: its overall accuracy equally weights the ten digits. It is exploratory when repeatedly consulted. Do not tune hyperparameters or choose epochs on test accuracy; use validation.',
        'producer': {'script': 'build_dataset.py', 'script_sha256': digest(Path(__file__).read_bytes()),
                     'numpy': np.__version__, 'torch': torch.__version__},
    }
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps({'output': str(args.output), 'default_raw_bytes': manifest['default_total_bytes'],
                      'default_gzip_bytes': manifest['default_total_gzip_bytes'],
                      'optional_full_test_gzip_bytes': full_test['gzip_bytes'],
                      'manifest_sha256': digest((args.output / 'manifest.json').read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
