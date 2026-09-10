"""CPU-only integrity audit of the exact frozen 48-run half-drop cohort.

This checks bytes, provenance, and timing without evaluating prediction arrays.
Optional --download-authoritative retrieves server result.json files from Modal
Volume storage; it never invokes GPU functions or modifies remote artifacts.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import time

HERE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = HERE / 'results/halfdrop'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def authoritative_download(root, specs):
    import modal
    remote = modal.Volume.from_name('gradient-dissent-ciresan-20260909',
                                    environment_name='gradient-dissent-ciresan')

    def one(spec):
        path = root / 'raw' / spec['run_id'] / 'result.json'
        if not path.exists():
            temporary = path.with_name('result.json.partial')
            with temporary.open('wb') as stream:
                remote.read_file_into_fileobj(f"/runs/{spec['run_id']}/result.json", stream)
            temporary.replace(path)
        local = read(root / (spec['run_id'] + '.json'))
        local.pop('local_dispatch_elapsed_seconds', None)
        assert read(path) == local, f"Remote/local result mismatch: {spec['run_id']}"
        return {'run_id': spec['run_id'], 'file': str(path.relative_to(root)),
                'sha256': sha(path), 'bytes': path.stat().st_size}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, specs))
    save(root / 'authoritative-result-download.json', {
        'complete': True, 'gpu_calls': 0, 'files': records,
        'only_local_addition': 'local_dispatch_elapsed_seconds',
        'finished_unix': time.time()})


def verify(root, source_root=HERE):
    root = Path(root)
    specs = read(root / 'main-manifest.json')
    frozen = read(root / 'main-source-freeze.json')
    assert sha(root / 'main-manifest.json') == frozen['manifest_sha256']
    assert len(specs) == 48
    assert {(s['drop_mask'], s['seed']) for s in specs} == {
        (m, s) for m in range(16) for s in (201, 202, 203)}
    review = read(root / 'root-review.json')
    assert review['approved'] is True and review['reviewer'] == 'root'
    assert review['manifest_sha256'] == sha(root / 'main-manifest.json')
    assert review['source_freeze_sha256'] == sha(root / 'main-source-freeze.json')
    assert review['qualification_result_sha256'] == sha(root / 'halfdrop-qualification-v1.json')
    transport = read(root / 'main-transport-receipt.json')
    assert len(transport['records']) == 48
    assert {r['run_id'] for r in transport['records']} == {s['run_id'] for s in specs}
    assert all(r['status'] == 'complete' for r in transport['records'])
    authoritative = read(root / 'authoritative-result-download.json')
    assert authoritative['complete'] is True and len(authoritative['files']) == 48
    assert {r['run_id'] for r in authoritative['files']} == {s['run_id'] for s in specs}

    paired, files, weights, timings, submissions, completion_times = {}, [], [], [], [], []
    timing_keys = ('training_seconds', 'epoch_preparation_seconds', 'graph_setup_seconds',
                   'test_export_seconds', 'total_run_seconds', 'local_dispatch_elapsed_seconds')
    for spec in specs:
        run_id = spec['run_id']
        for name, expected in spec['source_sha256'].items():
            path = (source_root / name).resolve()
            assert path.is_relative_to(source_root.resolve()) and sha(path) == expected
        result = read(root / (run_id + '.json'))
        assert result['complete'] is True and result['status'] == 'complete'
        assert result['diverged'] is False and result['completed_epochs'] == 100
        assert result['spec'] == spec and result['source_sha256'] == spec['source_sha256']
        assert result['test_accessed'] is True
        assert 'A100' in result['hardware']['gpu']
        assert 38 * 1024**3 < result['hardware']['gpu_memory_bytes'] < 45 * 1024**3
        assert result['artifacts']['best.pt'] == result['artifacts']['checkpoint.pt']
        assert all(result['checkpoints'][name]['sha256'] == result['artifacts'][name]['sha256']
                   for name in ('best.pt', 'final.pt', 'checkpoint.pt'))
        validation = read(root / (run_id + '-validation.json'))
        assert validation == result['validation_view']
        assert validation == read(root / 'raw' / run_id / 'validation.json')
        assert validation['test_accessed'] is False
        assert [row['epoch'] for row in result['history']] == list(range(1, 101))
        provenance = {k: validation[k] for k in (
            'initial_parameters_sha256', 'initialization_rng_sha256',
            'executed_epoch_order_sha256', 'raw_mask_draws_sha256', 'dataset')}
        if spec['seed'] in paired:
            assert paired[spec['seed']] == provenance, f'Paired provenance mismatch: {run_id}'
        else:
            paired[spec['seed']] = provenance
        receipt = read(root / (run_id + '-download.json'))
        assert receipt['status'] == 'complete'
        assert {r['file'] for r in receipt['files']} == set(result['artifacts'])
        for record in receipt['files']:
            name = record['file']
            assert Path(name).name == name
            expected = result['artifacts'][name]
            assert record['sha256'] == expected['sha256'] and record['bytes'] == expected['bytes']
            if name.endswith('.pt'):
                assert record['status'] == 'retained_on_volume'
                weights.append({'run_id': run_id, **record})
            else:
                path = root / 'raw' / run_id / name
                assert record['status'] == 'downloaded_verified'
                assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
                files.append({'run_id': run_id, **record})
        server_path = root / 'raw' / run_id / 'result.json'
        without_dispatch = dict(result)
        without_dispatch.pop('local_dispatch_elapsed_seconds')
        assert read(server_path) == without_dispatch
        server_receipt = next(r for r in authoritative['files'] if r['run_id'] == run_id)
        assert sha(server_path) == server_receipt['sha256']
        assert server_path.stat().st_size == server_receipt['bytes']
        timings.append({'run_id': run_id, **{k: result[k] for k in timing_keys},
                        'telemetry_seconds': result['telemetry']['seconds']})
        dispatch = read(root / (run_id + '-dispatch.json'))
        submissions.append(dispatch['submitted_unix'])
        completion_times.append(dispatch['submitted_unix'] + result['local_dispatch_elapsed_seconds'])
    assert sum(r['file'].endswith('.npz') for r in files) == 96
    summary = {
        'complete': True, 'verified_unix': time.time(), 'run_count': 48,
        'source_manifest_sha256': sha(root / 'main-manifest.json'),
        'source_freeze_sha256': sha(root / 'main-source-freeze.json'),
        'downloaded_declared_files': len(files),
        'downloaded_declared_bytes': sum(r['bytes'] for r in files),
        'authoritative_result_files': len(authoritative['files']),
        'authoritative_result_bytes': sum(r['bytes'] for r in authoritative['files']),
        'prediction_npz_files': 96, 'weight_files_retained_on_volume': len(weights),
        'test_prediction_arrays_evaluated': False,
        'same_seed_initialization_order_and_raw_draw_hashes_exact': True,
        'paired_seeds': sorted(paired),
        'timing_totals_seconds': {k: sum(t[k] for t in timings)
                                  for k in (*timing_keys, 'telemetry_seconds')},
        'first_submission_to_last_local_result_seconds': max(completion_times) - min(submissions),
        'timing_caveat': 'Totals sum overlapping calls; they are not elapsed study wall time or invoice measurements. Training_seconds excludes telemetry, setup, and evaluation. Total_run_seconds is the trainer clock; local_dispatch_elapsed_seconds additionally includes dispatch/startup/return. Test-export_seconds is already included in total_run_seconds.',
        'per_run_timing': timings,
    }
    save(root / 'download-verification.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--download-authoritative', action='store_true')
    args = parser.parse_args()
    if args.download_authoritative:
        authoritative_download(args.root, read(args.root / 'main-manifest.json'))
    record = verify(args.root)
    print(json.dumps({k: record[k] for k in ('complete', 'run_count',
          'downloaded_declared_files', 'prediction_npz_files', 'authoritative_result_files')}))


if __name__ == '__main__':
    main()
