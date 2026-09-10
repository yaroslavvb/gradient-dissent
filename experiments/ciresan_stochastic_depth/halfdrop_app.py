"""Bounded half-drop study; every call receives a cumulative budget reservation.

modal run -e gradient-dissent-ciresan experiments/ciresan_stochastic_depth/halfdrop_app.py --stage qualification
Main uses --stage main, after a root-review.json receipt approves frozen inputs.
No W&B mutations, schedules, deployments, or automatic experiment retries.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import sys
import threading
import time

import modal

if not modal.is_local():
    sys.path.insert(0, '/opt/ciresan_stochastic_depth')
from telemetry_app import image, volume, reserve, ENVIRONMENT, ROOT, VOLUME

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/halfdrop'
GUARD_USD = 24.0
app = modal.App('gradient-dissent-ciresan-halfdrop')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_sources(spec, directory=HERE):
    directory = Path(directory).resolve()
    if not spec.get('source_sha256'):
        raise ValueError('A source freeze is required')
    for name, expected in spec['source_sha256'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError(f'Frozen source mismatch: {name}')


def validate_jobs(jobs, stage):
    if stage == 'qualification':
        if len(jobs) != 1 or jobs[0].get('run_id') != 'halfdrop-qualification-v1':
            raise ValueError('Exactly one bounded qualification call is required')
        parent = jobs[0]
        if parent.get('stage') != 'pilot' or parent.get('kind') != 'qualification':
            raise ValueError('Qualification may not open official test data')
        subs = parent.get('subjobs', [])
        if len(subs) != 2 or {s.get('drop_mask') for s in subs} != {0, 15}:
            raise ValueError('Require both fixed two-epoch qualification extremes')
        for s in subs:
            if s.get('stage') != 'pilot' or s.get('epochs') != 2 or s.get('seed') != 201:
                raise ValueError('Unexpected qualification subrun')
    elif stage == 'main':
        expected = {(mask, seed) for mask in range(16) for seed in (201, 202, 203)}
        if len(jobs) != 48 or {(j.get('drop_mask'), j.get('seed')) for j in jobs} != expected:
            raise ValueError('Main requires the exact 48 paired subset runs')
        for j in jobs:
            if (j.get('stage') != 'evaluate' or j.get('epochs') != 100
                    or j.get('run_id') != f"halfdrop-m{j['drop_mask']:02d}-s{j['seed']}-v1"
                    or j.get('lr') != .01 or j.get('batch_size') != 64 or j.get('momentum') != .9):
                raise ValueError('Unexpected main protocol setting')
    else:
        raise ValueError('Use qualification or main')
    for j in jobs:
        if j.get('gpu') != 'A100-40GB' or j.get('timeout_seconds') != 180 or not j.get('source_sha256'):
            raise ValueError('Unexpected or unfrozen paid resource request')


def billing_snapshot():
    cost = float(modal.Environment.from_name(ENVIRONMENT).billing.summary().metered_cost)
    if not 0 <= cost < float('inf'):
        raise ValueError('Invalid metered cost')
    return {'environment': ENVIRONMENT, 'metered_cost_usd': cost,
            'queried_unix': time.time(), 'spend_guard_usd': GUARD_USD}


def download_artifacts(result):
    """CPU-only transport; retain weights on the volume, verify other raw files."""
    source_run = result['run_id']
    target = OUT/'raw'/source_run
    target.mkdir(parents=True, exist_ok=True)
    remote = modal.Volume.from_name(VOLUME, environment_name=ENVIRONMENT)
    records = []
    for name, info in result.get('artifacts', {}).items():
        if Path(name).name != name:
            raise ValueError('Artifact filename escapes its run directory')
        row = {'file': name, 'sha256': info['sha256'], 'bytes': info['bytes'],
               'volume_path': f'/runs/{source_run}/{name}'}
        if Path(name).suffix == '.pt':
            row['status'] = 'retained_on_volume'
        else:
            path = target/name
            if not path.exists():
                temporary = path.with_name(name+'.partial')
                with temporary.open('wb') as stream:
                    remote.read_file_into_fileobj(row['volume_path'], stream)
                temporary.replace(path)
            if sha(path) != info['sha256'] or path.stat().st_size != info['bytes']:
                raise AssertionError('Downloaded artifact hash/size mismatch')
            row['status'] = 'downloaded_verified'
        records.append(row)
    record = {'run_id': source_run, 'status': 'complete', 'files': records,
              'transport_finished_unix': time.time(), 'weights_downloaded': False,
              'volume': VOLUME, 'environment': ENVIRONMENT}
    write_json(OUT/(source_run+'-download.json'), record)
    return record


@app.function(image=image, gpu='A100-40GB', cpu=(2, 2), memory=(8192, 8192),
              timeout=180, startup_timeout=90, retries=0, max_containers=4,
              scaledown_window=2, volumes={ROOT: volume})
def train(spec):
    verify_sources(spec, '/opt/ciresan_stochastic_depth')
    from halfdrop.trainer import run
    if spec.get('kind') == 'qualification':
        subresults = []
        for sub in spec['subjobs']:
            verify_sources(sub, '/opt/ciresan_stochastic_depth')
            result = run(sub, Path(ROOT), progress_commit=volume.commit)
            if result.get('dataset', {}).get('test_included') is not False:
                raise AssertionError('Qualification returned an included official-test dataset')
            if result.get('test_accessed') is not False:
                raise AssertionError('Qualification must affirm no official-test access')
            if result.get('selected_test') is not None or result.get('final_test') is not None:
                raise AssertionError('Qualification must not score the official test set')
            subresults.append(result)
        result = {'run_id': spec['run_id'], 'spec': spec, 'stage': 'pilot',
                  'subresults': subresults, 'official_test_accessed': False,
                  'complete': all(r.get('complete') is True for r in subresults),
                  'status': 'complete' if all(r.get('complete') is True for r in subresults) else 'failed'}
    else:
        result = run(spec, Path(ROOT), progress_commit=volume.commit)
    volume.commit()
    return json.dumps(result, allow_nan=False)


@app.local_entrypoint()
def launch(stage: str = 'qualification'):
    if stage not in ('qualification', 'main'):
        raise ValueError('Use qualification or main')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / f'{stage}-manifest.json'
    frozen = OUT / f'{stage}-source-freeze.json'
    freeze = json.loads(frozen.read_text())
    if sha(manifest) != freeze['manifest_sha256']:
        raise ValueError('Manifest changed after freeze')
    if freeze.get('cpu_qualification', {}).get('passed') is not True:
        raise ValueError('CPU qualification required before paid dispatch')
    jobs = json.loads(manifest.read_text())
    validate_jobs(jobs, stage)
    for spec in jobs:
        verify_sources(spec)
    if stage == 'main':
        review = json.loads((OUT/'root-review.json').read_text())
        if (review.get('approved') is not True or review.get('reviewer') != 'root'
                or review.get('manifest_sha256') != sha(manifest)
                or review.get('source_freeze_sha256') != sha(frozen)
                or review.get('qualification_result_sha256') != sha(OUT/'halfdrop-qualification-v1.json')):
            raise ValueError('Root must review the exact frozen main protocol and qualification')
    opening = billing_snapshot()
    if opening['metered_cost_usd'] >= GUARD_USD:
        raise RuntimeError('Metered usage has reached the $24 guard')
    with (OUT/f'{stage}-budget-opening.json').open('x') as f:
        json.dump(opening, f, indent=2); f.write('\n')
    reserve(jobs, f'halfdrop-{stage}')
    ledger = HERE/'results/telemetry/budget-ledger.json'
    write_json(OUT/f'{stage}-reservation-receipt.json', {
        'shared_ledger': 'results/telemetry/budget-ledger.json',
        'shared_ledger_sha256_after_reservation': sha(ledger),
        'reserved_upper_usd': json.loads(ledger.read_text())['reserved_upper_usd'],
        'new_run_ids': [j['run_id'] for j in jobs], 'no_reservations_released': True})
    calls = []; call_lock = threading.Lock(); transport_futures = []
    transport_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    stopping = threading.Event(); budget_halted = threading.Event()

    def cancel_outstanding():
        with call_lock:
            pending = list(calls)
        for call in pending:
            try:
                call.cancel(terminate_containers=True)
            except Exception as exc:
                print('CANCEL STATUS', type(exc).__name__, flush=True)

    def monitor_budget():
        # Opening meter was just checked; avoid an immediate duplicate request.
        while not stopping.wait(40):
            try:
                record = billing_snapshot()
                write_json(OUT/f'{stage}-billing-latest.json', record)
                if record['metered_cost_usd'] >= GUARD_USD:
                    budget_halted.set(); cancel_outstanding()
                    print('BUDGET STOP: cancelling this study at $24 metered cost.', flush=True)
                    return
            except Exception as exc:
                print('BILLING MONITOR unavailable; reservations remain in force:', type(exc).__name__, flush=True)

    def execute(spec):
        started = time.time(); call = None
        path = OUT/(spec['run_id']+'.json')
        try:
            if budget_halted.is_set():
                raise RuntimeError('Budget monitor blocked further dispatch')
            call = train.spawn(spec)
            with call_lock:
                calls.append(call)
            if budget_halted.is_set():
                call.cancel(terminate_containers=True)
                raise RuntimeError('Budget monitor cancelled dispatch')
            write_json(OUT/(spec['run_id']+'-dispatch.json'), {
                'run_id': spec['run_id'], 'call_id': call.object_id, 'app_id': app.app_id,
                'submitted_unix': started,
                'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()})
            print('SUBMITTED', spec['run_id'], call.object_id, flush=True)
            result = json.loads(call.get(timeout=330))
            result['local_dispatch_elapsed_seconds'] = time.time()-started
            if isinstance(result.get('validation_view'), dict):
                write_json(OUT/(spec['run_id']+'-validation.json'), result['validation_view'])
            write_json(path, result)
            print('COMPLETED', spec['run_id'], round(time.time()-started, 2), flush=True)
            for payload in result.get('subresults', [result]):
                if payload.get('artifacts'):
                    transport_futures.append(transport_pool.submit(download_artifacts, payload))
            return result
        except Exception as exc:
            if call is not None:
                try:
                    call.cancel(terminate_containers=True)
                except Exception:
                    pass
            result = {'run_id': spec['run_id'], 'spec': spec, 'error_type': type(exc).__name__,
                      'error': str(exc), 'local_dispatch_elapsed_seconds': time.time()-started}
            write_json(path, result)
            print('FAILED', spec['run_id'], type(exc).__name__, flush=True)
            return result

    monitor = threading.Thread(target=monitor_budget, daemon=True); monitor.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(execute, jobs))
    finally:
        stopping.set(); monitor.join(timeout=2)
    transport_records = []
    for future in transport_futures:
        try:
            record = future.result()
            transport_records.append({'run_id': record['run_id'], 'status': record['status'],
                                      'artifact_count': len(record['files'])})
        except Exception as exc:
            transport_records.append({'status': 'failed', 'error_type': type(exc).__name__})
    transport_pool.shutdown(wait=True)
    write_json(OUT/f'{stage}-transport-receipt.json', {'records': transport_records,
               'note': 'Transport failures require only a CPU download retry, never a repeated GPU experiment.'})
    try:
        closing = billing_snapshot()
        closing.update({'metered_opening_usd': opening['metered_cost_usd'],
                        'metered_increment_at_call_completion_usd': closing['metered_cost_usd']-opening['metered_cost_usd'],
                        'note': 'Preliminary meter; app shutdown and billing lag checked separately.'})
    except Exception as exc:
        closing = {'metered_read_status': 'unavailable', 'error_type': type(exc).__name__,
                   'note': 'Read-only closing meter must be retried after app shutdown.'}
    write_json(OUT/f'{stage}-billing-at-call-completion.json', closing)
    failures = sum('error' in r or r.get('complete') is not True for r in results)
    print('HALFDROP COMPLETE', stage, len(results), 'jobs;', failures, 'errors.', flush=True)
    if failures:
        raise RuntimeError(f'{failures} jobs failed; reservations retained; no automatic retries')
