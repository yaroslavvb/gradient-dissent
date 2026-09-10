"""Three bounded, frozen-checkpoint inference jobs; never trains or logs to W&B.

Run with the root uv environment:
modal run -e gradient-dissent-ciresan experiments/ciresan_stochastic_depth/full_depth_app.py
The immutable manifest/freeze and CPU qualification must already exist.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import sys
import threading
import time

import modal

# Reuse both the cached runtime and the existing cumulative reservation ledger.
# Importing these definitions does not invoke any preparation/training function.
from telemetry_app import image, volume, reserve, ENVIRONMENT, ROOT

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'full-depth'
MANIFEST = OUT / 'manifest.json'
FREEZE = OUT / 'source-freeze.json'
GUARD_USD = 24.0
app = modal.App('gradient-dissent-ciresan-full-depth')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_jobs(jobs):
    if len(jobs) != 3 or {j.get('seed') for j in jobs} != {101, 102, 103}:
        raise ValueError('Exactly the three frozen seeds are required')
    for spec in jobs:
        if (spec.get('run_id') != f"full-depth-s{spec['seed']}-v1"
                or spec.get('timeout_seconds') != 180 or spec.get('gpu') != 'A100-40GB'
                or spec.get('stage') != 'inference' or not spec.get('source_sha256')):
            raise ValueError('Unexpected inference resources, identity or missing source freeze')
        expected = {f'{recipe}-{state}-s{spec["seed"]}'
                    for recipe in ('plain', 'residual', 'sd_constant', 'sd_annealed', 'residual_unit_dropout')
                    for state in ('selected', 'final')}
        models = spec.get('models', [])
        if (len(models) != 10 or {m.get('id') for m in models} != expected
                or spec.get('batch_size') != 2048):
            raise ValueError('Require all ten frozen checkpoint states and batch size 2048')


def verify_sources(spec, directory=HERE):
    directory = Path(directory).resolve()
    for name, expected in spec['source_sha256'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError(f'Frozen source mismatch: {name}')


def billing_snapshot():
    cost = float(modal.Environment.from_name(ENVIRONMENT).billing.summary().metered_cost)
    if not 0 <= cost < float('inf'):
        raise ValueError('Invalid metered cost')
    return {'environment': ENVIRONMENT, 'metered_cost_usd': cost,
            'queried_unix': time.time(), 'spend_guard_usd': GUARD_USD}


@app.function(image=image, gpu='A100-40GB', cpu=(2, 2), memory=(8192, 8192),
              timeout=180, startup_timeout=90, retries=0,
              max_containers=3, scaledown_window=2, volumes={ROOT: volume})
def evaluate(spec):
    sys.path.insert(0, '/opt/ciresan_stochastic_depth')
    verify_sources(spec, '/opt/ciresan_stochastic_depth')
    from layerdrop_hypotheses.full_depth import run
    result = run(spec, Path(ROOT), progress_commit=volume.commit)
    volume.commit()
    return json.dumps(result, allow_nan=False)


@app.local_entrypoint()
def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE.read_text())
    if sha(MANIFEST) != freeze['manifest_sha256']:
        raise ValueError('Manifest changed after source freeze')
    if freeze.get('cpu_qualification', {}).get('passed') is not True:
        raise ValueError('CPU qualification is required before paid dispatch')
    jobs = json.loads(MANIFEST.read_text())
    validate_jobs(jobs)
    for spec in jobs:
        verify_sources(spec)
    # Fail closed if the opening meter is unavailable. Existing ledger history
    # remains untouched; reserve appends three IDs and refuses reuse/retry.
    opening = billing_snapshot()
    if opening['metered_cost_usd'] >= GUARD_USD:
        raise RuntimeError('Environment metered cost already reached the $24 guard')
    opening_path = OUT / 'budget-opening.json'
    with opening_path.open('x') as f:
        json.dump(opening, f, indent=2)
        f.write('\n')
    reserve(jobs, 'full-depth-inference')
    ledger = HERE / 'results' / 'telemetry' / 'budget-ledger.json'
    write_json(OUT / 'reservation-receipt.json', {
        'shared_ledger': 'results/telemetry/budget-ledger.json',
        'shared_ledger_sha256_after_reservation': sha(ledger),
        'reserved_upper_usd': json.loads(ledger.read_text())['reserved_upper_usd'],
        'new_run_ids': [j['run_id'] for j in jobs],
        'no_reservations_released': True})
    calls = []
    call_lock = threading.Lock()
    stopping = threading.Event()
    budget_halted = threading.Event()

    def cancel_outstanding():
        with call_lock:
            pending = list(calls)
        for call in pending:
            try:
                call.cancel(terminate_containers=True)
            except Exception as exc:
                print('CANCEL STATUS', type(exc).__name__, flush=True)

    def monitor_budget():
        while not stopping.is_set():
            try:
                record = billing_snapshot()
                write_json(OUT / 'billing-latest.json', record)
                if record['metered_cost_usd'] >= GUARD_USD:
                    budget_halted.set()
                    cancel_outstanding()
                    print('BUDGET STOP: cancelling this phase at $24 metered cost.', flush=True)
                    return
            except Exception as exc:
                # Conservative reservations remain in force during a meter
                # outage; no further jobs or retries are introduced here.
                print('BILLING MONITOR unavailable:', type(exc).__name__, flush=True)
            stopping.wait(20)

    def execute(spec):
        started = time.time()
        result_path = OUT / (spec['run_id'] + '.json')
        call = None
        try:
            if budget_halted.is_set():
                raise RuntimeError('Budget monitor blocked dispatch')
            call = evaluate.spawn(spec)
            with call_lock:
                calls.append(call)
            # Close the small spawn/monitor race by checking again after append.
            if budget_halted.is_set():
                call.cancel(terminate_containers=True)
                raise RuntimeError('Budget monitor cancelled dispatch')
            write_json(OUT / (spec['run_id'] + '-dispatch.json'), {
                'run_id': spec['run_id'], 'call_id': call.object_id,
                'app_id': app.app_id, 'submitted_unix': started,
                'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()})
            print('SUBMITTED', spec['run_id'], call.object_id, flush=True)
            result = json.loads(call.get(timeout=330))
            result['local_dispatch_elapsed_seconds'] = time.time() - started
            write_json(result_path, result)
            print('COMPLETED', spec['run_id'], round(time.time()-started, 2), flush=True)
            return result
        except Exception as exc:
            if call is not None:
                try:
                    call.cancel(terminate_containers=True)
                except Exception:
                    pass
            result = {'run_id': spec['run_id'], 'spec': spec,
                      'error_type': type(exc).__name__, 'error': str(exc),
                      'local_dispatch_elapsed_seconds': time.time()-started}
            write_json(result_path, result)
            print('FAILED', spec['run_id'], type(exc).__name__, flush=True)
            return result

    monitor = threading.Thread(target=monitor_budget, daemon=True)
    monitor.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(execute, jobs))
    finally:
        stopping.set()
        monitor.join(timeout=2)
    closing = billing_snapshot()
    closing.update({'metered_opening_usd': opening['metered_cost_usd'],
                    'metered_increment_at_call_completion_usd': closing['metered_cost_usd']-opening['metered_cost_usd'],
                    'note': 'Preliminary meter at call completion; app shutdown and billing lag checked separately.'})
    write_json(OUT / 'billing-at-call-completion.json', closing)
    failures = sum('error' in r for r in results)
    print('FULL DEPTH COMPLETE', len(results), 'jobs;', failures, 'errors; no automatic retries.', flush=True)
    if failures:
        raise RuntimeError(f'{failures} inference jobs failed; reservations retained')
