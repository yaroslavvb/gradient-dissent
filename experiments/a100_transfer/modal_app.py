"""Bounded A100 experiments; all paid invocations pass the local reservation ledger.

Run with: modal run -e gradient-dissent-a100 experiments/a100_transfer/modal_app.py --stage prep
Use no deployment, schedule, nonpreemptible instances, or region premiums.
"""
from pathlib import Path
import concurrent.futures
import fcntl
import threading
import hashlib
import json
import sys
import time
import modal

HERE = Path(__file__).resolve().parent
ENVIRONMENT = 'gradient-dissent-a100'
VOLUME = 'gradient-dissent-a100-20260909'
ROOT = '/work'
# Published Modal rates checked 2026-09-09. Resource limits, not just requests.
RATE = .000694 + 2 * .0000131 + 8 * .00000222
CPU_RATE = 2 * .0000131 + 8 * .00000222
CAP = 50.0
RESERVATION_CEILING = 42.0  # Leave $8 for build, egress, billing lag and interruption overhead.
app = modal.App('gradient-dissent-a100-transfer')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.8.0', 'numpy==2.2.6', 'pyarrow==21.0.0', 'tiktoken==0.11.0')
         .env({'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'})
         .add_local_dir(HERE, remote_path='/opt/a100_transfer',
                        ignore=['results', 'data', 'checkpoints', '.venv', '__pycache__']))
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)

@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=1800,
              startup_timeout=90, retries=0, max_containers=1, scaledown_window=2,
              volumes={ROOT: volume})
def prepare():
    sys.path.insert(0, '/opt/a100_transfer')
    from language import prepare_language
    from vision import load_cifar100
    from parallel_cifar import download_archive
    start = time.time()
    data = Path(ROOT) / 'data'
    data.mkdir(parents=True, exist_ok=True)
    language = prepare_language(data / 'language')
    retrieval = download_archive(data / 'vision')
    vision = load_cifar100(data / 'vision', download=False)['metadata']
    vision['retrieval'] = retrieval
    result = {'language': language, 'vision': vision, 'seconds': time.time() - start}
    (Path(ROOT) / 'dataset-manifest.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result

@app.function(image=image, gpu='A100-80GB', cpu=(2, 2), memory=(8192, 8192),
              timeout=1800, startup_timeout=90, retries=0,
              max_containers=9, scaledown_window=2, volumes={ROOT: volume})
def train(spec):
    sys.path.insert(0, '/opt/a100_transfer')
    from train import run
    result = run(spec, Path(ROOT), progress_commit=volume.commit)
    volume.commit()
    return json.dumps(result, allow_nan=False)


def _reserve_unlocked(jobs, kind):
    """Reserve worst-case invocation+startup time BEFORE remote dispatch.

    Completed reservations are never released: failed attempts still count.
    A new invocation (including a manual retry) needs a new reservation.
    """
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    path = out / 'budget-ledger.json'
    ledger = json.loads(path.read_text()) if path.exists() else {
        'cap_usd': CAP, 'reservation_ceiling_usd': RESERVATION_CEILING,
        'pricing_url': 'https://modal.com/pricing', 'pricing_checked': '2026-09-09',
        'a100_80gb_usd_per_second': .000694, 'cpu_max_cores': 2, 'ram_max_gib': 8,
        'gpu_container_rate_upper_usd_per_second': RATE,
        'note': 'Reservations are conservative upper estimates, not invoices. Extra $8 buffer covers image build, egress and interruptions.',
        'reservations': []}
    new = []
    for job in jobs:
        charge_rate = CPU_RATE if kind=='prep' else (.000583+CPU_RATE if job.get('gpu') in ('A100','A100-40GB') else RATE)
        seconds = job['timeout_seconds'] + 130
        new.append({'id': job['run_id'], 'kind': kind, 'timeout_seconds': job['timeout_seconds'],
                    'startup_idle_allowance_seconds': 130, 'gpu':job.get('gpu','none' if kind=='prep' else 'A100-80GB'), 'reserved_rate_usd_per_second':charge_rate, 'reserved_usd': seconds * charge_rate,
                    'created_unix': time.time(), 'spec_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()})
    existing_ids = {r['id'] for r in ledger['reservations']}
    assert not existing_ids.intersection(r['id'] for r in new), 'Run IDs already reserved; inspect prior state before retrying.'
    assert len({r['id'] for r in new}) == len(new)
    total = sum(r['reserved_usd'] for r in ledger['reservations'] + new)
    if total > RESERVATION_CEILING:
        raise RuntimeError(f'Would reserve ${total:.2f}, above ${RESERVATION_CEILING:.2f} ceiling. No jobs submitted.')
    ledger['reservations'] += new
    ledger['reserved_upper_usd'] = total
    path.write_text(json.dumps(ledger, indent=2) + '\n')
    print(f'RESERVED aggregate ${total:.4f}; absolute ceiling ${CAP:.2f}', flush=True)

def reserve(jobs, kind):
    lock = HERE / 'results' / '.budget.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _reserve_unlocked(jobs, kind)

@app.local_entrypoint()
def launch(stage: str = 'prep', manifest: str = ''):
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    if stage == 'prep':
        reserve([{'run_id': 'data-preparation-v1', 'timeout_seconds': 1800}], 'prep')
        result = prepare.remote()
        (out / 'dataset-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        print('DATA READY', flush=True)
        return
    if stage not in ['pilot', 'tune', 'evaluate'] or not manifest:
        raise ValueError('Use --stage pilot|tune|evaluate --manifest <JSON list>')
    jobs = json.loads(Path(manifest).read_text())
    assert all(j['stage'] == stage for j in jobs)
    reserve(jobs, stage)
    calls = []
    stop_monitor = threading.Event()
    budget_halted = threading.Event()
    def monitor_budget():
        while not stop_monitor.is_set():
            try:
                cost = float(modal.Environment.from_name(ENVIRONMENT).billing.summary().metered_cost)
                (out / 'billing-latest.json').write_text(json.dumps({'metered_cost_usd':cost,'queried_unix':time.time()}, indent=2)+'\n')
                if cost >= 42:
                    budget_halted.set()
                    print('BUDGET STOP: environment metered usage reached $42; cancelling outstanding calls.', flush=True)
                    for call in list(calls): call.cancel(terminate_containers=True)
                    return
            except Exception as error:
                print('BILLING MONITOR unavailable:', type(error).__name__, flush=True)
            stop_monitor.wait(30)
    monitor = threading.Thread(target=monitor_budget, daemon=True)
    monitor.start()
    def execute(spec):
        start = time.time()
        path = out / (spec['run_id'] + '.json')
        try:
            if budget_halted.is_set(): raise RuntimeError('Budget monitor stopped further dispatch.')
            call = train.with_options(timeout=spec['timeout_seconds'],gpu=spec.get('gpu','A100-80GB')).spawn(spec)
            calls.append(call)
            print('SUBMITTED', spec['run_id'], call.object_id, flush=True)
            result = json.loads(call.get())
            result['local_dispatch_elapsed_seconds'] = time.time() - start
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print('COMPLETED', spec['run_id'], 'seconds', round(time.time()-start, 1), flush=True)
            return result
        except Exception as exc:
            result = {'run_id': spec['run_id'], 'spec': spec, 'error_type': type(exc).__name__,
                      'error': str(exc), 'local_dispatch_elapsed_seconds': time.time()-start}
            path.write_text(json.dumps(result, indent=2) + '\n')
            print('FAILED', spec['run_id'], type(exc).__name__, str(exc), flush=True)
            return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(execute, jobs))
    stop_monitor.set()
    monitor.join(timeout=2)
    errors = sum('error' in result for result in results)
    print('STAGE COMPLETE', stage, 'jobs', len(jobs), 'errors', errors, flush=True)
    if errors: raise RuntimeError(f'{errors} failed jobs; results and reservations retained, no automatic retries.')
