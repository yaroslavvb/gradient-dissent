"""Record a read-only final environment meter and stopped-app receipt.

Requires the complete local transport verification. Never launches or stops jobs,
changes reservations, or reads credentials. Run after the Modal CLI has exited.
"""
from pathlib import Path
import json
import subprocess
import sys
import time
import modal

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'results/halfdrop'
ENVIRONMENT = 'gradient-dissent-ciresan'


def read(path):
    return json.loads(path.read_text())


def main():
    verification = read(OUT / 'download-verification.json')
    assert verification['complete'] is True and verification['run_count'] == 48
    watchdog = read(OUT / 'watchdog-verification.json')
    assert watchdog['status'] == 'complete' and watchdog['verified_runs'] == 48
    specs = read(OUT / 'main-manifest.json')
    main_ids = {read(OUT / (s['run_id'] + '-dispatch.json'))['app_id'] for s in specs}
    qualification = read(OUT / 'qualification-budget-closing.json')
    owned_ids = main_ids | {qualification['app']['app_id']}
    apps = json.loads(subprocess.check_output([
        str(Path(sys.executable).with_name('modal')), 'app', 'list',
        '-e', ENVIRONMENT, '--json'], text=True))
    owned = [a for a in apps if a['app_id'] in owned_ids]
    assert {a['app_id'] for a in owned} == owned_ids, 'An owned app is absent from status readback'
    assert all(a['state'] == 'stopped' and str(a['tasks']) == '0' for a in owned), 'Owned app still active'
    meter = float(modal.Environment.from_name(ENVIRONMENT).billing.summary().metered_cost)
    assert 0 <= meter < float('inf')
    opening = read(OUT / 'preparation-opening.json')
    main_opening = read(OUT / 'main-budget-opening.json')
    ledger = read(HERE / 'results/telemetry/budget-ledger.json')
    record = {
        'complete': True, 'closed_unix': time.time(), 'environment': ENVIRONMENT,
        'main_runs_complete': 48, 'main_runs_expected': 48,
        'qualification_gpu_calls': 1, 'qualification_two_epoch_subruns': 2,
        'qualification_official_test_accessed': False,
        'all_owned_apps_stopped': True, 'owned_apps': owned,
        'all_returned_apps_stopped': all(a['state'] == 'stopped' and str(a['tasks']) == '0' for a in apps),
        'all_returned_apps': apps,
        'metered_opening_environment_usd': opening['metered_cost_usd'],
        'metered_main_opening_environment_usd': main_opening['metered_cost_usd'],
        'metered_closing_environment_usd': meter,
        'metered_environment_increment_main_snapshot_usd': meter - main_opening['metered_cost_usd'],
        'metered_environment_increment_qualification_and_main_snapshot_usd': meter - opening['metered_cost_usd'],
        'metered_qualification_increment_snapshot_usd': qualification['metered_increment_qualification_usd'],
        'main_new_reserved_upper_usd': opening['main_48_new_reserved_upper_usd'],
        'qualification_new_reserved_upper_usd': opening['one_qualification_call_upper_usd'],
        'shared_reserved_upper_usd': ledger['reserved_upper_usd'],
        'reservation_stop_guard_usd': 24.0, 'user_shared_cap_usd': 30.0,
        'no_reservations_released': True,
        'timing_totals_seconds': verification['timing_totals_seconds'],
        'first_submission_to_last_local_result_seconds': verification['first_submission_to_last_local_result_seconds'],
        'timing_definition': verification['timing_caveat'],
        'attribution_caveat': 'Before/after environment meter snapshots, not an itemized invoice for this cohort. Provider posting can lag; unseen concurrent activity can also affect environment differences. Opening and mid-study app snapshots found no other active apps, and the closing states are included. These differences must not be described as exact settled cohort charges.',
        'gpu_jobs_created_by_closing_check': 0,
    }
    with (OUT / 'budget-closing.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: record[k] for k in ('main_runs_complete', 'all_owned_apps_stopped',
        'all_returned_apps_stopped', 'metered_closing_environment_usd',
        'metered_environment_increment_qualification_and_main_snapshot_usd', 'shared_reserved_upper_usd')}))


if __name__ == '__main__':
    main()
