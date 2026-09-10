"""Move only this session's verified-source scalar imports to the requested project.

The authenticated server schema defines MoveRunsInput and moveRuns. This is
a one-time migration, not training. Persist the returned task before polling;
never resubmit an uncertain mutation. Final source/history verification is
performed by upload_wandb.py after switching its destination.
"""
import contextlib
import io
import json
from pathlib import Path

import requests
import wandb

from upload_wandb import REGISTRY, ENTITY, DEFAULT_GROUP, atomic_json, now

SOURCE = 'gradient-dissent-ciresan'
DESTINATION = 'gradient-dissent'
RECEIPT = REGISTRY.with_name('wandb-project-migration.json')


def main():
    if RECEIPT.exists():
        raise RuntimeError('Migration receipt already exists; reconcile before any retry')
    if REGISTRY.with_suffix('.lock').exists():
        raise RuntimeError('Stop the upload driver before changing its destination')
    registry = json.loads(REGISTRY.read_text())
    assert registry['project'] == SOURCE and registry['entity'] == ENTITY
    expected = {r['wandb_run_id']: r for r in registry['runs']}
    filters = {'$and': [{'name': {'$in': list(expected)}}, {'group': DEFAULT_GROUP}]}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api = wandb.Api(timeout=20)
        source_runs = list(api.runs(f'{ENTITY}/{SOURCE}', filters=filters, per_page=50))
        destination_runs = list(api.runs(f'{ENTITY}/{DESTINATION}', filters=filters, per_page=50))
    if destination_runs:
        raise RuntimeError('Destination already contains matching IDs; reconcile before moving')
    if not source_runs:
        raise RuntimeError('No matching source runs')
    ids = []
    for run in source_runs:
        record = expected[run.id]
        assert list(run.path) == [ENTITY, SOURCE, run.id]
        assert run.name == record['scientific_run_id'] and run.group == DEFAULT_GROUP
        assert run.state == 'finished', 'Do not move active imports'
        assert run.summary['telemetry/source_sha256'] == record['source_sha256']
        assert run.summary['telemetry/content_sha256'] == record['content_sha256']
        assert run.summary['telemetry/history_rows'] == record['history_rows_expected']
        ids.append(run.id)
    inputs = {'sourceEntityName': ENTITY, 'sourceProjectName': SOURCE,
              'destinationEntityName': ENTITY, 'destinationProjectName': DESTINATION,
              'filters': json.dumps({'$and': [{'name': {'$in': sorted(ids)}}, {'group': DEFAULT_GROUP}]})}
    query = 'mutation MoveTelemetry($input:MoveRunsInput!) { moveRuns(input:$input) { task { id name state progress } } }'
    receipt = {'status': 'submitting', 'requested_utc': now(), 'source_project': SOURCE,
               'destination_project': DESTINATION, 'entity': ENTITY, 'run_ids': sorted(ids),
               'source_metadata_preflight_verified': True, 'mutation': query, 'variables': inputs,
               'note': 'Only these exact scalar-import IDs were selected. No model artifacts were uploaded.'}
    atomic_json(RECEIPT, receipt)
    # Credentials go only to their intended API endpoint, never the receipt.
    response = requests.post('https://api.wandb.ai/graphql', auth=('api', api.api_key),
                             json={'query': query, 'variables': {'input': inputs}}, timeout=30)
    response.raise_for_status()
    result = response.json()
    if result.get('errors'):
        receipt['status'] = 'api_error_requires_reconciliation'
        atomic_json(RECEIPT, receipt)
        raise RuntimeError('Move API returned errors; inspect state before retrying')
    receipt.update(status='submitted', task=result['data']['moveRuns']['task'])
    atomic_json(RECEIPT, receipt)
    print(json.dumps({'status': receipt['status'], 'run_count': len(ids), 'task': receipt['task']}))


if __name__ == '__main__':
    main()
