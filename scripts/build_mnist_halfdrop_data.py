#!/usr/bin/env python3
"""Export completed, verified half-drop statistics and six lazy gallery files.

No training, cloud use, HTML generation, or test-driven model selection.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO/'experiments/ciresan_stochastic_depth/results/halfdrop'
DEFAULT_OUT = REPO/'docs/ciresan-stochastic-depth/training-50'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Invalid JSON number: '+value)))


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite export value')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def verified_file(record, repo):
    path = Path(record['path'])
    path = path if path.is_absolute() else repo/path
    require(path.is_file() and sha(path) == record['sha256'], 'Changed or missing evidence: '+str(path))
    return path


def build(results=DEFAULT_RESULTS, repo=REPO):
    results, repo = Path(results).resolve(), Path(repo).resolve()
    source = results/'analysis.json'
    data = read_json(source)
    finite_tree(data)
    require(data['status'] == 'complete' and data['verification']['complete_runs'] == 48, 'Only complete48-run analysis can be exported')
    require(data['seeds'] == [201, 202, 203] and len(data['models']) == 32, 'Wrong seed/endpoint panel')
    expected = {(endpoint, mask) for endpoint in ('selected', 'final') for mask in range(16)}
    require({(r['endpoint'], r['drop_mask']) for r in data['models']} == expected, 'Incomplete aggregate subset panel')
    require(len(data['by_k']) == 10 and len(data['paths']) == 24 and len(data['edges']) == 64, 'Missing count/path/edge records')
    require(sum(p['validation_selected'] for p in data['paths']) == 1, 'Exactly one validation-frozen path required')
    provenance = data['provenance']
    for key in ('manifest', 'source_freeze', 'order_manifest', 'protocol', 'analyzer', 'paired_errors'):
        verified_file(provenance[key], repo)
    for record in provenance['raw_files']:
        verified_file(record, repo)
    for model in data['models']:
        require([r['seed'] for r in model['seeds']] == [201, 202, 203], 'Seed row order changed')
        require(model['metrics']['accuracy']['n'] == 3 and model['metrics']['ce']['n'] == 3, 'Incomplete seed statistics')
        require(model['inference_affine_fraction'] == 1, 'Study must use full-depth evaluation')
        require(len(model['per_class']) == 10, 'Missing class rows')
    gallery_text = {}
    require(set(data['gallery_files']) == {f'{s}-{e}' for s in (201, 202, 203) for e in ('selected', 'final')}, 'Missing lazy gallery files')
    for key, name in data['gallery_files'].items():
        require(Path(name).name == name, 'Gallery filename escaped output directory')
        path = verified_file(provenance['gallery_files'][name], repo)
        gallery = read_json(path)
        require(f'{gallery["seed"]}-{gallery["endpoint"]}' == key and len(gallery['models']) == 16, 'Gallery identity/panel mismatch')
        for model in gallery['models']:
            require(100 <= len(model['examples']) <= 140, 'Gallery exceeds frozen size bounds')
            for example in model['examples']:
                require(str(example['index']) in gallery['images'], 'Missing gallery pixels')
                require(len(example['probs']) == len(example['baseline_probs']) == 10, 'Expected ten-class paired probabilities')
                for field in ('probs', 'baseline_probs'):
                    require(all(0 <= v <= 1 for v in example[field]) and abs(sum(example[field])-1) < 1e-10, 'Invalid softmax vector')
        finite_tree(gallery)
        gallery_text[name] = json.dumps(gallery, separators=(',', ':'), allow_nan=False)+'\n'
    output = copy.deepcopy(data)
    output['provenance']['analysis'] = {'path': str(source.relative_to(repo)) if source.is_relative_to(repo) else str(source), 'sha256': sha(source)}
    output['provenance']['exporter'] = {'path': 'scripts/build_mnist_halfdrop_data.py', 'sha256': sha(__file__)}
    # Only read already verified W&B URLs. This exporter never authenticates or uploads.
    registry = results/'wandb-upload-manifest.json'
    links = {}
    if registry.is_file():
        record = read_json(registry)
        valid_ids = {s['run_id'] for m in output['models'] for s in m['seeds']}
        for run in record.get('runs', []):
            url = run.get('verified_wandb_url')
            if run.get('status') != 'verified' or not url:
                continue
            parsed = urlparse(url)
            require(parsed.scheme == 'https' and parsed.netloc == 'wandb.ai' and '/runs/' in parsed.path, 'Unexpected W&B destination')
            scientific_id = run['scientific_run_id']
            require(scientific_id in valid_ids, 'W&B link references an unplanned scientific run')
            links[scientific_id] = url
        output['provenance']['wandb_upload_manifest'] = {'path': str(registry.relative_to(repo)) if registry.is_relative_to(repo) else str(registry), 'sha256': sha(registry)}
    output['wandb_runs'] = links
    return output, gallery_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    data, galleries = build(args.results_dir, args.repo)
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False)
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        files = {'data.json': encoded+'\n', 'data.js': 'window.HALFDROP_DATA = '+encoded+';\n', **galleries}
        # Complete serialization/validation before replacing any public asset.
        pending = []
        for name, content in files.items():
            destination = args.output_dir/name
            temporary = destination.with_name(destination.name+'.tmp')
            temporary.write_text(content)
            pending.append((temporary, destination))
        for temporary, destination in pending:
            temporary.replace(destination)
    print(json.dumps({'status': 'verified', 'runs': 48, 'aggregate_models': 32, 'paths': 24, 'galleries': len(galleries), 'wandb_links': len(data['wandb_runs']), 'written': not args.check_only}))


if __name__ == '__main__':
    main()
