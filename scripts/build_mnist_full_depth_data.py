#!/usr/bin/env python3
"""Export verified, completed six-affine MNIST inference results for the page.

Only consumes recorded results. Missing runs, changed frozen sources/galleries,
invalid abstention fields, or disagreement with any embedded old16-mask result
prevent both output files from being written. No model inference or cloud calls.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('experiments/ciresan_stochastic_depth')
RESULTS = EXPERIMENT/'results/full-depth'
PAGE = Path('docs/ciresan-stochastic-depth/hypotheses')
RECIPES = ('residual', 'sd_constant', 'sd_annealed', 'residual_unit_dropout', 'plain')
SEEDS = (101, 102, 103)
LAYER_MACS = (1960000, 5000000, 3000000, 1500000, 500000, 5000)
FULL_MACS = sum(LAYER_MACS)
CE_ATOL = 3e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonstandard JSON constant {value} in {path}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def finite_tree(value, path='root'):
    if isinstance(value, float):
        require(math.isfinite(value), f'Nonfinite value: {path}')
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f'{path}.{key}')
    elif isinstance(value, list):
        for i, child in enumerate(value):
            finite_tree(child, f'{path}[{i}]')


def fraction(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f'Invalid fraction: {label}')


def close(a, b, atol, label):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol,
            f'{label}: {a} != {b} (absolute tolerance {atol})')


def unique_by(items, key, label):
    require(isinstance(items, list), f'{label} must be a list')
    result = {}
    for item in items:
        require(item[key] not in result, f'Duplicate {label}: {item[key]}')
        result[item[key]] = item
    return result


def check_mask(row, dense_accuracy, class_counts):
    mask = row['id']; covered = mask >= 32
    require(type(mask) is int and 0 <= mask < 64, 'Mask ID outside0..63')
    selected = sum(c for j, c in enumerate(LAYER_MACS) if mask & (1 << j))
    executed = selected if covered else 0
    require(row['bits'] == [(mask >> j) & 1 for j in range(6)], f'Mask bits differ: {mask}')
    require(row['retained'] == row['selected_affines'] == mask.bit_count(), 'Selected affine count differs')
    require(row['executed_affines'] == (mask.bit_count() if covered else 0), 'Executed affine count differs')
    require(row['raw_macs'] == executed and row['nominal_selected_macs'] == selected, 'Affine MAC count differs')
    close(row['cost'], executed/FULL_MACS, 1e-15, 'Executed MAC fraction')
    close(row['nominal_selected_cost'], selected/FULL_MACS, 1e-15, 'Selected MAC fraction')
    require(row['n'] == 10000 and row['predictions_emitted'] == (10000 if covered else 0), 'Prediction denominator differs')
    require(row['coverage'] == int(covered) and row['abstentions'] == (0 if covered else 10000), 'Coverage differs')
    fraction(row['correct_output_rate'], 'correct_output_rate')
    require(type(row['correct_outputs']) is int and 0 <= row['correct_outputs'] <= 10000, 'Invalid correct count')
    close(row['correct_output_rate'], row['correct_outputs']/10000, 1e-15, 'Correct-output count/rate')
    fraction(row['harm'], 'harm'); fraction(row['repair'], 'repair')
    close(dense_accuracy-row['harm']+row['repair'], row['correct_output_rate'], 1e-12, 'Harm/repair identity')
    if covered:
        require(row['accuracy'] == row['correct_output_rate'], 'Covered accuracy must equal correct-output rate')
        require(type(row['ce']) in (int, float) and row['ce'] >= 0, 'Covered CE must be nonnegative')
    else:
        require(row['accuracy'] is None and row['ce'] is None, 'Abstention accuracy/CE must be null')
        require(row['correct_output_rate'] == 0 and row['repair'] == 0, 'Abstention cannot emit a correct answer')
    classes = unique_by(row['per_class'], 'digit', 'per-class row')
    require(set(classes) == set(range(10)), 'All ten true classes must be present')
    for digit, cls in classes.items():
        require(cls['n'] == class_counts[digit] and cls['coverage'] == int(covered), 'Class denominator/coverage differs')
        fraction(cls['correct_output_rate'], 'class correct-output rate')
        fraction(cls['dense_accuracy'], 'class dense accuracy')
        fraction(cls['harm'], 'class harm'); fraction(cls['repair'], 'class repair')
        close(cls['dense_accuracy']-cls['harm']+cls['repair'], cls['correct_output_rate'], 1e-12, 'Class harm/repair identity')
        if covered:
            require(cls['accuracy'] == cls['correct_output_rate'], 'Covered class accuracy differs')
            require(type(cls['ce']) in (int, float) and cls['ce'] >= 0, 'Covered class CE must be nonnegative')
        else:
            require(cls['accuracy'] is None and cls['ce'] is None and cls['correct_output_rate'] == 0,
                    'Abstention class fields are invalid')
    close(sum(c['n']*c['correct_output_rate'] for c in classes.values())/10000,
          row['correct_output_rate'], 1e-12, 'Per-class weighted correct-output rate')


def check_model(model, original, frozen, sources):
    finite_tree(model)
    for field in ('id', 'recipe', 'state', 'seed', 'epoch', 'checkpoint_sha256'):
        require(model[field] == frozen[field] == original[field], f'Model provenance mismatch: {model["id"]}/{field}')
    require(model['frozen_model'] == frozen, f'Frozen gallery metadata differs: {model["id"]}')
    require(model['source_sha256'] == sources, 'Per-model source hashes differ')
    require(model['n'] == original['n'] == 10000, 'Expected full official test panel')
    require(model['dense_accuracy'] == original['dense_accuracy'], 'Dense accuracy differs from existing page')
    direct = model['direct_forward_parity']
    require(direct['all_kept_bitwise_equal'] is True and direct['n'] == 10000 and direct['batch_size'] == 2048,
            'Missing all-kept bitwise forward verification')
    parity = model['archived_panel_parity']
    for key in ('passed', 'all_16_predictions_exact', 'labels_exact', 'dense_source_error_indices_exact'):
        require(parity[key] is True, f'Missing archival check: {key}')
    require(parity['embedded_masks'] == list(range(33, 64, 2)) and parity['prediction_comparisons'] == 160000,
            'Incorrect original-mask embedding/number of comparisons')
    require(parity['mean_ce_atol'] == CE_ATOL and parity['max_abs_mean_ce_difference'] <= CE_ATOL
            and parity['dense_source_mean_ce_difference'] <= CE_ATOL, 'CE verification/tolerance changed')
    masks = unique_by(model['masks'], 'id', 'extended mask')
    old_masks = unique_by(original['masks'], 'id', 'original mask')
    require(set(masks) == set(range(64)) and set(old_masks) == set(range(16)), 'Incomplete mask panel')
    require([r['id'] for r in model['masks']] == list(range(64)), 'Mask rows must be in integer order')
    class_counts = {c['digit']: c['n'] for c in old_masks[15]['per_class']}
    require(sum(class_counts.values()) == 10000, 'Original class counts do not cover the test set')
    for row in model['masks']:
        check_mask(row, model['dense_accuracy'], class_counts)
    for old_id in range(16):
        new, old = masks[33+2*old_id], old_masks[old_id]
        require(new['accuracy'] == old['accuracy'], f'Embedded accuracy differs: {model["id"]}/mask{old_id}')
        close(new['ce'], old['ce'], CE_ATOL, 'Embedded mean CE')
        close(new['cost'], old['cost'], 1e-15, 'Embedded mask cost')
        for field in ('harm', 'repair'):
            require(new[field] == old[field], f'Embedded {field} differs')
        for cls, prior in zip(new['per_class'], old['per_class']):
            for field in ('digit', 'n', 'accuracy', 'harm', 'repair'):
                require(cls[field] == prior[field], f'Embedded per-class {field} differs')
    examples = unique_by(model['examples'], 'index', 'extended example')
    old_examples = unique_by(original['examples'], 'index', 'original example')
    require(list(examples) == frozen['gallery_indices'] == list(old_examples), 'Existing gallery indices/order changed')
    for index, example in examples.items():
        old = old_examples[index]
        require(example['label'] == old['label'], 'Gallery true label differs')
        require(len(example['pred']) == len(example['ce']) == 64, 'Gallery requires all64 mask predictions/CE')
        for mask, (pred, ce) in enumerate(zip(example['pred'], example['ce'])):
            if mask < 32:
                require(pred == -1 and ce is None, 'Gallery abstention must use pred=-1 and CE=null')
            else:
                require(type(pred) is int and 0 <= pred < 10 and type(ce) in (int, float) and ce >= 0,
                        'Invalid covered gallery prediction/CE')
        for old_id in range(16):
            require(example['pred'][33+2*old_id] == old['pred'][old_id],
                    f'Embedded gallery prediction differs: {model["id"]}/example{index}/mask{old_id}')
    return copy.deepcopy(model)


def build(repo=REPO, manifest_file=RESULTS/'manifest-v2.json', freeze_file=RESULTS/'source-freeze-v2.json'):
    repo = Path(repo).resolve(); result_dir = repo/RESULTS
    manifest_path = repo/manifest_file; freeze_path = repo/freeze_file
    gallery_path = result_dir/'gallery-freeze.json'; existing_path = repo/PAGE/'data.json'
    freeze = read_json(freeze_path); gallery = read_json(gallery_path); manifest = read_json(manifest_path)
    existing = read_json(existing_path)
    require(sha(manifest_path) == freeze['manifest_sha256'], 'Manifest changed after freeze')
    require(sha(gallery_path) == freeze['gallery_freeze_sha256'], 'Gallery freeze changed')
    require(sha(existing_path) == gallery['source_site_sha256'], 'Original data.json changed after gallery freeze')
    require(freeze['cpu_qualification']['passed'] is True, 'CPU qualification missing')
    sources = freeze['source_sha256']
    for name, digest in sources.items():
        require(sha(repo/EXPERIMENT/name) == digest, f'Frozen executed source changed: {name}')
    expected_ids = {f'{r}-{state}-s{s}' for r in RECIPES for state in ('selected', 'final') for s in SEEDS}
    old_models = unique_by(existing['models'], 'id', 'existing model')
    frozen_models = unique_by(gallery['models'], 'id', 'frozen model')
    require(set(old_models) == set(frozen_models) == expected_ids, 'Expected the original30 states')
    specs = unique_by(manifest, 'seed', 'dispatch seed')
    require(set(specs) == set(SEEDS), 'Expected all three frozen jobs')
    normalized = {}; raw_jobs = []; datasets = []
    for seed in SEEDS:
        spec = specs[seed]
        require(Path(spec['run_id']).name == spec['run_id'], 'Invalid run filename in frozen manifest')
        path = manifest_path.parent/(spec['run_id']+'.json')
        raw = read_json(path); finite_tree(raw)
        require(raw.get('status') == 'complete' and raw.get('passed') is True and 'error' not in raw,
                f'Incomplete or failed result: {path.name}')
        require(raw['run_id'] == spec['run_id'] and raw['spec'] == spec, 'Result spec differs from frozen dispatch')
        require(raw['source_sha256'] == sources == spec['source_sha256'], 'Executed source hashes differ')
        require(raw['hardware']['batch_size'] == 2048 and raw['dataset']['test_size'] == 10000,
                'Result batch size/test panel differs')
        records = unique_by(raw['models'], 'id', 'job model')
        expected = {m['id'] for m in spec['models']}
        require(set(records) == expected and len(records) == 10, 'Missing/unplanned state in completed job')
        progress = raw['records']
        require(len(progress) == 10 and {r['id'] for r in progress} == expected
                and all(r['all_parity_passed'] is True for r in progress), 'Completed record inventory differs')
        for model_id, model in records.items():
            require(model_id not in normalized, 'Duplicate checkpoint state across jobs')
            normalized[model_id] = check_model(model, old_models[model_id], frozen_models[model_id], sources)
        datasets.append(raw['dataset'])
        raw_jobs.append({'run_id': raw['run_id'], 'path': str(path.relative_to(repo)), 'sha256': sha(path),
                         'elapsed_seconds': raw['elapsed_seconds'], 'hardware': raw['hardware']})
    require(set(normalized) == expected_ids and len(normalized) == 30, 'Incomplete30-state result panel')
    require(all(d == datasets[0] for d in datasets), 'Dataset metadata differs across seeds')
    models = [normalized[m['id']] for m in existing['models']]
    output = {'schema_version': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'models': models, 'method_labels': existing.get('method_labels', existing.get('methodslabels', {})),
              'mask_bits': ['stem', 'body1', 'body2', 'body3', 'body4', 'head'],
              'full_affine_macs': FULL_MACS, 'layer_affine_macs': list(LAYER_MACS),
              'units': {'accuracy': 'fraction among emitted predictions; null for abstention',
                        'correct_output_rate': 'correct predictions divided by all inputs',
                        'coverage': 'fraction of inputs receiving a prediction',
                        'ce': 'nats per emitted prediction; null for abstention',
                        'cost': 'executed affine MAC fraction; zero for short-circuit abstention',
                        'nominal_selected_cost': 'affine MAC fraction implied by switches before abstention short-circuit'},
              'provenance': {'raw_jobs': raw_jobs, 'source_sha256': sources,
                             'manifest': {'path': str(manifest_path.relative_to(repo)), 'sha256': sha(manifest_path)},
                             'source_freeze': {'path': str(freeze_path.relative_to(repo)), 'sha256': sha(freeze_path)},
                             'gallery_freeze': {'path': str(gallery_path.relative_to(repo)), 'sha256': sha(gallery_path)},
                             'original_data': {'path': str(existing_path.relative_to(repo)), 'sha256': sha(existing_path)},
                             'exporter': {'path': 'scripts/build_mnist_full_depth_data.py', 'sha256': sha(Path(__file__))},
                             'gallery_selection': gallery['selection'],
                             'verification': {'complete_models': 30, 'mask_rows': 1920,
                                              'archived_prediction_comparisons': 4800000,
                                              'embedded_masks': list(range(33, 64, 2)),
                                              'all_old16_accuracy_and_gallery_predictions_exact': True,
                                              'checkpoint_and_gallery_freeze_exact': True,
                                              'nonstandard_json_numbers': False}},
              'limitations': ['Exploratory inference interventions on reused official MNIST test data; no retraining.',
                              'Removing the stem introduces an arbitrary untrained zero-padding input bypass.',
                              'Removing the classifier abstains; it does not classify randomly or use hidden coordinates as logits.',
                              'Head-absent masks have undefined classification accuracy/CE, not zero accuracy.',
                              'The original illustrated gallery is not a representative accuracy sample.',
                              'Affine MAC counts exclude other work and are not measured latency or energy.']}
    finite_tree(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--manifest', type=Path, default=RESULTS/'manifest-v2.json',
                        help='Explicit successful dispatch manifest; never picks runs by glob or outcome')
    parser.add_argument('--freeze', type=Path, default=RESULTS/'source-freeze-v2.json')
    parser.add_argument('--check-only', action='store_true', help='Validate all results without writing assets')
    args = parser.parse_args()
    output = build(args.repo, args.manifest, args.freeze)
    encoded = json.dumps(output, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    if not args.check_only:
        directory = args.repo/PAGE; directory.mkdir(parents=True, exist_ok=True)
        # All validation and both serializations complete before touching either artifact.
        js = 'window.HYPOTHESIS_EXTENDED = '+encoded+';\n'
        pending = []
        for name, content in [('extended-data.json', encoded+'\n'), ('extended-data.js', js)]:
            destination = directory/name; temp = directory/(name+'.tmp')
            temp.write_text(content); pending.append((temp, destination))
        for temp, destination in pending:
            temp.replace(destination)
    print(json.dumps({'status': 'verified', 'models': len(output['models']), 'mask_rows': 1920,
                      'gallery_examples': sum(len(m['examples']) for m in output['models']),
                      'json_bytes': len(encoded.encode()), 'wrote_assets': not args.check_only}))


if __name__ == '__main__':
    main()
