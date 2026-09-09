#!/usr/bin/env python3
"""Build the A100 report only from complete, verified seed-level measurements.

No service calls, training, experiment edits, or deployment. A missing/invalid
summary leaves the existing report untouched. --self-test writes only to /tmp.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "experiments/a100_transfer/results/summary.json"
DEFAULT_OUTPUT = ROOT / "docs/a100-transfer"
DEFAULT_FINDINGS = ROOT / "experiments/a100_transfer/FINDINGS.json"
REPO = "https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer"
TASKS = ("vit_cifar100", "convnext_cifar100", "gpt_wikitext103")
RECIPES = ("dense", "constant_ild", "decreasing_ild")
NAMES = {"dense": "Dense", "constant_ild": "Constant ILD", "decreasing_ild": "Decreasing ILD"}
LABELS = {"vit_cifar100": "ViT · CIFAR-100", "convnext_cifar100": "ConvNeXt · CIFAR-100",
          "gpt_wikitext103": "GPT · WikiText-103"}
COLORS = {"dense": "#263747", "constant_ild": "#a74c16", "decreasing_ild": "#255bd5"}
METRICS = {"ce": "Test cross-entropy (nats)", "excess_ce": "CE change from full depth (nats)",
           "accuracy": "Test top-1 accuracy (%)", "accuracy_change": "Accuracy change from full depth (pp)"}


def require(value, message):
    if not value:
        raise ValueError(message)


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=4):
    return f"{value:,.{digits}f}"


def interval_text(row, digits=4, factor=1):
    return f"{row['mean']*factor:+.{digits}f} [{row['ci95_low']*factor:+.{digits}f}, {row['ci95_high']*factor:+.{digits}f}]"


def scientific_input_payload(data):
    """Stable experimental provenance; deliberately excludes clocks and billing."""
    v = data['verification']
    return {
        'raw_files': sorted(({'run_id':r['run_id'],'sha256':r['sha256']} for r in v['raw_files']), key=lambda r:r['run_id']),
        'evaluation_manifest_sha256': v['evaluation_manifest_sha256'],
        'executed_core_source_sha256': v['executed_core_source_sha256'],
        'tuning_manifest': v['tuning_manifest'],
    }


def scientific_input_sha256(data):
    encoded = json.dumps(scientific_input_payload(data), sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_findings(data, path=DEFAULT_FINDINGS, *, synthetic=False):
    """Optional plain-text interpretation, pinned to the exact scientific inputs.

    FINDINGS.json has exactly two fields: scientific_input_sha256 and paragraphs.
    Timestamp/billing refreshes are permitted; findings must contain no currency.
    Synthetic renderer fixtures never read or include the real findings file.
    """
    if synthetic or not path.exists():
        return None
    raw = path.read_bytes()
    content = json.loads(raw)
    require(set(content) == {'scientific_input_sha256','paragraphs'}, 'Unexpected FINDINGS.json fields')
    require(content['scientific_input_sha256'] == scientific_input_sha256(data),
            'FINDINGS.json scientific-input digest is stale; review findings against the current measurements')
    paragraphs = content['paragraphs']
    require(isinstance(paragraphs, list) and paragraphs and all(isinstance(p,str) and p.strip() for p in paragraphs),
            'FINDINGS.json requires nonempty plain-text paragraphs')
    require(all(not re.search(r'[$€£¥]|\b(?:USD|dollars?|euros?|GBP)\b', p, re.I) for p in paragraphs),
            'FINDINGS.json must contain no financial amounts; generated cost records are maintained separately')
    return {'content':content,'raw':raw,'sha256':hashlib.sha256(raw).hexdigest()}


def findings_html(findings):
    if findings is None:
        return ''
    return ('<section id="scientific-interpretation" class="note"><h2>Scientific interpretation</h2>'
            + ''.join('<p>'+esc(p)+'</p>' for p in findings['content']['paragraphs'])
            + '<p class="small"><a href="findings.json" download>Download the interpretation and scientific-input digest</a></p></section>')


def validate_interval(row, label):
    require(row.get("n") == 3 and row.get("df") == 2, f"{label}: requires the three declared final seeds")
    values = row.get("values", [])
    require(len(values) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
            f"{label}: invalid seed values")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    radius = 4.3026527299 * sd / math.sqrt(3)
    for key, expected in (("mean", mean), ("std", sd), ("ci95_low", mean-radius), ("ci95_high", mean+radius)):
        require(math.isclose(row[key], expected, rel_tol=1e-7, abs_tol=1e-9), f"{label}: interval arithmetic differs")


def validate_summary(data, *, synthetic=False):
    verification = data.get("verification", {})
    require(verification.get("passed") is True, "Analysis audit did not pass")
    require(verification.get("expected_final_runs") == verification.get("complete_final_runs") == 27,
            "All 27 locked final runs must be complete")
    require(bool(data.get("synthetic_fixture")) == synthetic, "Synthetic/measurement provenance mismatch")
    require({f["task"] for f in data["families"]} == set(TASKS) and len(data["families"]) == 3,
            "Expected all three model families")
    require(len(data["curves"]) == 81 and len(data["primary_comparisons"]) == 6,
            "Expected all 81 recipe/intervention cells and six primary comparisons")
    for family in data["families"]:
        require(len(family["seeds"]) == len(set(family["seeds"])) == 3, "Final seeds must be unique")
        require(len(family["mask_panel"]) == 9 and len({tuple(v) for v in family["mask_panel"].values()}) == 9,
                "The intervention panel must contain nine different fixed masks")
        require(family["mask_panel"]["full"] == list(range(family["prunable_count"])), "Invalid full-depth mask")
        if family['task'] == 'gpt_wikitext103':
            panel = family['test_panel']
            require(isinstance(panel['windows'], int) and 0 < panel['windows'] <= panel['all_available_windows'],
                    "Invalid recorded GPT test-window count")
            require(panel['context'] == family['context'] and panel['target_tokens'] == panel['windows'] * family['context'],
                    "GPT test targets must use actual scored windows, not a requested cap")
        for recipe in RECIPES:
            rows = [r for r in data["curves"] if r["task"] == family["task"] and r["recipe"] == recipe]
            require({r["mask_name"] for r in rows} == set(family["mask_panel"]) and len(rows) == 9,
                    "Missing/duplicate recipe intervention")
            for row in rows:
                require(row["seeds"] == family["seeds"], "Seed order differs")
                require(row["retained"] == len(family["mask_panel"][row["mask_name"]]), "Kept count differs")
                for metric in METRICS:
                    validate_interval(row[metric], f"{family['task']}/{recipe}/{row['mask_name']}/{metric}")
                full = next(r for r in rows if r["mask_name"] == "full")
                for metric, raw in (("excess_ce", "ce"), ("accuracy_change", "accuracy")):
                    require(all(math.isclose(delta, a-b, rel_tol=1e-7, abs_tol=1e-9)
                                for delta,a,b in zip(row[metric]["values"], row[raw]["values"], full[raw]["values"])),
                            "Within-model depth change differs from raw measurements")
    require(len(data["paired_comparisons"]) == 216, "Expected all paired task/recipe/mask/metric comparisons")
    pair_keys = set()
    for row in data["paired_comparisons"]:
        validate_interval(row, "paired comparison")
        key = (row["task"], row["recipe"], row["mask_name"], row["metric"])
        require(key not in pair_keys and row["recipe"] in RECIPES[1:] and row["reference"] == "dense",
                "Duplicate or invalid paired cell")
        pair_keys.add(key)
        treatment = curve(data, row["task"], row["recipe"], row["mask_name"])
        dense = curve(data, row["task"], "dense", row["mask_name"])
        require(row["seeds"] == treatment["seeds"] == dense["seeds"], "Paired seed ordering differs")
        require(all(math.isclose(delta, a-b, rel_tol=1e-7, abs_tol=1e-9)
                    for delta,a,b in zip(row["values"], treatment[row["metric"]]["values"], dense[row["metric"]]["values"])),
                "Treatment-minus-dense arithmetic differs")
    for row in data["primary_comparisons"]:
        require(row["is_primary"] and row["mask_name"] == "primary_two_thirds" and row["metric"] == "excess_ce",
                "Unexpected primary endpoint")
        validate_interval(row, "primary comparison")
        require(row == paired(data, row["task"], row["recipe"]), "Primary summary differs from paired results")
    require({(r["task"], r["recipe"]) for r in data["primary_comparisons"]}
            == {(t, r) for t in TASKS for r in RECIPES[1:]}, "Incomplete primary cells")
    require(len(data["learning_and_resources"]) == len(data["tuning"]) == 9, "Incomplete resource/tuning cells")
    for row in data["tuning"]:
        rates = row["learning_rates"]
        require(len(rates) == len(set(rates)) == 3, "Expected three different candidate learning rates")
        for key in ("mean_validation_ce", "mean_validation_accuracy"):
            require({float(k) for k in row[key]} == set(rates), "Candidate validation metrics do not match the LR grid")
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in row[key].values()),
                    "Invalid candidate validation metric")
        require(all(v <= 1 for v in row["mean_validation_accuracy"].values()), "Validation accuracy must be a fraction")
        chosen = min(rates, key=lambda lr: (lr_metric(row, "mean_validation_ce", lr), lr))
        require(row["selected_lr"] == chosen, "Selected LR must minimize validation CE, not accuracy")
    for row in data["learning_and_resources"]:
        for key, value in row.items():
            if isinstance(value, dict) and "values" in value:
                validate_interval(value, key)
    for row in data["layerscale_by_stage"]:
        for key in ("mean_abs_before", "mean_abs_after", "max_abs_after"):
            validate_interval(row[key], "LayerScale "+key)
    budget = data["budget"]
    require(0 <= budget["reserved_upper_usd"] <= budget["reservation_ceiling_usd"] <= budget["cap_usd"] <= 50,
            "Invalid budget ledger")
    require(len(verification["raw_files"]) == 27, "Missing raw-file provenance")
    require(set(verification["executed_core_source_sha256"]) == {"language.py", "vision.py", "train.py"},
            "Missing frozen core-source hashes")
    accepted = {}
    for key, task in (('initialization_numerical_audit','convnext_cifar100'),
                      ('vit_initialization_numerical_audit','vit_cifar100')):
        audit = verification.get(key)
        if audit is None:
            require(task != 'convnext_cifar100', "Missing qualified CNN initialization audit")
            continue
        require(audit["passed"] is True and audit["posthoc_verification_adjustment"] is True and audit['task'] == task,
                "Initialization audit is incomplete or belongs to a different task")
        require(audit["vision_source_sha256"] == verification["executed_core_source_sha256"]["vision.py"],
                "Initialization audit used different executed vision source")
        require(audit["thresholds"]["max_absolute_difference"] <= 1e-7
                and audit["thresholds"]["relative_l2_difference"] <= 1e-6, "Initialization audit bounds were loosened")
        f = next(f for f in data["families"] if f["task"] == task)
        declared_seeds = set(f['seeds']) | {s for r in data['tuning'] if r['task'] == task for s in r['tuning_seeds']}
        require(set(audit['audited_seeds']) and set(audit['audited_seeds']) <= declared_seeds
                and set(audit['audited_seeds']) == {r['seed'] for r in audit['seed_audits']},
                "Initialization audit seed coverage is invalid")
        if task == 'convnext_cifar100':
            require(set(audit['audited_seeds']) == declared_seeds, "CNN audit must retain all original seed comparisons")
        for row in audit['seed_audits']:
            require(row['parameters'] == f['parameters'] and len(set(row['state_hashes'])) == 2,
                    "Initialization audit must compare all parameters of both observed variants")
            require(0 <= row['max_absolute_difference'] <= audit['thresholds']['max_absolute_difference']
                    and 0 <= row['relative_l2_difference'] <= audit['thresholds']['relative_l2_difference'],
                    "Initialization difference exceeds the accepted numerical bounds")
            accepted[(task,row['seed'])] = set(row['state_hashes'])
        require(audit['max_absolute_difference'] == max(r['max_absolute_difference'] for r in audit['seed_audits'])
                and audit['max_relative_l2_difference'] == max(r['relative_l2_difference'] for r in audit['seed_audits']),
                "Initialization audit extrema do not match seed records")
        require(audit['evidence_files'], "Initialization evidence is missing")
    for row in verification["initialization_pairing"]:
        if row['status'] == 'exact':
            require(len(set(row['observed_hashes'])) == 1, "Exact initialization pairing contains multiple hashes")
        else:
            require(row['status'] == 'audited_numerical_close' and (row['task'],row['seed']) in accepted
                    and set(row['observed_hashes']) <= accepted[(row['task'],row['seed'])],
                    "Numerical initialization exception lacks an accepted task/seed/hash audit")


def family(data, task):
    return next(row for row in data["families"] if row["task"] == task)


def curve(data, task, recipe, mask="full"):
    return next(row for row in data["curves"] if row["task"] == task and row["recipe"] == recipe and row["mask_name"] == mask)


def paired(data, task, recipe, mask="primary_two_thirds", metric="excess_ce"):
    return next(row for row in data["paired_comparisons"] if row["task"] == task and row["recipe"] == recipe
                and row["mask_name"] == mask and row["metric"] == metric)


def lr_metric(row, key, lr):
    # JSON turns numeric map keys into strings; compare their numeric values.
    return next(value for rate, value in row[key].items() if float(rate) == lr)


def initialization_audits(data):
    return [data['verification'][key] for key in ('initialization_numerical_audit','vit_initialization_numerical_audit')
            if data['verification'].get(key) is not None]


def initialization_evidence(data):
    return [item for audit in initialization_audits(data) for item in audit['evidence_files']]


def initialization_audit_paragraph(audit):
    parameters = audit["seed_audits"][0]["parameters"]
    name = 'ConvNeXt' if audit['task'] == 'convnext_cifar100' else 'ViT'
    return (f"{name}: initialization for the audited seeds is same-seed and numerically close, not byte-identical. "
        f"Independent CPU reconstruction from the frozen source compared all {parameters:,} parameters for "
        f"{len(audit['audited_seeds'])} audited {'seed' if len(audit['audited_seeds']) == 1 else 'seeds'} "
        f"({', '.join(map(str,audit['audited_seeds']))}). "
        f"The worst absolute difference was {audit['max_absolute_difference']:.3g}; the worst relative L2 difference was "
        f"{audit['max_relative_l2_difference']:.3g}, within required bounds of "
        f"{audit['thresholds']['max_absolute_difference']:.0e} and {audit['thresholds']['relative_l2_difference']:.0e}, respectively. "
        "The two variants have identical post-initialization RNG states.")


def initialization_paragraphs(data):
    return ["GPT retains byte-identical same-seed initial states. Vision exceptions below are restricted to separately audited seeds and hashes; all other pairings must match exactly.",
        *[initialization_audit_paragraph(audit) for audit in initialization_audits(data)],
        "Accepting these exact, independently audited vision hashes is a posthoc verification adjustment made independently of test outcomes. "
        "It does not change the frozen executed training code, learning-rate selection, masks, final seeds, or test-metric requirements; "
        "unrecognized initialization hashes still fail. These bounds describe initial tensors, not a bound on later training divergence.",
        "The ConvNeXt worker probes locate the first observed difference at CPU erfinv after identical uniform draws, consistent with math-library roundoff "
        "in the inverse-error-function transform. They do not establish the exact dispatch path. "
        "These were CPU probes on existing workers, without additional GPU training jobs.",
    ]


def initialization_html(data):
    links = [f'<a href="{REPO}/initialization-audit.md">Initialization audit narrative ↗</a>']
    links.extend(f'<a href="{REPO}/results/{esc(Path(item["file"]).name)}">{esc(Path(item["file"]).name)} ↗</a>'
                 for item in initialization_evidence(data))
    return ('<aside id="initialization-audit" class="note"><h3>Initialization matching: a disclosed verification adjustment.</h3>'
            + ''.join('<p>'+esc(text)+'</p>' for text in initialization_paragraphs(data))
            + '<div class="downloads">'+''.join(links)+'</div></aside>')


def verdict(row):
    if row["ci95_high"] < 0:
        return "less", "The interval is entirely below zero: a lower CE change on pruning than with dense training."
    if row["ci95_low"] > 0:
        return "more", "The interval is entirely above zero: a higher CE change on pruning than with dense training."
    return "unresolved", "The interval includes zero: the direction is unresolved by these three seeds; this is not an equivalence result."


def headline(data):
    vision = [r for r in data["primary_comparisons"] if r["task"] != "gpt_wikitext103"]
    counts = {key: sum(verdict(r)[0] == key for r in vision) for key in ("less", "more", "unresolved")}
    vit = [r for r in vision if r['task'] == 'vit_cifar100']
    cnn = [r for r in vision if r['task'] == 'convnext_cifar100']
    if (all(verdict(r)[0] == 'more' for r in vit) and all(verdict(r)[0] == 'unresolved' for r in cnn)
            and all(curve(data,'vit_cifar100',recipe,'primary_two_thirds')['excess_ce']['mean'] < 0 for recipe in RECIPES)):
        return "Pruning lowers mean ViT CE for all recipes, most for dense training; ConvNeXt's primary CE intervals include zero."
    if counts["less"] == 4:
        return "Both image architectures show reduced pruning damage for both dropout recipes."
    if counts["more"] == 4:
        return "Both image architectures show increased pruning damage for both dropout recipes."
    if counts["unresolved"] == 4:
        return "The image-transfer comparisons remain unresolved with three seeds."
    return (f"The four image-model comparisons have {counts['less']} intervals below zero, "
            f"{counts['more']} above zero, and {counts['unresolved']} crossing zero for the paired CE change on pruning.")


def mask_order(f):
    panel = f["mask_panel"]
    structural = [k for k in panel if k in {"full", "primary_two_thirds", "one_third", "half", "five_sixths"} or k.startswith("prefix_")]
    return sorted(structural, key=lambda k: len(panel[k])) + [f"random_keeps_first_{i}" for i in range(3)] + ["delete_first_only"]


def mask_label(name, *, short=False):
    if name == "full":
        return "Full"
    if name == "primary_two_thirds":
        return "2/3 · primary" if short else "Primary · two-thirds retained"
    if name.startswith("random_keeps_first_"):
        return f"Random {int(name[-1])+1}" if short else f"Random two-thirds {int(name[-1])+1} · first block kept"
    if name == "delete_first_only":
        return "Delete first" if short else "Delete first block only · outside training support"
    return {"one_third": "1/3 · stagewise", "half": "1/2 · stagewise", "five_sixths": "5/6 · stagewise"}.get(name, name.replace("prefix_", "Prefix "))


def primary_table(data):
    rows = []
    for task in TASKS:
        for recipe in RECIPES[1:]:
            p = paired(data, task, recipe)
            status, text = verdict(p)
            rows.append(f'<tr data-primary="{task}/{recipe}" data-direction="{status}"><td>{LABELS[task]}</td><td>{NAMES[recipe]}</td>'
                        f'<td class="num">{interval_text(p)}</td><td class="num">{", ".join(f"{x:+.4f}" for x in p["values"])}</td>'
                        f'<td>{esc(text)}</td></tr>')
    return '<table><thead><tr><th>Architecture / data</th><th>Recipe</th><th>Paired effect [95% CI]</th><th>Three seed differences</th><th>What the interval says</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def exposure_table(data):
    rows = []
    for task in TASKS:
        f = family(data, task)
        language = task == "gpt_wikitext103"
        exposure = f"{f['training_targets_or_images']:,} token presentations<br>{f['tokens_per_parameter']:.3f} tokens / parameter" if language else f"{f['training_targets_or_images']:,} image presentations<br>{f['image_epochs']:.2f} equivalent passes with replacement"
        target = f['test_panel']['target_tokens'] if language else f['test_panel']['examples']
        panel_label = f"{target:,} target tokens<br>{f['test_panel']['windows']:,} actual windows" if language else f"{target:,} images"
        shape = f"{f['context']:,}-token context" if language else f"{f['image_size']} × {f['image_size']} pixels"
        rows.append(f'<tr><td>{LABELS[task]}</td><td class="num">{f["parameters"]:,}</td><td class="num">{f["steps"]:,} × {f["batch_size"]:,}</td>'
                    f'<td>{exposure}</td><td>{shape}</td><td class="num">{panel_label}</td></tr>')
    return '<table><thead><tr><th>Family</th><th>Parameters</th><th>Steps × batch</th><th>Training exposure per run</th><th>Input</th><th>Fixed test panel</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def resource_table(data):
    rows = []
    for task in TASKS:
        for recipe in RECIPES:
            r = next(r for r in data["learning_and_resources"] if r['task'] == task and r['recipe'] == recipe)
            rows.append(f'<tr><td>{LABELS[task]}</td><td>{NAMES[recipe]}</td><td class="num">{r["validation_before_ce"]["mean"]:.4f} → {r["validation_final_ce"]["mean"]:.4f}</td>'
                        f'<td class="num">{r["last_minibatch_ce"]["mean"]:.4f}</td><td class="num">{r["peak_allocated_gib"]["mean"]:.2f}</td>'
                        f'<td class="num">{r["peak_reserved_gib"]["mean"]:.2f}</td><td class="num">{r["train_seconds"]["mean"]:.1f}</td></tr>')
    return '<table><thead><tr><th>Family</th><th>Recipe</th><th>Validation CE<br>initial → final</th><th>Last minibatch CE</th><th>Peak allocated GiB</th><th>Peak reserved GiB</th><th>Training seconds</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def tuning_table(data):
    rows = []
    candidates = []
    for task in TASKS:
        for recipe in RECIPES:
            r = next(r for r in data["tuning"] if r['task'] == task and r['recipe'] == recipe)
            rows.append(f'<tr><td>{LABELS[task]}</td><td>{NAMES[recipe]}</td><td class="num">{", ".join(f"{x:g}" for x in r["learning_rates"])}</td>'
                        f'<td class="num">{r["selected_lr"]:g}</td><td>{"yes" if r["selected_at_boundary"] else "no"}</td>'
                        f'<td>{", ".join(map(str,r["tuning_seeds"]))}</td><td>{"matched" if r["same_horizon_as_final"] else "proxy horizon"}</td></tr>')
            for lr in r['learning_rates']:
                candidates.append(f'<tr data-task="{task}" data-recipe="{recipe}" data-lr="{lr:g}"><td>{LABELS[task]}</td><td>{NAMES[recipe]}</td>'
                                  f'<td class="num">{lr:g}</td><td class="num">{lr_metric(r,"mean_validation_ce",lr):.4f}</td>'
                                  f'<td class="num">{100*lr_metric(r,"mean_validation_accuracy",lr):.2f}%</td>'
                                  f'<td>{"yes — lowest CE" if lr == r["selected_lr"] else "no"}</td></tr>')
    return ('<table><thead><tr><th>Family</th><th>Recipe</th><th>Learning-rate grid</th><th>Selected LR</th><th>Grid boundary?</th><th>Tuning seeds</th><th>Training horizon</th></tr></thead><tbody>'
            +''.join(rows)+'</tbody></table><h3>Every candidate: CE and accuracy can favor different rates.</h3>'
            '<p class="small">Full-depth validation scores below are means over the listed tuning seeds. The lowest validation CE determines the selected learning rate; top-1 accuracy is a diagnostic and never overrides that fixed rule. Accuracy can improve while CE worsens. These are tuning scores, not final test results.</p>'
            '<table id="tuning-validation"><thead><tr><th>Family</th><th>Recipe</th><th>Candidate LR</th><th>Validation CE (nats)</th><th>Validation top-1 accuracy</th><th>Selected by CE?</th></tr></thead><tbody>'
            +''.join(candidates)+'</tbody></table>')


def layerscale_table(data):
    rows = []
    for r in sorted(data["layerscale_by_stage"], key=lambda r: (RECIPES.index(r["recipe"]), r["stage_zero_based"])):
        rows.append(f'<tr><td>{NAMES[r["recipe"]]}</td><td>{r["stage_zero_based"]+1}</td><td class="num">{r["mean_abs_before"]["mean"]:.6g}</td>'
                    f'<td class="num">{r["mean_abs_after"]["mean"]:.6g}</td><td class="num">{r["max_abs_after"]["mean"]:.6g}</td></tr>')
    return '<table><thead><tr><th>Recipe</th><th>Stage</th><th>Initial mean |γ|</th><th>Final mean |γ|</th><th>Mean of per-seed stage maxima</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def method_cards(data):
    result = []
    for task in TASKS:
        f = family(data, task)
        if task == "gpt_wikitext103":
            text = f"A randomly initialized, tied-head GPT-2-style decoder with {f['prunable_count']} blocks and width {f['model_config']['n_embd']}, trained on WikiText-103 with GPT-2 tokens. {f['tokens_per_parameter']:.3f} token presentations per parameter is far below the reviewed paper's 20-token-per-parameter principal regime. Fixed document-contained test windows are not canonical full-corpus perplexity."
        elif task == "vit_cifar100":
            text = f"A {f['prunable_count']}-block, width-{f['model_config']['width']} ViT-style classifier, with {f['model_config']['patch_size']}×{f['model_config']['patch_size']} patches and mean patch pooling. There is no class token. The primary intervention keeps the first eight blocks and the original final readout. This changes architecture and modality relative to GPT, but shares CIFAR-100 with the CNN experiment."
        else:
            text = "A ConvNeXt-style CNN with 18 residual blocks in stages of 3, 3, 9, 3. The primary mask retains stage prefixes of 2, 2, 6, 2; downsampling transitions remain. This is stagewise thinning, not early exit. Native 32×32 CIFAR images are resized to 64×64; interpolation adds no observed information."
        result.append(f'<article><h3>{LABELS[task]}</h3><p>{esc(text)}</p><p class="small">{esc(", ".join(f["gpu_models"]))} · {esc(f["precision"])}</p></article>')
    return ''.join(result)


def plots(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#202a35", "axes.labelcolor": "#202a35", "svg.fonttype": "none"})
    for task in TASKS:
        f = family(data, task)
        order = mask_order(f)
        for metric, metric_label in METRICS.items():
            factor = 100 if metric.startswith("accuracy") else 1
            fig, ax = plt.subplots(figsize=(11.6, 5.4), layout="constrained")
            for offset, recipe in zip((-.22, 0, .22), RECIPES):
                rows = [curve(data, task, recipe, mask)[metric] for mask in order]
                y = np.array([r["mean"]*factor for r in rows])
                low = np.array([r["ci95_low"]*factor for r in rows])
                high = np.array([r["ci95_high"]*factor for r in rows])
                # Categorical interventions are not joined by an interpolating line.
                ax.errorbar(np.arange(9)+offset, y, yerr=[y-low,high-y], fmt="o", markersize=4,
                            capsize=3, elinewidth=1.2, color=COLORS[recipe], label=NAMES[recipe])
            ax.axvspan(1.55, 2.45, color="#edf2fc", zorder=-2)  # Primary is third in structural order.
            ax.axvline(4.5, lw=.8, c="#aab5c0")
            ax.axvline(7.5, lw=.8, c="#aab5c0")
            if metric in ("excess_ce", "accuracy_change"):
                ax.axhline(0, c="#9aabbc", lw=.8)
            labels = [mask_label(mask, short=True)+f"\n{len(f['mask_panel'][mask])}/{f['prunable_count']} blocks" for mask in order]
            ax.set_xticks(np.arange(9), labels, rotation=28, ha="right", fontsize=8)
            ax.set_ylabel(metric_label)
            ax.set_title(LABELS[task]+" · nine predefined interventions", loc="left", weight="bold", pad=38)
            if data.get("synthetic_fixture"):
                fig.suptitle("SYNTHETIC RENDERER FIXTURE — NOT EXPERIMENTAL RESULTS", color="#b32828", fontsize=12)
            ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1,1.01), borderaxespad=0, fontsize=9)
            caption = "Mean and 95% Student-t interval across 3 trained seeds; secondary masks are descriptive."
            if metric == 'accuracy':
                caption += "\nIntervals are untruncated; bounds outside 0–100% are not observed accuracies."
            fig.supxlabel(caption, fontsize=9)
            for extension in ("svg", "png"):
                fig.savefig(out / f"{task}-{metric}.{extension}", dpi=180)
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout="constrained")
    for ax, task in zip(axes, TASKS):
        for i, recipe in enumerate(RECIPES[1:]):
            r = paired(data, task, recipe)
            ax.scatter(r["values"], [i+.12]*3, s=22, color=COLORS[recipe], alpha=.6)
            ax.errorbar(r["mean"], i, xerr=[[r["mean"]-r["ci95_low"]],[r["ci95_high"]-r["mean"]]],
                        fmt="D", capsize=4, markersize=5, color=COLORS[recipe])
        ax.axvline(0, c="#80909e", lw=1)
        ax.set_yticks([0,1], [NAMES[r] for r in RECIPES[1:]], fontsize=9)
        ax.set_ylim(1.45,-.4)
        ax.set_title(LABELS[task], loc="left", fontsize=11, weight="bold")
        ax.set_xlabel("Paired difference in CE change (nats)\nNegative = lower CE change than dense", fontsize=9)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("SYNTHETIC RENDERER FIXTURE — NOT EXPERIMENTAL RESULTS" if data.get("synthetic_fixture")
                 else "Predefined two-thirds-depth intervention · paired 95% t intervals", fontsize=13)
    caption = "Small dots: individual seed differences. Separate x scales and task-specific loss units. Intervals are not multiplicity-adjusted."
    if all(curve(data,'vit_cifar100',recipe,'primary_two_thirds')['excess_ce']['mean'] < 0 for recipe in RECIPES):
        caption += "\nViT pruning lowers CE in every recipe: a positive paired effect means a smaller CE improvement after dropout."
    fig.supxlabel(caption, fontsize=9)
    for extension in ("svg", "png"):
        fig.savefig(out / f"primary-effects.{extension}", dpi=180)
    plt.close(fig)


CSS = r"""
:root{--paper:#fcfcfa;--ink:#202a35;--muted:#5c6c7a;--line:#d7dfe5;--blue:#255bd5;--orange:#a74c16;--soft:#eff3f7;--green:#176646;--serif:Georgia,'Times New Roman',serif;--sans:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 var(--sans)}a{color:var(--blue);text-underline-offset:3px}header{border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:20px;padding:17px 5vw}header a{font-size:14px;text-decoration:none}.brand{font-weight:750;color:var(--ink);letter-spacing:.03em}main{max-width:1260px;margin:auto;padding:42px 42px 65px}.eyebrow{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 15px}h1{font:clamp(38px,4.8vw,64px)/1.07 var(--serif);letter-spacing:-.04em;max-width:1040px;margin:0 0 20px}h2{font:34px/1.18 var(--serif);letter-spacing:-.02em;margin:0 0 20px}h3{font-size:18px;margin:12px 0 12px}p{margin:0 0 16px;max-width:940px}.lead{font-size:20px;line-height:1.5;max-width:1040px}.facts{display:flex;flex-wrap:wrap;gap:12px 28px;margin:22px 0 10px;font-size:14px}.facts strong{font-size:17px}.jump{display:flex;flex-wrap:wrap;gap:12px 25px;font-size:14px;margin:24px 0 0}.section{padding:42px 0;border-bottom:1px solid var(--line);scroll-margin-top:20px}.controls{display:flex;flex-wrap:wrap;gap:18px;margin:22px 0 16px}.control{display:grid;gap:7px;flex:1;min-width:210px}label,legend{font-size:14px;color:var(--muted)}select,button{font:inherit;color:var(--ink);background:var(--paper);border:1px solid #a3b1c0;padding:9px 11px;border-radius:0;cursor:pointer}select{width:100%}button:hover{background:var(--soft)}:focus-visible{outline:3px solid var(--blue);outline-offset:3px}.recipe-controls{display:flex;flex-wrap:wrap;gap:17px;border:0;padding:0;margin:15px 0}.recipe-controls label{color:var(--ink);display:flex;align-items:center;gap:7px}.recipe-controls input{accent-color:var(--c);width:16px;height:16px}.legend-mark{display:inline-block;background:var(--c);width:22px;height:3px}.viz{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:16px 0;margin:19px 0;overflow-x:auto}.viz svg{display:block;width:100%;min-width:750px;height:auto}.viz.compact svg{min-width:600px}.axis{font:14px var(--sans);fill:var(--muted)}.chart-title{font:15px var(--sans);fill:var(--ink);font-weight:600}.tick{font:13px var(--sans);fill:var(--muted)}.small{font-size:14px;line-height:1.6;color:var(--muted)}.note{border-left:3px solid var(--blue);padding:3px 0 3px 19px;margin:22px 0}.note p:last-child{margin:0}.split{display:grid;grid-template-columns:1fr 1fr;gap:32px}.methods{display:grid;grid-template-columns:repeat(3,1fr);gap:27px}.methods p{font-size:15px}.table-wrap{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5}th{text-align:left;border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);padding:11px 10px;vertical-align:bottom}td{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}.num{text-align:right;font-variant-numeric:tabular-nums}.compact-table{font-size:13px}.formula{font:23px/1.6 var(--serif);border-block:1px solid var(--line);padding:13px 0;margin:20px 0}.downloads{display:flex;gap:14px 24px;flex-wrap:wrap;font-size:14px;margin:16px 0}.status{display:inline-block;border:1px solid var(--line);padding:3px 9px;font-size:13px;color:var(--muted)}.outcome-less{color:var(--green)}.outcome-more{color:#9e381a}.result{font-size:18px;line-height:1.5}.fixture{border:4px solid #b32828;padding:20px;color:#b32828;font-size:24px;font-weight:bold}.money{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;border-block:1px solid var(--line);padding:25px 0;margin:25px 0}.money strong{display:block;font:36px/1.2 var(--serif)}.money span{display:block;font-size:14px;color:var(--muted);margin-top:8px}details{border-top:1px solid var(--line);padding:16px 0}summary{font-weight:650;cursor:pointer}details p{margin-top:14px}.hash{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere;word-break:break-all}.code{background:var(--soft);padding:18px;overflow:auto;font:13px/1.65 ui-monospace,monospace}.ref-list{padding-left:20px;font-size:14px}.ref-list li{margin-bottom:12px}figure{margin:22px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:var(--muted);margin-top:10px}footer{border-top:1px solid var(--line);padding:22px 5vw;font-size:13px;color:var(--muted)}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}html{scroll-behavior:smooth}@media(max-width:760px){main{padding:30px 20px}.methods,.split{grid-template-columns:1fr;gap:10px}.money{grid-template-columns:1fr;gap:18px}h2{font-size:29px}.section{padding:33px 0}header{padding:15px 20px}.lead{font-size:18px}.control{min-width:180px}}@media print{header,.controls,.recipe-controls,.jump,button{display:none}main{padding:0}.section{break-inside:avoid;padding:25px 0}body{font-size:11pt}a{color:inherit}.viz svg{min-width:0}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""


APP_JS = r"""
'use strict';
(() => {
  const data=window.A100_DATA, $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
  const recipes=['dense','constant_ild','decreasing_ild'];
  const names={dense:'Dense',constant_ild:'Constant ILD',decreasing_ild:'Decreasing ILD'};
  const colors={dense:'#263747',constant_ild:'#a74c16',decreasing_ild:'#255bd5'};
  const metricNames={ce:'Test cross-entropy (nats)',excess_ce:'CE change from full depth (nats)',accuracy:'Test top-1 accuracy (%)',accuracy_change:'Accuracy change from full depth (pp)'};
  const number=(x,n=4)=>Number(x).toFixed(n), signed=(x,n=4)=>(x>=0?'+':'')+number(x,n);
  const interval=(r,factor=1)=>`${signed(r.mean*factor)} [${signed(r.ci95_low*factor)}, ${signed(r.ci95_high*factor)}]`;
  const clear=node=>node.replaceChildren();
  function el(tag, attrs={}, text=null, parent=null){const node=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))node.setAttribute(k,v);if(text!==null)node.textContent=text;if(parent)parent.append(node);return node;}
  function cell(tr,text,cls=''){const td=document.createElement('td');td.textContent=text;td.className=cls;tr.append(td);return td;}
  const family=()=>data.families.find(f=>f.task===$('task').value);
  const getRow=(f,recipe,mask)=>data.curves.find(r=>r.task===f.task&&r.recipe===recipe&&r.mask_name===mask);
  const pair=(f,recipe,mask='primary_two_thirds',metric='excess_ce')=>data.paired_comparisons.find(r=>r.task===f.task&&r.recipe===recipe&&r.mask_name===mask&&r.metric===metric);
  function label(name,short=false){if(name==='full')return'Full';if(name==='primary_two_thirds')return short?'2/3 · primary':'Primary · two-thirds retained';if(name.startsWith('random_keeps_first_'))return`Random ${Number(name.at(-1))+1}${short?'':' · two-thirds, first block kept'}`;if(name==='delete_first_only')return short?'Delete first':'Delete first block only';return({one_third:'1/3 · stagewise',half:'1/2 · stagewise',five_sixths:'5/6 · stagewise'})[name]||name.replace('prefix_','Prefix ');}
  function order(f){const structured=Object.keys(f.mask_panel).filter(k=>['full','primary_two_thirds','one_third','half','five_sixths'].includes(k)||k.startsWith('prefix_'));return structured.sort((a,b)=>f.mask_panel[a].length-f.mask_panel[b].length).concat([0,1,2].map(i=>`random_keeps_first_${i}`),['delete_first_only']);}
  function range(values,zero=false){let low=Math.min(...values),high=Math.max(...values);if(zero){low=Math.min(low,0);high=Math.max(high,0);}const pad=Math.max((high-low)*.12,.005);return[low-pad,high+pad];}
  function axes(svg,{left=80,right=980,top=35,bottom=355,limits,label:axisLabel,vertical=true}){const[lo,hi]=limits,scale=x=>bottom-(x-lo)/(hi-lo)*(bottom-top);for(let i=0;i<=5;i++){const value=lo+(hi-lo)*i/5,y=scale(value);el('line',{x1:left,x2:right,y1:y,y2:y,stroke:'#dfe5eb'},null,svg);el('text',{x:left-12,y:y+4,'text-anchor':'end',class:'tick'},number(value,Math.abs(hi-lo)<1?3:2),svg);}el('text',{x:left,y:18,class:'chart-title'},axisLabel,svg);return scale;}
  function activeRecipes(){return recipes.filter(r=>$('recipe-'+r).checked);}
  function drawCurves(){const f=family(),metric=$('metric').value,masks=order(f),active=activeRecipes(),factor=metric.startsWith('accuracy')?100:1;const svg=$('curve');clear(svg);const rows=active.flatMap(recipe=>masks.map(mask=>getRow(f,recipe,mask)));const limits=range(rows.flatMap(r=>[r[metric].ci95_low*factor,r[metric].ci95_high*factor]),['excess_ce','accuracy_change'].includes(metric));const y=axes(svg,{limits,label:metricNames[metric]});const x=i=>115+i*101;const primary=masks.indexOf('primary_two_thirds');el('rect',{x:x(primary)-42,y:27,width:84,height:342,fill:'#edf2fc'},null,svg);if(['excess_ce','accuracy_change'].includes(metric))el('line',{x1:80,x2:980,y1:y(0),y2:y(0),stroke:'#8294a5','stroke-dasharray':'4 4'},null,svg);
    for(let i=0;i<masks.length;i++){const mask=masks[i];el('text',{x:x(i),y:391,'text-anchor':'middle',class:'tick',transform:`rotate(24 ${x(i)} 391)`},label(mask,true),svg);el('text',{x:x(i),y:440,'text-anchor':'middle',class:'tick'},`${f.mask_panel[mask].length}/${f.prunable_count}`,svg);}
    el('text',{x:530,y:473,'text-anchor':'middle',class:'axis'},'Fixed intervention · original residual blocks retained',svg);
    for(const recipe of active){const offset=(recipes.indexOf(recipe)-1)*18;for(let i=0;i<masks.length;i++){const r=getRow(f,recipe,masks[i]),s=r[metric],px=x(i)+offset,low=y(s.ci95_low*factor),high=y(s.ci95_high*factor);el('line',{x1:px,x2:px,y1:low,y2:high,stroke:colors[recipe],'stroke-width':1.6},null,svg);for(const py of[low,high])el('line',{x1:px-4,x2:px+4,y1:py,y2:py,stroke:colors[recipe],'stroke-width':1.6},null,svg);const dot=el('circle',{cx:px,cy:y(s.mean*factor),r:4.5,fill:colors[recipe],tabindex:0,'data-point':`${recipe}/${masks[i]}`,'aria-label':`${names[recipe]}, ${label(masks[i])}: ${interval(s,factor)}`},null,svg);el('title',{},`${names[recipe]} · ${label(masks[i])}\nMean [95% t CI]: ${interval(s,factor)}`,dot);const choose=()=>{stop();$('mask').value=masks[i];drawMask();};dot.addEventListener('click',choose);dot.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});}}
    $('download-svg').href=`${f.task}-${metric}.svg`;$('download-png').href=`${f.task}-${metric}.png`;
    const body=$('curve-table').querySelector('tbody');clear(body);for(const mask of masks)for(const recipe of active){const r=getRow(f,recipe,mask),tr=document.createElement('tr');cell(tr,label(mask));cell(tr,names[recipe]);cell(tr,`${r.retained}/${f.prunable_count}`,'num');cell(tr,interval(r[metric],factor),'num');cell(tr,r[metric].values.map(v=>number(v*factor)).join(', '),'num');body.append(tr);}
    $('curve-summary').textContent=`${f.label}: ${active.length} recipes × 9 fixed masks. Dots are seed means; bars are 95% Student-t intervals across three training seeds. No curve is forced to be monotonic.`;
  }
  function drawPrimary(){const f=family(),svg=$('primary');clear(svg);const rr=recipes.slice(1).map(r=>pair(f,r));const[lo,hi]=range(rr.flatMap(r=>[r.ci95_low,r.ci95_high,...r.values]),true);const x=v=>225+(v-lo)/(hi-lo)*(960-225);el('line',{x1:x(0),x2:x(0),y1:30,y2:175,stroke:'#7c8d9d','stroke-dasharray':'4 4'},null,svg);for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5;el('text',{x:x(v),y:205,'text-anchor':'middle',class:'tick'},number(v,Math.abs(hi-lo)<1?4:2),svg);}rr.forEach((r,i)=>{const py=60+i*77;el('text',{x:205,y:py+5,'text-anchor':'end',class:'chart-title'},names[r.recipe],svg);el('line',{x1:x(r.ci95_low),x2:x(r.ci95_high),y1:py,y2:py,stroke:colors[r.recipe],'stroke-width':3},null,svg);for(const end of[r.ci95_low,r.ci95_high])el('line',{x1:x(end),x2:x(end),y1:py-6,y2:py+6,stroke:colors[r.recipe],'stroke-width':2},null,svg);el('path',{d:`M ${x(r.mean)} ${py-7} l 7 7 l -7 7 l -7 -7 Z`,fill:colors[r.recipe],'data-primary-point':r.recipe},null,svg);r.values.forEach((v,j)=>{const dot=el('circle',{cx:x(v),cy:py+20,r:3.5,fill:colors[r.recipe],opacity:.6,'data-seed':r.seeds[j]},null,svg);el('title',{},`Seed ${r.seeds[j]}: ${signed(v)}`,dot);});});el('text',{x:590,y:245,'text-anchor':'middle',class:'axis'},'Paired difference in CE change (nats) · negative means a lower change than dense',svg);
    const qualityDirection=(r,label)=>r.ci95_high<0?`${label} CE is lower than its matched dense-trained counterpart.`:r.ci95_low>0?`${label} CE is higher than its matched dense-trained counterpart.`:`The ${label.toLowerCase()} CE difference includes zero.`;const result=$('primary-narrative');clear(result);for(const r of rr){const p=document.createElement('p');const direction=r.ci95_high<0?'less':r.ci95_low>0?'more':'unresolved';p.className='outcome-'+direction;p.dataset.direction=direction;p.textContent=`${names[r.recipe]}: ${interval(r)}. `+(direction==='less'?'The interval lies below zero: a lower CE change on pruning than with dense training.':direction==='more'?'The interval lies above zero: a higher CE change on pruning than with dense training.':'The interval includes zero: direction unresolved; not an equivalence finding.');const denseChange=getRow(f,'dense','primary_two_thirds').excess_ce.mean,ownChange=getRow(f,r.recipe,'primary_two_thirds').excess_ce.mean;if(denseChange<0&&ownChange<0)p.textContent+=' Pruning lowers CE in both recipes; '+(r.mean>0?'the positive paired mean denotes a smaller CE improvement after dropout.':'the negative paired mean denotes a larger CE improvement after dropout.');p.textContent+=' '+qualityDirection(pair(f,r.recipe,'primary_two_thirds','ce'),'Pruned-model')+' '+qualityDirection(pair(f,r.recipe,'full','ce'),'Full-model');result.append(p);}
    const body=$('quality-table').querySelector('tbody');clear(body);for(const recipe of recipes){const full=getRow(f,recipe,'full'),p=getRow(f,recipe,'primary_two_thirds'),tr=document.createElement('tr');cell(tr,names[recipe]);cell(tr,number(full.ce.mean),'num');cell(tr,number(p.ce.mean),'num');cell(tr,number(full.accuracy.mean*100,2)+'%','num');cell(tr,number(p.accuracy.mean*100,2)+'%','num');cell(tr,recipe==='dense'?'reference':interval(pair(f,recipe,'full','ce')),'num');cell(tr,recipe==='dense'?'reference':interval(pair(f,recipe,'primary_two_thirds','ce')),'num');body.append(tr);}
    $('primary-definition').textContent=f.task==='convnext_cifar100'?'CNN intervention: keep stage prefixes 2, 2, 6, 2 from stages 3, 3, 9, 3; all downsampling transitions and the original readout remain.':'Transformer intervention: retain the first 8 of 12 blocks, then apply the unchanged final normalization and readout.';
  }
  function drawMask(){const f=family(),mask=$('mask').value,kept=f.mask_panel[mask],svg=$('mask-viz');clear(svg);const step=f.prunable_count===18?45:65,start=80;for(let i=0;i<f.prunable_count;i++){const extra=f.task==='convnext_cifar100'?(i>=3?15:0)+(i>=6?15:0)+(i>=15?15:0):0,px=start+i*step+extra,on=kept.includes(i);el('rect',{x:px,y:30,width:step-8,height:42,fill:on?'#255bd5':'#eef2f5',stroke:on?'#255bd5':'#aeb9c3','data-block':i,'data-kept':on},null,svg);el('text',{x:px+(step-8)/2,y:57,'text-anchor':'middle',fill:on?'#fff':'#61717f','font-size':14},i+1,svg);}el('text',{x:80,y:102,class:'axis'},`${label(mask)} · ${kept.length}/${f.prunable_count} residual blocks retained`,svg);$('mask-description').textContent=(mask==='delete_first_only'?'The first block is always active during ILD training. Deleting it tests a pattern outside that training support. ':mask.startsWith('random')?'This random mask was fixed in advance and is shared across recipes and seeds. It is not selected by test performance. ':'This structural intervention was fixed before final evaluation. ')+(f.task==='convnext_cifar100'?'Stage gaps mark transitions, which are never deleted. ':'Block order and final readout are preserved. ')+`Zero-based retained indices: [${kept.join(', ')}].`;
    const body=$('mask-table').querySelector('tbody');clear(body);for(const recipe of recipes){const r=getRow(f,recipe,mask),tr=document.createElement('tr');cell(tr,names[recipe]);cell(tr,interval(r.ce),'num');cell(tr,interval(r.excess_ce),'num');cell(tr,interval(r.accuracy,100),'num');cell(tr,interval(r.accuracy_change,100),'num');body.append(tr);}
  }
  let timer=null;
  function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play-masks').textContent='Play mask tour';$('play-masks').setAttribute('aria-pressed','false');}
  function taskChanged(){stop();const f=family(),old=$('mask').value;clear($('mask'));for(const name of order(f)){const option=document.createElement('option');option.value=name;option.textContent=label(name);$('mask').append(option);}$('mask').value=f.mask_panel[old]?old:'primary_two_thirds';const lang=f.task==='gpt_wikitext103';$('task-context').textContent=`${f.parameters.toLocaleString('en-US')} parameters · ${f.prunable_count} residual blocks · ${f.steps.toLocaleString('en-US')} steps × batch ${f.batch_size} · ${lang?f.context.toLocaleString('en-US')+'-token context':f.image_size+'×'+f.image_size+' input'} · ${lang?'token':'image-class'} top-1 accuracy.${lang?' '+f.test_panel.windows.toLocaleString('en-US')+' actual test windows / '+f.test_panel.target_tokens.toLocaleString('en-US')+' target tokens.':''}`;drawPrimary();drawCurves();drawMask();}
  $('task').addEventListener('change',taskChanged);$('metric').addEventListener('change',drawCurves);$('mask').addEventListener('change',()=>{stop();drawMask();});for(const recipe of recipes)$('recipe-'+recipe).addEventListener('change',event=>{if(!activeRecipes().length)event.target.checked=true;drawCurves();});$('play-masks').addEventListener('click',()=>{if(timer!==null){stop();return;}$('play-masks').textContent='Pause mask tour';$('play-masks').setAttribute('aria-pressed','true');timer=setInterval(()=>{const options=[...$('mask').options];$('mask').selectedIndex=($('mask').selectedIndex+1)%options.length;drawMask();},1000);});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
  taskChanged();window.A100_REPORT={redraw:taskChanged,stop,maskOrder:()=>order(family())};
})();
"""


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A verified A100 experiment testing layer-dropout depth robustness across GPT, ViT, and ConvNeXt, with paired seeds, fixed interventions and complete cost provenance."><title>Depth robustness across architectures · Gradient Dissent</title><link rel="stylesheet" href="style.css"><script defer src="data.js"></script><script defer src="app.js"></script></head>
<body><a class="sr-only" href="#explore">Skip to results</a><header><a class="brand" href="../">GRADIENT DISSENT</a><a href="https://github.com/yaroslavvb/gradient-dissent/tree/main/experiments/a100_transfer">Code &amp; raw runs ↗</a></header><main>
@@FIXTURE@@<p class="eyebrow">Independent A100 experiment · verified final measurements</p><h1>Does depth robustness survive a change of architecture?</h1><p class="lead">@@HEADLINE@@</p><p>We train GPT, a vision transformer, and a convolutional network with two layer-dropout recipes and a dense baseline. The test is whether removing a fixed fraction of residual blocks causes less damage—and whether the retained model is actually better.</p>
<div class="facts"><span><strong>3</strong> architectures</span><span><strong>27</strong> final runs</span><span><strong>3</strong> paired seeds per comparison</span><span><strong>9</strong> fixed interventions per model</span><span><strong>27</strong> validation-only tuning runs</span></div>
@@FINDINGS_SECTION@@
<nav class="jump" aria-label="Report sections"><a href="#explore">Primary result</a><a href="#interventions">All nine interventions</a><a href="#methods">Training and transfer</a><a href="#resources">Memory and cost</a><a href="#audit">Audit and downloads</a></nav>

<section class="section" id="explore"><h2>Pruning damage and useful prediction are separate questions.</h2><div class="controls"><div class="control"><label for="task">Architecture and data</label><select id="task"><option value="vit_cifar100">ViT · CIFAR-100</option><option value="convnext_cifar100">ConvNeXt · CIFAR-100</option><option value="gpt_wikitext103">GPT · WikiText-103</option></select></div></div><p id="task-context" class="small"></p>
<div class="formula">Primary effect = (CE<sub>two-thirds</sub> − CE<sub>full</sub>)<sub>dropout</sub> − (CE<sub>two-thirds</sub> − CE<sub>full</sub>)<sub>dense</sub></div><p id="primary-definition"></p><div class="viz compact"><svg id="primary" viewBox="0 0 1020 265" role="img" aria-label="Primary paired effects and 95 percent confidence intervals"></svg></div><div id="primary-narrative" class="result" aria-live="polite"></div><p class="small">Diamonds are paired means; small dots are individual training-seed differences. Bars are 95% Student-t intervals with 2 degrees of freedom. Six primary intervals are shown across this report, without multiplicity adjustment. Seed intervals omit dataset and learning-rate-selection uncertainty.</p>
<h3>Keep the intact model and the pruned model in view.</h3><div class="table-wrap"><table id="quality-table"><thead><tr><th>Recipe</th><th>Full CE</th><th>Primary CE</th><th>Full accuracy</th><th>Primary accuracy</th><th>Full CE vs dense<br>paired effect [95% CI]</th><th>Primary CE vs dense<br>paired effect [95% CI]</th></tr></thead><tbody></tbody></table></div><p>A smaller pruning penalty can coexist with worse predictions. Absolute full and pruned losses are therefore reported alongside robustness. For a CE difference, negative favors the dropout-trained model. When both full-to-pruned CE changes are negative, a positive paired effect means a smaller CE improvement, not worse absolute pruned CE. We do not interpret an interval crossing zero as equivalence. Symmetric t intervals are untruncated; accuracy bounds outside 0–100% are not observed accuracies.</p>
<details><summary>All six primary effects and their three paired seed values</summary><div class="table-wrap">@@PRIMARY_TABLE@@</div></details><div class="downloads"><a href="primary-effects.svg" download>Primary figure · SVG</a><a href="primary-effects.png" download>Primary figure · PNG</a><a href="paired.csv" download>All paired comparisons · CSV</a></div></section>

<section class="section" id="interventions"><h2>Nine interventions, fixed before final evaluation.</h2><p>The same original readout is used for every retained network. No fine-tuning, adapters, output calibration, or inverse-survival rescaling is applied after pruning. These are nine specified masks, not an exhaustive subset search.</p><div class="controls"><div class="control"><label for="metric">Metric</label><select id="metric"><option value="ce">Raw cross-entropy</option><option value="excess_ce">CE change from full depth</option><option value="accuracy">Top-1 accuracy</option><option value="accuracy_change">Accuracy change from full depth</option></select></div></div>
<fieldset class="recipe-controls"><legend class="sr-only">Visible training recipes</legend><label style="--c:#263747"><input id="recipe-dense" type="checkbox" checked><span class="legend-mark"></span>Dense</label><label style="--c:#a74c16"><input id="recipe-constant_ild" type="checkbox" checked><span class="legend-mark"></span>Constant ILD</label><label style="--c:#255bd5"><input id="recipe-decreasing_ild" type="checkbox" checked><span class="legend-mark"></span>Decreasing ILD</label></fieldset>
<div class="viz"><svg id="curve" viewBox="0 0 1020 490" role="img" aria-label="Test metrics for nine predefined layer masks"></svg></div><p id="curve-summary" class="small" aria-live="polite"></p><p class="small">Interventions are categorical. Equal retained counts can remove different blocks; no interpolation between masks is implied. A negative change from full depth is allowed. Accuracy is token top-1 for GPT and image-class top-1 for vision; raw values should not be averaged across families.</p><div class="downloads"><a id="download-svg" href="vit_cifar100-ce.svg" download>Selected metric · SVG</a><a id="download-png" href="vit_cifar100-ce.png" download>Selected metric · PNG</a><a href="data.json" download>All seed-level statistics · JSON</a></div>
<details><summary>Numeric table for every displayed point</summary><div class="table-wrap"><table id="curve-table"><thead><tr><th>Intervention</th><th>Recipe</th><th>Kept blocks</th><th>Mean [95% CI]</th><th>Three seed measurements</th></tr></thead><tbody></tbody></table></div></details>
<h3>Inspect the retained blocks.</h3><div class="controls"><div class="control"><label for="mask">Fixed intervention</label><select id="mask"></select></div><button id="play-masks" aria-pressed="false">Play mask tour</button></div><div class="viz compact"><svg id="mask-viz" viewBox="0 0 1020 125" role="img" aria-label="Original blocks retained by the selected mask"></svg></div><p id="mask-description" class="small"></p><div class="table-wrap"><table id="mask-table"><thead><tr><th>Recipe</th><th>Raw CE [95% CI]</th><th>CE change [95% CI]</th><th>Accuracy % [95% CI]</th><th>Accuracy change pp [95% CI]</th></tr></thead><tbody></tbody></table></div></section>

<section class="section" id="methods"><h2>Transfer across model families, under a declared budget.</h2><div class="methods">@@METHOD_CARDS@@</div><p class="note">Here, “transfer” means an effect replicating across settings; no pretrained weights transfer between families. The two image architectures use the same CIFAR-100 dataset and split. They test transfer across architecture, not replication across two independent vision datasets. All models are trained from scratch; GPT here is a short-run architectural baseline, not a reproduction of the paper's large-language-model pretraining.</p><div class="table-wrap">@@EXPOSURE_TABLE@@</div><p class="small">Token and image counts are presentations, not unique examples. Training samples with replacement. “Equivalent passes” means presentations divided by training-set size, with replacement; it does not mean shuffled epochs or guaranteed complete passes. The 10,000 official CIFAR test images are kept out of tuning; GPT uses the actual recorded document-contained test windows and target count shown above. A requested maximum is only a cap; valid spans can be fewer.</p>
<div class="split"><div><h3>Two recipes with the same expected masking rate</h3><p>Increasing layer dropout (ILD) drops later blocks more often and always retains the first. Constant ILD uses maximum probability 0.4. Decreasing ILD starts at maximum 0.8 and linearly reaches zero. Both schedules imply a 20% expected masking rate for example-block residual contributions, averaged over depth and training steps; realized Bernoulli masks vary. Decreasing ILD reaches exactly one zero-dropout terminal step, with no extended dense continuation.</p><p class="small">These schedules differ in maximum probability and variance as well as temporal order. Their comparison does not isolate time ordering alone. Training uses compute-then-mask execution, so this expected 20% masks residual contributions after execution; it is not saved execution or a measured 20% compute reduction.</p></div><div><h3>Matched comparisons, independently tuned recipes</h3><p>Within each family, recipes share training exposure, initialization seeds, data streams, optimizer family and evaluation panels. Each selects a learning rate using full-depth validation CE only, with equal grids and a separate tuning seed. No test or pruned metric chooses the learning rate.</p><p class="small">The result compares the complete recipe plus its chosen learning rate. Final three-seed intervals condition on the selection made from one tuning seed; they do not include search uncertainty.</p></div></div>@@INITIALIZATION_NOTE@@<details><summary>Learning-rate grids, selections and boundary diagnostics</summary><div class="table-wrap">@@TUNING_TABLE@@</div></details>
<p class="small">This larger experiment includes neither separately trained smaller dense models nor alternating-dropout training. It cannot establish that pruning beats training a smaller network directly, or settle the paper's distinct alternating-mask claim. Earlier toy controls do not replace these controls at A100 scale.</p><h3>ConvNeXt: test the easy identity-path explanation.</h3><p>ConvNeXt's residual LayerScale starts small. A network that barely uses its residual branches can appear robust to deleting them. Inspect the measured full-model learning and gamma magnitudes before attributing small pruning damage to useful learned redundancy. The residual contribution is γ × f(h), so gamma magnitude alone cannot identify how much the branch is used. Neither small gamma nor gamma growth establishes functional irrelevance or importance; direct residual-activation attribution is absent.</p><p>The <a href="https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py">original ConvNeXt implementation</a> already uses depth-increasing stochastic depth. Applying layer dropout to a CNN is therefore an established idea. This experiment tests full/pruned behavior of these specified schedules, at this scale and geometry.</p><details><summary>LayerScale before and after training, by stage</summary><div class="table-wrap">@@GAMMA_TABLE@@</div><p class="small">Values summarize absolute gamma within a stage and then across training seeds. The last column averages each seed's within-stage maximum; it is not the maximum across all runs.</p></details></section>

<section class="section" id="resources"><h2>Measured memory; an explicit cost ledger.</h2><div class="table-wrap">@@RESOURCE_TABLE@@</div><p class="small">Values above are means across the three final seeds. GPU peaks cover the recorded training window. Allocated memory measures live tensors, including resident data. Reserved memory can include allocator cache inherited from earlier calls or prevalidation; resetting peak counters does not clear that cache. Reserved values therefore do not establish a minimum VRAM requirement. Last-minibatch CE is noisy, not full-training-set loss. Full intervals and raw values are in the downloadable JSON.</p><p class="note">These are execution and provenance measurements, not a controlled efficiency comparison. The masking implementation executes dense branches. The experiment establishes no reduction in FLOPs, wall time, GPU memory, energy, or dollars. No joules were measured.</p><div class="money"><div><strong>$@@METERED@@</strong><span>@@METERED_LABEL@@</span></div><div><strong>$@@RESERVED@@</strong><span>Conservative invocation reservations; not spend</span></div><div><strong>$@@CAP@@</strong><span>User-authorized external-compute ceiling</span></div></div><p class="small">@@BILLING_NOTE@@ The ledger records @@RESERVATION_COUNT@@ invocation reservations, including the audit trail of preparation, pilots, tuning and final work. Reserved amounts are neither invoices nor measured charges.</p><div class="downloads"><a href="@@REPO@@/results/budget-ledger.json">Full reservation ledger ↗</a><a href="@@REPO@@/results/billing-latest.json">Saved metered snapshot ↗</a></div></section>

<section class="section" id="audit"><h2>Evidence you can inspect and reproduce.</h2><p class="status">All 27 declared final runs passed the analysis audit.</p><p>The generator refuses partial cells, failed audit flags, missing seed measurements, or inconsistent interval arithmetic. The upstream analysis checks paired initial states against exact matching or the separately disclosed vision numerical allowlists, matched data streams, completed training horizons, source and dataset hashes, fixed mask panels, GPU all-kept equivalence, equal learning-rate searches and reservation arithmetic.</p><details><summary>Frozen source hashes and audit checks</summary>@@SOURCE_HASHES@@<ul>@@CHECKS@@</ul></details><details><summary>What these results cannot establish</summary><ul>@@LIMITS@@</ul></details>
<figure><img src="primary-effects.png" alt="Exportable three-panel primary effect figure with paired confidence intervals"><figcaption>Scientific figure export: paired treatment-minus-dense effects on the predefined intervention, with the three individual seed differences.</figcaption></figure><div class="downloads"><a href="report.md" download>Concise report · Markdown</a><a href="data.json" download>Verified summary · JSON</a><a href="paired.csv" download>Paired effects · CSV</a><a href="manifest.json" download>Report provenance · JSON</a><a href="@@REPO@@/results">Every saved raw run ↗</a></div>
<pre class="code">python experiments/a100_transfer/analyze.py
python scripts/build_a100_report.py
node scripts/check_a100_report.cjs</pre><p class="small">These commands analyze existing completed measurements and rebuild the report; they do not launch cloud training. All visualization code and assets are local, with no CDN dependency.</p>
<ol class="ref-list"><li><a href="https://arxiv.org/pdf/2609.05275v1">Elhoushi et al., Don't Drop Dropout, September 2026 extended PDF.</a> The empirical motivation, especially depth-elastic inference and layer/time schedules.</li><li><a href="https://arxiv.org/abs/1603.09382">Huang et al., Deep Networks with Stochastic Depth (2016)</a> and <a href="https://arxiv.org/abs/1909.11556">Fan et al., Reducing Transformer Depth on Demand with Structured Dropout (2020).</a> Existing CNN stochastic depth and transformer pruning without fine-tuning precede the reviewed paper.</li><li><a href="@@REPO@@/interpretation-audit.md">Interpretation audit written before final results.</a> Scope, confounds and missing controls.</li><li><a href="@@REPO@@/protocol.json">Frozen experiment protocol.</a> Exposures, schedule, tuning rules, primary endpoint and budget boundary.</li><li><a href="@@REPO@@/evaluation-manifest.json">Locked final-run manifest.</a> Every declared final seed and selected recipe configuration.</li><li><a href="@@REPO@@/vision-notes.md">Vision architecture and dataset provenance.</a> ConvNeXt transitions, LayerScale, CIFAR preprocessing and ViT readout.</li><li><a href="@@REPO@@/language-notes.md">Language architecture and dataset provenance.</a> GPT-style initialization, WikiText103, tokenizer and document-contained windows.</li><li><a href="../">Critical paper review.</a> Mathematical counterexamples, paper audit and energy/locality connections.</li><li><a href="../depth-robustness/">Earlier local toy experiments.</a> A different scale, dataset and mask enumeration; not pooled into this report's seed statistics.</li></ol></section>
</main><footer>Built from verified measurements · summary SHA256 <span class="hash">@@SUMMARY_SHA@@</span> · generated @@GENERATED@@</footer></body></html>
"""


def markdown(data, findings=None):
    def absolute(row, *, factor=1, digits=4):
        return (f"{row['mean']*factor:.{digits}f} "
                f"[{row['ci95_low']*factor:.{digits}f}, {row['ci95_high']*factor:.{digits}f}]")

    lines = ["# Depth robustness across architectures", ""]
    if data.get("synthetic_fixture"):
        lines.extend(["**SYNTHETIC RENDERER FIXTURE — NOT EXPERIMENTAL RESULTS.**", ""])
    if findings is not None:
        lines.extend(['## Scientific interpretation', ''])
        lines.extend(p+'\n' for p in findings['content']['paragraphs'])
        lines.extend([f"Scientific-input SHA-256: `{findings['content']['scientific_input_sha256']}`. "
                      "This binds the interpretation to the raw runs, evaluation manifest, core source and tuning manifest, excluding timestamps and billing.", '',
                      '## Primary comparison', ''])
    lines.extend([headline(data), "", f"Verified summary generated: {data['generated_utc']}.", "",
             "27 completed final runs; three paired seeds per recipe and architecture; nine predefined masks. "
             "The primary effect is treatment-minus-dense in CE(two-thirds retained) minus CE(full). Negative means less pruning damage. "
             "The six primary intervals are paired 95% Student-t intervals with 2 degrees of freedom, unadjusted for multiple comparisons. "
             "They condition on the selected learning rates and fixed data split. An interval including zero does not establish equivalence.", "",
             "| Family | Recipe | Paired primary effect [95% CI] | Interpretation |", "|---|---|---|---|"])
    for task in TASKS:
        for recipe in RECIPES[1:]:
            r = paired(data, task, recipe)
            lines.append(f"| {LABELS[task]} | {NAMES[recipe]} | {interval_text(r)} | {verdict(r)[1]} |")

    lines.extend(["", "## Absolute predictive quality", "",
                  "Each cell is the mean [95% Student-t interval] across the three final training seeds. "
                  "CE is in nats per GPT-2 token for language and nats per image for vision. Accuracy is token top-1 or image-class top-1, in percent. "
                  "These task-specific metrics should not be pooled across families. A smaller pruning penalty can coexist with worse predictions. "
                  "The symmetric t intervals are untruncated: an accuracy bound outside 0–100% is not an observed accuracy.", "",
                  "| Family | Recipe | Full CE | Primary-pruned CE | Full accuracy (%) | Primary-pruned accuracy (%) |",
                  "|---|---|---|---|---|---|"])
    for task in TASKS:
        for recipe in RECIPES:
            full, pruned = curve(data, task, recipe, "full"), curve(data, task, recipe, "primary_two_thirds")
            lines.append(f"| {LABELS[task]} | {NAMES[recipe]} | {absolute(full['ce'])} | {absolute(pruned['ce'])} "
                         f"| {absolute(full['accuracy'],factor=100,digits=2)} | {absolute(pruned['accuracy'],factor=100,digits=2)} |")
    lines.extend(["", "The GPT and ViT primary intervention retains the first 8 of 12 blocks and the unchanged final readout. "
                  "ConvNeXt retains stage prefixes of 2/2/6/2 from stages of 3/3/9/3 blocks, with all downsampling transitions; this is stagewise thinning, not early exit. "
                  "The same nine masks are used across recipes and seeds, without post-pruning adaptation. First-block deletion is a separate stress test outside the ILD training support.", "",
                  "## Model size and training exposure", "",
                  "Exposures below are per final run and are matched across recipes within each family. "
                  "Counts are sampled presentations, not unique examples. Equivalent passes mean presentations divided by training-set size, with replacement; they are not shuffled epochs or guaranteed complete passes.", "",
                  "| Family | Parameters | Residual blocks | Steps × batch | Input | Training presentations | Exposure ratio | Fixed test panel |",
                  "|---|---|---|---|---|---|---|---|"])
    for task in TASKS:
        f = family(data, task)
        lang = task == "gpt_wikitext103"
        shape = f"{f['context']:,} tokens" if lang else f"{f['image_size']}×{f['image_size']} pixels"
        unit = "tokens" if lang else "images"
        ratio = f"{f['tokens_per_parameter']:.3f} tokens / parameter" if lang else f"{f['image_epochs']:.2f} equivalent passes with replacement"
        targets = f['test_panel']['target_tokens'] if lang else f['test_panel']['examples']
        panel_label = f"{targets:,} target tokens in {f['test_panel']['windows']:,} actual windows" if lang else f"{targets:,} images"
        lines.append(f"| {LABELS[task]} | {f['parameters']:,} | {f['prunable_count']} | {f['steps']:,} × {f['batch_size']:,} "
                     f"| {shape} | {f['training_targets_or_images']:,} {unit} | {ratio} | {panel_label} |")
    lines.extend(["", "All models are randomly initialized. The GPT arm is a short-budget architectural baseline, far below the reviewed paper's principal 20-token-per-parameter regime; it is not full-scale GPT-2 pretraining. "
                  "The recorded number of scored test windows and target tokens is shown above; a requested maximum is a cap, not a guarantee of that many valid document-contained spans. "
                  "These test windows are not canonical full-corpus perplexity. "
                  "The ViT uses mean patch pooling without a class token. ConvNeXt bilinearly resizes CIFAR images from 32×32 to 64×64; interpolation adds no new image information. "
                  "The two image families share CIFAR-100 and its split, so they are not independent dataset replications. “Transfer” means replication of an effect, not reuse of trained weights.", "",
                  "## Initialization verification adjustment", ""])
    lines.extend(paragraph + "\n" for paragraph in initialization_paragraphs(data))
    lines.extend([f"[Initialization audit narrative]({REPO}/initialization-audit.md) · "
                  + " · ".join(f"[{Path(item['file']).name}]({REPO}/results/{Path(item['file']).name})" for item in initialization_evidence(data)), "",
                  "## Recipe selection", "",
                  "Each family/recipe selects a learning rate from an equal three-rate grid using full-depth validation CE and a separate tuning seed. "
                  "There are 27 validation-only tuning runs, distinct from the 27 final runs. No test or pruned score chooses the learning rate. "
                  "Final comparisons concern independently tuned recipes; they do not isolate dropout at an identical learning rate. Boundary winners are disclosed below.", "",
                  "| Family | Recipe | Candidate learning rates | Selected learning rate | Grid boundary? | Tuning seeds | Tuning horizon |",
                  "|---|---|---|---|---|---|---|"])
    for task in TASKS:
        for recipe in RECIPES:
            r = next(r for r in data['tuning'] if r['task'] == task and r['recipe'] == recipe)
            lines.append(f"| {LABELS[task]} | {NAMES[recipe]} | {', '.join(f'{x:g}' for x in r['learning_rates'])} "
                         f"| {r['selected_lr']:g} | {'yes' if r['selected_at_boundary'] else 'no'} "
                         f"| {', '.join(map(str,r['tuning_seeds']))} | {'matched to final' if r['same_horizon_as_final'] else 'proxy horizon'} |")
    lines.extend(["", "## Candidate validation measurements", "",
                  "These are full-depth validation means over the listed tuning seeds, not final test results. "
                  "The selected rate is fixed by lowest validation CE; accuracy is shown as a diagnostic and never changes the selection. "
                  "Accuracy can improve while CE worsens, so the objective matters. Top-1 is token accuracy for GPT and image-class accuracy for vision, in percent.", "",
                  "| Family | Recipe | Candidate LR | Validation CE (nats) | Validation top-1 accuracy (%) | Selected by CE? |",
                  "|---|---|---|---|---|---|"])
    for task in TASKS:
        for recipe in RECIPES:
            r = next(r for r in data['tuning'] if r['task'] == task and r['recipe'] == recipe)
            for lr in r['learning_rates']:
                lines.append(f"| {LABELS[task]} | {NAMES[recipe]} | {lr:g} | {lr_metric(r,'mean_validation_ce',lr):.4f} "
                             f"| {100*lr_metric(r,'mean_validation_accuracy',lr):.2f} | {'yes — lowest CE' if lr == r['selected_lr'] else 'no'} |")
    lines.extend(["", "Constant ILD uses depth-increasing omission probabilities with maximum 0.4 and the first block always retained. "
                  "Decreasing ILD linearly anneals the maximum from 0.8 to zero, reaching exactly one zero-dropout terminal step, without an extended dense continuation. "
                  "Both schedules imply a 20% expected masking rate for example-block residual contributions, averaged over depth and training steps; realized Bernoulli masks vary. "
                  "All branches execute before masking, so this is masked contribution, not saved execution. "
                  "The recipes also differ in maximum strength and variance; the comparison does not isolate temporal order alone.", "",
                  "## Cost and execution scope", ""])
    budget = data['budget']
    billing = budget['latest_metered_usage']
    if billing:
        queried = datetime.fromtimestamp(billing['queried_unix'], timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        lines.append(f"Latest saved metered usage: **${billing['metered_cost_usd']:.2f}**, queried **{queried}**. "
                     "This timestamped snapshot may lag; it is not a final invoice or settled charge.")
    else:
        lines.append("No metered-usage snapshot was available when the verified summary was generated. Spending is not inferred from reservations.")
    lines.extend(["", f"The reservation-ledger snapshot included in the summary generated **{data['generated_utc']}** records "
                  f"**${budget['reserved_upper_usd']:.2f}** across **{budget['reservation_count']} invocation reservations**. "
                  f"The internal reservation ceiling is **${budget['reservation_ceiling_usd']:.2f}** and the user-authorized external-compute cap is **${budget['cap_usd']:.2f}**. "
                  "Reservations are conservative budget allocations, not spending or invoices. The summary-generation timestamp is not a billing-query timestamp.", "",
                  f"[Reservation ledger]({REPO}/results/budget-ledger.json)" +
                  (f" · [Saved metered snapshot]({REPO}/results/billing-latest.json)" if billing else ""), "",
                  "Training computes dense residual branches before applying masks. Expected omission is therefore not a measured saving in FLOPs, time, GPU memory, energy or dollars. "
                  "No joules were measured. Recorded GPU memory and training time are execution measurements rather than a controlled efficiency comparison.", "",
                  "## Limits", ""])
    lines.extend("- "+item for item in data["limitations"])
    lines.extend(["", "ConvNeXt's existing stochastic depth is relevant prior work; applying dropout to a CNN is not itself new. "
                  "Its residual contribution is γ × f(h): LayerScale magnitude alone cannot establish branch importance or irrelevance. "
                  "This larger study includes neither separately trained smaller dense models nor alternating-dropout training; earlier toy controls do not supply those missing A100-scale comparisons."])
    lines.extend(["", f"[Code and raw measurements]({REPO}) · [Verified statistics](data.json) · [All paired comparisons](paired.csv)", ""])
    return "\n".join(lines)


def paired_csv(data):
    import csv
    import io
    stream = io.StringIO()
    columns = ["task","recipe","reference","mask_name","metric","is_primary","n","df","mean","std","ci95_low","ci95_high","seeds_json","seed_differences_json"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in data["paired_comparisons"]:
        writer.writerow({**{k:row[k] for k in columns if k not in ("seeds_json","seed_differences_json")},
                         "seeds_json":json.dumps(row["seeds"]), "seed_differences_json":json.dumps(row["values"])})
    return stream.getvalue()


def build(data, summary_bytes, output, *, synthetic=False):
    validate_summary(data, synthetic=synthetic)
    findings = load_findings(data, synthetic=synthetic)
    output = Path(output).resolve()
    if synthetic:
        require(not output.is_relative_to(ROOT / "docs"), "Synthetic fixtures must never be written under published docs")
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    budget = data["budget"]
    billing = budget["latest_metered_usage"]
    if billing:
        metered, metered_label = f"{billing['metered_cost_usd']:.2f}", "Latest saved metered-usage snapshot"
        queried = datetime.fromtimestamp(billing["queried_unix"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        billing_note = f"Metered snapshot queried {queried}; it may lag and is not a final invoice."
    else:
        metered, metered_label = "—", "No metered-usage snapshot available"
        billing_note = "No metered-usage snapshot was available. Spending is not inferred from reservation totals."
    substitutions = {
        "FIXTURE": '<p class="fixture">SYNTHETIC TEST FIXTURE — NOT EXPERIMENTAL RESULTS</p>' if synthetic else "",
        "HEADLINE": esc(headline(data)), "PRIMARY_TABLE": primary_table(data), "METHOD_CARDS": method_cards(data),
        "EXPOSURE_TABLE": exposure_table(data), "RESOURCE_TABLE": resource_table(data),
        "TUNING_TABLE": tuning_table(data), "GAMMA_TABLE": layerscale_table(data),
        "INITIALIZATION_NOTE": initialization_html(data),
        "FINDINGS_SECTION": findings_html(findings),
        "METERED": metered, "METERED_LABEL": metered_label, "RESERVED": f"{budget['reserved_upper_usd']:.2f}",
        "CAP": f"{budget['cap_usd']:.2f}", "BILLING_NOTE": esc(billing_note),
        "RESERVATION_COUNT": str(budget["reservation_count"]), "REPO": REPO,
        "SOURCE_HASHES": ''.join(f'<p><strong>{esc(name)}</strong><br><span class="hash">{esc(value)}</span></p>' for name,value in data["verification"]["executed_core_source_sha256"].items()),
        "CHECKS": ''.join('<li>'+esc(s)+'</li>' for s in data["verification"]["checks"]),
        "LIMITS": ''.join('<li>'+esc(s)+'</li>' for s in data["limitations"]),
        "SUMMARY_SHA": summary_sha, "GENERATED": esc(data["generated_utc"]),
    }
    page = TEMPLATE
    for key, value in substitutions.items():
        page = page.replace('@@'+key+'@@', value)
    require('@@' not in page, "Unfilled HTML template")
    # Render and validate into staging before replacing any published artifact.
    with tempfile.TemporaryDirectory(prefix="a100-report-build-") as temporary:
        staging = Path(temporary)
        plots(data, staging)
        (staging / "index.html").write_text(page)
        (staging / "style.css").write_text(CSS)
        (staging / "app.js").write_text(APP_JS)
        (staging / "data.json").write_bytes(summary_bytes)
        (staging / "data.js").write_text('window.A100_DATA = '+json.dumps(data, allow_nan=False)+';\n')
        (staging / "paired.csv").write_text(paired_csv(data))
        (staging / "report.md").write_text(markdown(data, findings))
        if findings is not None:
            (staging / 'findings.json').write_bytes(findings['raw'])
        manifest = {"source": "experiments/a100_transfer/results/summary.json", "summary_sha256": summary_sha,
                    "scientific_input_sha256": scientific_input_sha256(data),
                    "scientific_findings": {'file':'findings.json','source':'experiments/a100_transfer/FINDINGS.json',
                                            'sha256':findings['sha256'],'scientific_input_sha256':findings['content']['scientific_input_sha256']} if findings else None,
                    "synthetic_fixture": synthetic, "complete_final_runs": 27,
                    "primary_endpoint": data["statistical_method"],
                    "evaluation_manifest_sha256": data["verification"]["evaluation_manifest_sha256"],
                    "core_source_sha256": data["verification"]["executed_core_source_sha256"],
                    "initialization_numerical_audit": data["verification"]["initialization_numerical_audit"],
                    "vit_initialization_numerical_audit": data["verification"].get("vit_initialization_numerical_audit"),
                    "raw_files": data["verification"]["raw_files"], "figures": {}}
        for path in sorted([*staging.glob("*.png"), *staging.glob("*.svg")]):
            manifest["figures"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "backing_data": "data.json", "interval": "95% Student-t across 3 training seeds"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2)+'\n')
        output.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            shutil.copyfile(path, output / path.name)
        if findings is None:
            (output / 'findings.json').unlink(missing_ok=True)
    if synthetic:
        print(f"Built synthetic renderer fixture at {output}; no measured run results; 13 PNG/SVG figure pairs")
    else:
        print(f"Built {output}: 27 verified final runs, six paired primary comparisons, 13 PNG/SVG figure pairs")


def synthetic_fixture():
    """Only for renderer QA; the build guard prevents publication under docs/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("a100_analysis_fixture_helpers", ROOT / "experiments/a100_transfer/analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    stat = analysis.interval
    seeds = [2000,2001,2002]
    data = {"synthetic_fixture": True, "generated_utc": "2000-01-01T00:00:00+00:00",
            "verification": {"passed": True, "expected_final_runs": 27, "complete_final_runs": 27,
                             "evaluation_manifest_sha256": "a"*64, "executed_core_source_sha256": {k:"b"*64 for k in ("train.py","language.py","vision.py")},
                             "tuning_manifest":{"file":"synthetic-tuning-manifest.json","sha256":"f"*64,"runs":27,"initialization_pairing":[]},
                             "raw_files": [{"run_id":f"synthetic-{i}","file":f"synthetic-{i}.json","sha256":"c"*64} for i in range(27)],
                             "checks": ["SYNTHETIC UI TEST ONLY"]}, "families": [], "curves": [], "paired_comparisons": [],
            "primary_comparisons": [], "learning_and_resources": [], "tuning": [], "layerscale_by_stage": [],
            "statistical_method": "SYNTHETIC renderer test; never a measurement",
            "budget": {"cap_usd":50,"reserved_upper_usd":30,"reservation_count":58,"reservation_ceiling_usd":42,
                       "latest_metered_usage":{"metered_cost_usd":12.5,"queried_unix":946684800}},
            "limitations": ["SYNTHETIC renderer test: no values are real experimental results."]}
    for ti, task in enumerate(TASKS):
        n = 18 if task == "convnext_cifar100" else 12
        masks = analysis.expected_named_masks(task, n)
        # Deterministic distinct masks, not meant to reproduce the experiment RNG.
        masks.update({f"random_keeps_first_{j}": [0]+list(range(2+j,2+j+2*n//3-1)) for j in range(3)})
        data["families"].append({"task":task,"label":LABELS[task],"seeds":seeds,"parameters":124_439_808 if ti==2 else 27_897_028 if ti==1 else 20_000_000,
            "prunable_count":n,"mask_panel":masks,"steps":3200,"batch_size":32,"context":1024,"image_size":64 if n==18 else 32,
            "training_targets_or_images":104_857_600 if ti==2 else 320000,"tokens_per_parameter":.84 if ti==2 else None,
            "image_epochs":None if ti==2 else 7.11,"test_panel":{"target_tokens":131072,"windows":128,"all_available_windows":128,"context":1024} if ti==2 else {"examples":10000},
            "gpu_models":["SYNTHETIC A100"],"precision":"synthetic bfloat16",
            "model_config":{"n_embd":768,"width":768,"patch_size":4}})
        for ri, recipe in enumerate(RECIPES):
            for mask, kept in masks.items():
                damage = (n-len(kept))/n * (1.2 if ri==0 else .6+ti*.4) + (.3 if mask=='delete_first_only' else 0)
                ce = [1.8+ti*.4+ri*.03+damage+v+(v*ri*.15 if mask!='full' else 0) for v in (-.04,.01,.03)]
                excess = [damage+v*(ri*.15) for v in (-.04,.01,.03)]
                if mask == 'full':
                    excess = [0.,0.,0.]
                accuracy = [max(.02,.62-damage*.15-ri*.01+v) for v in (-.02,.005,.015)]
                change = [0.,0.,0.] if mask=='full' else [-damage*.15]*3
                data['curves'].append({'task':task,'recipe':recipe,'mask_name':mask,'retained':len(kept),'seeds':seeds,
                    'ce':stat(ce),'excess_ce':stat(excess),'accuracy':stat(accuracy),'accuracy_change':stat(change)})
            data['learning_and_resources'].append({'task':task,'recipe':recipe,'lr':.001,
                **{k:stat([v*.98,v,v*1.02]) for k,v in {'validation_before_ce':4.6,'validation_final_ce':2.3,'last_minibatch_ce':2.1,
                    'train_seconds':300,'peak_allocated_gib':14,'peak_reserved_gib':18,'clipped_step_fraction':.05}.items()}})
            data['tuning'].append({'task':task,'recipe':recipe,'tuning_seeds':[1000],'learning_rates':[.0003,.001,.003],
                                  'mean_validation_ce':{.0003:2.3,.001:2.1,.003:2.4},
                                  'mean_validation_accuracy':{.0003:.55,.001:.56,.003:.59},
                                  'selected_lr':.001,'selected_at_boundary':False,'same_horizon_as_final':True})
            if n==18:
                for stage in range(4):
                    data['layerscale_by_stage'].append({'task':task,'recipe':recipe,'stage_zero_based':stage,
                        'mean_abs_before':stat([1e-6]*3),'mean_abs_after':stat([.01,.012,.014]),'max_abs_after':stat([.04,.05,.06])})
        for recipe in RECIPES[1:]:
            for mask in masks:
                for metric in METRICS:
                    diffs = [a-b for a,b in zip(curve(data,task,recipe,mask)[metric]['values'],curve(data,task,'dense',mask)[metric]['values'])]
                    row = {'task':task,'recipe':recipe,'reference':'dense','mask_name':mask,'metric':metric,'seeds':seeds,
                           'is_primary':mask=='primary_two_thirds' and metric=='excess_ce',**stat(diffs)}
                    data['paired_comparisons'].append(row)
                    if row['is_primary']:
                        data['primary_comparisons'].append(row)
    audit_seeds = [1000, *seeds]
    data['verification']['initialization_numerical_audit'] = {
        'passed':True, 'posthoc_verification_adjustment':True, 'task':'convnext_cifar100',
        'audited_seeds':audit_seeds, 'torch':'SYNTHETIC', 'vision_source_sha256':'b'*64,
        'thresholds':{'max_absolute_difference':1e-7,'relative_l2_difference':1e-6},
        'max_absolute_difference':7e-9, 'max_relative_l2_difference':5e-9,
        'evidence_files':[{'file':name,'sha256':'d'*64} for name in (
            'initialization-numerical-audit.json','initialization-variant-manifests.json',
            'initialization-worker-probes.json','initialization-kernel-probes.json')],
        'seed_audits':[{'seed':seed,'state_hashes':['e'*64,'f'*64],'parameters':27_897_028,
                       'changed_elements':100,'max_absolute_difference':7e-9,
                       'relative_l2_difference':5e-9,'rng_sha256':'c'*64} for seed in audit_seeds],
        'interpretation':'SYNTHETIC fixture only; no empirical numerical audit.'}
    data['verification']['initialization_pairing'] = [
        {'task':task,'seed':seed,'status':'audited_numerical_close' if task=='convnext_cifar100' else 'exact',
         'observed_hashes':['e'*64,'f'*64] if task=='convnext_cifar100' else ['e'*64],
         'run_hashes':{}} for task in TASKS for seed in seeds]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--scientific-digest", action="store_true", help="Print the stable scientific-input SHA-256 only; do not build the report")
    args = parser.parse_args()
    if args.self_test:
        data = synthetic_fixture()
        path = Path(tempfile.mkdtemp(prefix="a100-report-SYNTHETIC-"))
        build(data, json.dumps(data,indent=2).encode(), path, synthetic=True)
        print(f"SYNTHETIC renderer fixture only; inspect/check temporary output: {path}")
        return
    require(args.summary.exists(), "Verified summary.json is not available; no report written")
    raw = args.summary.read_bytes()
    data = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: "+value)))
    if args.scientific_digest:
        validate_summary(data)
        print(scientific_input_sha256(data))
        return
    build(data, raw, args.output)


if __name__ == "__main__":
    main()
