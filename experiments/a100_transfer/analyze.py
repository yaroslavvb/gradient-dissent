#!/usr/bin/env python3
"""Fail-closed verification and seed-level statistics for completed A100 runs.

Reads the locked evaluation manifest and raw JSON results; does not train,
contact a service, select masks, or silently omit missing/failed final runs.
Reports are written only after every expected final run passes verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPES = ("dense", "constant_ild", "decreasing_ild")
CORE_SOURCES = ("train.py", "language.py", "vision.py")
T_CRITICAL = {2: 12.7062047364, 3: 4.3026527299, 4: 3.1824463053,
              5: 2.7764451052, 6: 2.5705818356, 7: 2.4469118511,
              8: 2.3646242516, 9: 2.3060041352, 10: 2.2621571629}
TASK_LABELS = {"gpt_wikitext103": "GPT / WikiText-103",
               "vit_cifar100": "ViT / CIFAR-100",
               "convnext_cifar100": "ConvNeXt / CIFAR-100"}
REPO = "https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/a100_transfer"
# Post hoc initialization diagnosis, independent of trained/test outcomes. This
# is an exact allowlist of measured variants, never a general hash bypass.
CNN_INITIALIZATION_SOURCE = "edd014c05c041793025db1f64224de194ae0345c47b4e6720a631404f3848bf0"
CNN_INITIALIZATION_TORCH = "2.8.0+cu128"
CNN_INITIALIZATION_PAIRS = {
    1000: ("0b281e0b27872b80dabc53a2c9f5de1730e9d2de6801fbcaeeea4ed679d43939", "14ceace68e98d7bbbdf8c42894b2e2adba144734b024aadad678c73680370af8"),
    2000: ("779fb93eab45b3be10560e4ea5284fbcf4a3720743c4bf26ff434148ea8acc1a", "4aa96a3d184a042bf0599a2a4e9953a15ab04c99f2f6c00d677fbe8d7bf939e5"),
    2001: ("88bb29bc604531546ffba0538c2613605dca544fc9fc18349e5a6b0ce2d8b9e1", "b01dcf0da2b2e4b28c746d3b541771bb93dfc9fd47782eacf66177229d44e283"),
    2002: ("62e4cf6632275a306c622c0fb7c8f09c15d09d563d6d8a4d1437430148df41b8", "bf54efd51c77baa142139a5fb4f8f40fc00256565ec0f5a73ab3bc1803874bbe"),
}
CNN_INITIALIZATION_EVIDENCE = {
    "initialization-numerical-audit.json": "e2e8b144d47e7c465e7c178994b239dd2ccd523276025ab1628bdd191fd0a7e8",
    "initialization-variant-manifests.json": "64bd44b980b7d204be439208d007194713c9f152ca27a83642183d423ce28fe4",
    "initialization-worker-probes.json": "cc55ae7133d1e2f8434490e1220be68b12983326e7e53a3890bd4c6d7ae94489",
}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def load(path):
    require(path.exists(), f"Missing required file: {path}")
    def invalid(value):
        raise AuditError(f"Non-finite JSON constant {value} in {path}")
    return json.loads(path.read_text(), parse_constant=invalid)


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"Non-finite/non-numeric {label}: {value!r}")
    return float(value)


def close(a, b, label, tolerance=1e-7):
    require(abs(float(a)-float(b)) <= tolerance, f"Mismatch for {label}: {a} vs {b}")


def checked_hash(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid SHA-256 for {label}")
    return value


def cnn_tensor_inventory():
    """Independent shape inventory for the frozen 100-class ConvNeXt-Tiny."""
    shapes = {"downsample.0.0.weight": [96,3,4,4], "downsample.0.0.bias": [96],
              "downsample.0.1.norm.weight": [96], "downsample.0.1.norm.bias": [96],
              "norm.weight": [768], "norm.bias": [768], "head.weight": [100,768], "head.bias": [100]}
    for stage, (depth, width) in enumerate(zip((3,3,9,3), (96,192,384,768))):
        if stage:
            prefix = f"downsample.{stage}"
            shapes.update({prefix+".0.norm.weight": [width//2], prefix+".0.norm.bias": [width//2],
                           prefix+".1.weight": [width,width//2,2,2], prefix+".1.bias": [width]})
        for block in range(depth):
            prefix = f"stages.{stage}.{block}."
            shapes.update({prefix+"gamma": [width], prefix+"depthwise.weight": [width,1,7,7],
                           prefix+"depthwise.bias": [width], prefix+"norm.weight": [width], prefix+"norm.bias": [width],
                           prefix+"expand.weight": [4*width,width], prefix+"expand.bias": [4*width],
                           prefix+"project.weight": [width,4*width], prefix+"project.bias": [width]})
    require(len(shapes) == 182 and sum(math.prod(s) for s in shapes.values()) == 27897028,
            "Internal ConvNeXt audit inventory mismatch")
    return shapes


def verify_initialization_audit(results_dir, core_hashes):
    """Verify immutable numerical evidence before admitting known CNN hashes."""
    require(core_hashes["vision.py"] == CNN_INITIALIZATION_SOURCE,
            "Initialization audit applies only to the frozen vision source")
    evidence, provenance = {}, []
    for name, expected_sha in CNN_INITIALIZATION_EVIDENCE.items():
        path = results_dir/name
        value = load(path)
        digest = sha(path.read_bytes())
        require(digest == expected_sha, f"Initialization evidence changed: {name}; unaudited evidence is not accepted")
        evidence[name] = value
        provenance.append({"file": name, "sha256": digest})
    numerical = evidence["initialization-numerical-audit.json"]
    require(numerical.get("passed") is True and numerical.get("torch") == CNN_INITIALIZATION_TORCH and
            numerical.get("vision_source_sha256") == CNN_INITIALIZATION_SOURCE, "Initialization numerical audit version/status mismatch")
    thresholds = {"max_absolute_difference": 1e-7, "relative_l2_difference": 1e-6}
    require(numerical.get("thresholds") == thresholds, "Initialization audit tolerance changed")
    variants = evidence["initialization-variant-manifests.json"]
    require(len(variants) == 2 and {v["kind"] for v in variants} == {"avx512", "avx2"}, "Unexpected initialization variants")
    variants = {v["kind"]: v for v in variants}
    seeds = set(CNN_INITIALIZATION_PAIRS)
    rng = {}
    for index, kind in enumerate(("avx512", "avx2")):
        variant = variants[kind]
        require(variant.get("torch") == CNN_INITIALIZATION_TORCH and variant.get("source_sha256") == CNN_INITIALIZATION_SOURCE and
                variant.get("cpu_capability") == kind.upper() and variant.get("committed") is True,
                f"Initialization variant provenance mismatch: {kind}")
        rows = variant["rows"]
        require(len(rows) == len(seeds) and {r["seed"] for r in rows} == seeds, f"Initialization variant seeds mismatch: {kind}")
        for row in rows:
            seed = row["seed"]
            require(row["state_sha256"] == CNN_INITIALIZATION_PAIRS[seed][index], f"Unrecognized initialization hash for {kind}/{seed}")
            require(row["file"] == f"/work/initialization-audit/{kind}/{seed}.pt", "Initialization state artifact path mismatch")
            checked_hash(row["rng_sha256"], f"initialization RNG {kind}/{seed}")
            if seed in rng:
                require(rng[seed] == row["rng_sha256"], f"Post-initialization RNG differs for seed {seed}")
            rng[seed] = row["rng_sha256"]
    shapes = cnn_tensor_inventory()
    seed_audits = []
    rows = numerical["rows"]
    require(len(rows) == len(seeds) and {r["seed"] for r in rows} == seeds, "Numerical audit seed set mismatch")
    for row in rows:
        seed = row["seed"]
        require(row.get("task") == "convnext_cifar100" and row.get("passed") is True and
                tuple(row["state_hashes"]) == CNN_INITIALIZATION_PAIRS[seed], f"Numerical audit identity mismatch for seed {seed}")
        tensors = row["tensors"]
        require(len(tensors) == len(shapes) and {t["name"] for t in tensors} == set(shapes), f"Incomplete/duplicate tensor inventory for seed {seed}")
        maximum, squared_diff, squared_norm, numel, changed = 0., 0., 0., 0, 0
        for tensor in tensors:
            name, shape = tensor["name"], tensor["shape"]
            size = math.prod(shape)
            require(shape == shapes[name] and tensor["dtype"] == "torch.float32" and tensor["numel"] == size,
                    f"Numerical audit shape/dtype/count mismatch: {seed}/{name}")
            count = tensor["changed_elements"]
            require(isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= size, f"Invalid changed count: {seed}/{name}")
            mx = finite(tensor["max_absolute_difference"], f"{seed}/{name} max difference")
            sd = finite(tensor["squared_difference"], f"{seed}/{name} squared difference")
            sn = finite(tensor["squared_reference_norm"], f"{seed}/{name} squared norm")
            require(mx >= 0 and sd >= 0 and sn >= 0 and mx <= thresholds["max_absolute_difference"], f"Out-of-bound tensor difference: {seed}/{name}")
            require((count == 0) == (mx == 0 and sd == 0), f"Inconsistent zero-difference count: {seed}/{name}")
            require(mx*mx <= sd*(1+1e-12) and sd <= count*mx*mx*(1+1e-12), f"Inconsistent tensor difference norm: {seed}/{name}")
            maximum = max(maximum, mx)
            squared_diff += sd
            squared_norm += sn
            numel += size
            changed += count
        relative = math.sqrt(squared_diff/squared_norm)
        require(numel == row["parameters"] == 27897028 and changed == row["changed_elements"], f"Numerical audit aggregate counts differ for seed {seed}")
        require(maximum == row["max_absolute_difference"] and math.isclose(relative, row["relative_l2_difference"], rel_tol=1e-12, abs_tol=1e-20),
                f"Numerical audit aggregate bounds differ for seed {seed}")
        require(relative <= thresholds["relative_l2_difference"], f"Relative initialization difference exceeds bound for seed {seed}")
        seed_audits.append({"seed": seed, "state_hashes": row["state_hashes"], "parameters": numel, "changed_elements": changed,
                            "max_absolute_difference": maximum, "relative_l2_difference": relative, "rng_sha256": rng[seed]})
    probes = evidence["initialization-worker-probes.json"]["rows"]
    require(len(probes) == 9 and {p["state_sha256"] for p in probes} == set(CNN_INITIALIZATION_PAIRS[1000]), "Worker probes do not reproduce both allowed tuning hashes")
    for probe in probes:
        index = CNN_INITIALIZATION_PAIRS[1000].index(probe["state_sha256"])
        require(probe["seed"] == 1000 and probe["torch"] == CNN_INITIALIZATION_TORCH and
                probe["source_sha256"] == CNN_INITIALIZATION_SOURCE and probe["rng_sha256"] == rng[1000] and
                probe["cpu_capability"] == ("AVX512", "AVX2")[index], "Worker probe source/seed/RNG/version mismatch")
        require(len(probe["tensors"]) == len(shapes) and {t["name"]: t["shape"] for t in probe["tensors"]} == shapes,
                "Worker probe tensor inventory mismatch")
        for tensor in probe["tensors"]:
            checked_hash(tensor["sha256"], "worker probe tensor")
    return {"passed": True, "posthoc_verification_adjustment": True, "task": "convnext_cifar100", "audited_seeds": sorted(seeds),
            "torch": CNN_INITIALIZATION_TORCH, "vision_source_sha256": CNN_INITIALIZATION_SOURCE, "thresholds": thresholds,
            "max_absolute_difference": max(r["max_absolute_difference"] for r in seed_audits),
            "max_relative_l2_difference": max(r["relative_l2_difference"] for r in seed_audits),
            "evidence_files": provenance, "seed_audits": seed_audits,
            "interpretation": "Post hoc verification adjustment based only on reconstructed, untrained CPU states, independent of test outcomes. ConvNeXt permits only the two exact recorded hashes per audited seed, whose complete tensors are numerically close and whose post-init RNG states match. GPT/ViT require bit-identical paired initialization. This does not establish identical training trajectories or eliminate host-dependent numerical variation."}


def verify_initialization_pairing(runs, audit):
    require(bool(runs), "Empty initialization pairing group")
    task, seed = runs[0]["spec"]["task"], runs[0]["spec"]["seed"]
    require(all(r["spec"]["task"] == task and r["spec"]["seed"] == seed for r in runs), "Mixed initialization pairing group")
    hashes = {r["run_id"]: checked_hash(r["metadata"]["initial_state_sha256"], r["run_id"]+" initialization") for r in runs}
    observed = sorted(set(hashes.values()))
    if task == "convnext_cifar100":
        require(audit.get("passed") is True and seed in CNN_INITIALIZATION_PAIRS and set(observed) <= set(CNN_INITIALIZATION_PAIRS[seed]),
                f"{task}/{seed}: initialization hash is not an audited variant")
        for run in runs:
            require(run["metadata"].get("torch") == CNN_INITIALIZATION_TORCH and run["metadata"]["source_sha256"]["vision.py"] == CNN_INITIALIZATION_SOURCE,
                    f"{run['run_id']}: initialization audit does not cover this runtime/source")
    else:
        require(len(observed) == 1, f"{task}/{seed}: initial states are not bit-identically paired")
    return {"task": task, "seed": seed, "status": "exact" if len(observed) == 1 else "audited_numerical_close",
            "observed_hashes": observed, "run_hashes": hashes}


def interval(values):
    values = [finite(v, "interval value") for v in values]
    n = len(values)
    require(n in T_CRITICAL, f"Student-t critical value not declared for n={n}; no normal approximation substituted")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    radius = T_CRITICAL[n]*sd/math.sqrt(n)
    return {"n": n, "df": n-1, "mean": mean, "std": sd,
            "ci95_low": mean-radius, "ci95_high": mean+radius, "values": values}


def manifest_jobs(value):
    jobs = value if isinstance(value, list) else value.get("runs", value.get("jobs"))
    require(isinstance(jobs, list) and bool(jobs), "Evaluation manifest must contain a nonempty list of jobs")
    require(all(isinstance(j, dict) and j.get("stage") == "evaluate" for j in jobs),
            "Every locked final job must have stage=evaluate")
    ids = [j.get("run_id") for j in jobs]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), "Duplicate/missing run IDs in manifest")
    return jobs


def validate_design(jobs):
    by_task = defaultdict(list)
    for spec in jobs:
        require(spec["task"] in TASK_LABELS, f"Unknown family {spec['task']}")
        require(spec["recipe"] in RECIPES, f"Unknown recipe {spec['recipe']}")
        by_task[spec["task"]].append(spec)
    for task, group in by_task.items():
        cells = [(g["recipe"], g["seed"]) for g in group]
        require(len(cells) == len(set(cells)), f"Duplicate final recipe/seed cell in {task}")
        seed_sets = {recipe: {g["seed"] for g in group if g["recipe"] == recipe} for recipe in RECIPES}
        reference = seed_sets["dense"]
        require(reference and all(s == reference for s in seed_sets.values()), f"Unpaired recipe seeds in {task}")
        require(len(reference) in T_CRITICAL, f"Insufficient/unsupported number of final seeds in {task}")
        # Only recipe, selected LR, run ID, seed and harmless logging/time guards
        # may differ within a family. Data/training controls must match.
        equal_keys = ("steps", "batch_size", "context", "image_size", "weight_decay", "betas",
                      "validation_windows", "test_windows", "validation_examples", "eval_batch_size", "warmup_steps")
        for key in equal_keys:
            require(len({canonical(g.get(key)) for g in group}) == 1, f"Unequal {key} within family {task}")
        for recipe in RECIPES:
            require(len({g["lr"] for g in group if g["recipe"] == recipe}) == 1,
                    f"Selected LR differs across final seeds for {task}/{recipe}")
    return by_task


def expected_named_masks(task, n):
    require(n == (18 if task == "convnext_cifar100" else 12), f"Unexpected prunable count for {task}: {n}")
    masks = {"full": list(range(n))}
    if task == "convnext_cifar100":
        def stages(counts):
            return [j for start, count in zip((0, 3, 6, 15), counts) for j in range(start, start+count)]
        masks.update({"primary_two_thirds": stages((2, 2, 6, 2)), "one_third": stages((1, 1, 3, 1)),
                      "half": stages((2, 1, 4, 2)), "five_sixths": stages((3, 2, 8, 2))})
    else:
        masks["primary_two_thirds"] = list(range(2*n//3))
        masks.update({f"prefix_{k}": list(range(k)) for k in (n//3, n//2, 5*n//6)})
    masks["delete_first_only"] = list(range(1, n))
    return masks


def validate_rows(result):
    run_id, spec, metadata = result["run_id"], result["spec"], result["metadata"]
    rows = result.get("rows")
    require(isinstance(rows, list) and len(rows) == 9, f"{run_id}: final panel must have all9 masks")
    n = metadata["prunable_count"]
    expected = expected_named_masks(spec["task"], n)
    names = set(expected) | {f"random_keeps_first_{i}" for i in range(3)}
    require({r.get("mask_name") for r in rows} == names, f"{run_id}: missing/unexpected/duplicate mask names")
    indices_seen = []
    panel = result["test_panel"]
    targets = panel["target_tokens"] if spec["task"] == "gpt_wikitext103" else panel["examples"]
    require(isinstance(targets, int) and targets > 0, f"{run_id}: invalid test target count")
    if spec["task"] != "gpt_wikitext103":
        require(targets == 10000 and panel["split"] == "test", f"{run_id}: image final evaluation must use all10000 officialtest images")
    else:
        require(panel["split"] == "test" and panel["context"] == spec["context"], f"{run_id}: language test split/context mismatch")
        require(targets == panel["windows"]*spec["context"], f"{run_id}: language target/window mismatch")
        starts = panel["starts"]
        require(len(starts) == panel["windows"] and all(b-a >= spec["context"]+1 for a,b in zip(starts,starts[1:])),
                f"{run_id}: duplicated/overlapping or incorrectly counted test windows")
    full = next(r for r in rows if r["mask_name"] == "full")
    for row in rows:
        label = f"{run_id}/{row['mask_name']}"
        kept = row["kept_indices_zero_based"]
        require(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for i in kept), f"{label}: invalid retained indices")
        require(kept == sorted(set(kept)) and len(kept) == row["retained"], f"{label}: unordered/duplicate/count mismatch")
        require(row["targets"] == targets, f"{label}: target counts differ across masks")
        if row["mask_name"] in expected:
            require(kept == expected[row["mask_name"]], f"{label}: mismatch with prescribed structural intervention")
        else:
            require(len(kept) == 2*n//3 and 0 in kept, f"{label}: random mask must keep first and two-thirds of blocks")
        ce = finite(row["ce"], label+" CE")
        accuracy = finite(row["accuracy"], label+" accuracy")
        require(ce >= 0 and 0 <= accuracy <= 1, f"{label}: metric out of range")
        close(row["excess_ce"], ce-full["ce"], label+" excess CE", 1e-10)
        # Accuracy is computed from an integer number of correct predictions;
        # allow float32 division rounding accumulated over target count.
        require(abs(accuracy*targets-round(accuracy*targets)) <= max(.005, targets*1e-7), f"{label}: accuracy incompatible with target count")
        indices_seen.append(tuple(kept))
    require(len(indices_seen) == len(set(indices_seen)), f"{run_id}: duplicate structural masks in panel")
    close(full["excess_ce"], 0., run_id+" full anchor", 0.)
    return {r["mask_name"]: r for r in rows}


def validate_run(result, spec, dataset_hashes):
    run_id = spec["run_id"]
    require("error" not in result, f"Final run failed: {run_id}: {result.get('error')}")
    require(result.get("run_id") == run_id and result.get("spec") == spec, f"{run_id}: raw run/spec differs from locked manifest")
    metadata = result["metadata"]
    for key in ("initial_state_sha256", "dataset_manifest_sha256"):
        checked_hash(metadata.get(key), run_id+" "+key)
    require(metadata["dataset_manifest_sha256"] == dataset_hashes[spec["task"]], f"{run_id}: executed dataset hash differs from saved preparation manifest")
    checked_hash(result.get("training_data_stream_sha256"), run_id+" training stream")
    for source in CORE_SOURCES:
        checked_hash(metadata["source_sha256"].get(source), run_id+" "+source)
    correctness = result.get("correctness", {})
    require(correctness.get("gpu_all_kept_equals_default") is True and correctness.get("checks_use_validation_only") is True,
            f"{run_id}: missing passed GPU all-kept/default equivalence check")
    require(len(metadata["prunable_ids"]) == metadata["prunable_count"] and len(set(metadata["prunable_ids"])) == metadata["prunable_count"],
            f"{run_id}: invalid prunable IDs")
    steps, batch, n = spec["steps"], spec["batch_size"], metadata["prunable_count"]
    require(steps > 1 and batch > 0, f"{run_id}: invalid training size")
    close(result["nominal_mean_dropout"], 0. if spec["recipe"] == "dense" else .2, run_id+" nominal dropout", 1e-6)
    require(result["dense_example_blocks"] == steps*batch*n, f"{run_id}: dense work accounting mismatch")
    active = result["realized_active_example_blocks"]
    require(steps*batch <= active <= steps*batch*n, f"{run_id}: active work outside valid always-first-kept bounds")
    if spec["recipe"] == "dense":
        require(active == steps*batch*n, f"{run_id}: dense run skipped blocks")
    total = steps*batch*(spec["context"] if spec["task"] == "gpt_wikitext103" else 1)
    require(result["training_targets_or_images"] == total, f"{run_id}: training target count mismatch")
    curve = result["curve"]
    require(curve and curve[0]["step"] == 1 and curve[-1]["step"] == steps,
            f"{run_id}: did not complete declared terminal step")
    require(all(a["step"] < b["step"] for a,b in zip(curve,curve[1:])), f"{run_id}: invalid learning-curve step sequence")
    for point in curve:
        require(finite(point["train_ce"], run_id+" train CE") >= 0, f"{run_id}: negative train loss")
    for split in ("validation_before", "validation"):
        require(finite(result[split]["ce"], run_id+" "+split+" CE") >= 0, f"{run_id}: invalid validation CE")
        require(0 <= finite(result[split]["accuracy"], run_id+" "+split+" accuracy") <= 1, f"{run_id}: invalid validation accuracy")
    for key in ("train_seconds", "container_function_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        require(finite(result[key], run_id+" "+key) > 0, f"{run_id}: nonpositive resource metric")
    require(result["peak_allocated_bytes"] <= result["peak_reserved_bytes"], f"{run_id}: allocated memory exceeds reserved memory")
    return validate_rows(result)


def dataset_hashes(dataset):
    # Mirrors only the serialization contract, not the training computations.
    language = sha((json.dumps(dataset["language"], indent=2)+"\n").encode())
    # CPU preparation adds download/mirror provenance after load_cifar100 has
    # returned. The trainer fingerprints the loader metadata, which does not
    # contain that retrieval-only field. Preserve every other key verbatim.
    vision_metadata = {key: value for key, value in dataset["vision"].items() if key != "retrieval"}
    vision = sha(json.dumps(vision_metadata, sort_keys=True).encode())
    return {"gpt_wikitext103": language, "vit_cifar100": vision, "convnext_cifar100": vision}


def verify_tuning(results_dir, evaluation_jobs, core_hashes, data_hashes, initialization_audit=None):
    """Verify complete equal LR comparisons when a tuning manifest is present."""
    manifest = results_dir.parent / "tuning-manifest.json"
    if not manifest.exists():
        manifest = results_dir.parent / "tune-manifest.json"
    require(manifest.exists(), "Missing locked tuning-manifest.json (or tune-manifest.json); cannot verify selected learning rates")
    value = load(manifest)
    jobs = value if isinstance(value, list) else value.get("runs", value.get("jobs"))
    require(jobs and all(j["stage"] == "tune" for j in jobs), "Tuning manifest is empty or includes another stage")
    ids = [j["run_id"] for j in jobs]
    require(len(ids) == len(set(ids)), "Duplicate tuning run IDs")
    if initialization_audit is None:
        initialization_audit = verify_initialization_audit(results_dir, core_hashes)
    by_family = defaultdict(list)
    for spec in jobs:
        run = load(results_dir / (spec["run_id"]+".json"))
        require("error" not in run and run.get("spec") == spec, f"Tuning failure/spec mismatch: {spec['run_id']}")
        require(run.get("rows") == [] and run.get("test_panel") is None, f"Tuning run scored test data: {spec['run_id']}")
        metadata = run["metadata"]
        require({k: metadata["source_sha256"][k] for k in CORE_SOURCES} == core_hashes, f"Tuning/evaluation code mismatch: {spec['run_id']}")
        require(metadata["dataset_manifest_sha256"] == data_hashes[spec["task"]], f"Tuning/evaluation data mismatch: {spec['run_id']}")
        require(run["curve"][-1]["step"] == spec["steps"], f"Incomplete tuning run: {spec['run_id']}")
        finite(run["validation"]["ce"], spec["run_id"]+" tuning validation CE")
        accuracy = finite(run["validation"]["accuracy"], spec["run_id"]+" tuning validation accuracy")
        require(0 <= accuracy <= 1, f"Tuning validation accuracy outside[0,1]: {spec['run_id']}")
        by_family[spec["task"]].append(run)
    summaries, pairing = [], []
    for task in sorted({j["task"] for j in evaluation_jobs}):
        runs = by_family[task]
        require(runs, f"No tuning runs for {task}")
        grids = {recipe: {(r["spec"]["seed"], r["spec"]["lr"]) for r in runs if r["spec"]["recipe"] == recipe} for recipe in RECIPES}
        require(grids["dense"] and all(g == grids["dense"] for g in grids.values()), f"Unequal tuning grid or seeds for {task}")
        require(len(runs) == sum(len(grid) for grid in grids.values()), f"Duplicated tuning cells for {task}")
        tuning_seeds = sorted({s for s,_ in grids["dense"]})
        for seed in tuning_seeds:
            paired = [r for r in runs if r["spec"]["seed"] == seed]
            pairing.append(verify_initialization_pairing(paired, initialization_audit))
            require(len({checked_hash(r["training_data_stream_sha256"], r["run_id"]+" training stream") for r in paired}) == 1,
                    f"{task}/{seed}: tuning data streams are not paired")
            require(len({canonical(r["validation_panel"]) for r in paired}) == 1,
                    f"{task}/{seed}: tuning validation panels differ")
        final_seeds = {j["seed"] for j in evaluation_jobs if j["task"] == task}
        require(not final_seeds.intersection(tuning_seeds), f"Tuning and evaluation seeds overlap for {task}")
        rates = sorted({lr for _,lr in grids["dense"]})
        for recipe in RECIPES:
            scores = {lr: statistics.mean(r["validation"]["ce"] for r in runs if r["spec"]["recipe"] == recipe and r["spec"]["lr"] == lr) for lr in rates}
            accuracies = {lr: statistics.mean(r["validation"]["accuracy"] for r in runs if r["spec"]["recipe"] == recipe and r["spec"]["lr"] == lr) for lr in rates}
            chosen = min(scores, key=lambda lr: (scores[lr],lr))
            actual = {j["lr"] for j in evaluation_jobs if j["task"] == task and j["recipe"] == recipe}
            require(actual == {chosen}, f"Selected LR is not the full-validation winner for {task}/{recipe}: expected{chosen}, got{actual}")
            matched_steps = {r["spec"]["steps"] for r in runs if r["spec"]["recipe"] == recipe} == {j["steps"] for j in evaluation_jobs if j["task"] == task}
            summaries.append({"task": task, "recipe": recipe, "tuning_seeds": tuning_seeds,
                              "learning_rates": rates, "mean_validation_ce": scores,
                              "mean_validation_accuracy": accuracies, "selected_lr": chosen,
                              "selected_at_boundary": chosen in (rates[0], rates[-1]), "same_horizon_as_final": matched_steps})
    return summaries, {"file": manifest.name, "sha256": sha(manifest.read_bytes()), "runs": len(jobs),
                       "initialization_pairing": pairing}


def verify_budget(results_dir, expected_jobs):
    ledger_path = results_dir / "budget-ledger.json"
    ledger = load(ledger_path)
    reservations = ledger["reservations"]
    require(len({r["id"] for r in reservations}) == len(reservations), "Duplicate budget reservation IDs")
    total = sum(finite(r["reserved_usd"], "reservation USD") for r in reservations)
    close(total, ledger["reserved_upper_usd"], "reservation sum", 1e-6)
    require(total <= ledger["reservation_ceiling_usd"] <= ledger["cap_usd"] <= 50., "Reservation or authorization cap exceeded")
    by_id = {r["id"]: r for r in reservations}
    for job in expected_jobs:
        require(job["run_id"] in by_id, f"Missing reservation for final run {job['run_id']}")
        require(by_id[job["run_id"]]["spec_sha256"] == sha(json.dumps(job, sort_keys=True).encode()),
                f"Reservation spec hash mismatch for {job['run_id']}")
    billing_path = results_dir / "billing-latest.json"
    billing = load(billing_path) if billing_path.exists() else None
    if billing is not None:
        require(0 <= finite(billing["metered_cost_usd"], "metered USD") <= 50., "Metered usage snapshot exceeds authorization cap")
    return {"cap_usd": ledger["cap_usd"], "reserved_upper_usd": total, "reservation_count": len(reservations),
            "reservation_ceiling_usd": ledger["reservation_ceiling_usd"], "latest_metered_usage": billing,
            "ledger_sha256": sha(ledger_path.read_bytes()),
            "interpretation": "Reservations are conservative resource allocations, not invoices or measured spend. Metered usage is a timestamped potentially lagging snapshot, not a final invoice."}


def stage_layerscale(run):
    stages = defaultdict(lambda: {"before": [], "after": [], "max_after": []})
    before, after = run["layerscale_before"], run["layerscale_after"]
    require(set(before) == set(after), f"{run['run_id']}: LayerScale parameter set changed")
    for name, value in after.items():
        match = re.fullmatch(r"stages\.(\d+)\.(\d+)\.gamma", name)
        require(match is not None, f"Unexpected LayerScale parameter name: {name}")
        group = stages[int(match[1])]
        group["before"].append(finite(before[name]["mean_abs"], name+" initial LayerScale"))
        group["after"].append(finite(value["mean_abs"], name+" final LayerScale"))
        group["max_after"].append(finite(value["max_abs"], name+" max LayerScale"))
    if run["spec"]["task"] == "convnext_cifar100":
        require(set(stages) == {0,1,2,3} and [len(stages[s]["after"]) for s in range(4)] == [3,3,9,3],
                f"{run['run_id']}: missing ConvNeXt LayerScale records")
    else:
        require(not stages, f"Unexpected LayerScale records in {run['spec']['task']}")
    return {s: {"mean_abs_before": statistics.mean(v["before"]), "mean_abs_after": statistics.mean(v["after"]),
                "max_abs_after": max(v["max_after"])} for s,v in stages.items()}


def summarize(results_dir, manifest_path):
    jobs = manifest_jobs(load(manifest_path))
    design = validate_design(jobs)
    expected_ids = {j["run_id"] for j in jobs}
    missing = [j["run_id"] for j in jobs if not (results_dir/(j["run_id"]+".json")).exists()]
    require(not missing, "Missing final runs; no report written: "+", ".join(missing))
    for path in results_dir.glob("*.json"):
        value = load(path)
        if isinstance(value, dict) and value.get("spec", {}).get("stage") == "evaluate" and "error" not in value:
            require(value.get("run_id") in expected_ids, f"Unmanifested successful final run {path.name}; do not silently select final seeds")
    dataset_path = results_dir / "dataset-manifest.json"
    dataset = load(dataset_path)
    data_hashes = dataset_hashes(dataset)
    runs, masks, files = [], {}, []
    for spec in jobs:
        path = results_dir/(spec["run_id"]+".json")
        run = load(path)
        masks[run["run_id"]] = validate_run(run, spec, data_hashes)
        runs.append(run)
        files.append({"run_id": run["run_id"], "file": path.name, "sha256": sha(path.read_bytes())})
    code = {k: runs[0]["metadata"]["source_sha256"][k] for k in CORE_SOURCES}
    for run in runs:
        require({k: run["metadata"]["source_sha256"][k] for k in CORE_SOURCES} == code,
                f"Mixed executed code versions: {run['run_id']}")
    for name, digest in code.items():
        require(sha((results_dir.parent/name).read_bytes()) == digest,
                f"Current {name} differs from executed source; restore the actual executed version before publishing")
    initialization_audit = verify_initialization_audit(results_dir, code)
    initialization_pairing = []
    for task, group in design.items():
        family_runs = [r for r in runs if r["spec"]["task"] == task]
        reference = family_runs[0]
        for r in family_runs[1:]:
            for key in ("parameters", "model_config", "prunable_count", "prunable_ids", "dataset_manifest_sha256", "precision"):
                require(r["metadata"][key] == reference["metadata"][key], f"{task}: inconsistent {key}")
            require(r["test_panel"] == reference["test_panel"] and r["validation_panel"] == reference["validation_panel"],
                    f"{task}: heldout examples/windows differ")
            require({k: v["kept_indices_zero_based"] for k,v in masks[r["run_id"]].items()} ==
                    {k: v["kept_indices_zero_based"] for k,v in masks[reference["run_id"]].items()},
                    f"{task}: masks differ across recipes or seeds")
        for seed in sorted({j["seed"] for j in group}):
            paired = [r for r in family_runs if r["spec"]["seed"] == seed]
            initialization_pairing.append(verify_initialization_pairing(paired, initialization_audit))
            require(len({r["training_data_stream_sha256"] for r in paired}) == 1, f"{task}/{seed}: training streams are not paired")
    tuning, tune_manifest = verify_tuning(results_dir, jobs, code, data_hashes, initialization_audit)
    budget = verify_budget(results_dir, jobs)
    by_cell = {(r["spec"]["task"],r["spec"]["recipe"],r["spec"]["seed"]):r for r in runs}
    curves, comparisons, family_info, learning, gamma_rows = [], [], [], [], []
    for task in sorted(design):
        group = design[task]
        seeds = sorted({j["seed"] for j in group})
        first = by_cell[(task,"dense",seeds[0])]
        n = first["metadata"]["prunable_count"]
        spec = first["spec"]
        family_info.append({"task": task, "label": TASK_LABELS[task], "seeds": seeds,
                            "parameters": first["metadata"]["parameters"], "model_config": first["metadata"]["model_config"],
                            "gpu_models": sorted({r["metadata"].get("gpu", "unrecorded") for r in runs if r["spec"]["task"] == task}),
                            "precision": first["metadata"]["precision"],
                            "software": {key:first["metadata"].get(key) for key in ("python","torch","numpy")},
                            "steps": spec["steps"], "batch_size": spec["batch_size"], "context": spec.get("context"), "image_size": spec.get("image_size"),
                            "training_targets_or_images": first["training_targets_or_images"],
                            "image_epochs": first["training_targets_or_images"]/first["metadata"]["training_examples_or_corpus_tokens"] if task != "gpt_wikitext103" else None,
                            "tokens_per_parameter": first["training_targets_or_images"]/first["metadata"]["parameters"] if task == "gpt_wikitext103" else None,
                            "test_panel": first["test_panel"], "test_panel_sha256": sha(canonical(first["test_panel"]).encode()),
                            "dataset_manifest_sha256": data_hashes[task], "prunable_count": n,
                            "prunable_ids": first["metadata"]["prunable_ids"], "mask_panel": {k:v["kept_indices_zero_based"] for k,v in masks[first["run_id"]].items()}})
        for recipe in RECIPES:
            cell_runs = [by_cell[(task,recipe,seed)] for seed in seeds]
            learning.append({"task": task, "recipe": recipe, "lr": cell_runs[0]["spec"]["lr"],
                             "validation_before_ce": interval(r["validation_before"]["ce"] for r in cell_runs),
                             "validation_final_ce": interval(r["validation"]["ce"] for r in cell_runs),
                             "last_minibatch_ce": interval(r["curve"][-1]["train_ce"] for r in cell_runs),
                             "train_seconds": interval(r["train_seconds"] for r in cell_runs),
                             "peak_allocated_gib": interval(r["peak_allocated_bytes"]/2**30 for r in cell_runs),
                             "peak_reserved_gib": interval(r["peak_reserved_bytes"]/2**30 for r in cell_runs),
                             "clipped_step_fraction": interval(r["clipped_steps"]/r["spec"]["steps"] for r in cell_runs)})
            layers = [stage_layerscale(r) for r in cell_runs]
            if task == "convnext_cifar100":
                for stage in range(4):
                    gamma_rows.append({"task": task, "recipe": recipe, "stage_zero_based": stage,
                                       "mean_abs_before": interval(s[stage]["mean_abs_before"] for s in layers),
                                       "mean_abs_after": interval(s[stage]["mean_abs_after"] for s in layers),
                                       "max_abs_after": interval(s[stage]["max_abs_after"] for s in layers)})
            for mask_name in masks[first["run_id"]]:
                values = {metric: [] for metric in ("ce","accuracy","excess_ce","accuracy_change")}
                for run in cell_runs:
                    row = masks[run["run_id"]][mask_name]
                    full = masks[run["run_id"]]["full"]
                    for metric in ("ce","accuracy","excess_ce"):
                        values[metric].append(row[metric])
                    values["accuracy_change"].append(row["accuracy"]-full["accuracy"])
                curves.append({"task": task, "recipe": recipe, "mask_name": mask_name,
                               "retained": masks[first["run_id"]][mask_name]["retained"], "seeds": seeds,
                               **{metric:interval(v) for metric,v in values.items()}})
                if recipe != "dense":
                    for metric in values:
                        diffs = []
                        for seed, value in zip(seeds, values[metric]):
                            dense_run = by_cell[(task,"dense",seed)]
                            dense_row = masks[dense_run["run_id"]][mask_name]
                            dense_full = masks[dense_run["run_id"]]["full"]
                            dense_value = dense_row[metric] if metric != "accuracy_change" else dense_row["accuracy"]-dense_full["accuracy"]
                            diffs.append(value-dense_value)
                        comparisons.append({"task": task, "recipe": recipe, "reference": "dense", "mask_name": mask_name,
                                            "metric": metric, "seeds": seeds,
                                            "is_primary": mask_name == "primary_two_thirds" and metric == "excess_ce",
                                            **interval(diffs)})
    return {"title": "A100 depth robustness transfer experiment", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "verification": {"passed": True, "expected_final_runs": len(jobs), "complete_final_runs": len(runs),
                             "evaluation_manifest_sha256": sha(manifest_path.read_bytes()), "tuning_manifest": tune_manifest,
                             "dataset_preparation_manifest_sha256": sha(dataset_path.read_bytes()), "executed_core_source_sha256": code,
                             "initialization_numerical_audit": initialization_audit, "initialization_pairing": initialization_pairing,
                             "raw_files": files, "checks": ["exact locked run specs and complete paired cells", "no silently discarded successful final runs",
                                 "bit-identical GPT/ViT initialization; only exact allowlisted, numerically audited ConvNeXt variants", "identical paired data streams", "executed core source matches published source",
                                 "executed dataset manifests match preparation artifacts", "identical heldout panels and masks",
                                 "nominal20% mean omission for ILD", "completed terminal steps", "passed GPU full/allkeep equivalence",
                                 "unique prescribed mask panel and fixed target counts", "equal full-validation-only LR search", "reservation ledger arithmetic/specs"]},
            "statistical_method": "Within each architecture, paired treatment-minus-dense differences across independently trained seeds. Primary difference is (CEprimary−CEfull)treatment−(CEprimary−CEfull)dense.95% Student-t intervals with df=n−1; no mask/example pseudoreplication and no multiple-comparison adjustment.",
            "families": family_info, "tuning": tuning, "learning_and_resources": learning, "curves": curves,
            "paired_comparisons": comparisons, "primary_comparisons": [r for r in comparisons if r["is_primary"]],
            "layerscale_by_stage": gamma_rows, "budget": budget,
            "limitations": ["Few final training seeds; intervals are fragile and unadjusted across primary/secondary comparisons.",
                "Learning-rate selection uncertainty and heldout dataset uncertainty are not included in seed intervals; tuning uses its recorded independent seed count.",
                "ConvNeXt same-seed initial weights can differ by CPU host. A post hoc audit accepts only two measured, numerically close hashes per seed with matching post-init RNG; identical training trajectories are not established. GPT/ViT initialization remains bit-identically paired.",
                "Compute-then-mask executes dense branches; these runs do not establish FLOP, latency, memory, energy, or dollar savings.",
                "Peak reserved GPU memory can include allocator cache from previous calls or prevalidation; it is not the model's required VRAM. Peak allocated memory describes live tensors in this implementation, including resident data.",
                "Two-thirds retained residual blocks does not imply two-thirds FLOPs; ConvNeXt stage transitions always remain.",
                "Fixed image resizing adds no observed information; CIFAR spatial geometry differs from ImageNet.",
                "Small ConvNeXt LayerScale and undertraining can make deletion appear harmless; inspect full learning and recorded gamma magnitudes.",
                "Language uses its recorded fixed document-contained context windows, not canonical full-corpus perplexity.",
                "This is training from scratch under a short declared compute budget, not a reproduction of the target paper's LLM training scale."]}


def render_markdown(summary):
    out = ["# A100 depth robustness transfer experiment", "",
           f"Verified **{summary['verification']['complete_final_runs']} completed final runs** against the locked evaluation manifest. All reported statistics below derive from the saved raw results.", "",
           "The primary endpoint is treatment minus dense in **CE after the predefined two-thirds-depth intervention minus the same model's full-depth CE**. Negative values indicate less loss increase after pruning. Confidence intervals are paired 95% Student-t intervals across training seeds, unadjusted for multiple comparisons.", "",
           "| Family | Treatment | Seeds | Primary effect (CE) | Paired 95% interval | Individual paired effects |",
           "|---|---|---:|---:|---|---|"]
    for row in summary["primary_comparisons"]:
        out.append(f"| {TASK_LABELS[row['task']]} | {row['recipe']} | {row['n']} | {row['mean']:.5f} | [{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] | {', '.join(f'{x:.5f}' for x in row['values'])} |")
    out += ["", "## Full quality and the retained model", "",
            "CE is nats per target token for language and nats per image for vision. Accuracy is token top-1 for language and class top-1 for vision. Their raw scales should not be averaged across families.", "",
            "| Family | Treatment | Full CE | Primary CE | CE increase | Full accuracy | Primary accuracy | Accuracy change |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for family in summary["families"]:
        for recipe in RECIPES:
            rows = [r for r in summary["curves"] if r["task"] == family["task"] and r["recipe"] == recipe]
            full = next(r for r in rows if r["mask_name"] == "full")
            primary = next(r for r in rows if r["mask_name"] == "primary_two_thirds")
            out.append(f"| {family['label']} | {recipe} | {full['ce']['mean']:.5f} | {primary['ce']['mean']:.5f} | {primary['excess_ce']['mean']:.5f} | {100*full['accuracy']['mean']:.2f}% | {100*primary['accuracy']['mean']:.2f}% | {100*primary['accuracy_change']['mean']:+.3f} pp |")
    out += ["", "Full and primary CE/accuracy intervals, secondary mask curves, and all seed values are in [summary.json](summary.json). Paired differences in raw quality and accuracy are in [paired.csv](paired.csv). A narrow pruning-damage difference alone does not establish a full-quality improvement or equivalence.", "",
            "## Training exposure, memory and learning", "",
            "| Family | Parameters | Steps × batch | Exposure | Input | Primary retained blocks |",
            "|---|---:|---|---|---|---|"]
    for family in summary["families"]:
        language = family["task"] == "gpt_wikitext103"
        exposure = f"{family['training_targets_or_images']:,} tokens; {family['tokens_per_parameter']:.3f} tokens/parameter" if language else f"{family['training_targets_or_images']:,} image presentations; {family['image_epochs']:.2f} equivalent passes"
        shape = f"context {family['context']}" if language else f"{family['image_size']}×{family['image_size']}"
        kept = len(family["mask_panel"]["primary_two_thirds"])
        out.append(f"| {family['label']} | {family['parameters']:,} | {family['steps']} × {family['batch_size']} | {exposure} | {shape} | {kept}/{family['prunable_count']} |")
    out += ["", "| Family | Recipe | Initial validation CE | Final validation CE | Last minibatch CE | Peak allocated GiB | Peak reserved GiB | Mean train seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["learning_and_resources"]:
        out.append(f"| {TASK_LABELS[row['task']]} | {row['recipe']} | {row['validation_before_ce']['mean']:.4f} | {row['validation_final_ce']['mean']:.4f} | {row['last_minibatch_ce']['mean']:.4f} | {row['peak_allocated_gib']['mean']:.2f} | {row['peak_reserved_gib']['mean']:.2f} | {row['train_seconds']['mean']:.1f} |")
    out += ["", "Image sampling is with replacement; equivalent passes do not mean shuffled epochs. Memory and training time are provenance measurements for these runs, not a controlled efficiency comparison. Reserved memory can include allocator cache inherited from earlier calls or prevalidation; it is not a minimum VRAM requirement. Last-minibatch loss is noisy and is not full-training-set loss. Compute-then-mask evaluates all training branches. Every model keeps its original final normalization/head, and inference applies no inverse-survival scaling or classifier adaptation.", "",
            "## Learning-rate selection", "",
            "| Family | Recipe | Tuning seeds | Grid: LR → CE / accuracy | Selected LR | Grid boundary? | Full training horizon? |",
            "|---|---|---|---|---:|---|---|"]
    for row in summary["tuning"]:
        ce_by_lr = {float(k):v for k,v in row["mean_validation_ce"].items()}
        accuracy_by_lr = {float(k):v for k,v in row["mean_validation_accuracy"].items()}
        grid = "; ".join(f"{lr:g} → {ce_by_lr[lr]:.4f} / {100*accuracy_by_lr[lr]:.2f}%" for lr in row["learning_rates"])
        out.append(f"| {TASK_LABELS[row['task']]} | {row['recipe']} | {', '.join(map(str,row['tuning_seeds']))} | {grid} | {row['selected_lr']:g} | {'yes' if row['selected_at_boundary'] else 'no'} | {'yes' if row['same_horizon_as_final'] else 'no: proxy horizon'} |")
    out += ["", "Learning rates minimize terminal full-depth validation CE with equal grids per recipe. Accuracy is reported alongside CE to expose disagreements between the objectives; it does not affect selection. Selection uses the recorded tuning seeds, separate from final seeds. A one-seed search and boundary winners add uncertainty not represented by the final seed intervals."]
    if summary["layerscale_by_stage"]:
        out += ["", "## ConvNeXt LayerScale diagnostic", "",
                "Small residual scales can create trivial pruning robustness in an undertrained network. The table reports means of absolute gamma within each stage, then across seeds; maxima are the per-run stage maxima averaged across seeds. This diagnostic complements full-model learning and does not by itself establish useful learned residual computation.", "",
                "| Recipe | Stage (zero-based) | Initial mean absolute gamma | Final mean absolute gamma | Mean of per-seed stage maxima |",
                "|---|---:|---:|---:|---:|"]
        for row in summary["layerscale_by_stage"]:
            out.append(f"| {row['recipe']} | {row['stage_zero_based']} | {row['mean_abs_before']['mean']:.6g} | {row['mean_abs_after']['mean']:.6g} | {row['max_abs_after']['mean']:.6g} |")
    budget = summary["budget"]
    out += ["", "## Budget and audit", "",
            f"The authorization cap is **${budget['cap_usd']:.2f}**. The ledger contains {budget['reservation_count']} invocation reservations totaling **${budget['reserved_upper_usd']:.4f}**. Reservations are conservative allocations, **not invoices or measured spend**."]
    billing = budget["latest_metered_usage"]
    if billing:
        queried = datetime.fromtimestamp(billing["queried_unix"], timezone.utc).isoformat()
        out += ["", f"The latest saved Modal metered-usage snapshot is **${billing['metered_cost_usd']:.4f}**, queried {queried}. It may lag and is not a final invoice."]
    else:
        out += ["", "No metered-usage snapshot was available; measured spend is not inferred from reservations."]
    out += ["", f"[Budget ledger]({REPO}/results/budget-ledger.json) · [Locked evaluation manifest]({REPO}/evaluation-manifest.json) · [Analysis source]({REPO}/analyze.py)", "",
            "Verification requires every locked final run, bit-identical GPT/ViT initialization, exact allowlisted numerical variants for ConvNeXt initialization, identical paired data streams, executed data/source hashes matching the saved artifacts, completed steps, unique fixed mask panels, identical target counts, and passed GPU full-mask/default equivalence. Raw-file SHA-256 values are included in summary.json. Failed attempts and earlier pilots remain in the result/ledger audit trail and are not statistical replicates.", "",
            "## Initialization audit qualification", ""]
    initialization = summary["verification"]["initialization_numerical_audit"]
    out += [f"A **post hoc verification adjustment**, based on untrained CPU initializations and independent of test outcomes, permits two exact recorded ConvNeXt initialization hashes for each of seeds {', '.join(map(str,initialization['audited_seeds']))}. Every tensor was compared across the two variants: maximum absolute difference **{initialization['max_absolute_difference']:.10g}**, maximum relative L2 difference **{initialization['max_relative_l2_difference']:.10g}**, with identical post-initialization RNG states. The analyzer recomputes these bounds from per-tensor statistics and checks pinned evidence SHA-256 values, the frozen source, PyTorch version, shapes, dtypes, and the full parameter count. Unknown hashes fail verification. GPT and ViT retain bit-identical paired initialization.", "",
            "These are seed-paired, numerically close ConvNeXt initializations, not bit-identical weights across all runs. This evidence does not show that subsequent training trajectories are identical or quantify downstream host effects. The LR grid, CE-only selection, evaluation seeds, training code and primary endpoint were unchanged.", "",
            " · ".join(f"[{p['file']}]({REPO}/results/{p['file']})" for p in initialization["evidence_files"]), "",
            "## Limits of the result", ""]
    out.extend("- "+limit for limit in summary["limitations"])
    out += ["", "[Vision implementation/provenance]({}/vision-notes.md) · [Language implementation/provenance]({}/language-notes.md)".format(REPO,REPO), ""]
    return "\n".join(out)


def paired_csv(summary):
    buffer = io.StringIO()
    fields = ["task","recipe","reference","mask_name","metric","is_primary","n","df","mean","std","ci95_low","ci95_high","seeds_json","seed_differences_json"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in summary["paired_comparisons"]:
        writer.writerow({**{key:row[key] for key in fields if key not in ("seeds_json","seed_differences_json")},
                         "seeds_json": json.dumps(row["seeds"]), "seed_differences_json": json.dumps(row["values"])})
    return buffer.getvalue()


def self_test():
    result = interval([1.,2.,3.])
    close(result["mean"], 2., "synthetic paired mean")
    close(result["ci95_high"]-2., 4.3026527299/math.sqrt(3), "three-seed t interval")
    require(expected_named_masks("convnext_cifar100",18)["primary_two_thirds"] == [0,1,3,4,6,7,8,9,10,11,15,16], "ConvNeXt primary mask")
    require(expected_named_masks("vit_cifar100",12)["primary_two_thirds"] == list(range(8)), "ViT primary mask")
    for bad in ([1.], [1.,float("nan"),3.]):
        try:
            interval(bad)
        except AuditError:
            pass
        else:
            raise AssertionError("Failed to reject invalid statistical input")
    try:
        manifest_jobs([])
    except AuditError:
        pass
    else:
        raise AssertionError("Failed to reject empty manifest")
    print("PASS: independent t calculation, prescribed primary masks, invalid-statistics/empty-manifest rejection. Synthetic self-test only; no report files written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE/"evaluation-manifest.json")
    parser.add_argument("--results-dir", type=Path, default=HERE/"results")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        summary = summarize(args.results_dir, args.manifest)
        # Prepare every output fully before replacing any prior artifact.
        outputs = {"summary.json": json.dumps(summary, indent=2, allow_nan=False)+"\n",
                   "paired.csv": paired_csv(summary), "REPORT.md": render_markdown(summary)}
        if not args.check_only:
            for name, content in outputs.items():
                temporary = args.results_dir/(name+".tmp")
                temporary.write_text(content)
            for name in outputs:
                (args.results_dir/(name+".tmp")).replace(args.results_dir/name)
        print(f"PASS: {summary['verification']['complete_final_runs']} final runs verified; "+("no outputs written" if args.check_only else "summary.json, paired.csv, REPORT.md written"))
    except (AuditError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"AUDIT FAILED: {error}. No new report was published.", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
