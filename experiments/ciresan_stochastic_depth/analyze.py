"""Read-only, standard-library analysis of the Ciresan-width MNIST study.

By default, only pilot/tune result filenames named in local manifests are opened.
Official-test result files require --final. No dataset, checkpoint, cloud API, or
W&B connection is opened. Outputs are sanitized aggregates and scientific scalars.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent
RECIPE_ORDER = ("plain", "residual", "sd_constant", "sd_annealed", "residual_unit_dropout")
LABELS = {"plain": "Plain MLP", "residual": "Residual dense",
          "sd_constant": "Residual + constant SD", "sd_annealed": "Residual + decreasing SD",
          "residual_unit_dropout": "Residual + unit dropout", "unit_dropout": "Plain + unit dropout"}
SPEC_KEYS = ("run_id", "stage", "cohort", "variant", "recipe", "seed", "lr", "epochs",
             "batch_size", "eval_every", "train_size", "momentum", "pmax", "input_scale",
             "output_relu", "shrinkage", "precision")
HARDWARE_KEYS = ("gpu", "torch", "cuda_runtime", "tf32", "float32_matmul_precision", "dtype", "cpu_threads")
TIMING_KEYS = ("training_seconds", "total_run_seconds", "selected_training_seconds",
               "training_seconds_to_validation_98", "epoch_preparation_seconds", "graph_setup_seconds")
HASH = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
# Two-sided 95% Student-t quantiles; no normal approximation for tiny seed sets.
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
        6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
        11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
        16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
        21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899, 25: 2.059539,
        26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230, 30: 2.042272}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def stats(values, ci=False):
    """Sample SD is undefined for n<2; missing/censored observations stay missing."""
    values = [float(x) for x in values if numeric(x)]
    n = len(values)
    mean = statistics.mean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    result = {"n": n, "mean": mean, "sample_sd": sd, "values": values}
    if ci:
        critical = T975.get(n - 1)
        half = critical * sd / math.sqrt(n) if critical is not None and sd is not None else None
        result.update(df=n - 1 if n else None, t_critical=critical,
                      ci95_low=mean - half if half is not None else None,
                      ci95_high=mean + half if half is not None else None)
    return result


def issue(run_id, code, detail=None):
    result = {"run_id": run_id, "code": code}
    if detail is not None:
        result["detail"] = detail
    return result


def manifest_jobs(path):
    data = read_json(path)
    jobs = data if isinstance(data, list) else data.get("runs", data.get("jobs", [])) if isinstance(data, dict) else None
    if not isinstance(jobs, list) or not all(isinstance(x, dict) for x in jobs):
        raise ValueError("Manifest must contain a list of run specifications")
    for spec in jobs:
        if not isinstance(spec.get("run_id"), str) or not SAFE_ID.fullmatch(spec["run_id"]):
            raise ValueError("Manifest contains an invalid run_id")
    return jobs


def inventory(results_dir, manifests_dir, final, final_manifest):
    """Open only controlled-study manifest IDs, including when final is enabled.

    Archived tuning, optimization, and qualification files are never parsed here.
    The explicit final gate is resolved before opening any evaluate-stage result.
    """
    manifests, specs, problems = [], {}, []
    paths = [manifests_dir / name for name in ("pilot-manifest.json", "graph-tune-manifest.json")]
    if final:
        paths.append(final_manifest)
    for path in paths:
        if not path.exists():
            continue
        try:
            jobs = manifest_jobs(path)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            problems.append(issue(None, "invalid_manifest", path.name))
            continue
        manifests.append({"file": path.name, "sha256": digest(path), "runs": len(jobs)})
        for spec in jobs:
            allowed_stage = "evaluate" if path == final_manifest else "pilot" if path.name == "pilot-manifest.json" else "tune"
            if spec.get("stage") != allowed_stage:
                problems.append(issue(spec["run_id"], "manifest_stage_mismatch", path.name))
                continue
            if spec["run_id"] in specs and specs[spec["run_id"]] != spec:
                problems.append(issue(spec["run_id"], "conflicting_manifest_specs"))
            elif spec["run_id"] in specs:
                problems.append(issue(spec["run_id"], "duplicate_manifest_run_id"))
            specs[spec["run_id"]] = spec
    paths = [results_dir / (rid + ".json") for rid in sorted(specs) if (results_dir / (rid + ".json")).exists()]
    ignored = sorted(p.name for p in results_dir.glob("*.json") if p.stem not in specs and p.name != "analysis.json")
    records, sources = [], []
    for path in paths:
        try:
            data = read_json(path)
        except (ValueError, UnicodeError):
            problems.append(issue(path.stem, "invalid_result_json"))
            continue
        if not isinstance(data, dict) or not isinstance(data.get("spec"), dict):
            problems.append(issue(path.stem, "missing_result_spec"))
            continue
        spec = data["spec"]
        if not final and spec.get("stage") not in ("pilot", "tune"):
            problems.append(issue(path.stem, "result_stage_disagrees_with_nonfinal_manifest"))
            continue
        if data.get("run_id") != spec.get("run_id") or path.stem != spec.get("run_id"):
            problems.append(issue(path.stem, "result_run_id_mismatch"))
            continue
        if spec["run_id"] in specs and spec != specs[spec["run_id"]]:
            problems.append(issue(spec["run_id"], "result_spec_disagrees_with_manifest"))
            continue
        if spec.get("stage") != "evaluate" and any(k in data for k in ("selected_test", "final_test")):
            problems.append(issue(spec["run_id"], "unexpected_test_fields_in_nonfinal_result"))
            continue
        sources.append({"file": path.name, "run_id": spec["run_id"], "sha256": digest(path)})
        records.append(data)
    return records, sources, specs, manifests, problems, ignored


def hardware_record(raw):
    """Only scientific runtime identifiers; never copy host IDs or environment."""
    hardware = raw.get("hardware", {})
    return {k: hardware[k] for k in HARDWARE_KEYS if k in hardware and isinstance(hardware[k], (str, int, float, bool))}


def timing_identity(run):
    hardware = run.get("hardware", {})
    source = run.get("source_sha256", {})
    identity = {"hardware": hardware, "source_sha256": source}
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def matched_timing(a, b):
    required = ("gpu", "torch", "cuda_runtime", "dtype", "tf32", "cpu_threads")
    return (all(k in r.get("hardware", {}) for r in (a, b) for k in required)
            and bool(a.get("source_sha256")) and bool(b.get("source_sha256"))
            and timing_identity(a) == timing_identity(b))


def check_evaluation(value, name, problems, rid, with_errors=False):
    if not isinstance(value, dict):
        problems.append(issue(rid, "missing_evaluation", name))
        return None
    n = value.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        problems.append(issue(rid, "invalid_evaluation_count", name))
        return None
    if not numeric(value.get("loss")) or value["loss"] < 0 or not numeric(value.get("accuracy")) or not 0 <= value["accuracy"] <= 1:
        problems.append(issue(rid, "invalid_evaluation_metric", name))
        return None
    result = {"loss": value["loss"], "accuracy": value["accuracy"], "accuracy_pct": 100 * value["accuracy"], "n": n}
    if with_errors:
        wrong, errors = value.get("wrong_indices"), value.get("errors")
        valid_indices = (isinstance(wrong, list) and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < n for x in wrong))
        if (not valid_indices or len(set(wrong)) != len(wrong) or not isinstance(errors, int)
                or isinstance(errors, bool) or errors != len(wrong)
                or not math.isclose(value["accuracy"], (n - errors) / n, rel_tol=0, abs_tol=1e-9)):
            problems.append(issue(rid, "test_error_indices_inconsistent", name))
            return None
        result.update(errors=errors, wrong_indices_verified=True)
    return result


def summarize_run(raw, final):
    spec, problems = raw["spec"], []
    rid = spec["run_id"]
    sources = {k: v for k, v in raw.get("source_sha256", {}).items()
               if k in ("train_optimized.py", "model_data.py", "optimization/stochastic_graph.py",
                        "optimization/test_stochastic_graph.py", "train.py") and isinstance(v, str) and HASH.fullmatch(v)}
    result = {"run_id": rid, "spec": {k: spec[k] for k in SPEC_KEYS if k in spec}, "status": "ok",
              "hardware": hardware_record(raw), "source_sha256": sources}
    if "error" in raw or raw.get("diverged"):
        result["status"] = "diverged" if raw.get("diverged") else "failed"
        # Never copy arbitrary exception text, which can include paths or secrets.
        code = "training_diverged" if raw.get("diverged") else "job_failed"
        error_type = raw.get("error_type")
        result["failure"] = issue(rid, code, error_type if isinstance(error_type, str) and re.fullmatch(r"[A-Za-z0-9_.]+", error_type) else None)
        result["training_seconds"] = raw.get("training_seconds") if numeric(raw.get("training_seconds")) else None
        result["total_run_seconds"] = raw.get("total_run_seconds") if numeric(raw.get("total_run_seconds")) else None
        rows = [r for r in raw.get("history", []) if isinstance(r, dict)]
        result["stopped_epoch"] = max((r.get("epoch", 0) for r in rows if isinstance(r.get("epoch"), int)), default=None)
        # Partial observations describe the failed trajectory, never an LR candidate.
        observed = [r for r in rows if isinstance(r.get("validation"), dict)]
        if observed:
            result["last_observed_validation"] = check_evaluation(observed[-1]["validation"], "last_observed_validation", problems, rid)
            result["last_observed_validation_epoch"] = observed[-1].get("epoch")
        return result, [result["failure"]]
    rows = raw.get("history", [])
    good = [r for r in rows if isinstance(r, dict) and isinstance(r.get("validation"), dict)]
    if not good:
        return result | {"status": "invalid"}, [issue(rid, "missing_validation_history")]
    epochs = [r.get("epoch") for r in good]
    if any(not isinstance(x, int) or x <= 0 for x in epochs) or epochs != sorted(set(epochs)):
        problems.append(issue(rid, "invalid_validation_epochs"))
    checked = []
    for row in good:
        val = check_evaluation(row.get("validation"), "validation", problems, rid)
        probe = check_evaluation(row.get("dense_train_probe"), "dense_train_probe", problems, rid)
        if val is not None and probe is not None:
            checked.append({"epoch": row["epoch"], "validation": val, "dense_train_probe": probe,
                            "training_seconds": row.get("training_seconds"),
                            "stochastic_training_loss": row.get("stochastic_training_loss")})
    if not checked:
        return result | {"status": "invalid"}, problems
    selected = min(checked, key=lambda r: r["validation"]["loss"])
    last = checked[-1]
    if raw.get("best_epoch") != selected["epoch"] or not numeric(raw.get("best_validation_loss")) or not math.isclose(raw["best_validation_loss"], selected["validation"]["loss"], rel_tol=1e-10, abs_tol=1e-12):
        problems.append(issue(rid, "checkpoint_selection_inconsistent"))
    if last["epoch"] != spec.get("epochs"):
        problems.append(issue(rid, "training_horizon_incomplete"))
    times = [r["training_seconds"] for r in checked]
    if any(not numeric(t) or t < 0 for t in times) or (all(numeric(t) for t in times) and times != sorted(times)):
        problems.append(issue(rid, "invalid_training_times"))
    total_train = raw.get("training_seconds")
    if not numeric(total_train) or total_train <= 0 or (numeric(times[-1]) and not math.isclose(total_train, times[-1], rel_tol=1e-8, abs_tol=1e-6)):
        problems.append(issue(rid, "training_runtime_inconsistent"))
    hashes = {"initial_parameters_sha256": raw.get("initial_parameters_sha256"),
              "training_data_sha256": raw.get("dataset", {}).get("training_data_sha256"),
              "train_indices_sha256": raw.get("dataset", {}).get("train_indices_sha256"),
              "val_indices_sha256": raw.get("dataset", {}).get("val_indices_sha256")}
    if spec.get("stage") in ("tune", "evaluate") or "executed_epoch_order_sha256" in raw:
        hashes["executed_epoch_order_sha256"] = raw.get("executed_epoch_order_sha256")
    for key, value in hashes.items():
        if not isinstance(value, str) or not HASH.fullmatch(value):
            problems.append(issue(rid, "missing_or_invalid_hash", key))
    threshold = next((r for r in checked if r["validation"]["accuracy"] >= .98), None)
    result.update(hashes=hashes, parameter_count=raw.get("parameter_count"),
                  steps_per_epoch=raw.get("steps_per_epoch"), training_examples_per_epoch=raw.get("training_examples_per_epoch"),
                  training_seconds=total_train, total_run_seconds=raw.get("total_run_seconds"),
                  selected_epoch=selected["epoch"], last_epoch=last["epoch"],
                  validation_98_reached=threshold is not None,
                  epoch_to_validation_98=threshold["epoch"] if threshold else None,
                  training_seconds_to_validation_98=threshold["training_seconds"] if threshold else None,
                  selected_training_seconds=selected["training_seconds"],
                  selected_validation=selected["validation"], final_validation=last["validation"],
                  selected_dense_train_probe=selected["dense_train_probe"], final_dense_train_probe=last["dense_train_probe"],
                  final_stochastic_training_loss=last["stochastic_training_loss"],
                  skipped_batch_counts=raw.get("skipped_batch_counts"))
    for key in ("epoch_preparation_seconds", "graph_setup_seconds"):
        result[key] = raw.get(key) if numeric(raw.get(key)) else None
    result["executed_epoch_masks_sha256"] = raw.get("executed_epoch_masks_sha256") if isinstance(raw.get("executed_epoch_masks_sha256"), str) and HASH.fullmatch(raw["executed_epoch_masks_sha256"]) else None
    if final and spec.get("stage") == "evaluate":
        for endpoint in ("selected", "final"):
            train = check_evaluation(raw.get(endpoint + "_train"), endpoint + "_train", problems, rid)
            test = check_evaluation(raw.get(endpoint + "_test"), endpoint + "_test", problems, rid, with_errors=True)
            if train and train["n"] != raw.get("dataset", {}).get("train_size"):
                problems.append(issue(rid, "full_train_count_mismatch", endpoint))
            if test and test["n"] != raw.get("dataset", {}).get("test_size"):
                problems.append(issue(rid, "test_count_mismatch", endpoint))
            if train and test:
                val = result[endpoint + "_validation"]
                result[endpoint + "_train"], result[endpoint + "_test"] = train, test
                result[endpoint + "_generalization"] = {
                    "validation_minus_train_loss": val["loss"] - train["loss"],
                    "test_minus_train_loss": test["loss"] - train["loss"],
                    "train_minus_validation_accuracy_pp": 100 * (train["accuracy"] - val["accuracy"]),
                    "train_minus_test_accuracy_pp": 100 * (train["accuracy"] - test["accuracy"])}
        if not raw.get("dataset", {}).get("test_included"):
            problems.append(issue(rid, "evaluation_missing_test_dataset_provenance"))
    elif raw.get("dataset", {}).get("test_included"):
        problems.append(issue(rid, "nonfinal_run_opened_test_dataset"))
    result["status"] = "invalid" if problems else "ok"
    return result, problems


def group_name(run):
    spec = run["spec"]
    if spec.get("cohort") == "main":
        return spec["recipe"]
    if spec.get("variant"):
        return spec["variant"]
    return "{recipe}__{head}__scale{scale:g}__shrink{shrink:g}".format(
        recipe=spec["recipe"], head="relu_head" if spec.get("output_relu", False) else "linear_head",
        scale=spec.get("input_scale", 1 / 255), shrink=spec.get("shrinkage", 0))


def aggregate_group(all_runs):
    good = [r for r in all_runs if r["status"] == "ok"]
    result = {"label": LABELS.get(all_runs[0]["spec"]["recipe"], all_runs[0]["spec"]["recipe"]),
              "run_ids": [r["run_id"] for r in all_runs], "valid_run_ids": [r["run_id"] for r in good],
              "seeds": [r["spec"]["seed"] for r in good], "n": len(good),
              "selected_learning_rates": sorted(set(r["spec"]["lr"] for r in good)),
              "failure_count": len(all_runs) - len(good), "metrics": {}, "endpoints": {},
              "timing_by_hardware": [],
              "timing_interpretation": "Pooled runtime means are descriptive across assigned workers. Use timing_by_hardware and hardware-matched seed pairs for speed comparisons; hardware assignment was not randomized."}
    for key in ("training_seconds", "total_run_seconds", "selected_epoch", "selected_training_seconds",
                "epoch_to_validation_98", "training_seconds_to_validation_98", "final_stochastic_training_loss",
                "parameter_count", "training_examples_per_epoch", "steps_per_epoch", "epoch_preparation_seconds", "graph_setup_seconds"):
        result["metrics"][key] = stats(r.get(key) for r in good)
    timing_groups = defaultdict(list)
    for r in good:
        timing_groups[timing_identity(r)].append(r)
    for key, rows in sorted(timing_groups.items()):
        result["timing_by_hardware"].append({**json.loads(key), "run_ids": [r["run_id"] for r in rows],
            "seeds": [r["spec"]["seed"] for r in rows], "metrics": {k: stats(r.get(k) for r in rows) for k in TIMING_KEYS}})
    result["validation_98"] = {"reached": sum(r["validation_98_reached"] for r in good),
                               "not_reached": sum(not r["validation_98_reached"] for r in good),
                               "interpretation": "First scheduled validation observation >=98%; means include achievers only, not unreached/censored runs."}
    for endpoint in ("selected", "final"):
        entry = {}
        for split in ("train", "validation", "test", "dense_train_probe"):
            available = [r[endpoint + "_" + split] for r in good if endpoint + "_" + split in r]
            if available:
                entry[split] = {key: stats(x[key] for x in available if key in x)
                                for key in ("loss", "accuracy_pct", "errors") if any(key in x for x in available)}
                entry[split]["example_counts"] = sorted(set(x["n"] for x in available))
        gap = [r[endpoint + "_generalization"] for r in good if endpoint + "_generalization" in r]
        if gap:
            entry["generalization"] = {key: stats(x[key] for x in gap) for key in gap[0]}
        result["endpoints"][endpoint] = entry
    return result


def pairing_checks(runs):
    groups = defaultdict(list)
    for r in runs:
        if r["spec"].get("stage") == "evaluate" and r["spec"].get("cohort") in ("main", "fidelity") and r["status"] == "ok":
            groups[(r["spec"]["cohort"], r["spec"]["seed"])].append(r)
    checks, problems = [], []
    for (cohort, seed), rows in sorted(groups.items()):
        names = [group_name(r) for r in rows]
        check = {"cohort": cohort, "seed": seed, "run_ids": [r["run_id"] for r in rows], "n": len(rows), "hashes": {}}
        if len(names) != len(set(names)):
            problems.append(issue(None, "duplicate_recipe_seed", {"cohort": cohort, "seed": seed}))
        for key in ("initial_parameters_sha256", "training_data_sha256", "train_indices_sha256", "val_indices_sha256", "executed_epoch_order_sha256"):
            vals = {r["run_id"]: r["hashes"][key] for r in rows}
            matches = len(set(vals.values())) == 1
            check["hashes"][key] = {"matches": matches, "run_hashes": vals}
            if not matches:
                problems.append(issue(None, "paired_hash_mismatch", {"cohort": cohort, "seed": seed, "field": key}))
        exposure = {(r["spec"].get("epochs"), r["spec"].get("batch_size", 64), r["training_examples_per_epoch"]) for r in rows}
        check["same_training_exposure"] = len(exposure) == 1
        if len(exposure) != 1:
            problems.append(issue(None, "paired_training_exposure_mismatch", {"cohort": cohort, "seed": seed}))
        check["passed"] = (len(rows) >= 2 and len(names) == len(set(names)) and check["same_training_exposure"]
                           and all(x["matches"] for x in check["hashes"].values()))
        check["interpretation"] = "Hashes verify exact initialization, training data, split indices, and the ordered stream of all executed epoch permutations. Equal batch size/drop-last then fixes minibatch membership/order; mask hashes intentionally differ between recipes."
        checks.append(check)
    return checks, problems


def paired_comparisons(runs, raw_by_id, checks):
    """Only verified, same-cohort/seed pairs; no pooled examples or cross-cohort pairing."""
    passed = {(c["cohort"], c["seed"]) for c in checks if c["passed"]}
    by_cohort = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        s = r["spec"]
        if r["status"] == "ok" and s.get("stage") == "evaluate" and (s.get("cohort"), s["seed"]) in passed:
            by_cohort[s["cohort"]][group_name(r)][s["seed"]] = r
    output = []
    for cohort, groups in sorted(by_cohort.items()):
        controls = [name for name, rows in groups.items() if next(iter(rows.values()))["spec"]["recipe"] == "residual"]
        if len(controls) != 1:
            continue
        control = controls[0]
        for name, rows in sorted(groups.items()):
            if name == control:
                continue
            seeds = sorted(set(rows) & set(groups[control]))
            pairs = [(rows[s], groups[control][s]) for s in seeds]
            for endpoint in ("selected", "final"):
                valid, disagreements = [], []
                for a, b in pairs:
                    at, bt = a.get(endpoint + "_test"), b.get(endpoint + "_test")
                    if not at or not bt or at["n"] != bt["n"]:
                        continue
                    valid.append((a, b))
                    ai = set(raw_by_id[a["run_id"]][endpoint + "_test"]["wrong_indices"])
                    bi = set(raw_by_id[b["run_id"]][endpoint + "_test"]["wrong_indices"])
                    disagreements.append({"seed": a["spec"]["seed"], "test_n": at["n"],
                                          "treatment_wrong_control_correct": len(ai - bi),
                                          "treatment_correct_control_wrong": len(bi - ai), "both_wrong": len(ai & bi)})
                entry = {"cohort": cohort, "treatment": name, "control": control, "endpoint": endpoint,
                         "seeds": [a["spec"]["seed"] for a, _ in valid],
                         "run_pairs": [{"seed": a["spec"]["seed"], "treatment": a["run_id"], "control": b["run_id"]} for a, b in valid],
                         "test_accuracy_delta_pp": stats((100 * (a[endpoint + "_test"]["accuracy"] - b[endpoint + "_test"]["accuracy"]) for a, b in valid), ci=True),
                         "test_error_count_delta": stats((a[endpoint + "_test"]["errors"] - b[endpoint + "_test"]["errors"] for a, b in valid), ci=True),
                         "test_loss_delta": stats((a[endpoint + "_test"]["loss"] - b[endpoint + "_test"]["loss"] for a, b in valid), ci=True),
                         "error_disagreements": disagreements}
                timed = [(a, b) for a, b in valid if matched_timing(a, b)]
                entry["timing_pairing"] = {"matched_seeds": [a["spec"]["seed"] for a, b in timed],
                    "excluded_pairs": [{"seed": a["spec"]["seed"], "treatment_hardware": a.get("hardware", {}),
                        "control_hardware": b.get("hardware", {}), "reason": "hardware_or_source_mismatch_or_missing"}
                        for a, b in valid if not matched_timing(a, b)]}
                for key in ("training_seconds", "selected_training_seconds", "epoch_to_validation_98", "training_seconds_to_validation_98"):
                    eligible = valid if key == "epoch_to_validation_98" else timed
                    complete = [(a, b) for a, b in eligible if numeric(a.get(key)) and numeric(b.get(key))]
                    entry[key + "_delta"] = stats((a[key] - b[key] for a, b in complete), ci=True)
                    entry[key + "_delta"]["paired_seeds"] = [a["spec"]["seed"] for a, b in complete]
                entry["training_speedup_ratio"] = stats((b["training_seconds"] / a["training_seconds"] for a, b in timed), ci=True)
                timed_groups = defaultdict(list)
                for a, b in timed:
                    timed_groups[timing_identity(a)].append((a, b))
                entry["timing_by_hardware"] = [{**json.loads(key), "seeds": [a["spec"]["seed"] for a, b in group],
                    "training_seconds_delta": stats((a["training_seconds"] - b["training_seconds"] for a, b in group), ci=True),
                    "training_speedup_ratio": stats((b["training_seconds"] / a["training_seconds"] for a, b in group), ci=True)}
                    for key, group in sorted(timed_groups.items())]
                entry["interpretation"] = ("Treatment minus residual dense: positive accuracy delta is better; negative error/loss/time delta is better. "
                    "Quality pairs need not share assigned GPU variant. All time/speed comparisons exclude pairs with different or unknown reported GPU/runtime/source; hardware-specific strata are retained. "
                    "Speedup is control training seconds / treatment training seconds, above one faster. A pooled matched-pair mean remains descriptive when its pairs use different A100 variants. "
                    "Threshold-time comparisons include only pairs where both reached 98%; not an unconditional time-to-target estimate.")
                output.append(entry)
    return output


def selection_summary(path, runs, sources):
    candidates = []
    for r in runs:
        if r["spec"].get("stage") == "tune":
            candidates.append({"run_id": r["run_id"], "recipe": r["spec"]["recipe"], "seed": r["spec"]["seed"],
                               "lr": r["spec"]["lr"], "status": r["status"], "selected_epoch": r.get("selected_epoch"),
                               "best_validation_loss": r.get("selected_validation", {}).get("loss"),
                               "accuracy_at_best_validation_loss_pct": r.get("selected_validation", {}).get("accuracy_pct")})
    selected, problems, data = {}, [], {}
    if path.exists():
        try:
            data = read_json(path)
            if not isinstance(data, dict):
                raise ValueError("Expected mapping")
        except (ValueError, UnicodeError):
            data = {}
            problems.append(issue(None, "invalid_selection_json"))
        mapping = data.get("selected_learning_rates", data.get("selected_lrs", data.get("selected", {})))
        if isinstance(mapping, list):
            mapping = {x["recipe"]: x for x in mapping if isinstance(x, dict) and "recipe" in x}
        if not isinstance(mapping, dict):
            problems.append(issue(None, "invalid_selection_schema"))
        else:
            for recipe, value in mapping.items():
                lr = value.get("lr", value.get("selected_lr")) if isinstance(value, dict) else value
                if numeric(lr) and lr > 0 and recipe in LABELS:
                    selected[recipe] = lr
            if not selected:
                problems.append(issue(None, "selection_file_has_no_recognized_learning_rates"))
    audits = []
    for recipe, lr in sorted(selected.items()):
        eligible = [r for r in candidates if r["recipe"] == recipe and r["status"] == "ok"]
        winners = [r for r in eligible if r["best_validation_loss"] == min(x["best_validation_loss"] for x in eligible)] if eligible else []
        passed = bool(winners) and math.isclose(min(r["lr"] for r in winners), lr, rel_tol=1e-12, abs_tol=0)
        audits.append({"recipe": recipe, "selected_lr": lr, "ce_selection_verified": passed,
                       "lowest_validation_ce_run_ids": [r["run_id"] for r in winners]})
        if not passed:
            problems.append(issue(None, "selected_lr_not_validation_ce_winner", recipe))
    pinned = data.get("candidate_results_sha256")
    source_check = None
    if pinned is not None:
        actual = {s["run_id"]: s["sha256"] for s in sources if s["run_id"] in {r["run_id"] for r in candidates}}
        source_check = isinstance(pinned, dict) and pinned == actual
        if not source_check:
            problems.append(issue(None, "selection_candidate_source_hash_mismatch"))
    if "excluded_diverged_candidates" in data:
        observed = sorted(r["run_id"] for r in candidates if r["status"] == "diverged")
        if sorted(data["excluded_diverged_candidates"]) != observed:
            problems.append(issue(None, "selection_divergence_exclusion_mismatch"))
    return {"selection_file": {"file": path.name, "sha256": digest(path)} if path.exists() else None,
            "selected_learning_rates": selected, "candidates": candidates, "audits": audits,
            "candidate_source_hashes_verified": source_check,
            "criterion": "Among completed nondiverged candidates, lowest validation CE over scheduled checkpoints, then lower LR for an exact tie; accuracy does not select LR."}, problems


def frozen_source_checks(manifests_dir, runs, manifests):
    path = manifests_dir / "graph-protocol-freeze.json"
    if not path.exists():
        return {"available": False, "interpretation": "No graph protocol freeze supplied; source fingerprints are retained but no freeze comparison was possible."}, []
    problems = []
    try:
        frozen = read_json(path)["sha256"]
        required = ("train_optimized.py", "model_data.py", "optimization/stochastic_graph.py")
        if any(not isinstance(frozen.get(k), str) or not HASH.fullmatch(frozen[k]) for k in required):
            raise ValueError("Invalid source pin")
    except (ValueError, TypeError, KeyError):
        return {"available": True, "passed": False}, [issue(None, "invalid_graph_source_freeze")]
    audited = []
    for r in runs:
        if r["spec"].get("stage") not in ("tune", "evaluate") or r["status"] == "failed":
            continue
        passed = all(r.get("source_sha256", {}).get(k) == frozen[k] for k in required)
        audited.append({"run_id": r["run_id"], "passed": passed})
        if not passed:
            problems.append(issue(r["run_id"], "executed_source_disagrees_with_graph_freeze"))
    tune = next((m for m in manifests if m["file"] == "graph-tune-manifest.json"), None)
    tune_matches = tune is not None and tune["sha256"] == frozen.get("graph-tune-manifest.json")
    if not tune_matches:
        problems.append(issue(None, "tuning_manifest_disagrees_with_graph_freeze"))
    return {"available": True, "file": path.name, "sha256": digest(path), "passed": not problems,
        "executed_core_expected_sha256": {k: frozen[k] for k in required}, "run_checks": audited,
        "tuning_manifest_hash_matches": tune_matches}, problems


def analyze(results_dir=HERE / "results", output_dir=None, final=False, require_complete=False,
            manifests_dir=HERE, final_manifest=None, selection=None):
    results_dir, manifests_dir = Path(results_dir), Path(manifests_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir
    final_manifest = Path(final_manifest) if final_manifest is not None else manifests_dir / "final-manifest.json"
    selection = Path(selection) if selection is not None else (results_dir / "selection.json" if (results_dir / "selection.json").exists() else manifests_dir / "selection.json")
    if require_complete and not final:
        raise ValueError("--require-complete requires --final; test results are gated explicitly")
    raw, sources, specs, manifests, problems, ignored = inventory(results_dir, manifests_dir, final, final_manifest)
    runs, failures = [], []
    for record in raw:
        try:
            run, errors = summarize_run(record, final)
        except (KeyError, TypeError, ValueError, AttributeError):
            run = {"run_id": record["run_id"], "spec": {k: record["spec"][k] for k in SPEC_KEYS if k in record["spec"]}, "status": "invalid"}
            errors = [issue(record["run_id"], "malformed_result_schema")]
        runs.append(run)
        failures.extend(errors)
    checks, pairing_problems = pairing_checks(runs)
    problems += pairing_problems
    selection_info, selection_problems = selection_summary(selection, runs, sources)
    problems += selection_problems
    source_checks, source_problems = frozen_source_checks(manifests_dir, runs, manifests)
    problems += source_problems
    expected = manifest_jobs(final_manifest) if final and final_manifest.exists() else []
    result_by_id = {r["run_id"]: r for r in runs}
    expected_ids = [s["run_id"] for s in expected]
    missing = [rid for rid in expected_ids if rid not in result_by_id]
    invalid = [rid for rid in expected_ids if rid in result_by_id and result_by_id[rid]["status"] != "ok"]
    expected_tunes = [rid for rid, s in specs.items() if s.get("stage") == "tune"]
    missing_tunes = sorted(rid for rid in expected_tunes if rid not in result_by_id)
    invalid_records = [r["run_id"] for r in runs if r["status"] == "invalid"]
    if expected and len(set(expected_ids)) != len(expected_ids):
        problems.append(issue(None, "duplicate_final_manifest_run_ids"))
    if expected and not selection_info["selected_learning_rates"]:
        problems.append(issue(None, "final_evaluation_missing_frozen_lr_selection"))
    for r in runs:
        s = r["spec"]
        if s.get("stage") != "evaluate":
            continue
        if s.get("cohort") not in ("main", "fidelity"):
            problems.append(issue(r["run_id"], "unrecognized_evaluation_cohort"))
        if expected and r["run_id"] not in expected_ids:
            problems.append(issue(r["run_id"], "unexpected_evaluation_run"))
        if s.get("cohort") == "main" and selection_info["selected_learning_rates"]:
            lr = selection_info["selected_learning_rates"].get(s["recipe"])
            if lr is None or not math.isclose(lr, s["lr"], rel_tol=1e-12, abs_tol=0):
                problems.append(issue(r["run_id"], "evaluation_lr_disagrees_with_selection"))
    cohorts = {"main": {"groups": {}}, "fidelity": {"groups": {}}}
    for cohort in cohorts:
        grouped = defaultdict(list)
        for r in runs:
            if r["spec"].get("stage") == "evaluate" and r["spec"].get("cohort") == cohort:
                grouped[group_name(r)].append(r)
        cohorts[cohort]["groups"] = {key: aggregate_group(sorted(rows, key=lambda r: r["spec"]["seed"])) for key, rows in sorted(grouped.items())}
    verified = not problems and not invalid_records and not invalid
    complete = bool(expected) and not missing and not missing_tunes and verified
    doc = {"schema_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "mode": "final_enabled" if final else "validation_only", "test_results_included": final,
           "verification": {"passed": verified, "complete": complete,
                            "final_manifest_present": final_manifest.exists() if final else False,
                            "expected_runs": len(expected), "missing_run_ids": missing, "invalid_run_ids": invalid,
                            "missing_tuning_run_ids": missing_tunes, "invalid_record_ids": invalid_records,
                            "expected_final_cohorts": {c: {"runs": sum(s.get("cohort") == c for s in expected),
                                "seeds": sorted({s["seed"] for s in expected if s.get("cohort") == c})} for c in ("main", "fidelity")},
                            "failure_interpretation": "Reported scientific divergence or job failure in an exploratory pilot/tune is retained but does not invalidate otherwise verified final results. Invalid records, missing final runs and divergent/failed final runs do block completeness.",
                            "issues": problems, "pairing": checks, "frozen_graph_source": source_checks},
           "counts": dict(Counter(r["spec"].get("stage", "unknown") for r in runs)),
           "source_files": sources, "manifests": manifests, "failures": failures,
           "scope": {"included_manifest_names": [m["file"] for m in manifests], "excluded_result_filenames": ignored,
               "interpretation": "Only pilot-manifest.json, graph-tune-manifest.json and the explicit final manifest authorize result reads. Other filenames are listed without opening their contents; archived old tuning, optimization and qualification do not enter study selection or outcomes."},
           "selection": selection_info, "cohorts": cohorts,
           "paired_comparisons": paired_comparisons(runs, {r["run_id"]: r for r in raw}, checks) if final else [],
           "runs": runs,
           "interpretation": [
               "Validation-selected checkpoint is primary; last fixed-epoch checkpoint is secondary. This resumed study selects LRs/checkpoints using validation only. Its official test set was previously inspected during separate baseline optimization, so the overall investigation is exploratory rather than globally test-naive.",
               "Main and fidelity cohorts are separate; plain versus residual also changes architecture.",
               "Means and sample SD concern seeds. Paired 95% t intervals are conditional on one fixed data split, the same test examples, and validation-selected LRs; no multiple-comparison correction.",
               "Repeated seeds reuse the same test examples; they are not independent dataset replications. Confirmation seeds 101–103 were also used in the preceding baseline optimization, so they are not globally untouched initializations. CI including zero is not proof of equivalence. The resumed final manifest uses three seeds per arm, with df=2 for complete paired intervals.",
               "Training seconds include gather, graph execution, optimizer/loop diagnostics; exclude loading/evaluation/checkpointing, graph setup and per-epoch mask/order/scaling preparation. Separate graph/preparation timings and total run time are retained. None is billed runtime or energy.",
               "Assigned A100 variants differ. Per-arm pooled runtimes are descriptive; speed effects require the same reported hardware/runtime and source within a seed pair, with separate hardware strata. Matching model names still does not control clock, worker contention or thermal state.",
               "First observed validation accuracy >=98% is discretized by evaluation cadence. Unreached runs remain censored; achiever-only means may be selection-biased.",
               "Initialization, data/split, and executed epoch-order hashes verify paired inputs for the resumed runs; equal batch size and drop-last determine minibatch order. Mask streams are independent and intentionally recipe-specific. Older pilot runs predate the order digest.",
               "Outputs omit checkpoints, wrong-index arrays, host identifiers and arbitrary exception messages; scientific GPU/runtime identifiers and paired disagreement counts are retained."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("analysis.json", json.dumps(doc, indent=2, allow_nan=False) + "\n"), ("summary.md", markdown(doc))):
        temp = output_dir / (name + ".tmp")
        temp.write_text(content)
        temp.replace(output_dir / name)
    return doc, (not require_complete or complete)


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if numeric(value) else "—"


def mean_sd(value, digits=3):
    if not value or not value.get("n"):
        return "—"
    return f"{fmt(value['mean'], digits)} ± {fmt(value['sample_sd'], digits)} (n={value['n']})"


def interval(value, digits=3):
    if not value or not value.get("n"):
        return "—"
    return f"{fmt(value['mean'], digits)} [{fmt(value.get('ci95_low'), digits)}, {fmt(value.get('ci95_high'), digits)}], n={value['n']}"


def markdown(doc):
    final = doc["test_results_included"]
    lines = ["# Ciresan-width MNIST stochastic-depth analysis", "", f"Generated {doc['generated_utc']}.", "",
             ("Official-test analysis explicitly enabled with `--final`." if final else "Validation-only analysis. Evaluate-stage files and official-test metrics are excluded."), "",
             f"Frozen final manifest complete: **{doc['verification']['complete']}**. Verification passed: **{doc['verification']['passed']}**.", "",
             "Only the original pilot manifest and resumed graph-tuning manifest enter validation analysis; the final manifest is additionally enabled by `--final`. Archived tuning, qualification and baseline optimization are excluded.", "",
             "Values are seed means ± sample SD. Missing SD/CI is undefined, not zero. Primary: validation-CE-selected checkpoint. Secondary: final fixed-epoch checkpoint.", ""]
    if final:
        for cohort in ("main", "fidelity"):
            lines += [f"## {cohort.title()} cohort", ""]
            planned = doc["verification"]["expected_final_cohorts"][cohort]
            lines += [f"Final manifest: {planned['runs']} runs; seeds {planned['seeds']}.", ""]
            groups = doc["cohorts"][cohort]["groups"]
            if not groups:
                lines += ["No eligible results available.", ""]
                continue
            for endpoint, title in (("selected", "Primary: validation-selected checkpoint"), ("final", "Secondary: last fixed-epoch checkpoint")):
                lines += [f"### {title}", "", "| Arm | Test accuracy (%) | Errors / test set | Test CE | Full-train CE | Validation CE | Validation − train CE | Full-train accuracy (%) |", "|---|---|---|---|---|---|---|---|"]
                for name, group in groups.items():
                    entry = group["endpoints"][endpoint]
                    get = lambda split, key: entry.get(split, {}).get(key)
                    lines.append(f"| {name} | {mean_sd(get('test','accuracy_pct'),2)} | {mean_sd(get('test','errors'),1)} / {entry.get('test',{}).get('example_counts',[])} | {mean_sd(get('test','loss'),4)} | {mean_sd(get('train','loss'),4)} | {mean_sd(get('validation','loss'),4)} | {mean_sd(get('generalization','validation_minus_train_loss'),4)} | {mean_sd(get('train','accuracy_pct'),2)} |")
                lines.append("")
            lines += ["### Exposure and measured training time", "", "Pooled runtime means below describe the assigned workers; they do not isolate a method speedup across different A100 variants. Hardware-specific measurements follow.", "", "| Arm | Valid / failed | LR | Selected epoch | Pooled training seconds | Pooled total run seconds | First observed epoch ≥98% validation | Pooled training seconds to ≥98% | Reached / not reached |", "|---|---|---|---|---|---|---|---|---|"]
            for name, group in groups.items():
                m, threshold = group["metrics"], group["validation_98"]
                lines.append(f"| {name} | {group['n']} / {group['failure_count']} | {group['selected_learning_rates']} | {mean_sd(m['selected_epoch'],1)} | {mean_sd(m['training_seconds'],2)} | {mean_sd(m['total_run_seconds'],2)} | {mean_sd(m['epoch_to_validation_98'],1)} | {mean_sd(m['training_seconds_to_validation_98'],2)} | {threshold['reached']} / {threshold['not_reached']} |")
            lines += ["", "Time-to-98% summaries include achievers only; unreached runs are censored. Training time excludes loading, evaluation, checkpointing, graph setup and per-epoch preparation; it is not billed time or energy.", "", "| Arm / GPU | Seeds | Training seconds | Total run seconds | Graph setup seconds | Epoch preparation seconds |", "|---|---|---|---|---|---|"]
            for name, group in groups.items():
                for h in group["timing_by_hardware"]:
                    m = h["metrics"]
                    lines.append(f"| {name} / {h['hardware'].get('gpu','unknown')} | {h['seeds']} | {mean_sd(m['training_seconds'],2)} | {mean_sd(m['total_run_seconds'],2)} | {mean_sd(m['graph_setup_seconds'],2)} | {mean_sd(m['epoch_preparation_seconds'],2)} |")
            lines.append("")
        lines += ["## Paired effects versus residual dense", "", "All quality effects use same-cohort, same-seed pairs with verified initialization, data and executed epoch-order hashes. Accuracy is treatment minus control in percentage points; errors are treatment minus control in counts. Positive accuracy and negative error deltas favor the treatment. Brackets are 95% t intervals over paired seed differences. Time/speed columns additionally require matching GPU/runtime/source, so their n may be smaller or zero.", "", "| Cohort / arm | Endpoint | Test accuracy Δ (pp) [95% CI] | Error count Δ [95% CI] | Matched training seconds Δ [95% CI] | Matched training speedup ratio [95% CI] |", "|---|---|---|---|---|---|"]
        for p in doc["paired_comparisons"]:
            lines.append(f"| {p['cohort']} / {p['treatment']} | {p['endpoint']} | {interval(p['test_accuracy_delta_pp'])} | {interval(p['test_error_count_delta'],1)} | {interval(p['training_seconds_delta'],2)} | {interval(p['training_speedup_ratio'])} |")
        lines += ["", "Full-horizon training-time effects are repeated at both endpoints for context; they are the same measurements. Speedup is control time divided by treatment time. Matching within pairs does not standardize the GPU across all pairs; hardware-specific effects and every excluded timing pair, conditional time-to-target pair and error-disagreement count are in analysis.json.", ""]
    lines += ["## Pilots and LR candidates (validation only)", "", "| Run | Stage | Recipe | LR | Status | Best validation CE | Accuracy at CE-selected epoch (%) | Selected epoch | Final train-probe CE / accuracy (%) | Training seconds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["runs"]:
        if r["spec"].get("stage") not in ("pilot", "tune"):
            continue
        s, val, probe = r["spec"], r.get("selected_validation", {}), r.get("final_dense_train_probe", {})
        lines.append(f"| {r['run_id']} | {s.get('stage')} | {s.get('recipe')} | {s.get('lr')} | {r['status']} | {fmt(val.get('loss'))} | {fmt(val.get('accuracy_pct'),2)} | {r.get('selected_epoch','—')} | {fmt(probe.get('loss'))} / {fmt(probe.get('accuracy_pct'),2)} | {fmt(r.get('training_seconds'),2)} |")
    lines += ["", "Training-probe metrics above use the deterministic dense-inference probe, not the full training set or stochastic minibatch loss. Diverged candidates are attempted outcomes and are ineligible for LR selection; their partial trajectories do not count as completed training.", "", "### Frozen LR selection", "", "LR selection uses validation CE alone among completed, nondiverged candidates, with lower LR breaking exact ties; validation accuracy is descriptive. No selected LR is inferred if selection.json is absent.", ""]
    for recipe, lr in sorted(doc["selection"]["selected_learning_rates"].items()):
        lines.append(f"- {recipe}: {lr:g}")
    if not doc["selection"]["selected_learning_rates"]:
        lines.append("No selection.json learning-rate choices available.")
    lines += ["", "## Verification and failures", ""]
    for check in doc["verification"]["pairing"]:
        lines.append(f"- {check['cohort']} seed {check['seed']}: {check['n']} runs, pairing passed={check['passed']}.")
    lines += [doc["verification"]["failure_interpretation"], ""]
    for name in ("missing_run_ids", "invalid_run_ids", "missing_tuning_run_ids", "invalid_record_ids"):
        if doc["verification"][name]:
            lines.append(f"- {name}: {', '.join(doc['verification'][name])}")
    for failure in doc["failures"] + doc["verification"]["issues"]:
        lines.append(f"- {failure.get('run_id') or 'Study'}: {failure['code']}" + (f" ({failure['detail']})" if "detail" in failure else ""))
    for r in doc["runs"]:
        if r["status"] == "diverged":
            last = r.get("last_observed_validation", {})
            lines.append(f"- {r['run_id']}: stopped at epoch {r.get('stopped_epoch','—')}; last valid validation observation at epoch {r.get('last_observed_validation_epoch','—')}: CE {fmt(last.get('loss'))}, accuracy {fmt(last.get('accuracy_pct'),2)}%. These are partial, ineligible observations.")
    if not doc["failures"] and not doc["verification"]["issues"]:
        lines.append("No recorded job/metric failures or verification issues among inspected files. Missing not-yet-dispatched runs are not successful results.")
    lines += ["", "### Excluded files", "", "The following filenames were inventoried without opening their contents. They are outside the controlled-study manifest scope (including archived, optimization, qualification and ancillary metadata files):", "", ", ".join(f"`{name}`" for name in doc["scope"]["excluded_result_filenames"]) or "None.", "", "## Interpretation limits", ""]
    lines += ["- " + text for text in doc["interpretation"]]
    lines += ["", "Raw result filenames and SHA-256 digests, per-seed scientific measurements, selection checks, and paired error-disagreement counts are retained in analysis.json. No checkpoints or private metadata are copied.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifests-dir", type=Path, default=HERE)
    parser.add_argument("--final", action="store_true", help="Explicitly enable evaluate-stage/official-test analysis")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every frozen final-manifest run verifies")
    parser.add_argument("--final-manifest", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.require_complete and not args.final:
        parser.error("--require-complete requires --final")
    doc, passed = analyze(**vars(args))
    print(json.dumps({"mode": doc["mode"], "counts": doc["counts"], "complete": doc["verification"]["complete"],
                      "failures": len(doc["failures"]), "issues": len(doc["verification"]["issues"]),
                      "output": str(args.output_dir or args.results_dir)}))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
