"""Summarize the explicitly test-monitored MNIST optimization search.

Standard library only; reads local JSON, never launches training or cloud calls.
Every attempt remains visible. Confirmations and exploratory candidates are kept
separate, and timing means never mix different recipes or reported GPU models.
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
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parent.parent
TARGET_ERRORS = 137
TEST_N = 10000
HISTORICAL_SECONDS = 388.48669505119324
HISTORICAL_URL = "https://wandb.ai/yaroslavvb/train_ciresan/runs/ts4k9n55"
RAW_BASE = "https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/"
RECIPE_DEFAULTS = {"batch_size": 64, "lr": .001, "shrinkage": .00002,
                   "input_scale": 1., "output_relu": True, "precision": "fp32",
                   "recipe": "plain", "dropout": 0., "lr_schedule": {}}
IMPLEMENTATION_DEFAULTS = {"mode": "eager", "fused": False, "compile_mode": "reduce-overhead"}
HASH = re.compile(r"^[a-f0-9]{64}$")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_object(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sample(values):
    values = [float(x) for x in values if finite(x)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "values": values}


def effective(spec):
    recipe = {key: spec.get(key, default) for key, default in RECIPE_DEFAULTS.items()}
    # The runner fixes these independently of user-visible spec fields.
    recipe.update(momentum=.9, parameter_dtype="float32", training_pool_n=60000,
                  widths=[784, 2500, 2000, 1500, 1000, 500, 10],
                  schedule_semantics="Listed epoch factor multiplies initial LR and initial direct shrinkage; graph is recaptured.")
    implementation = {key: spec.get(key, default) for key, default in IMPLEMENTATION_DEFAULTS.items()}
    if implementation["mode"] != "compile":
        implementation["compile_mode"] = None
    return recipe, implementation


def same_recipe(a, b):
    if set(a) != set(b):
        return False
    return all(math.isclose(a[k], b[k], rel_tol=1e-11, abs_tol=1e-15)
               if finite(a[k]) and finite(b[k]) else a[k] == b[k] for k in a)


def changes(recipe):
    result = []
    for key, default in RECIPE_DEFAULTS.items():
        value = recipe[key]
        equals = math.isclose(value, default, rel_tol=1e-11, abs_tol=1e-15) if finite(value) and finite(default) else value == default
        if not equals:
            result.append({"field": key, "reference": default, "value": value})
    return result


def hardware(raw):
    value = raw.get("hardware", {})
    return {key: value[key] for key in ("gpu", "torch", "cuda", "tf32", "precision_api", "cpu_threads", "triton") if key in value}


def comparable_hardware(a, b):
    return all(a.get(k) is not None and a.get(k) == b.get(k) for k in ("gpu", "torch", "cuda", "tf32"))


def dataset_hash(raw):
    files = raw.get("dataset", {}).get("files", {})
    selected = {name: {k: meta[k] for k in ("md5", "sha256", "bytes") if k in meta}
                for name, meta in files.items() if isinstance(meta, dict) and name.endswith(".gz")}
    return sha_object(selected) if len(selected) == 4 else None


def score(value):
    if not isinstance(value, dict) or value.get("n") != TEST_N:
        raise ValueError("score_must_use_10000_test_examples")
    errors = value.get("errors")
    if not isinstance(errors, int) or isinstance(errors, bool) or not 0 <= errors <= TEST_N:
        raise ValueError("invalid_test_error_count")
    if not finite(value.get("accuracy")) or not math.isclose(value["accuracy"], 1 - errors / TEST_N, rel_tol=0, abs_tol=1e-10):
        raise ValueError("test_accuracy_error_count_mismatch")
    if not finite(value.get("loss")) or value["loss"] < 0:
        raise ValueError("invalid_test_loss")
    return {"n": TEST_N, "errors": errors, "accuracy": value["accuracy"],
            "accuracy_pct": 100 * value["accuracy"], "loss": value["loss"]}


def evaluation(row):
    if not isinstance(row.get("epoch"), int) or row["epoch"] < 1:
        raise ValueError("invalid_epoch")
    for key in ("training_seconds", "run_wall_seconds"):
        if not finite(row.get(key)) or row[key] < 0:
            raise ValueError("invalid_" + key)
    if row["run_wall_seconds"] + 1e-6 < row["training_seconds"]:
        raise ValueError("invocation_time_less_than_training_time")
    return {"epoch": row["epoch"], "test": score(row["test"]),
            "training_seconds": row["training_seconds"], "run_wall_seconds": row["run_wall_seconds"]}


def failure_reason(raw):
    """Only allowlisted reason classes; never copy arbitrary remote error text."""
    text = str(raw.get("error", ""))
    missing = re.search(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]", text)
    if missing:
        return {"category": "missing_module", "module": missing.group(1)}
    if "precision" in text.lower() and "mix" in text.lower():
        return {"category": "precision_api_compatibility_failure"}
    if "timeout" in text.lower():
        return {"category": "timeout"}
    return {"category": "remote_failure"}


def summarize_accuracy(raw, path):
    spec = raw["spec"]
    recipe, implementation = effective(spec)
    rid = raw.get("run_id", path.stem)
    result = {"run_id": rid, "file": path.name, "sha256": sha_bytes(path.read_bytes()),
              "kind": "accuracy", "cohort": "confirmation" if rid.startswith("opt-confirm") else "diagnostic" if rid.startswith("opt-diagnostic") else "exploratory",
              "confirmation_group": spec.get("confirmation_group"),
              "confirmation_type": spec.get("confirmation_type", "unspecified"),
              "seed": spec.get("seed", 1), "recipe": recipe, "implementation": implementation,
              "recipe_changes_from_source_style_tf32_b64": changes(recipe),
              "max_epochs": spec.get("max_epochs", 100), "eval_every": spec.get("eval_every", 1),
              "stop_at_target": spec.get("stop_at_target", True), "hardware": hardware(raw),
              "initial_parameters_sha256": raw.get("initial_parameters_sha256"),
              "dataset_files_sha256": dataset_hash(raw),
              "source_sha256": {k: v for k, v in raw.get("source_sha256", {}).items() if isinstance(v, str) and HASH.fullmatch(v)},
              "verification_issues": [], "warnings": [], "target_reached": False,
              "first_target": None, "preceding_evaluation": None, "last_evaluation": None,
              "best_accuracy_evaluation": None,
              "timing": {k: raw.get(k) for k in ("setup_seconds", "setup_helper_seconds", "training_seconds",
                          "total_run_seconds", "local_dispatch_elapsed_seconds")}}
    result["schedule_events"] = [{k: event[k] for k in ("epoch", "factor", "recapture_seconds") if k in event}
                                 for event in raw.get("schedule_events", []) if isinstance(event, dict)]
    if "error" in raw:
        result.update(status="failed", failure=failure_reason(raw))
        return result
    try:
        if not isinstance(result["max_epochs"], int) or result["max_epochs"] <= 0:
            raise ValueError("invalid_epoch_budget")
        rows = [evaluation(row) for row in raw.get("history", [])]
        if not rows:
            if raw.get("diverged"):
                result.update(status="diverged", failure={"category": "nonfinite_minibatch_loss_before_first_evaluation"})
                return result
            raise ValueError("no_evaluations")
        for key in ("epoch", "training_seconds", "run_wall_seconds"):
            vals = [r[key] for r in rows]
            if vals != sorted(vals) or (key == "epoch" and len(set(vals)) != len(vals)):
                raise ValueError("nonmonotone_" + key)
        if rows[-1]["epoch"] > result["max_epochs"]:
            raise ValueError("evaluated_epoch_exceeds_budget")
        qualifying = [(i, r) for i, r in enumerate(rows) if r["test"]["errors"] <= TARGET_ERRORS]
        first = qualifying[0] if qualifying else None
        if bool(raw.get("threshold")) != bool(first):
            raise ValueError("stored_threshold_disagrees_with_history")
        if first and evaluation(raw["threshold"]) != first[1]:
            raise ValueError("stored_threshold_is_not_first_qualifying_evaluation")
        result.update(first_target=first[1] if first else None,
                      preceding_evaluation=rows[first[0] - 1] if first and first[0] > 0 else None,
                      target_reached=first is not None, last_evaluation=rows[-1],
                      best_accuracy_evaluation=min(rows, key=lambda r: r["test"]["errors"]),
                      minimum_ce_evaluation=min(rows, key=lambda r: r["test"]["loss"]),
                      evaluation_count=len(rows),
                      scored_trajectory_sha256=sha_object([{"epoch": r["epoch"], "test": r["test"]} for r in rows]),
                      completed_epoch_lower_bound=rows[-1]["epoch"],
                      steps_per_epoch=raw.get("steps_per_epoch"),
                      train_examples_per_epoch=raw.get("train_examples_per_epoch"), parameter_count=raw.get("parameter_count"))
        if raw.get("diverged"):
            result["status"] = "reached_then_diverged" if first else "diverged"
            result["warnings"].append("Last scored checkpoint precedes divergent weights; do not label it final-model quality.")
        else:
            result["status"] = "reached" if first else "censored" if rows[-1]["epoch"] >= result["max_epochs"] else "incomplete"
        for key, last in (("training_seconds", rows[-1]["training_seconds"]), ("total_run_seconds", rows[-1]["run_wall_seconds"])):
            if not finite(raw.get(key)) or raw[key] + 1e-6 < last:
                raise ValueError("result_" + key + "_precedes_last_evaluation")
        if not raw.get("diverged") and raw.get("final_test") is not None and score(raw["final_test"]) != rows[-1]["test"]:
            raise ValueError("final_test_disagrees_with_last_evaluation")
        if raw.get("last_evaluated_test") is not None and score(raw["last_evaluated_test"]) != rows[-1]["test"]:
            raise ValueError("last_evaluated_test_disagrees_with_history")
        if "last_class_diagnostics" in raw:
            diagnostic = raw["last_class_diagnostics"]
            keys = ("head_preactivation_max_by_class", "positive_logit_fraction_by_class", "prediction_class_counts",
                    "true_class_counts", "class_accuracy", "last_minibatch_head_row_gradient_norms")
            if any(not isinstance(diagnostic.get(k), list) or len(diagnostic[k]) != 10
                   or any(not finite(v) for v in diagnostic[k]) for k in keys):
                raise ValueError("invalid_class_diagnostic_vectors")
            for key in ("prediction_class_counts", "true_class_counts"):
                if sum(diagnostic[key]) != TEST_N or any(not isinstance(v, int) or v < 0 for v in diagnostic[key]):
                    raise ValueError("class_diagnostic_count_mismatch")
            class_correct = [round(a * n) for a, n in zip(diagnostic["class_accuracy"], diagnostic["true_class_counts"])]
            if sum(class_correct) != TEST_N - rows[-1]["test"]["errors"]:
                raise ValueError("class_diagnostic_accuracy_mismatch")
            result["digit6_diagnostic"] = {"true_count": diagnostic["true_class_counts"][6],
                "correct_count": class_correct[6], "prediction_count": diagnostic["prediction_class_counts"][6],
                "accuracy_pct": 100 * class_correct[6] / diagnostic["true_class_counts"][6],
                "head_preactivation_max": diagnostic["head_preactivation_max_by_class"][6],
                "positive_logit_fraction": diagnostic["positive_logit_fraction_by_class"][6],
                "last_minibatch_head_row_gradient_norm": diagnostic["last_minibatch_head_row_gradient_norms"][6],
                "scope": "Test-set class diagnostics at the last scored model; gradient is one last training minibatch only."}
        if not isinstance(result["initial_parameters_sha256"], str) or not HASH.fullmatch(result["initial_parameters_sha256"]):
            raise ValueError("missing_initialization_hash")
        if result["dataset_files_sha256"] is None or raw.get("dataset", {}).get("train_n") != 60000:
            raise ValueError("missing_full_mnist_dataset_provenance")
        if not result["source_sha256"]:
            result["warnings"].append("This early result lacks executed-source hashes; result bytes and scored trajectory are hashed.")
        expected_events = {int(epoch): factor for epoch, factor in recipe["lr_schedule"].items() if int(epoch) <= rows[-1]["epoch"]}
        observed_events = {event.get("epoch"): event.get("factor") for event in result["schedule_events"]}
        if expected_events != observed_events:
            raise ValueError("schedule_events_disagree_with_executed_epochs")
        result["timing"]["schedule_recapture_seconds"] = sum(event.get("recapture_seconds", 0.) for event in result["schedule_events"])
        result["timing"].update(target_training_seconds=first[1]["training_seconds"] if first else None,
                                target_run_wall_seconds=first[1]["run_wall_seconds"] if first else None,
                                last_score_training_seconds=rows[-1]["training_seconds"],
                                last_score_run_wall_seconds=rows[-1]["run_wall_seconds"])
        if finite(raw.get("local_dispatch_elapsed_seconds")):
            result["timing"]["dispatch_minus_invocation_total_seconds"] = raw["local_dispatch_elapsed_seconds"] - raw["total_run_seconds"]
    except (KeyError, TypeError, ValueError) as exc:
        result["status"] = "invalid"
        # Validation exceptions are our own constant codes, not source content.
        result["verification_issues"].append(str(exc) if re.fullmatch(r"[a-z0-9_]+", str(exc)) else "malformed_result")
    return result


def summarize_microbench(raw, path):
    spec = raw.get("spec", {})
    result = {"run_id": raw.get("run_id", path.stem), "file": path.name,
              "sha256": sha_bytes(path.read_bytes()), "kind": "microbench", "hardware": hardware(raw),
              "total_run_seconds": raw.get("total_run_seconds"),
              "local_dispatch_elapsed_seconds": raw.get("local_dispatch_elapsed_seconds"), "variants": [],
              "verification_issues": [], "kernel_tests": raw.get("kernel_tests")}
    if "error" in raw:
        return result | {"status": "failed", "failure": failure_reason(raw)}
    try:
        if not isinstance(raw.get("kernel_tests"), dict) or raw["kernel_tests"].get("passed") is not True:
            raise ValueError("kernel_correctness_tests_not_verified")
        for i, row in enumerate(raw.get("benchmarks", [])):
            merged = {**spec, **row["variant"]}
            recipe, implementation = effective(merged)
            times = row["step_seconds"]
            if not times or any(not finite(t) or t <= 0 for t in times):
                raise ValueError("invalid_step_timings")
            if not math.isclose(statistics.median(times), row["median_step_seconds"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("median_step_time_mismatch")
            if not isinstance(row.get("initial_parameters_sha256"), str) or not HASH.fullmatch(row["initial_parameters_sha256"]):
                raise ValueError("missing_initialization_hash")
            result["variants"].append({"variant_index": i, "recipe": recipe, "implementation": implementation,
                "setup_seconds": row["setup_seconds"], "step_seconds": times, "median_step_seconds": row["median_step_seconds"],
                "initial_parameters_sha256": row.get("initial_parameters_sha256"),
                "examples_per_second": recipe["batch_size"] / row["median_step_seconds"]})
        if not result["variants"]:
            raise ValueError("missing_benchmark_variants")
        result["status"] = "completed"
    except (KeyError, TypeError, ValueError) as exc:
        result["status"] = "invalid"
        result["verification_issues"].append(str(exc) if re.fullmatch(r"[a-z0-9_]+", str(exc)) else "malformed_result")
    return result


def kernel_comparisons(attempts, microbenchmarks):
    comparisons = []
    for bench in microbenchmarks:
        refs = [r for r in bench["variants"] if r["implementation"]["mode"] == "eager" and not changes(r["recipe"])]
        if bench["status"] != "completed" or not refs:
            continue
        ref = refs[0]
        for row in bench["variants"]:
            if row is ref or not same_recipe(ref["recipe"], row["recipe"]) or ref["initial_parameters_sha256"] != row["initial_parameters_sha256"]:
                continue
            comparisons.append({"kind": "within_invocation_microbench", "reference_run": bench["run_id"],
                "treatment_run": bench["run_id"], "reference_variant": ref["implementation"], "treatment_variant": row["implementation"],
                "hardware": bench["hardware"], "same_recipe": True, "same_initialization": True,
                "reference_step_seconds": ref["median_step_seconds"], "treatment_step_seconds": row["median_step_seconds"],
                "step_speedup": ref["median_step_seconds"] / row["median_step_seconds"],
                "interpretation": "Same-invocation repeated-batch microbenchmark, not convergence or independent-seed uncertainty."})
    refs = [r for r in attempts if r["status"] in ("reached", "censored") and r["implementation"]["mode"] == "eager" and not changes(r["recipe"])]
    for ref in refs:
        for row in attempts:
            if row is ref or row["cohort"] != ref["cohort"] or row["status"] not in ("reached", "censored"):
                continue
            matched = (row["seed"] == ref["seed"] and same_recipe(row["recipe"], ref["recipe"])
                       and row["initial_parameters_sha256"] == ref["initial_parameters_sha256"]
                       and row["dataset_files_sha256"] == ref["dataset_files_sha256"]
                       and comparable_hardware(row["hardware"], ref["hardware"])
                       and row["last_evaluation"]["epoch"] == ref["last_evaluation"]["epoch"])
            if not matched:
                continue
            comparisons.append({"kind": "matched_full_training_horizon", "reference_run": ref["run_id"],
                "treatment_run": row["run_id"], "seed": row["seed"], "hardware": row["hardware"],
                "same_recipe": True, "same_initialization": True, "same_dataset": True,
                "epochs": row["last_evaluation"]["epoch"],
                "scored_test_trajectories_identical": row["scored_trajectory_sha256"] == ref["scored_trajectory_sha256"],
                "reference_training_seconds": ref["timing"]["training_seconds"],
                "treatment_training_seconds": row["timing"]["training_seconds"],
                "training_speedup": ref["timing"]["training_seconds"] / row["timing"]["training_seconds"],
                "target_reached_by_reference": ref["target_reached"], "target_reached_by_treatment": row["target_reached"],
                "interpretation": "Fixed-horizon training speed, not a time-to-target speedup when the target was not reached. Separate invocations may have different setup/cache history."})
    return comparisons


def expected_manifest_runs(manifests_dir):
    expected, sources, issues = {}, [], []
    for path in sorted(manifests_dir.glob("*manifest*.json")):
        try:
            data = load(path)
            jobs = data if isinstance(data, list) else data.get("runs", data.get("jobs", []))
            for spec in jobs:
                if isinstance(spec, dict) and str(spec.get("run_id", "")).startswith("opt-"):
                    rid = spec["run_id"]
                    if rid in expected and expected[rid] != spec:
                        issues.append({"file": path.name, "run_id": rid, "code": "conflicting_manifest_specs"})
                    expected[rid] = spec
            sources.append({"file": path.name, "sha256": sha_bytes(path.read_bytes())})
        except (ValueError, TypeError, AttributeError):
            issues.append({"file": path.name, "code": "malformed_manifest"})
    return expected, sources, issues


def confirmation_groups(attempts):
    groups = defaultdict(list)
    for row in attempts:
        if row["cohort"] == "confirmation":
            fingerprint = sha_object({"recipe": row["recipe"], "implementation": row["implementation"],
                "confirmation_group": row["confirmation_group"], "confirmation_type": row["confirmation_type"],
                "gpu": row["hardware"].get("gpu"), "torch": row["hardware"].get("torch"),
                "cuda": row["hardware"].get("cuda"), "max_epochs": row["max_epochs"], "eval_every": row["eval_every"]})[:16]
            groups[fingerprint].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        hits = [r for r in rows if r["target_reached"] and r["status"] != "invalid"]
        example = rows[0]
        result.append({"group_id": key, "recipe": example["recipe"], "implementation": example["implementation"],
            "confirmation_group": example["confirmation_group"], "confirmation_type": example["confirmation_type"],
            "hardware": example["hardware"], "run_ids": [r["run_id"] for r in rows], "seeds": [r["seed"] for r in rows],
            "n_attempts": len(rows), "n_reached": len(hits), "status_counts": dict(Counter(r["status"] for r in rows)),
            "target_epoch": sample(r["first_target"]["epoch"] for r in hits),
            "target_training_seconds": sample(r["first_target"]["training_seconds"] for r in hits),
            "target_run_wall_seconds": sample(r["first_target"]["run_wall_seconds"] for r in hits),
            "target_accuracy_pct": sample(r["first_target"]["test"]["accuracy_pct"] for r in hits),
            "local_dispatch_elapsed_seconds": sample(r["timing"].get("local_dispatch_elapsed_seconds") for r in rows),
            "interpretation": "Confirmation attempts grouped by confirmation type, exact recipe, implementation, reported hardware and budget. Same-seed repeats are not fresh independent seeds. Target-time summaries include successful runs only; failures/censoring remain in denominator. Official test still monitored."})
    return result


def confirmation_outcomes(attempts):
    """Count outcomes across hardware without pooling timing measurements."""
    groups = defaultdict(list)
    for row in attempts:
        if row["cohort"] == "confirmation":
            key = sha_object({"group": row["confirmation_group"], "recipe": row["recipe"],
                              "implementation": row["implementation"], "confirmation_type": row["confirmation_type"],
                              "max_epochs": row["max_epochs"]})[:16]
            groups[key].append(row)
    return [{"group_id": key, "confirmation_group": rows[0]["confirmation_group"],
             "confirmation_type": rows[0]["confirmation_type"], "recipe": rows[0]["recipe"],
             "implementation": rows[0]["implementation"], "max_epochs": rows[0]["max_epochs"],
             "run_ids": [r["run_id"] for r in rows], "seeds": [r["seed"] for r in rows],
             "n_attempts": len(rows), "n_reached": sum(r["target_reached"] and r["status"] != "invalid" for r in rows),
             "status_counts": dict(Counter(r["status"] for r in rows)),
             "hardware_models": sorted(set(r["hardware"].get("gpu", "unknown") for r in rows)),
             "interpretation": "Outcome counts only; timing means are separated by reported hardware in confirmation_groups."}
            for key, rows in sorted(groups.items())]


def head_diagnostic_pairs(attempts):
    rows = [r for r in attempts if r["cohort"] == "diagnostic" and r["status"] not in ("failed", "invalid") and "digit6_diagnostic" in r]
    pairs = []
    for relu in [r for r in rows if r["recipe"]["output_relu"]]:
        for linear in [r for r in rows if not r["recipe"]["output_relu"] and r["seed"] == relu["seed"]]:
            checks = {"same_recipe_except_output_relu": same_recipe(relu["recipe"] | {"output_relu": False}, linear["recipe"]),
                      "same_implementation": relu["implementation"] == linear["implementation"],
                      "same_initialization": relu["initial_parameters_sha256"] == linear["initial_parameters_sha256"],
                      "same_dataset": relu["dataset_files_sha256"] == linear["dataset_files_sha256"],
                      "same_hardware": comparable_hardware(relu["hardware"], linear["hardware"]),
                      "same_executed_source": bool(relu["source_sha256"]) and relu["source_sha256"] == linear["source_sha256"],
                      "same_epoch": relu["last_evaluation"]["epoch"] == linear["last_evaluation"]["epoch"]}
            pairs.append({"relu_run_id": relu["run_id"], "linear_run_id": linear["run_id"], "seed": relu["seed"],
                "epoch": relu["last_evaluation"]["epoch"], "checks": checks, "pairing_passed": all(checks.values()),
                "relu_digit6": relu["digit6_diagnostic"], "linear_digit6": linear["digit6_diagnostic"],
                "relu_overall_accuracy_pct": relu["last_evaluation"]["test"]["accuracy_pct"],
                "linear_overall_accuracy_pct": linear["last_evaluation"]["test"]["accuracy_pct"],
                "interpretation": "A paired intervention on the output ReLU at this seed/horizon. Near-suppressed class-6 learning is not a literally all-zero gradient or permanently dead output: the measured ReLU head has some positive preactivations and a small nonzero final-minibatch gradient."})
    return pairs


def selected_search_status(path, expected, attempts):
    if not path.exists():
        return {"present": False, "verified": False}
    data = load(path)
    selected = {k: data.get(k) for k in ("frozen_utc", "groups", "fresh_seeds", "test_monitored_search", "search_closed_after_these_confirmations")}
    by_id = {r["run_id"]: r for r in attempts}
    audits = []
    for name in data.get("groups", []):
        jobs = [s for s in expected.values() if s.get("confirmation_group") == name and s.get("confirmation_type") == "fresh-seed"]
        ids = [s["run_id"] for s in sorted(jobs, key=lambda s: s["seed"])]
        observed = [by_id[rid] for rid in ids if rid in by_id]
        audits.append({"group": name, "run_ids": ids,
                       "seeds_match_selection": sorted(s["seed"] for s in jobs) == sorted(data.get("fresh_seeds", [])),
                       "all_attempts_present_and_verified": bool(ids) and len(observed) == len(ids) and all(r["status"] not in ("invalid", "incomplete") for r in observed),
                       "all_reached": bool(ids) and len(observed) == len(ids) and all(r["target_reached"] and r["status"] != "invalid" for r in observed)})
    return {"present": True, "file": path.name, "sha256": sha_bytes(path.read_bytes()), "selection": selected,
            "groups": audits, "verified": bool(audits) and data.get("test_monitored_search") is True
            and data.get("search_closed_after_these_confirmations") is True
            and all(x["seeds_match_selection"] and x["all_attempts_present_and_verified"] for x in audits)}


def build(results_dir=EXPERIMENT / "results", manifests_dir=HERE,
          output_json=EXPERIMENT / "results" / "optimization-analysis.json",
          output_markdown=REPO / "research" / "ciresan-optimization.md", finalize=False):
    results_dir, manifests_dir = Path(results_dir), Path(manifests_dir)
    attempts, microbenchmarks, issues, seen = [], [], [], set()
    expected, manifests, manifest_issues = expected_manifest_runs(manifests_dir)
    issues.extend(manifest_issues)
    for path in sorted(results_dir.glob("opt-*.json")):
        try:
            raw = load(path)
            if not isinstance(raw, dict) or not isinstance(raw.get("spec"), dict):
                raise ValueError("missing_spec")
            if raw.get("run_id") != path.stem or raw["spec"].get("run_id") != path.stem:
                raise ValueError("run_id_mismatch")
            if path.stem in expected and raw["spec"] != expected[path.stem]:
                issues.append({"run_id": path.stem, "code": "result_spec_differs_from_manifest"})
            seen.add(path.stem)
            if raw["spec"].get("kind") == "microbench":
                microbenchmarks.append(summarize_microbench(raw, path))
            else:
                attempts.append(summarize_accuracy(raw, path))
        except (ValueError, TypeError, KeyError, AttributeError):
            issues.append({"file": path.name, "code": "unreadable_or_malformed_result"})
    confirm_expected = sorted(rid for rid in expected if rid.startswith("opt-confirm"))
    by_id = {r["run_id"]: r for r in attempts}
    confirmations_complete = bool(confirm_expected) and all(rid in by_id and by_id[rid]["status"] not in ("incomplete", "invalid") for rid in confirm_expected)
    pending = sorted(set(expected) - seen)
    selected_search = selected_search_status(manifests_dir / "stable-selection.json", expected, attempts)
    diagnostic_pairs = head_diagnostic_pairs(attempts)
    if any(not pair["pairing_passed"] for pair in diagnostic_pairs):
        issues.append({"code": "head_diagnostic_pairing_failed"})
    if finalize and (pending or not confirmations_complete or issues or not selected_search["verified"]
                     or any(r["status"] in ("invalid", "incomplete") for r in attempts + microbenchmarks)):
        raise ValueError("Cannot finalize: expected results, completed confirmation attempts and valid records are required")
    valid_hits = [r for r in attempts if r["status"] == "reached" and r["cohort"] != "diagnostic"]
    best_training = min(valid_hits, key=lambda r: r["first_target"]["training_seconds"]) if valid_hits else None
    best_invocation = min(valid_hits, key=lambda r: r["first_target"]["run_wall_seconds"]) if valid_hits else None
    doc = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
           "status": "finalized_test_target_search" if finalize else "provisional",
           "question": "Time to first observed <=137 errors on official MNIST test10000, fitting pool60000; explicitly test-target optimization.",
           "historical_reference": {"run_id": "ts4k9n55", "url": HISTORICAL_URL,
               "first_target_wandb_runtime_seconds": HISTORICAL_SECONDS, "epoch": 94, "accuracy_pct": 98.63,
               "evaluation_rows_verified": 91, "qualifying_evaluations": 1, "gpu": None,
               "interpretation": "Historical elapsed W&B clock includes evaluation/logging; unknown GPU and incomplete executed-source provenance. Not a matched-hardware kernel baseline."},
           "attempts": attempts, "kernel_benchmarks": microbenchmarks,
           "kernel_only_comparisons": kernel_comparisons(attempts, microbenchmarks),
           "confirmation_groups": confirmation_groups(attempts),
           "confirmation_outcomes": confirmation_outcomes(attempts),
           "selected_search": selected_search, "head_diagnostic_pairs": diagnostic_pairs,
           "verification": {"passed": not issues and not any(r["status"] in ("invalid", "incomplete") for r in attempts + microbenchmarks),
                            "manifest_entries": len(expected), "result_attempts": len(attempts) + len(microbenchmarks),
                            "all_manifest_results_present": not pending, "selected_search_verified": selected_search["verified"]},
           "confirmation_status": {"expected_run_ids": confirm_expected, "complete": confirmations_complete,
               "missing_run_ids": [rid for rid in confirm_expected if rid not in by_id]},
           "manifest_sources": manifests, "pending_or_unavailable_result_ids": pending, "issues": issues,
           "counts": {"accuracy": dict(Counter(r["status"] for r in attempts)), "microbench": dict(Counter(r["status"] for r in microbenchmarks))},
           "exploratory_minima_not_recipe_selection": {"training_time_run": best_training["run_id"] if best_training else None,
               "invocation_to_score_run": best_invocation["run_id"] if best_invocation else None,
               "interpretation": "Descriptive minima over available attempts, potentially different hardware and setup. Do not choose a recipe solely from warm/cold setup differences."},
           "timing_definitions": {
               "training_seconds": "Synchronized minibatch loops: gather, graph-input copies, forward/backward, SGD and direct shrinkage. Excludes epoch permutation, evaluation, checkpoint writes, imports, data loading and graph/compiler preparation or schedule-driven recapture.",
               "run_wall_seconds": "Invocation start before benchmark import through availability of this score: includes invocation setup, loading/transfers, graph/compiler preparation, shuffling, training, prior progress writes and test evaluation. Excludes this score's subsequent checkpoint/progress writes, remote dispatch/container startup, image build and final volume commit.",
               "total_run_seconds": "Invocation through post-score result/checkpoint work to its final timer; still excludes driver dispatch/container startup and final volume commit.",
               "local_dispatch_elapsed_seconds": "Driver time from dispatch to returned result, including queue/startup/import/run/final volume commit/transfer. Whole-call time, not an exact timestamp of first target availability.",
               "setup_seconds": "Per-invocation setup until training begins; does not prove a cold process/cache. Container reuse can change this value.",
               "microbench_step_seconds": "Warm repeated-single-batch completed updates measured with synchronization; excludes compiler/setup cost and epoch gather/shuffle. Not time to accuracy."},
           "limitations": [
               "Official test outcomes actively guided configuration choices and early stopping. These results are test-target optimization, not unbiased held-out generalization estimates.",
               "A first qualifying epoch is an observed checkpoint; the preceding evaluation gives temporal resolution but cannot exclude an earlier transient within-epoch crossing.",
               "Censored means target not observed by the evaluation budget; it does not prove the recipe can never reach it. Missing/failed jobs are not successful fast results.",
               "Larger batch, BF16, output-head changes, LR and shrinkage scaling alter the optimization recipe. Only matched-recipe comparisons identify kernel implementation effects.",
               "All rows use unchanged Ciresan widths, but this search has no augmentation and is not the record-setting historical Ciresan training system.",
               "Reported A100-SXM4-40GB, A100-SXM4-80GB and A100 80GB PCIe devices are distinguished. No hardware-normalized speed inference is made across those variants.",
               "Confirmation seeds 101–103 differ from exploratory seed 1 but were already used with the fragile raw/ReLU recipe before the stable recipes were selected. Raw metadata fresh-seed means distinct from seed 1, not untouched by this search. These confirmations do not undo test-set-driven selection or make the 10000 examples independent across runs.",
               "Training, invocation-to-score, whole invocation and driver dispatch clocks answer different questions. None is an energy measurement or invoice."]}
    for path, text in ((Path(output_json), json.dumps(doc, indent=2, allow_nan=False) + "\n"), (Path(output_markdown), markdown(doc))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text)
        temp.replace(path)
    return doc


def f(value, places=3):
    return f"{value:.{places}f}" if finite(value) else "—"


def ms(value):
    return f"{f(value['mean'])} ± {f(value['sample_sd'])} (n={value['n']})" if value["n"] else "—"


def raw_link(run):
    return f"[{run['run_id']}]({RAW_BASE}{run['file']})"


def short_recipe(row):
    r, i = row["recipe"], row["implementation"]
    precision = "TF32" if r["precision"] == "fp32" else r["precision"].upper()
    return (f"{i['mode']}{'+fused' if i['fused'] else ''}, {precision}, B{r['batch_size']}, {'ReLU' if r['output_relu'] else 'linear'} head, "
            f"{r['recipe']} dropout={r['dropout']:g}, scale={r['input_scale']:.5g}, LR {r['lr']:g}, shrink {r['shrinkage']:.8g}, schedule={r['lr_schedule']}")


def markdown(doc):
    provisional = doc["status"] == "provisional"
    lines = ["# Ciresan-width MNIST: optimization results", "", f"Generated {doc['generated_utc']}. **{'Provisional: the optimization search has not been finalized.' if provisional else 'Search finalized; outcomes remain test-monitored.'}**", "",
             "The target is the first observed **98.63% official-test accuracy (at most 137 errors out of 10,000)**. These runs fit the full 60,000-example training pool and monitor the test set after scheduled epochs. Recipe choices respond to test results: this is explicitly a test-target speed search, separate from the interrupted validation-only stochastic-depth study.", "",
             f"The public [historical W&B run]({HISTORICAL_URL}) first and uniquely reached 98.63% at epoch 94 and **{HISTORICAL_SECONDS:.4f} seconds of logged W&B runtime**, among 91 verified evaluations. Its GPU is unknown; its clock includes logging/evaluation and is not isolated training time. Historical source/clock gaps are documented in the [history audit](../experiments/ciresan_stochastic_depth/history/README.md).", ""]
    lookup = {r["run_id"]: r for r in doc["attempts"]}
    selected_search = doc["selected_search"]
    if selected_search.get("verified"):
        lines += ["## Selected recipes: all three confirmation seeds", "",
                  "The two selected linear-head, unit-dropout recipes are modified training recipes. They use B256, BF16 training, dropout 0.2, initial shrinkage 0.00008, and a factor 0.1 applied to both LR and shrinkage at epoch 21. Raw pixels use LR 0.004; normalized pixels (÷255) use LR 0.12. Seeds 101–103 differ from exploratory seed 1 but were also used in earlier recipe checks; the test set was already known and repeatedly monitored.", ""]
        for group in selected_search["groups"]:
            rows = [lookup[rid] for rid in group["run_ids"]]
            hits = [r for r in rows if r["target_reached"]]
            if hits:
                train = [r["first_target"]["training_seconds"] for r in hits]
                wall = [r["first_target"]["run_wall_seconds"] for r in hits]
                lines.append(f"- **{group['group']}: {len(hits)}/{len(rows)} reached the target**; observed training time {min(train):.3f}–{max(train):.3f}s and invocation-to-score {min(wall):.3f}–{max(wall):.3f}s. These ranges span the hardware listed below; they are not hardware-normalized recipe comparisons.")
        lines += ["", "| Selected recipe | Seed | Reported GPU | Target epoch / accuracy | Training s | Invocation-to-score s | Whole invocation s | Dispatch-to-result s |", "|---|---|---|---|---|---|---|---|"]
        for group in selected_search["groups"]:
            for rid in group["run_ids"]:
                r = lookup[rid]; t = r["first_target"]
                lines.append(f"| {raw_link(r)} | {r['seed']} | {r['hardware'].get('gpu','unknown')} | {str(t['epoch'])+' / '+f(t['test']['accuracy_pct'],2)+'%' if t else r['status']} | {f(t['training_seconds'],6) if t else '—'} | {f(t['run_wall_seconds'],6) if t else '—'} | {f(r['timing']['total_run_seconds'],6)} | {f(r['timing']['local_dispatch_elapsed_seconds'],6)} |")
        lines += ["", "Only seed 102 has the same reported GPU (A100-SXM4-40GB) for both selected recipes; the other two pairs mix SXM4/PCIe or 40/80GB variants. Keep the per-run numbers rather than attributing a pooled difference entirely to input normalization or LR.", ""]
    for outcome in doc["confirmation_outcomes"]:
        if outcome["confirmation_type"] == "fresh-seed":
            lines += [f"**Confirmation check for {outcome['confirmation_group'] or outcome['group_id']}: {outcome['n_reached']}/{outcome['n_attempts']} reached the target** by {outcome['max_epochs']} epochs. Seeds: {outcome['seeds']}; outcomes: {outcome['status_counts']}. Same-seed repeats are counted separately; stable-recipe seeds were reused from earlier raw/ReLU checks. "
                      + ("The observed successes do not establish a robust recipe across these tested seeds." if outcome['n_reached'] < outcome['n_attempts'] else "This supports repeatability across these tested seeds; it does not undo test-target selection."), ""]
    minima = doc["exploratory_minima_not_recipe_selection"]
    if provisional and minima["training_time_run"]:
        r = lookup[minima["training_time_run"]]; t = r["first_target"]
        lines += [f"The smallest observed training-loop time so far is **{t['training_seconds']:.3f}s**: {raw_link(r)}, {short_recipe(r)}, seed {r['seed']}. It reaches **{t['test']['accuracy_pct']:.2f}% at epoch {t['epoch']}**, with **{t['run_wall_seconds']:.3f}s invocation-to-score**, {f(r['timing']['total_run_seconds'])}s whole invocation, and {f(r['timing']['local_dispatch_elapsed_seconds'])}s driver dispatch-to-result. This is an observed run, not a cross-seed guarantee or a pure kernel speedup.", ""]
    if provisional and minima["invocation_to_score_run"] and minima["invocation_to_score_run"] != minima["training_time_run"]:
        r = lookup[minima["invocation_to_score_run"]]
        lines += [f"The smallest invocation-to-score time belongs to a different attempt, {raw_link(r)} ({r['first_target']['run_wall_seconds']:.3f}s; {r['first_target']['training_seconds']:.3f}s training). This difference must be read alongside GPU model, setup time, head and numerical recipe; it is not sufficient to choose a faster recipe.", ""]
    lines += ["## Kernel effects that are actually isolated", ""]
    full = [r for r in doc["kernel_only_comparisons"] if r["kind"] == "matched_full_training_horizon"]
    for pair in full:
        lines += [f"- `{pair['reference_run']}` → `{pair['treatment_run']}`: same seed, recipe, initialization, dataset and reported {pair['hardware']['gpu']}; {pair['epochs']} epochs take {pair['reference_training_seconds']:.3f} → {pair['treatment_training_seconds']:.3f}s of training (**{pair['training_speedup']:.3f}×**). All scored test trajectories identical: **{pair['scored_test_trajectories_identical']}**. Target reached: reference={pair['target_reached_by_reference']}, treatment={pair['target_reached_by_treatment']}. This measures fixed-horizon training speed, not a successful time-to-target ratio."]
    if not full:
        lines.append("No verified full-horizon matched baseline comparison is available yet.")
    if doc["head_diagnostic_pairs"]:
        lines += ["", "## Output-head diagnostic: digit 6", ""]
        for pair in doc["head_diagnostic_pairs"]:
            relu, linear = pair["relu_digit6"], pair["linear_digit6"]
            lines += [f"At seed {pair['seed']}, epoch {pair['epoch']}, the paired [ReLU-head run]({RAW_BASE}{pair['relu_run_id']}.json) predicts **zero sixes and correctly classifies {relu['correct_count']}/{relu['true_count']} true sixes**. The [linear-head run]({RAW_BASE}{pair['linear_run_id']}.json) gets **{linear['correct_count']}/{linear['true_count']} sixes correct ({linear['accuracy_pct']:.2f}%)**. Overall test accuracy changes **{pair['relu_overall_accuracy_pct']:.2f}% → {pair['linear_overall_accuracy_pct']:.2f}%**. Initialization, data, implementation, recorded source, GPU and other recipe settings match; only the output ReLU differs. Pairing checks passed: **{pair['pairing_passed']}**.", "",
                      f"This directly supports an output-head failure in this diagnostic, not a claim that every stalled run has the same cause. The ReLU class 6 preactivation is positive on {100*relu['positive_logit_fraction']:.2f}% of test inputs (maximum {relu['head_preactivation_max']:.6f}), and its last-minibatch head-row gradient norm is {relu['last_minibatch_head_row_gradient_norm']:.6g}; the linear head's corresponding norm is {linear['last_minibatch_head_row_gradient_norm']:.6g}. Thus “permanently dead output” or “identically zero gradient” would overstate the evidence. The original fixed-seed history could hide this initialization sensitivity.", ""]
    lines += ["", "CUDA graphs reduce replay launch overhead, while fused SGD reduces optimizer orchestration. The relevant evidence here is the matched measurements, not an assumption that these features always help. [PyTorch CUDA graphs](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs), [SGD implementation choices](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html).", "", "## All accuracy attempts", "", "Time columns for successful rows refer to the first target score. For censored/diverged rows they refer to the last evaluated checkpoint and are explicitly not successful target times. Dispatch is always the whole returned call. Diagnostic runs have deliberately shorter horizons; their lack of a target crossing is not directly comparable to a 150/200-epoch search.", "", "| Attempt | Recipe | GPU | Status / cohort | Last / maximum epochs | First target epoch / accuracy | Best / last accuracy (%) | Training s at target or censor | Invocation s at target or censor | Setup s | Whole invocation s | Dispatch s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["attempts"]:
        t, last, best = r.get("first_target"), r.get("last_evaluation"), r.get("best_accuracy_evaluation")
        observed = t or last
        target = f"{t['epoch']} / {t['test']['accuracy_pct']:.2f}%" if t else "not observed"
        lines.append(f"| {raw_link(r)} | {short_recipe(r)} | {r['hardware'].get('gpu','unknown')} | {r['status']} / {r['cohort']} | {last['epoch'] if last else '—'} / {r['max_epochs']} | {target} | {f(best['test']['accuracy_pct'],2) if best else '—'} / {f(last['test']['accuracy_pct'],2) if last else '—'} | {f(observed['training_seconds']) if observed else '—'} | {f(observed['run_wall_seconds']) if observed else '—'} | {f(r['timing'].get('setup_seconds'))} | {f(r['timing'].get('total_run_seconds'))} | {f(r['timing'].get('local_dispatch_elapsed_seconds'))} |")
    lines += ["", "All attempts retain source-style raw 0–255 pixels unless their recipe fields say otherwise. Larger batch changes update count and momentum dynamics even when LR and multiplicative shrinkage are scaled by batch size. BF16 changes training arithmetic; evaluation uses the FP32 stored parameters under TF32. Linear heads, unit dropout and normalization are explicit recipe changes. Listed schedule factors multiply both initial LR and initial direct shrinkage, requiring graph recapture; recapture time is excluded from training-loop time and included in invocation time. The six affine layers and 11,972,510 parameters are retained.", "", "### First target and preceding observation", "", "| Run | Previous epoch / accuracy | Previous training / invocation s | First target epoch / accuracy | Target training / invocation s |", "|---|---|---|---|---|"]
    for r in doc["attempts"]:
        if not r.get("first_target") or r["status"] == "invalid":
            continue
        p, t = r["preceding_evaluation"], r["first_target"]
        lines.append(f"| {r['run_id']} | {str(p['epoch'])+' / '+f(p['test']['accuracy_pct'],2)+'%' if p else 'none observed'} | {f(p['training_seconds'])+' / '+f(p['run_wall_seconds']) if p else '—'} | {t['epoch']} / {t['test']['accuracy_pct']:.2f}% | {t['training_seconds']:.3f} / {t['run_wall_seconds']:.3f} |")
    lines += ["", "The preceding score describes observation cadence; accuracy need not change monotonically, and an unobserved transient earlier within an epoch cannot be ruled out.", "", "## Kernel microbenchmarks and compiler cost", "", "These repeatedly update one fixed real minibatch. Three timing repetitions are not three independent training seeds. Setup and steady-state timing are both shown; microbenchmark timings do not establish convergence.", "", "| Run / variant | GPU | Status | Warm median ms/update | Variant setup s | Whole invocation / dispatch s |", "|---|---|---|---|---|---|"]
    for bench in doc["kernel_benchmarks"]:
        if not bench["variants"]:
            lines.append(f"| {raw_link(bench)} | {bench['hardware'].get('gpu','unknown')} | {bench['status']} ({bench.get('failure',{}).get('category','')}) | — | — | {f(bench.get('total_run_seconds'))} / {f(bench.get('local_dispatch_elapsed_seconds'))} |")
        for row in bench["variants"]:
            label = short_recipe(row)
            if row["implementation"]["mode"] == "compile":
                label += " / " + str(row["implementation"]["compile_mode"])
            lines.append(f"| {raw_link(bench)} / {label} | {bench['hardware'].get('gpu','unknown')} | {bench['status']} | {1000*row['median_step_seconds']:.4f} | {row['setup_seconds']:.3f} | {f(bench.get('total_run_seconds'))} / {f(bench.get('local_dispatch_elapsed_seconds'))} |")
    lines += ["", "The compiler candidate compiles the model with eager optimizer orchestration; explicit CUDA graph captures the full update. Different capture scope, A100 memory variants and cache history limit cross-run attribution. A large compile/setup cost matters for this short workload even if later replays are faster. [PyTorch compile modes](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html), [compiler caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).", "", "## Confirmation runs", ""]
    cs = doc["confirmation_status"]
    lines += [f"Expected confirmation attempts: {len(cs['expected_run_ids'])}; all declared attempts complete: **{cs['complete']}**. Missing: {', '.join(cs['missing_run_ids']) or 'none declared/missing'}.", ""]
    if doc["confirmation_groups"]:
        lines += ["Target means ± sample SD below include successful seeds only. The attempted/reached denominator and censored/failed counts must accompany those means. Hardware-specific groups and seed-1 repeats versus other confirmation seeds are separate. The preserved raw label `fresh-seed` means different from exploratory seed 1; stable-recipe seeds 101–103 were already tested with the fragile raw/ReLU recipe.", "", "| Recipe / GPU / confirmation type | Reached / attempted | Statuses | Target epochs | Target training seconds | Target invocation seconds |", "|---|---|---|---|---|---|"]
        for group in doc["confirmation_groups"]:
            lines.append(f"| {short_recipe(group)} / {group['hardware'].get('gpu','unknown')} / {group['confirmation_type']} | {group['n_reached']} / {group['n_attempts']} | {group['status_counts']} | {ms(group['target_epoch'])} | {ms(group['target_training_seconds'])} | {ms(group['target_run_wall_seconds'])} |")
    else:
        lines.append("No confirmation result is available yet. Exploratory seed1 successes are insufficient to establish robust time-to-target.")
    lines += ["", "## Timing and interpretation", ""]
    lines += [f"- **{key}:** {text}" for key, text in doc["timing_definitions"].items()]
    lines += ["", "Container/cache reuse was not recorded as a reliable cold/warm flag. Differences in setup or driver delay must not be credited to SGD, precision or batch size. No cold-process benchmark or energy measurement is claimed.", ""]
    lines += ["- " + text for text in doc["limitations"]]
    lines += ["", "## Completeness and evidence", "", f"Accuracy statuses: {doc['counts']['accuracy']}. Kernel/compile statuses: {doc['counts']['microbench']}.", ""]
    if doc["pending_or_unavailable_result_ids"]:
        lines.append("Manifest entries without results (pending, cancelled or otherwise unavailable; status not inferred): " + ", ".join(doc["pending_or_unavailable_result_ids"]) + ".")
    for item in doc["issues"]:
        lines.append(f"- {item.get('run_id',item.get('file','study'))}: {item['code']}")
    for r in doc["attempts"] + doc["kernel_benchmarks"]:
        for text in r.get("verification_issues", []):
            lines.append(f"- {r['run_id']}: {text}")
    lines += ["", "The [machine-readable summary](../experiments/ciresan_stochastic_depth/results/optimization-analysis.json) includes result-byte hashes, effective recipes, initialization/data identity, source hashes when available, first/previous/last evaluations and all failures. It copies no checkpoints, arbitrary remote errors, private paths or credentials. The [implementation research and audit](../experiments/ciresan_stochastic_depth/optimization/research.md) documents graph-state restoration, precision-API compatibility and primary references.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results")
    p.add_argument("--manifests-dir", type=Path, default=HERE)
    p.add_argument("--output-json", type=Path, default=EXPERIMENT / "results" / "optimization-analysis.json")
    p.add_argument("--output-markdown", type=Path, default=REPO / "research" / "ciresan-optimization.md")
    p.add_argument("--finalize", action="store_true", help="Explicitly close the search only after all declared results and confirmations are present")
    args = p.parse_args()
    doc = build(**vars(args))
    print(json.dumps({"status": doc["status"], "counts": doc["counts"],
                      "verification_issues": len(doc["issues"]) + sum(len(r["verification_issues"]) for r in doc["attempts"] + doc["kernel_benchmarks"]),
                      "confirmation_complete": doc["confirmation_status"]["complete"]}))


if __name__ == "__main__":
    main()
