"""Export completed Ciresan telemetry after training; JSONL needs no W&B SDK.

Default mode is jsonl (no network). Optional offline/online modes lazily import
W&B and create a NEW run, never resume or edit historical runs. W&B's upload
runtime/step are not training clocks. The default chart axis is training_seconds.

Accepted preferred rows:
  telemetry_history: [{epoch, training_seconds, run_elapsed_seconds,
                       metrics: {flat_name: number}, diagnostics: {...}}]
The older train_optimized history schema has a narrow, explicit adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_PROJECT = "gradient-dissent"
DEFAULT_GROUP = "ciresan-telemetry-20260909"
# A detailed six-layer record includes ~837 named metrics plus ~565 nested
# diagnostic scalars. Keep all of them; this is a sanity bound, not truncation.
MAX_SCALAR_FIELDS = 2048
AXES = ("epoch", "training_seconds", "run_elapsed_seconds", "optimizer_step")
CONFIG_KEYS = {
    "recipe", "pmax", "input_scale", "output_relu", "lr", "shrinkage", "seed",
    "epochs", "batch_size", "momentum", "train_size", "precision", "optimizer",
    "schedule", "eval_every", "cohort", "stage", "widths", "train_pool",
    "stats_batch_size", "stats_num_batches", "stats_every", "diagnostic_every",
    "full_batch", "hess_samples", "hess_kfac", "compute_rho", "curv",
    "skip_stats", "log_spectra", "disable_hess", "train_steps", "stats_steps",
    "weight_decay", "nonlin", "bias", "dropout", "dataset_size",
    "max_epochs", "stop_at_target", "mode", "fused", "runner", "gpu",
    "telemetry", "telemetry_every", "telemetry_probe_n", "lr_schedule",
}
KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-/]*$")
SECRET_KEY = re.compile(r"api.?key|password|secret|access.?token|authorization|credential", re.I)
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe_config(spec):
    result = {}
    for key in CONFIG_KEYS:
        if key not in spec:
            continue
        value = spec[key]
        if isinstance(value, bool) or value is None or _number(value):
            result[key] = value
        elif isinstance(value, str) and len(value) <= 128 and not any(c in value for c in ("/", "\\", "\n")):
            result[key] = value
        elif key == "widths" and isinstance(value, list) and all(isinstance(x, int) and x > 0 for x in value):
            result[key] = value
        elif key == "lr_schedule" and isinstance(value, dict) and all(
                isinstance(k, str) and k.isdecimal() and _number(v) and v >= 0
                for k, v in value.items()):
            result[key] = value
    return result


def _metric_name(name):
    if not isinstance(name, str) or not KEY.fullmatch(name) or len(name) > 180 or SECRET_KEY.search(name):
        raise ValueError("Telemetry contains an invalid or credential-like metric name")
    if name in AXES:
        raise ValueError("Put measured clocks in row fields, not inside metrics")
    return name


def _flatten_diagnostics(value, prefix="diagnostics"):
    """Only finite scalar diagnostics; arrays/text remain in original results."""
    out, omitted = {}, 0
    if not isinstance(value, dict):
        return out, int(value is not None)
    for key, item in value.items():
        name = _metric_name(f"{prefix}/{key}")
        if _number(item):
            out[name] = item
        elif isinstance(item, dict):
            nested, count = _flatten_diagnostics(item, name)
            out.update(nested)
            omitted += count
        else:
            omitted += 1
    return out, omitted


def _legacy_metrics(row):
    mapped = {}
    if _number(row.get("stochastic_training_loss")):
        mapped["train/stochastic_ce"] = row["stochastic_training_loss"]
    for source_key, prefix in (("dense_train_probe", "train_probe"), ("validation", "validation"), ("test", "test")):
        entry = row.get(source_key)
        if not isinstance(entry, dict):
            continue
        if _number(entry.get("loss")):
            mapped[f"{prefix}/ce"] = entry["loss"]
        if _number(entry.get("accuracy")):
            if not 0 <= entry["accuracy"] <= 1:
                raise ValueError("Old history adapter expects accuracy fractions in [0,1]")
            mapped[f"{prefix}/accuracy_pct"] = 100*entry["accuracy"]
        if _number(entry.get("n")):
            mapped[f"{prefix}/n"] = entry["n"]
    return mapped


def prepare_payload(result, *, source_sha256=None, project=DEFAULT_PROJECT, group=DEFAULT_GROUP):
    if not isinstance(result, dict):
        raise ValueError("Result must be an object")
    if not ID.fullmatch(project) or project == "train_ciresan":
        raise ValueError("Use the new telemetry project; historical train_ciresan is read-only")
    if not ID.fullmatch(group):
        raise ValueError("Invalid new-run group")
    spec = result.get("spec", {})
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    source_id = result.get("run_id", spec.get("run_id"))
    if not isinstance(source_id, str) or not ID.fullmatch(source_id):
        raise ValueError("Missing or invalid scientific run_id")
    preferred = "telemetry_history" in result
    history = result.get("telemetry_history" if preferred else "history")
    if not isinstance(history, list) or not history:
        raise ValueError("Need a nonempty completed telemetry_history or supported history")
    rows, omitted = [], 0
    previous = {}
    schema = result.get("telemetry_schema", {})
    if not isinstance(schema, dict):
        raise ValueError("telemetry_schema must be an object when supplied")
    allow_legacy_val = schema.get("legacy_val_is_official_test_percent") is True
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("Each history row must be an object")
        row = {}
        for axis in AXES:
            value = record.get(axis)
            if value is None:
                continue
            if not _number(value) or value < 0 or value < previous.get(axis, 0):
                raise ValueError(f"Measured axis {axis} must be finite, nonnegative and nondecreasing")
            row[axis] = value
            previous[axis] = value
        if "epoch" not in row or "training_seconds" not in row:
            raise ValueError("Every exported row requires measured epoch and training_seconds")
        raw = record.get("metrics", {}) if preferred else _legacy_metrics(record)
        if not isinstance(raw, dict):
            raise ValueError("metrics must be a flat scalar object")
        for key, value in raw.items():
            _metric_name(key)
            if not _number(value):
                if value is None:
                    omitted += 1
                    continue
                raise ValueError("Metrics must contain finite numeric values or explicit null")
            if key in ("val_accuracy", "val_loss"):
                if not allow_legacy_val:
                    raise ValueError("Ambiguous val_* aliases require an explicit official-test-percent schema declaration")
                canonical = "test/accuracy_pct" if key == "val_accuracy" else "test/ce"
                if canonical in raw and raw[canonical] != value:
                    raise ValueError("Legacy official-test alias disagrees with canonical test metric")
                row[canonical] = value
            row[key] = value
        for key, value in list(row.items()):
            if key.endswith("/accuracy_pct") and not 0 <= value <= 100:
                raise ValueError("accuracy_pct metrics must use percent units in [0,100]")
            if key.endswith("/ce") and value < 0:
                raise ValueError("Cross-entropy cannot be negative")
        # Original names are explicit official-test aliases, never validation.
        if "test/accuracy_pct" in row:
            row["val_accuracy"] = row["test/accuracy_pct"]
        if "test/ce" in row and "test/loss" in row and row["test/ce"] != row["test/loss"]:
            raise ValueError("Official-test CE and loss aliases disagree")
        if "test/ce" in row or "test/loss" in row:
            row["val_loss"] = row.get("test/ce", row.get("test/loss"))
        diagnostics, skipped = _flatten_diagnostics(record.get("diagnostics", {}))
        collisions = set(row) & set(diagnostics)
        if collisions:
            raise ValueError("Duplicate metric names between metrics and diagnostics")
        row.update(diagnostics)
        omitted += skipped
        if len(row) > MAX_SCALAR_FIELDS:
            raise ValueError(f"Unexpectedly wide scalar telemetry row ({len(row)} > {MAX_SCALAR_FIELDS}); no values were truncated")
        rows.append(row)
    config = _safe_config(spec)
    config.update(scientific_run_id=source_id, telemetry_uploaded_after_training=True,
                  training_clock="training_seconds", run_clock="run_elapsed_seconds",
                  accuracy_units="percent for *_pct and legacy val_accuracy",
                  legacy_val_alias_semantics="official test only; validation/accuracy_pct is separate")
    content = {"config": config, "rows": rows}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"schema_version": 1, "project": project, "group": group,
            "run_id": "telemetry-"+digest[:20], "name": source_id, "config": config,
            "source_sha256": source_sha256, "content_sha256": digest,
            "metric_names": sorted(set().union(*(set(row)-set(AXES) for row in rows))),
            "axes": {"epoch": "Measured completed epoch", "training_seconds": "Measured cumulative training clock from source; never upload runtime",
                     "run_elapsed_seconds": "Measured original run elapsed clock, only where recorded; never synthesized",
                     "optimizer_step": "Measured optimizer count, only where recorded; exporter row index is separate"},
            "omitted_null_or_nonscalar_diagnostics_count": omitted,
            "rows": rows}


def write_jsonl(payload, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir/(payload["run_id"]+".jsonl")
    manifest = output_dir/(payload["run_id"]+".manifest.json")
    jsonl.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n" for row in payload["rows"]))
    metadata = {key: value for key, value in payload.items() if key != "rows"}
    metadata["history_rows"] = len(payload["rows"])
    metadata["jsonl_sha256"] = hashlib.sha256(jsonl.read_bytes()).hexdigest()
    metadata["jsonl_file"] = jsonl.name
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return {"jsonl_file": jsonl.name, "manifest_file": manifest.name, "row_count": len(payload["rows"])}


def publish(payload, *, mode="offline", entity=None, sdk_directory=None, axis="training_seconds", wandb_module=None):
    """Explicit optional SDK action; call after training, never from its timer."""
    if mode not in ("offline", "online") or axis not in AXES:
        raise ValueError("Expected offline/online SDK mode and a measured custom axis")
    if any(axis not in row for row in payload["rows"]):
        raise ValueError("Chosen chart axis is absent from some rows; do not invent elapsed times")
    if entity is not None and not ID.fullmatch(entity):
        raise ValueError("Invalid entity")
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("W&B SDK is not installed. JSONL export is complete and requires no login.") from exc
    directory = Path(sdk_directory) if sdk_directory else Path.home()/".cache"/"gradient-dissent"/"wandb-posthoc"
    directory.mkdir(parents=True, exist_ok=True)
    settings = wandb_module.Settings(console="off", disable_code=True, disable_git=True,
                                    x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
                                    x_save_requirements=False, init_timeout=60, finish_timeout=60,
                                    finish_timeout_raises=True)
    with wandb_module.init(project=payload["project"], entity=entity, group=payload["group"],
                           id=payload["run_id"], name=payload["name"], resume="never",
                           reinit="create_new",
                           job_type="post-training-telemetry-import", mode=mode, force=mode=="online",
                           config=payload["config"], save_code=False, sync_tensorboard=False,
                           dir=str(directory), settings=settings,
                           notes="Imported after training. Plot against measured epoch/training_seconds/run_elapsed_seconds; W&B runtime describes the importer.") as run:
        for step_axis in AXES:
            run.define_metric(step_axis, hidden=True)
        for metric in payload["metric_names"]:
            run.define_metric(metric, step_metric=axis, step_sync=False)
        for index, row in enumerate(payload["rows"]):
            run.log(row, step=index, commit=True)
        run.summary["telemetry/source_sha256"] = payload["source_sha256"]
        run.summary["telemetry/content_sha256"] = payload["content_sha256"]
        run.summary["telemetry/history_rows"] = len(payload["rows"])
        run.summary["telemetry/imported_after_training"] = True
        run.summary["telemetry/default_plot_axis"] = axis
        run_url = run.url if mode == "online" else None
    return {"mode": mode, "run_id": payload["run_id"], "project": payload["project"], "run_url": run_url,
            "history_rows": len(payload["rows"]), "note": "W&B upload runtime is not a training-time measurement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("jsonl", "offline", "online"), default="jsonl")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--entity")
    parser.add_argument("--axis", choices=AXES, default="training_seconds")
    parser.add_argument("--sdk-directory", type=Path)
    args = parser.parse_args()
    exports = []
    for source in args.results:
        raw = source.read_bytes()
        result = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        payload = prepare_payload(result, source_sha256=hashlib.sha256(raw).hexdigest(), project=args.project, group=args.group)
        export = write_jsonl(payload, args.output_dir)
        if args.mode != "jsonl":
            export["wandb"] = publish(payload, mode=args.mode, entity=args.entity, axis=args.axis, sdk_directory=args.sdk_directory)
        exports.append(export)
    print(json.dumps({"mode": args.mode, "exports": exports}, allow_nan=False))


if __name__ == "__main__":
    main()
