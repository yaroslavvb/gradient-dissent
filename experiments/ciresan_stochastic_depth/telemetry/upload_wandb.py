"""Post-training W&B upload with verified links and bounded recovery.

Default: prepare a local login-blocked registry, no SDK/network. --upload needs
an already-configured login; never prompt for credentials or launch compute.
Each isolated worker has a wall-time limit. An uncertain write is reconciled
against the SAME content-derived run ID before any retry or continuation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import netrc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

try:
    from .wandb_export import AXES, DEFAULT_PROJECT, DEFAULT_GROUP, prepare_payload
except ImportError:
    from wandb_export import AXES, DEFAULT_PROJECT, DEFAULT_GROUP, prepare_payload

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results/telemetry"
REGISTRY = RESULTS / "wandb-upload-manifest.json"
ENTITY = "yaroslavvb"
RECIPES = ("plain", "residual", "sd_constant", "sd_annealed", "residual_unit_dropout")
SOURCES = tuple(f"telemetry-main-{recipe}-s{seed}-v1" for recipe in RECIPES for seed in (101, 102, 103)) + tuple(
    f"telemetry-baseline-s{seed}-v1" for seed in (101, 102, 103))
WORKER_SECONDS = 180
MAX_ATTEMPTS = 3


class IntegrityError(ValueError):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def payload_for(source_id):
    if source_id not in SOURCES:
        raise IntegrityError("Only the 18 frozen telemetry source runs are eligible")
    raw = (RESULTS / (source_id + ".json")).read_bytes()
    source = json.loads(raw)
    if source.get("run_id") != source_id or source.get("diverged") or source.get("error"):
        raise IntegrityError("Expected a completed, nondivergent scientific source")
    if not source.get("trained_final_parameters_sha256") or not source.get("telemetry_history"):
        raise IntegrityError("Source training has not finalized")
    payload = prepare_payload(source, source_sha256=hashlib.sha256(raw).hexdigest())
    # Extra summary fields do not alter the existing history/config content
    # hash or content-derived ID. They copy frozen endpoints, never reselect.
    payload["scientific_summary"] = scientific_summary(source)
    return payload


def scientific_summary(source):
    output = {}
    def put(key, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            output[key] = value
    def evaluation(prefix, value):
        if not isinstance(value, dict):
            return
        for key in ("loss", "errors", "n"):
            put(prefix + "_" + key, value.get(key))
        if isinstance(value.get("accuracy"), (int, float)):
            put(prefix + "_accuracy_pct", 100 * value["accuracy"])
    put("selected/epoch", source.get("best_epoch"))
    put("selected/validation_loss", source.get("best_validation_loss"))
    evaluation("selected/test", source.get("selected_test"))
    evaluation("selected/train", source.get("selected_train"))
    evaluation("final/test", source.get("final_test"))
    evaluation("final/train", source.get("final_train"))
    put("time/training_seconds", source.get("training_seconds"))
    put("time/run_elapsed_seconds", source.get("total_run_seconds"))
    put("time/telemetry_seconds", source.get("telemetry", {}).get("seconds"))
    target = source.get("threshold")
    if isinstance(target, dict):
        put("target/epoch", target.get("epoch"))
        put("target/training_seconds", target.get("training_seconds"))
        put("target/run_elapsed_seconds", target.get("run_wall_seconds_with_telemetry", target.get("run_wall_seconds")))
        evaluation("target/test", target.get("test"))
    return output


def make_registry(previous=None):
    old = {r["scientific_run_id"]: r for r in (previous or {}).get("runs", [])}
    runs = []
    for source_id in SOURCES:
        payload = payload_for(source_id)
        record = {"scientific_run_id": source_id, "wandb_run_id": payload["run_id"],
                  "entity": ENTITY, "project": DEFAULT_PROJECT, "group": DEFAULT_GROUP,
                  "source_file": "experiments/ciresan_stochastic_depth/results/telemetry/" + source_id + ".json",
                  "source_sha256": payload["source_sha256"], "content_sha256": payload["content_sha256"],
                  "history_rows_expected": len(payload["rows"]), "status": "blocked_login",
                  "scientific_summary": payload["scientific_summary"],
                  "verified_wandb_url": None, "verification": None}
        prior = old.get(source_id)
        if prior:
            for key in ("wandb_run_id", "source_sha256", "content_sha256"):
                if prior.get(key) != record[key]:
                    raise IntegrityError("Source changed after creation of the upload registry")
            if prior.get('entity') != ENTITY or prior.get('project') != DEFAULT_PROJECT:
                record['previous_destination'] = {
                    'entity': prior.get('entity'), 'project': prior.get('project'),
                    'status_before_switch': prior.get('status'),
                    'attempts_before_switch': prior.get('attempts', 0),
                }
                # A moved run must be read back at its new destination.
                prior = None
        if prior:
            # Preparation never fabricates links or erases a prior verified result.
            for key in ("status", "verified_wandb_url", "verification", "attempts", "last_attempt_utc", "error_class", "previous_destination"):
                if key in prior:
                    record[key] = prior[key]
        runs.append(record)
    complete = all(r["status"] == "verified" for r in runs)
    return {"schema_version": 1, "updated_utc": now(), "entity": ENTITY, "project": DEFAULT_PROJECT,
            "group": DEFAULT_GROUP, "status": "verified" if complete else "blocked_login",
            "complete": complete, "expected_run_count": 18,
            "verified_run_count": sum(r["status"] == "verified" for r in runs),
            "clock_semantics": "Measured epoch/training_seconds/run_elapsed_seconds are custom axes; W&B importer runtime is not training time.",
            "upload_scope": "Sanitized scientific config and all exported scalar telemetry. No model weights, source files, environment dumps, or credentials uploaded.",
            "verification_rule": "An actual server run, exact source/config identity, all expected metric/clock cells, source/content hashes, and finished state must be read back before exposing its SDK-returned URL.",
            "runs": runs}


def configured_auth_present():
    """Presence only. Credential values stay local and are never returned/logged."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    path = Path(os.environ.get("NETRC", str(Path.home() / ".netrc")))
    if path.exists():
        try:
            auth = netrc.netrc(str(path)).authenticators("api.wandb.ai")
            if auth and auth[2]:
                return True
        except (OSError, netrc.NetrcParseError):
            pass
    # The SDK also supports configured credentials outside netrc. This lookup
    # only returns a boolean and does not initiate an interactive login.
    try:
        import wandb
        from wandb.sdk.lib.apikey import api_key
        return bool(api_key(wandb.Settings()))
    except Exception:
        return False


def same_number(a, b):
    return (isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a)
            and (a == b or math.isclose(a, b, rel_tol=1e-12, abs_tol=0.)))


def verified_prefix(expected_rows, actual_rows):
    """Require a contiguous exact scientific prefix; never overwrite a conflict."""
    by_step = {}
    for row in actual_rows:
        step = row.get("_step")
        if (not isinstance(step, (int, float)) or isinstance(step, bool)
                or not math.isfinite(step) or step != int(step) or step in by_step):
            raise IntegrityError("Remote history has absent or duplicate importer steps")
        by_step[int(step)] = row
    if sorted(by_step) != list(range(len(by_step))) or len(by_step) > len(expected_rows):
        raise IntegrityError("Remote history is not a contiguous expected prefix")
    cells = 0
    for step in range(len(by_step)):
        expected, actual = expected_rows[step], by_step[step]
        # W&B adds only underscore-prefixed runtime metadata. Reject any other
        # unexpected key to avoid adopting somebody else's conflicting run.
        extra = {k for k in actual if not k.startswith("_")} - set(expected)
        if extra:
            raise IntegrityError("Unexpected scientific fields in existing remote row")
        for key, value in expected.items():
            if key not in actual or not same_number(actual[key], value):
                raise IntegrityError("Remote scientific history conflicts with source")
            cells += 1
    return len(by_step), cells


def inspect_remote(api, payload):
    from wandb.apis.public.runs import RunNotFoundError
    from wandb.errors import CommError
    api.flush()
    try:
        run = api.run(f"{ENTITY}/{DEFAULT_PROJECT}/{payload['run_id']}")
    except (RunNotFoundError, CommError) as exc:
        # SDK 0.30 normalizes RunNotFoundError into CommError, retaining
        # the original typed exception in .exc. Other API errors must not
        # be mistaken for absence, especially auth/network failures.
        if not isinstance(exc, RunNotFoundError) and not isinstance(exc.exc, RunNotFoundError):
            raise
        # If this is a permissions failure disguised as absence, resume=never
        # still cannot overwrite an existing run and the write will fail.
        return {"exists": False}
    if list(run.path) != [ENTITY, DEFAULT_PROJECT, payload["run_id"]]:
        raise IntegrityError("Remote run path mismatch")
    if run.name != payload["name"] or run.group != DEFAULT_GROUP:
        raise IntegrityError("Remote run name/group does not identify this export")
    for key, value in payload["config"].items():
        if run.config.get(key) != value:
            raise IntegrityError("Remote scientific config conflicts with source")
    count, cells = verified_prefix(payload["rows"], list(run.scan_history(page_size=100, use_cache=False)))
    summary = dict(run.summary)
    for key, value in (("telemetry/source_sha256", payload["source_sha256"]),
                       ("telemetry/content_sha256", payload["content_sha256"])):
        if key in summary and summary[key] != value:
            raise IntegrityError("Remote provenance hash conflicts with source")
    all_hashes = (summary.get("telemetry/source_sha256") == payload["source_sha256"]
                  and summary.get("telemetry/content_sha256") == payload["content_sha256"]
                  and summary.get("telemetry/history_rows") == len(payload["rows"]))
    all_summaries = all(same_number(summary.get(key), value)
                        for key, value in payload["scientific_summary"].items())
    complete = count == len(payload["rows"]) and run.state == "finished" and all_hashes and all_summaries
    url = None
    if complete:
        # URL comes from the SDK object AFTER actual run and content retrieval.
        candidate = run.url
        parsed = urlparse(candidate)
        if parsed.scheme != "https" or parsed.netloc != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{DEFAULT_PROJECT}/runs/{payload['run_id']}":
            raise IntegrityError("Unexpected verified remote URL")
        url = candidate
    return {"exists": True, "state": run.state, "prefix_rows": count, "verified_cells": cells,
            "complete": complete, "verified_wandb_url": url,
            "verification": {"verified_at_utc": now(), "source_and_config_match": True,
                             "history_rows_retrieved": count, "scientific_cells_checked": cells,
                             "all_provenance_hashes_match": all_hashes,
                             "all_frozen_endpoint_and_timing_summaries_match": all_summaries,
                             "finished": run.state == "finished"}}


def write_remote(wandb, payload, prefix):
    """Continuation is allowed only after inspect_remote validates the prefix."""
    cache = Path.home() / ".cache/gradient-dissent/wandb-posthoc"
    cache.mkdir(parents=True, exist_ok=True)
    settings = wandb.Settings(console="off", disable_code=True, disable_git=True,
        x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
        x_save_requirements=False, init_timeout=45, finish_timeout=45, finish_timeout_raises=True,
        x_graphql_retry_max=1, x_file_stream_retry_max=2, x_file_transfer_retry_max=2)
    with wandb.init(entity=ENTITY, project=DEFAULT_PROJECT, group=DEFAULT_GROUP,
        id=payload["run_id"], name=payload["name"], resume="must" if prefix is not None else "never",
        reinit="create_new", mode="online", force=True, config=payload["config"],
        job_type="post-training-telemetry-import", save_code=False, sync_tensorboard=False,
        dir=str(cache), settings=settings,
        notes="Post-training scalar import. Use measured training_seconds/epoch/run_elapsed_seconds; W&B runtime is importer runtime.") as run:
        for axis in AXES:
            run.define_metric(axis, hidden=True)
        for metric in payload["metric_names"]:
            run.define_metric(metric, step_metric="training_seconds", step_sync=False)
        # Persist provenance before history so an interrupted import identifies
        # its exact source. Remote completion still requires every history cell.
        run.summary["telemetry/source_sha256"] = payload["source_sha256"]
        run.summary["telemetry/content_sha256"] = payload["content_sha256"]
        run.summary["telemetry/history_rows"] = len(payload["rows"])
        run.summary["telemetry/imported_after_training"] = True
        run.summary["telemetry/default_plot_axis"] = "training_seconds"
        run.summary["telemetry/github_source_result"] = "https://github.com/yaroslavvb/gradient-dissent/blob/main/experiments/ciresan_stochastic_depth/results/telemetry/" + payload["name"] + ".json"
        for key, value in payload["scientific_summary"].items():
            run.summary[key] = value
        for index in range(prefix or 0, len(payload["rows"])):
            run.log(payload["rows"][index], step=index, commit=True)


def worker(source_id):
    """One bounded attempt; errors return safe classes, never raw SDK messages."""
    if not configured_auth_present():
        return {"status": "blocked_login", "verified_wandb_url": None, "retryable": False}
    import wandb
    payload = payload_for(source_id)
    api = wandb.Api(overrides={"entity": ENTITY, "project": DEFAULT_PROJECT,
                               "base_url": "https://api.wandb.ai"}, timeout=20)
    # SDK authentication is verified by this actual authenticated server read;
    # no viewer/account fields are saved, printed, or uploaded.
    if api.viewer is None:
        return {"status": "blocked_login", "verified_wandb_url": None, "retryable": False}
    remote = inspect_remote(api, payload)
    if remote.get("complete"):
        return {"status": "verified", "verified_wandb_url": remote["verified_wandb_url"], "verification": remote["verification"]}
    if remote.get("exists") and remote.get("state") in {"running", "pending"}:
        # Another SDK process or an uncertain earlier write may still own it.
        # Do not steal or duplicate an active run. A later invocation reconciles.
        return {"status": "awaiting_remote_completion", "verified_wandb_url": None,
                "verification": remote["verification"], "retryable": True}
    prefix = remote["prefix_rows"] if remote.get("exists") else None
    write_error = None
    try:
        write_remote(wandb, payload, prefix)
    except Exception as exc:
        write_error = type(exc).__name__
    # Read after success OR an uncertain failure before reporting an outcome.
    for attempt in range(3):
        if attempt:
            time.sleep(2 * attempt)
        remote = inspect_remote(api, payload)
        if remote.get("complete"):
            return {"status": "verified", "verified_wandb_url": remote["verified_wandb_url"], "verification": remote["verification"]}
    return {"status": "pending_verification", "verified_wandb_url": None,
            "verification": remote.get("verification"), "error_class": write_error,
            "retryable": True}


def bounded_worker(source_id):
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", source_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=WORKER_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        return {"status": "uncertain_worker_timeout", "verified_wandb_url": None, "retryable": True}
    # Ignore SDK console text. Only this explicit machine-readable marker is
    # accepted, and raw stderr/errors are never copied into public artifacts.
    markers = [line[len("UPLOAD_RESULT "):] for line in stdout.splitlines() if line.startswith("UPLOAD_RESULT ")]
    if process.returncode or len(markers) != 1:
        return {"status": "worker_failed", "verified_wandb_url": None, "retryable": True}
    return json.loads(markers[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--worker", choices=SOURCES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            result = worker(args.worker)
        except IntegrityError as exc:
            result = {"status": "integrity_conflict", "error_class": type(exc).__name__,
                      "verified_wandb_url": None, "retryable": False}
        except Exception as exc:
            result = {"status": "api_or_sdk_error", "error_class": type(exc).__name__,
                      "verified_wandb_url": None, "retryable": True}
        print("UPLOAD_RESULT " + json.dumps(result, allow_nan=False))
        return
    previous = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else None
    registry = make_registry(previous)
    atomic_json(REGISTRY, registry)
    if not args.upload:
        print(json.dumps({"status": registry["status"], "registry": REGISTRY.name, "runs": 18,
                          "verified_run_count": registry["verified_run_count"], "network_used": False}))
        return
    if not configured_auth_present():
        print(json.dumps({"status": "blocked_login", "registry": REGISTRY.name, "network_used": False}))
        return
    # O_EXCL prevents concurrent drivers from racing the same new run IDs.
    lock = REGISTRY.with_suffix(".lock")
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        registry["status"] = "uploading"
        for record in registry["runs"]:
            for attempt in range(MAX_ATTEMPTS):
                outcome = bounded_worker(record["scientific_run_id"])
                record.update({k: v for k, v in outcome.items() if k != "retryable"})
                if record['status'] == 'verified':
                    record.pop('error_class', None)
                record["attempts"] = record.get("attempts", 0) + 1
                record["last_attempt_utc"] = now()
                registry["updated_utc"] = now()
                registry["verified_run_count"] = sum(r["status"] == "verified" for r in registry["runs"])
                atomic_json(REGISTRY, registry)
                print(json.dumps({"run": record["scientific_run_id"], "status": record["status"],
                                  "verified_count": registry["verified_run_count"]}), flush=True)
                if record["status"] == "verified" or not outcome.get("retryable"):
                    break
                if attempt + 1 < MAX_ATTEMPTS:
                    time.sleep(2 * (attempt + 1))
        registry["complete"] = registry["verified_run_count"] == 18
        registry["status"] = "verified" if registry["complete"] else "incomplete"
        registry["updated_utc"] = now()
        atomic_json(REGISTRY, registry)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
