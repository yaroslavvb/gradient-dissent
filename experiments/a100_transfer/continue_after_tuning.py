"""Supervised, fail-closed continuation of the already frozen experiment design.

Orchestration-only hardening: finite wait, failed/dead launcher detection,
exclusive final log creation, and refusal to restart an existing final stage.
This does not change the frozen core model/training files, learning-rate
selection, masks, seed set, or experiment budget. It performs no retries.

An instance already running before this edit retains its previously loaded
code. This file is for a future explicitly supervised continuation invocation.
"""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
POLL_SECONDS = 10
QUEUE_MARGIN_SECONDS = 1800
MISSING_LAUNCHER_GRACE_SECONDS = 60


def load_json(path):
    return json.loads(Path(path).read_text())


def assert_final_not_started(out):
    """Check before waiting/selection and again immediately before dispatch."""
    log = out / "evaluation.log"
    if log.exists():
        raise RuntimeError(
            f"Final-stage log already exists: {log}. It has been preserved. "
            "Recover or inspect the existing stage; do not relaunch billable jobs. "
            "If all final runs are complete, invoke analyze.py directly.")
    ledger_path = out / "budget-ledger.json"
    if ledger_path.exists():
        final = [r["id"] for r in load_json(ledger_path).get("reservations", [])
                 if r.get("kind") == "evaluate" or r.get("id", "").startswith("eval-")]
        if final:
            raise RuntimeError(
                "Final-stage reservations already exist; no logs were overwritten and no jobs submitted. "
                "Recover existing calls/results rather than creating duplicate invocations: "+", ".join(final))


def wait_deadline(jobs, out, protocol, now):
    """Conservative wave runtime plus startup and30-minute queue margin.

    Use reservation creation time when available, so restarting a waiter does
    not silently reset the stage's deadline. This deadline is a supervision
    guard; expiry never cancels or relaunches remote work automatically.
    """
    workers = int(protocol.get("maximum_parallel_a100s", 9))
    if workers < 1 or not jobs:
        raise ValueError("Invalid locked concurrency or empty tuning grid")
    waves = math.ceil(len(jobs)/workers)
    longest = max(float(j["timeout_seconds"]) for j in jobs)
    allowance = waves*(longest+130)+QUEUE_MARGIN_SECONDS
    ledger_path = out / "budget-ledger.json"
    started = now
    if ledger_path.exists():
        ids = {j["run_id"] for j in jobs}
        entries = [r for r in load_json(ledger_path).get("reservations", []) if r.get("id") in ids]
        if entries:
            started = min(float(r["created_unix"]) for r in entries)
    return started+allowance, allowance


def failed_stage_reason(text, jobs):
    """Recognize terminal launcher failures; ordinary billing warnings aren't fatal."""
    ids = {j["run_id"] for j in jobs}
    for run_id in re.findall(r"\bFAILED\s+(\S+)", text):
        if run_id in ids:
            return "Launcher reported failed tuning run "+run_id
    stage = re.search(r"STAGE COMPLETE\s+tune\s+jobs\s+\d+\s+errors\s+(\d+)", text)
    if stage and int(stage.group(1)):
        return "Launcher reported "+stage.group(1)+" tuning errors"
    for marker in ("Traceback (most recent call last):", "KeyboardInterrupt", "Stopping app - uncaught exception"):
        if marker in text:
            return "Tuning launcher log contains terminal failure marker: "+marker
    return None


def parse_launcher_presence(process_text, manifest_path):
    """Match an actual Modal CLI process, excluding shell wrappers and waiters."""
    for line in process_text.splitlines():
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        if not args:
            continue
        executable = Path(args[0]).name.lower()
        if executable in {"bash", "sh", "zsh", "dash"}:
            continue
        modal_index = next((i for i, token in enumerate(args) if Path(token).name == "modal"), None)
        if modal_index is None or args[modal_index+1:modal_index+2] != ["run"]:
            continue
        try:
            stage = args[args.index("--stage")+1]
            manifest = args[args.index("--manifest")+1]
        except (ValueError, IndexError):
            continue
        if stage == "tune" and Path(manifest).name == manifest_path.name and any(Path(t).name == "modal_app.py" for t in args):
            return True
    return False


def launcher_alive(manifest_path):
    try:
        result = subprocess.run(["ps", "-axo", "args="], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None  # Finite stage deadline and failure-log checks still apply.
    if result.returncode:
        return None
    return parse_launcher_presence(result.stdout, manifest_path)


def wait_for_tuning(jobs, out, manifest_path, protocol):
    deadline, allowance = wait_deadline(jobs, out, protocol, time.time())
    print(f"WAITING for all {len(jobs)} locked tuning runs; deadline Unix {deadline:.0f}; "
          f"stage allowance {allowance:.0f}s includes queue margin", flush=True)
    missing_launcher_since = None
    while True:
        completed, missing = [], []
        for spec in jobs:
            path = out / (spec["run_id"]+".json")
            if not path.exists():
                missing.append(spec["run_id"])
                continue
            try:
                result = load_json(path)
            except json.JSONDecodeError:
                missing.append(spec["run_id"])
                continue  # Launcher may be completing its write.
            if "error" in result:
                raise RuntimeError("Tuning failed; no final jobs submitted: "+spec["run_id"]+
                                   ". Inspect existing remote artifacts before considering any explicitly reserved retry.")
            if result.get("spec") != spec or result["curve"][-1]["step"] != spec["steps"]:
                raise RuntimeError("Tuning spec or terminal step differs from frozen grid: "+spec["run_id"])
            if result.get("rows") != [] or result.get("test_panel") is not None:
                raise RuntimeError("Tuning result unexpectedly includes test scoring: "+spec["run_id"])
            ce = result["validation"]["ce"]
            if not isinstance(ce, (int, float)) or not math.isfinite(ce):
                raise RuntimeError("Non-finite tuning validation CE: "+spec["run_id"])
            completed.append(result)
        if not missing:
            return completed
        log = out / "tuning.log"
        if log.exists():
            reason = failed_stage_reason(log.read_text(errors="replace"), jobs)
            if reason:
                raise RuntimeError(reason+"; waiting stopped without launching final jobs. See "+str(log))
        now = time.time()
        if now > deadline:
            raise RuntimeError("Tuning wait deadline exceeded; missing "+", ".join(missing)+
                               ". No remote work was cancelled or retried. Inspect tuning.log and existing calls.")
        alive = launcher_alive(manifest_path)
        if alive is False:
            if missing_launcher_since is None:
                missing_launcher_since = now
            elif now-missing_launcher_since >= MISSING_LAUNCHER_GRACE_SECONDS:
                raise RuntimeError("Tuning Modal launcher has been absent for60 seconds while results remain missing: "+
                                   ", ".join(missing)+". No final jobs submitted; inspect tuning.log and recover existing remote results.")
        elif alive is True:
            missing_launcher_since = None
        time.sleep(POLL_SECONDS)


def select_and_construct(jobs, completed):
    """Unchanged frozen rule: three full-duration LRs, one tuning seed."""
    selected = {}
    for task in sorted({j["task"] for j in jobs}):
        for recipe in ("dense", "constant_ild", "decreasing_ild"):
            group = [r for r in completed if r["spec"]["task"] == task and r["spec"]["recipe"] == recipe]
            if len(group) != 3:
                raise RuntimeError("Expected exactly3 locked tuning rates for "+task+"/"+recipe)
            chosen = min(group, key=lambda r: (r["validation"]["ce"], r["spec"]["lr"]))
            selected[task+"/"+recipe] = {"lr": chosen["spec"]["lr"], "validation_ce": chosen["validation"]["ce"],
                "validation_grid": [{"lr": r["spec"]["lr"], "ce": r["validation"]["ce"]} for r in group]}
    final = []
    for seed in (2000, 2001, 2002):
        for recipe in ("dense", "constant_ild", "decreasing_ild"):
            for task in sorted({j["task"] for j in jobs}):
                base = next(j for j in jobs if j["task"] == task and j["recipe"] == recipe)
                final.append(dict(base, run_id=f"eval-{task}-{recipe}-s{seed}", stage="evaluate", seed=seed,
                                  lr=selected[task+"/"+recipe]["lr"]))
    return selected, final


def main():
    OUT.mkdir(exist_ok=True)
    # Prevent two new-version continuation instances from racing to dispatch.
    with (OUT / ".continuation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another continuation process holds the lock; no jobs submitted") from error
        assert_final_not_started(OUT)
        manifest_path = HERE / "tuning-manifest.json"
        jobs = load_json(manifest_path)
        protocol = load_json(HERE / "protocol.json")
        completed = wait_for_tuning(jobs, OUT, manifest_path, protocol)
        assert_final_not_started(OUT)
        selected, final = select_and_construct(jobs, completed)
        path = HERE / "evaluation-manifest.json"
        if path.exists():
            if load_json(path) != final:
                raise RuntimeError("Existing final manifest differs; refuse overwrite")
        else:
            path.write_text(json.dumps(final, indent=2)+"\n")
        (OUT / "selected-lrs.json").write_text(json.dumps(selected, indent=2)+"\n")
        print("SELECTED", json.dumps({k:v["lr"] for k,v in selected.items()}), flush=True)
        print("STARTING", len(final), "final runs; selected manifest is frozen.", flush=True)
        # Exclusive creation preserves an earlier stage log even under a race.
        with (OUT / "evaluation.log").open("x") as log:
            subprocess.run(["modal", "run", "-e", "gradient-dissent-a100", str(HERE / "modal_app.py"),
                            "--stage", "evaluate", "--manifest", str(path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(HERE / "analyze.py")], check=True)
        print("ALL EXPERIMENTS AND INDEPENDENT DATA AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
