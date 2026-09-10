"""Curated post-training W&B report; publication requires 18 verified run links.

Default: write a local scientific panel plan, without network access.
--publish: inspect/reuse the uniquely identified report, save it, and read it
back before recording a real report URL. No GPU jobs or source/code capture.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results/telemetry"
REGISTRY = RESULTS / "wandb-upload-manifest.json"
PLAN = RESULTS / "wandb-report-plan.json"
RECEIPT = RESULTS / "wandb-report-manifest.json"
ENTITY = "yaroslavvb"
PROJECT = "gradient-dissent"
GROUP = "ciresan-telemetry-20260909"
TITLE = "Ciresan MNIST: measured training curves · 2026-09-09"
MARKER = "gradient-dissent-ciresan-curated-v1"
DESCRIPTION = f"{MARKER}; verified 18-run group {GROUP}. Measured clocks, separate held-out validation and official-test target replay."
COLORS = {"plain": "#72818a", "residual": "#1d647f", "sd_constant": "#c36b13",
          "sd_annealed": "#864d9b", "residual_unit_dropout": "#18845c", "unit_dropout": "#1d647f"}
PANEL_SPECS = (
    {"cohort": "controlled", "title": "Validation cross-entropy", "metric": "validation/loss", "axis": "training_seconds", "ylabel": "Mean cross-entropy (nats)", "log_y": True},
    {"cohort": "controlled", "title": "Validation accuracy", "metric": "validation/accuracy_pct", "axis": "training_seconds", "ylabel": "Accuracy (%)", "range_y": (0, 100)},
    {"cohort": "controlled", "title": "Full training-set cross-entropy", "metric": "train/loss", "axis": "training_seconds", "ylabel": "Mean cross-entropy (nats)", "log_y": True},
    {"cohort": "controlled", "title": "Full training-set accuracy", "metric": "train/accuracy_pct", "axis": "training_seconds", "ylabel": "Accuracy (%)", "range_y": (0, 100)},
    {"cohort": "diagnostics", "title": "First body branch: dense-probe gradient norm", "metric": "layer-1/probe/grad_l2", "axis": "training_seconds", "ylabel": "L2 norm of mean weight gradient", "log_y": True},
    {"cohort": "diagnostics", "title": "First body branch: gradient diversity", "metric": "layer-1/probe/weight_grad_diversity", "axis": "training_seconds", "ylabel": "E‖gᵢ‖² / ‖E gᵢ‖²", "log_y": True},
    {"cohort": "baseline", "title": "Official-test target replay: 98.63% target", "metric": "test/accuracy_pct", "axis": "training_seconds", "ylabel": "Official-test accuracy (%)", "range_y": (0, 100)},
    {"cohort": "baseline", "title": "Same official-test replay including setup and telemetry", "metric": "test/accuracy_pct", "axis": "run_elapsed_seconds", "ylabel": "Official-test accuracy (%)", "range_y": (0, 100)},
)


def atomic_json(path, data):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def expected_source_ids():
    recipes = ("plain", "residual", "sd_constant", "sd_annealed", "residual_unit_dropout")
    return {f"telemetry-main-{recipe}-s{seed}-v1" for recipe in recipes for seed in (101, 102, 103)} | {
        f"telemetry-baseline-s{seed}-v1" for seed in (101, 102, 103)}


def check_registry(registry, *, require_complete=False):
    if (registry.get("entity"), registry.get("project"), registry.get("group")) != (ENTITY, PROJECT, GROUP):
        raise ValueError("Unexpected registry project or group")
    runs = registry.get("runs", [])
    if len(runs) != 18 or {r.get("scientific_run_id") for r in runs} != expected_source_ids():
        raise ValueError("Report requires the exact 18 source runs")
    if len({r.get("wandb_run_id") for r in runs}) != 18:
        raise ValueError("Duplicate or absent W&B run IDs")
    if require_complete:
        if registry.get("complete") is not True or registry.get("verified_run_count") != 18 or registry.get("status") != "verified":
            raise ValueError("Publication waits for all 18 verified uploads")
        for r in runs:
            parsed = urlparse(r.get("verified_wandb_url") or "")
            check = r.get("verification") or {}
            if (r.get("status") != "verified" or parsed.scheme != "https" or parsed.netloc != "wandb.ai"
                    or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}/runs/{r['wandb_run_id']}"
                    or check.get("finished") is not True or check.get("all_provenance_hashes_match") is not True
                    or check.get("all_frozen_endpoint_and_timing_summaries_match") is not True
                    or check.get("history_rows_retrieved") != r.get("history_rows_expected")):
                raise ValueError("Registry contains an unverified run link")


def selected_runs(registry, cohort):
    wanted_baseline = cohort == "baseline"
    return [r for r in registry["runs"] if r["scientific_run_id"].startswith("telemetry-baseline-") == wanted_baseline]


def filters_for(registry, cohort):
    source_ids = sorted(r["scientific_run_id"] for r in selected_runs(registry, cohort))
    return f"Group == {GROUP!r} and State == 'finished' and Config('scientific_run_id') in {source_ids!r}"


def make_plan(registry):
    check_registry(registry)
    content = {"title": TITLE, "identifier": MARKER, "description": DESCRIPTION,
               "entity": ENTITY, "project": PROJECT, "group": GROUP, "panels": list(PANEL_SPECS),
               "filters": {cohort: filters_for(registry, cohort) for cohort in ("controlled", "diagnostics", "baseline")},
               "source_runs": [{k: r[k] for k in ("scientific_run_id", "wandb_run_id", "source_sha256", "content_sha256")} for r in registry["runs"]],
               "aggregation": "Individual seed traces; no smoothing, grouped mean, or confidence band.",
               "target_semantics": "98.63% official-test target is a labeled stopping goal, not a training-accuracy goal; no synthetic constant metric is added.",
               "clock_semantics": "training_seconds is measured optimization time; run_elapsed_seconds includes setup, evaluation and telemetry; W&B importer runtime is never used."}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"schema_version": 1, "status": "ready_to_publish" if registry.get("complete") else "waiting_for_verified_uploads",
            "plan_sha256": digest, **content}


def recipe_for(source_id):
    if source_id.startswith("telemetry-baseline-"):
        return "unit_dropout"
    return source_id.removeprefix("telemetry-main-").rsplit("-s", 1)[0]


def make_report(registry, *, wr=None):
    if wr is None:
        import wandb_workspaces.reports.v2 as wr
    check_registry(registry)
    def grid(cohort):
        records = selected_runs(registry, cohort)
        runset = wr.Runset(entity=ENTITY, project=PROJECT,
            name="Optimized baseline · 3 seeds" if cohort == "baseline" else "Controlled comparison · 5 methods × 3 seeds",
            filters=filters_for(registry, cohort),
            custom_run_colors={r["wandb_run_id"]: COLORS[recipe_for(r["scientific_run_id"])] for r in records},
            pinned_columns=["run:displayName"],
            visible_columns=["config:recipe.value", "config:seed.value", "config:lr.value",
                "summary:selected/epoch", "summary:selected/test_accuracy_pct", "summary:final/test_accuracy_pct",
                "summary:time/training_seconds", "summary:time/run_elapsed_seconds", "summary:time/telemetry_seconds"],
            lock_columns=True)
        panels = []
        for i, spec in enumerate(p for p in PANEL_SPECS if p["cohort"] == cohort):
            panels.append(wr.LinePlot(title=spec["title"], x=spec["axis"], y=[spec["metric"]],
                title_x="Measured training seconds" if spec["axis"] == "training_seconds" else "Measured run elapsed seconds",
                title_y=spec["ylabel"], log_y=spec.get("log_y", False), range_y=spec.get("range_y", (None, None)),
                smoothing_type="none", smoothing_factor=0., aggregate=False, ignore_outliers=False,
                groupby=None, groupby_rangefunc="none", max_runs_to_show=len(records),
                layout=wr.Layout(x=12*(i % 2), y=9*(i // 2), w=12, h=9)))
        return wr.PanelGrid(runsets=[runset], panels=panels)
    blocks = [
        wr.MarkdownBlock(text="These are post-training imports of 18 completed, read-back-verified A100 runs. All charts use **measured clocks saved during training**, not W&B upload runtime. Each trace is one seed; colors distinguish methods. There is no smoothing, seed aggregation, or confidence band. [Interactive scientific report](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/telemetry/) · [Conclusions](https://yaroslavvb.github.io/gradient-dissent/ciresan-stochastic-depth/conclusions/)."),
        wr.H2("Controlled comparison: train and validation learning curves"),
        wr.P("Five methods × three paired seeds; fixed 50,000 fitting / 10,000 validation split, batch 64, 100 epochs. Previously selected learning rates and the original validation-loss checkpoint rule are unchanged. Full-training-set evaluations are less frequent than the original five-epoch validation checks. Official-test diagnostics did not select the recipes."),
        grid("controlled"),
        wr.H2("Dense fixed-probe gradients"),
        wr.P("All branches are kept and unit dropout is disabled on the same 128 fitting examples within this cohort. The norm is that of the mean weight gradient, not the mean of per-example norms. Diversity is E||g_i||² / ||E g_i||²; large values can reflect cancellation or an almost-zero-gradient probe dominated by a few examples. These ordinary FP32-autograd moments are neither Hessian curvature nor a proof of deletion robustness. Zero/undefined values are absent on log axes."),
        grid("diagnostics"),
        wr.H2("Separate optimized baseline: official-test target replay"),
        wr.P("Three seeds; 60,000 fitting examples, normalized inputs, unit dropout 0.2, BF16 CUDA-graph training, batch 256, learning rate 0.12 reduced tenfold at epoch 21. Stop at the first official-test evaluation with at most 137 errors (at least 98.63% accuracy). The target is stated in the title, without adding a synthetic metric. Prior test-based recipe tuning and this stopping rule make these time-to-target replays, not independent held-out generalization estimates."),
        grid("baseline"),
        wr.H2("Metric and timing semantics"),
        wr.MarkdownBlock(text="- `train/loss` and `validation/loss` are mean cross-entropy in nats; `*_accuracy_pct` is percent. `validation/*` refers to the held-out training split.\n- `test/*` and legacy `val_accuracy` / `val_loss` refer to the official MNIST test set; the legacy names never mean the new validation split.\n- `training_seconds` excludes evaluation, diagnostics and checkpoint I/O. `run_elapsed_seconds` includes setup and those costs; remote dispatch/provisioning is a further separate clock.\n- W&B `_runtime` is importer runtime and `_step` is importer-row index. Neither is optimizer progress.\n- Frozen endpoint summaries are copied from source results: `selected/*` uses the original validation-selected checkpoint; `final/*` is the original final checkpoint; `target/*` is the baseline's first recorded threshold crossing. Source hashes and GitHub result links are recorded on every run.\n- Sparse diagnostics are connected across measured checkpoints. The report uses only the exact verified source IDs and group, excluding future or unrelated runs."),
    ]
    return wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, description=DESCRIPTION, blocks=blocks, width="fluid")


def report_signature(report):
    """Scientific content used for read-back checks; ignore generated block IDs."""
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces.reports.v2.interface import expr
    def canonical_filters(runset):
        # The SDK renders Group as Metric("Group") on read-back. Compare the
        # backend filter tree, including disabled flags, instead of spelling.
        stashed = getattr(runset, "_stashed_filters_v2", None)
        if stashed is not None and runset.filters == getattr(runset, "_stashed_filter_string", None):
            return stashed
        return expr.filters_tree_to_v2(expr.expr_to_filters(runset.filters))
    def metric_name(value):
        return value if isinstance(value, str) else getattr(value, "name", None)
    grids = []
    for block in report.blocks:
        if isinstance(block, wr.PanelGrid):
            grids.append({"runsets": [{"entity": r.entity, "project": r.project, "filters": canonical_filters(r)} for r in block.runsets],
                          "panels": [{"title": p.title, "x": metric_name(p.x), "y": [metric_name(y) for y in p.y],
                                      "smoothing_type": p.smoothing_type, "aggregate": p.aggregate,
                                      "log_y": p.log_y, "range_y": list(p.range_y), "max_runs_to_show": p.max_runs_to_show} for p in block.panels]})
    return {"title": report.title, "description": report.description, "entity": report.entity, "project": report.project, "grids": grids}


def read_visibility(api):
    """Read one schema-verified setting; null is unavailable, never private."""
    from wandb_workspaces._graphql import execute_graphql
    result = {"setting_changed": False, "anonymous_access_verified": False,
              "note": "Verified through the authenticated W&B account; anonymous access has not been established."}
    try:
        data = execute_graphql(api,
            "query ProjectVisibility($entity:String!,$name:String!) { project(entityName:$entity,name:$name) { public } }",
            {"entity": ENTITY, "name": PROJECT})
        value = (data.get("project") or {}).get("public")
        result["server_project_public_setting"] = value if isinstance(value, bool) else None
        result["setting_read_status"] = "available" if isinstance(value, bool) else "unavailable_null"
    except Exception as exc:
        result["setting_read_status"] = "unavailable"
        result["error_class"] = type(exc).__name__
    return result


def publish(registry):
    check_registry(registry, require_complete=True)
    import wandb
    import wandb_workspaces.reports.v2 as wr
    api = wandb.Api(timeout=30)
    desired = make_report(registry, wr=wr)
    existing = []
    # Match both exact display title and our stable ownership marker. Never
    # touch the user's other reports, even in the same project.
    for i, report in enumerate(api.reports(f"{ENTITY}/{PROJECT}", per_page=50)):
        if i >= 200:
            raise ValueError("Report inventory exceeds bounded duplicate-check scope")
        if report.display_name == TITLE:
            if MARKER not in (report.description or ""):
                raise ValueError("A same-title report lacks this report's ownership marker")
            existing.append(report)
    if len(existing) > 1:
        raise ValueError("Multiple matching reports exist; refusing another duplicate")
    if existing:
        loaded = wr.Report.from_url(existing[0].url)
        if (loaded.entity, loaded.project, loaded.title) != (ENTITY, PROJECT, TITLE) or MARKER not in loaded.description:
            raise ValueError("Existing report identity mismatch")
        loaded.blocks = desired.blocks
        loaded.description = desired.description
        loaded.width = desired.width
        desired = loaded
    desired.save(draft=False, clone=False)
    actual = wr.Report.from_url(desired.url)
    expected_signature = report_signature(desired)
    if report_signature(actual) != expected_signature:
        raise ValueError("Saved report scientific panel/filter read-back mismatch")
    parsed = urlparse(actual.url)
    if parsed.scheme != "https" or parsed.netloc != "wandb.ai" or not parsed.path.startswith(f"/{ENTITY}/{PROJECT}/reports/"):
        raise ValueError("Unexpected saved report URL")
    receipt = {"schema_version": 1, "status": "verified", "verified_report_url": actual.url,
               "title": TITLE, "identifier": MARKER, "entity": ENTITY, "project": PROJECT, "group": GROUP,
               "verified_run_count": 18, "panel_count": len(PANEL_SPECS), "reused_existing_report": bool(existing),
               "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "plan_sha256": make_plan(registry)["plan_sha256"],
               "upload_registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
               "verification": "Saved report reloaded through the official API; project/title, eight panel axes/metrics, exact run filters (canonical trees), smoothing and aggregation settings match.",
               "verified_panel_configuration": expected_signature,
               "visibility": read_visibility(api)}
    atomic_json(RECEIPT, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        registry = json.loads(REGISTRY.read_text())
        plan = make_plan(registry)
        atomic_json(PLAN, plan)
        if args.publish:
            # No API call occurs until complete verification passes here.
            check_registry(registry, require_complete=True)
            receipt = publish(registry)
            print(json.dumps({"status": receipt["status"], "report_url": receipt["verified_report_url"], "panel_count": receipt["panel_count"]}))
        else:
            print(json.dumps({"status": plan["status"], "plan_file": PLAN.name, "panel_count": len(PANEL_SPECS), "network_used": False}))
    except Exception as exc:
        # Never print raw API exceptions, account fields, settings or secrets.
        print(json.dumps({"status": "blocked_or_failed", "error_class": type(exc).__name__, "verified_report_url": None}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
