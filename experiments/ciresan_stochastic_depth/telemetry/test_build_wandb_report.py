"""Offline tests of the scientific scope and publication gate; no W&B calls."""
import copy
import unittest
from unittest.mock import patch

import build_wandb_report as b


def registry():
    runs = []
    for i, source_id in enumerate(sorted(b.expected_source_ids())):
        run_id = f"fixture-{i}"
        runs.append({"scientific_run_id": source_id, "wandb_run_id": run_id,
                     "source_sha256": "a" * 64, "content_sha256": "b" * 64,
                     "status": "verified", "history_rows_expected": 22,
                     "verified_wandb_url": f"https://wandb.ai/{b.ENTITY}/{b.PROJECT}/runs/{run_id}",
                     "verification": {"finished": True, "all_provenance_hashes_match": True,
                                      "all_frozen_endpoint_and_timing_summaries_match": True,
                                      "history_rows_retrieved": 22}})
    return {"entity": b.ENTITY, "project": b.PROJECT, "group": b.GROUP,
            "complete": True, "status": "verified", "verified_run_count": 18, "runs": runs}


class ReportTests(unittest.TestCase):
    def test_exact_user_destination(self):
        self.assertEqual((b.ENTITY, b.PROJECT), ("yaroslavvb", "gradient-dissent"))
        r = registry()
        r["project"] = "gradient-dissent-ciresan"
        with self.assertRaises(ValueError):
            b.check_registry(r, require_complete=True)

    def test_eighteen_verified_runs_required(self):
        b.check_registry(registry(), require_complete=True)
        for mutate in (
            lambda r: r.update(complete=False),
            lambda r: r.update(verified_run_count=17),
            lambda r: r["runs"].pop(),
            lambda r: r["runs"][0].update(status="uploading"),
            lambda r: r["runs"][0].update(verified_wandb_url=None),
            lambda r: r["runs"][0]["verification"].update(history_rows_retrieved=21),
            lambda r: r["runs"][0]["verification"].update(all_provenance_hashes_match=False),
            lambda r: r["runs"][0]["verification"].update(all_frozen_endpoint_and_timing_summaries_match=False),
        ):
            r = registry()
            mutate(r)
            with self.assertRaises(ValueError):
                b.check_registry(r, require_complete=True)

    def test_incomplete_publication_stops_before_sdk(self):
        r = registry()
        r["complete"] = False
        with patch.dict("sys.modules", {"wandb": None}):
            with self.assertRaises(ValueError):
                b.publish(r)

    def test_eight_panels_use_saved_clocks(self):
        plan = b.make_plan(registry())
        self.assertEqual(len(plan["panels"]), 8)
        self.assertEqual([p["axis"] for p in plan["panels"]], ["training_seconds"] * 7 + ["run_elapsed_seconds"])
        self.assertEqual([p["metric"] for p in plan["panels"][:4]],
                         ["validation/loss", "validation/accuracy_pct", "train/loss", "train/accuracy_pct"])
        self.assertEqual([p["metric"] for p in plan["panels"][-2:]], ["test/accuracy_pct"] * 2)
        self.assertEqual(len(b.selected_runs(registry(), "controlled")), 15)
        self.assertEqual(len(b.selected_runs(registry(), "baseline")), 3)

    def test_exact_scopes_are_backend_run_and_config_filters(self):
        from wandb_workspaces.reports.v2.interface import expr
        for cohort in ("controlled", "diagnostics", "baseline"):
            tree = expr.expr_to_filters(b.filters_for(registry(), cohort)).model_dump()
            leaves = tree["filters"]
            self.assertEqual(leaves[0]["key"], {"section": "run", "name": "group"})
            self.assertEqual(leaves[0]["value"], b.GROUP)
            self.assertEqual(leaves[1]["key"], {"section": "run", "name": "state"})
            self.assertEqual(leaves[1]["value"], "finished")
            self.assertEqual(leaves[2]["key"], {"section": "config", "name": "scientific_run_id"})
            self.assertEqual(leaves[2]["op"], "IN")
            self.assertEqual(set(leaves[2]["value"]), {r["scientific_run_id"] for r in b.selected_runs(registry(), cohort)})

    def test_sdk_roundtrip_and_mutation_detection(self):
        import wandb_workspaces.reports.v2 as wr
        report = b.make_report(registry())
        with patch("wandb_workspaces.reports.v2.interface.execute_graphql", return_value={"project": {"internalId": "offline-fixture"}}), patch("wandb_workspaces.reports.v2.interface._get_api", return_value=None):
            model = report._to_model()
        loaded = wr.Report._from_model(model)
        self.assertEqual(b.report_signature(report), b.report_signature(loaded))
        grids = [block for block in loaded.blocks if isinstance(block, wr.PanelGrid)]
        self.assertEqual([len(g.panels) for g in grids], [4, 2, 2])
        for g in grids:
            for p in g.panels:
                self.assertFalse(p.aggregate)
                self.assertEqual(p.smoothing_type, "none")
        changed = copy.deepcopy(loaded)
        next(g for g in changed.blocks if isinstance(g, wr.PanelGrid)).panels[0].x = "_runtime"
        self.assertNotEqual(b.report_signature(report), b.report_signature(changed))
        grids[0].runsets[0]._stashed_filters_v2["filters"][0]["disabled"] = True
        self.assertNotEqual(b.report_signature(report), b.report_signature(loaded))

    def test_plan_digest_deterministic(self):
        r = registry()
        self.assertEqual(b.make_plan(r)["plan_sha256"], b.make_plan(copy.deepcopy(r))["plan_sha256"])


if __name__ == "__main__":
    unittest.main()
