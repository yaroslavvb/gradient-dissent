"""CPU/no-network tests for identity, partial recovery and verified URL gating."""
import copy
from types import SimpleNamespace
import unittest

from . import upload_wandb as upload
from .wandb_export import prepare_payload


def payload():
    result = prepare_payload({"run_id": "scientific-test", "spec": {"seed": 101}, "telemetry_history": [
        {"epoch": 0, "training_seconds": 0., "metrics": {"validation/accuracy_pct": 10.}},
        {"epoch": 1, "training_seconds": 2., "metrics": {"validation/accuracy_pct": 90.}}]})
    result["source_sha256"] = "a" * 64
    result["scientific_summary"] = {"selected/epoch": 1, "selected/test_accuracy_pct": 89., "time/training_seconds": 2.}
    return result


def remote(p, count=2, state="finished"):
    rows = [{**r, "_step": i, "_runtime": .001*i, "_timestamp": 100.+i} for i, r in enumerate(p["rows"][:count])]
    summary = {"telemetry/source_sha256": p["source_sha256"], "telemetry/content_sha256": p["content_sha256"],
               "telemetry/history_rows": len(p["rows"]), **p["scientific_summary"]}
    return SimpleNamespace(path=[upload.ENTITY, upload.DEFAULT_PROJECT, p["run_id"]], name=p["name"],
        group=upload.DEFAULT_GROUP, config=copy.deepcopy(p["config"]), summary=summary, state=state,
        url=f"https://wandb.ai/{upload.ENTITY}/{upload.DEFAULT_PROJECT}/runs/{p['run_id']}",
        scan_history=lambda **kwargs: rows)


class FakeApi:
    def __init__(self, run): self.r = run
    def flush(self): pass
    def run(self, path): return self.r


class UploadTests(unittest.TestCase):
    def test_prefix_accepts_only_matching_measured_cells(self):
        p = payload()
        first = [{**p["rows"][0], "_step": 0., "_runtime": 1000}]
        count, cells = upload.verified_prefix(p["rows"], first)
        self.assertEqual(count, 1)
        self.assertEqual(cells, len(p["rows"][0]))
        for field, value in (("training_seconds", 1000), ("validation/accuracy_pct", 11)):
            changed = copy.deepcopy(first); changed[0][field] = value
            with self.assertRaises(upload.IntegrityError): upload.verified_prefix(p["rows"], changed)

    def test_missing_duplicate_extra_and_noncontiguous_rows_rejected(self):
        p = payload(); first = {**p["rows"][0], "_step": 0}
        cases = [[first, first], [{**first, "_step": 1}], [{**first, "unrelated_metric": 3.}],
                 [{k:v for k,v in first.items() if k != "epoch"}], [{**first, "_step": .5}]]
        for actual in cases:
            with self.subTest(actual=actual):
                with self.assertRaises(upload.IntegrityError): upload.verified_prefix(p["rows"], actual)

    def test_verified_remote_url_requires_all_history_hashes_and_finished(self):
        p = payload()
        for count, state in ((1, "finished"), (2, "running")):
            r = upload.inspect_remote(FakeApi(remote(p, count, state)), p)
            self.assertFalse(r["complete"]); self.assertIsNone(r["verified_wandb_url"])
        good = upload.inspect_remote(FakeApi(remote(p)), p)
        self.assertTrue(good["complete"])
        self.assertEqual(good["verified_wandb_url"], remote(p).url)
        self.assertTrue(good["verification"]["all_frozen_endpoint_and_timing_summaries_match"])

    def test_remote_provenance_or_identity_conflict_never_adopted(self):
        p = payload()
        for kind in ("path", "name", "group", "config", "hash", "url"):
            r = remote(p)
            if kind == "path": r.path[1] = "train_ciresan"
            elif kind == "name": r.name = "unrelated"
            elif kind == "group": r.group = "unrelated"
            elif kind == "config": r.config["seed"] = 999
            elif kind == "hash": r.summary["telemetry/source_sha256"] = "b"*64
            elif kind == "url": r.url = "https://example.com/unverified"
            with self.subTest(kind=kind):
                with self.assertRaises(upload.IntegrityError): upload.inspect_remote(FakeApi(r), p)

    def test_missing_or_wrong_endpoint_summary_not_declared_verified(self):
        p = payload()
        r = remote(p); del r.summary["selected/test_accuracy_pct"]
        self.assertFalse(upload.inspect_remote(FakeApi(r), p)["complete"])
        r = remote(p); r.summary["time/training_seconds"] = 2000.
        self.assertFalse(upload.inspect_remote(FakeApi(r), p)["complete"])

    def test_frozen_endpoint_summaries_preserve_test_units_and_clock(self):
        s = upload.scientific_summary({"best_epoch": 20, "best_validation_loss": .1,
            "selected_test": {"accuracy": .9863, "loss": .11}, "final_test": {"accuracy": .98, "loss": .12},
            "training_seconds": 6., "total_run_seconds": 15., "telemetry": {"seconds": 1.},
            "threshold": {"epoch": 21, "test": {"accuracy": .9863, "errors": 137},
                          "training_seconds": 5., "run_wall_seconds": 14., "run_wall_seconds_with_telemetry": 14.5}})
        self.assertEqual(s["selected/epoch"], 20)
        self.assertEqual(s["selected/test_accuracy_pct"], 98.63)
        self.assertEqual(s["target/run_elapsed_seconds"], 14.5)
        self.assertEqual(s["time/run_elapsed_seconds"], 15.)
        self.assertEqual(s["time/training_seconds"], 6.)

    def test_exact_eighteen_local_sources_have_no_fabricated_urls(self):
        registry = upload.make_registry()
        self.assertEqual(len(registry["runs"]), 18)
        self.assertEqual(len({r["wandb_run_id"] for r in registry["runs"]}), 18)
        self.assertTrue(all(r["status"] == "blocked_login" and r["verified_wandb_url"] is None for r in registry["runs"]))
        self.assertEqual(sum(r["history_rows_expected"] for r in registry["runs"]), 401)
        for r in registry["runs"]:
            self.assertIn("final/test_accuracy_pct", r["scientific_summary"])
            self.assertIn("time/training_seconds", r["scientific_summary"])

    def test_registry_refuses_source_mutation(self):
        registry = upload.make_registry()
        registry["runs"][0]["source_sha256"] = "0"*64
        with self.assertRaises(upload.IntegrityError): upload.make_registry(registry)


if __name__ == "__main__": unittest.main()
