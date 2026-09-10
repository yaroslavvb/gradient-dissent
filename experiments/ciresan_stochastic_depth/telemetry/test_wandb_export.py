"""CPU-only exporter checks; fake SDK, no login or network calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.ciresan_stochastic_depth.telemetry import wandb_export as exporter


def example():
    return {
        "run_id": "telemetry-main-residual-s101-test",
        "spec": {"recipe": "residual", "seed": 101, "batch_size": 64,
                 "lr_schedule": {"21": .1}, "epochs": 100,
                 "api_key": "SENSITIVE_SENTINEL", "source_path": "/local/private/sentinel"},
        "telemetry_history": [
            {"epoch": 1, "training_seconds": 1.25, "run_elapsed_seconds": 3.,
             "metrics": {"validation/loss": .12, "validation/accuracy_pct": 97.,
                         "layer-1/probe/gradient_fro": .03},
             "diagnostics": {"probe": {"n": 256, "undefined": None,
                                        "explanation": "private text", "vector": [1, 2]}}},
            {"epoch": 5, "training_seconds": 5.75, "run_elapsed_seconds": 9.,
             "metrics": {"validation/loss": .10, "validation/accuracy_pct": 98.,
                         "test/loss": .11, "test/accuracy_pct": 97.9}, "diagnostics": {}}
        ]}


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.definitions = []
        self.logged = []
        self.url = "https://wandb.ai/test/newproject/runs/newrun"
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finished = True

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log(self, row, **kwargs):
        self.logged.append((copy.deepcopy(row), kwargs))


class FakeSDK:
    def __init__(self):
        self.run = FakeRun()
        self.settings = None
        self.kwargs = None

    def Settings(self, **kwargs):
        self.settings = kwargs
        return kwargs

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


class ExportTests(unittest.TestCase):
    def test_measured_clocks_and_units_are_preserved(self):
        source = example()
        before = copy.deepcopy(source)
        payload = exporter.prepare_payload(source)
        self.assertEqual(source, before)
        for old, new in zip(source["telemetry_history"], payload["rows"]):
            for axis in ("epoch", "training_seconds", "run_elapsed_seconds"):
                self.assertEqual(old[axis], new[axis])
        self.assertNotIn("val_accuracy", payload["rows"][0])
        self.assertEqual(payload["rows"][1]["val_accuracy"], 97.9)
        self.assertEqual(payload["rows"][1]["val_loss"], .11)
        self.assertEqual(payload["config"]["lr_schedule"], {"21": .1})

    def test_config_and_nonscalar_diagnostics_not_copied(self):
        payload = exporter.prepare_payload(example())
        encoded = json.dumps(payload)
        for sentinel in ("SENSITIVE_SENTINEL", "/local/private", "private text", "api_key", "source_path"):
            self.assertNotIn(sentinel, encoded)
        self.assertEqual(payload["rows"][0]["diagnostics/probe/n"], 256)
        self.assertEqual(payload["omitted_null_or_nonscalar_diagnostics_count"], 3)

    def test_clocks_never_invented_and_missing_required_rejected(self):
        result = example()
        del result["telemetry_history"][0]["run_elapsed_seconds"]
        payload = exporter.prepare_payload(result)
        self.assertNotIn("run_elapsed_seconds", payload["rows"][0])
        with self.assertRaises(ValueError):
            exporter.publish(payload, axis="run_elapsed_seconds", wandb_module=FakeSDK())
        del result["telemetry_history"][0]["training_seconds"]
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)

    def test_nonmonotonic_or_invalid_axes_rejected(self):
        for field, value in (("training_seconds", 0.), ("epoch", -1),
                             ("run_elapsed_seconds", float("nan")), ("epoch", True)):
            with self.subTest(field=field, value=value):
                result = example()
                result["telemetry_history"][1][field] = value
                with self.assertRaises(ValueError):
                    exporter.prepare_payload(result)

    def test_ambiguous_legacy_val_rejected(self):
        result = example()
        result["telemetry_history"][0]["metrics"]["val_accuracy"] = 98.63
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)
        result["telemetry_schema"] = {"legacy_val_is_official_test_percent": True}
        payload = exporter.prepare_payload(result)
        self.assertEqual(payload["rows"][0]["test/accuracy_pct"], 98.63)
        self.assertEqual(payload["rows"][0]["validation/accuracy_pct"], 97.)

    def test_legacy_test_conflicts_rejected(self):
        result = example()
        result["telemetry_schema"] = {"legacy_val_is_official_test_percent": True}
        result["telemetry_history"][1]["metrics"]["val_accuracy"] = 99.
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)
        del result["telemetry_history"][1]["metrics"]["val_accuracy"]
        result["telemetry_history"][1]["metrics"]["test/ce"] = .9
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)

    def test_numeric_limits_and_credential_metric_names(self):
        for name, value in (("test/accuracy_pct", 101.), ("test/ce", -.1),
                            ("test/loss", float("inf")), ("api_key", 5),
                            ("training_seconds", 3.), ("bad name", 2),
                            ("metric", {"nested": 1}), ("_runtime", 20.)):
            with self.subTest(name=name):
                result = example()
                result["telemetry_history"][0]["metrics"][name] = value
                with self.assertRaises(ValueError):
                    exporter.prepare_payload(result)

    def test_narrow_old_history_adapter(self):
        result = {"run_id": "older-result", "spec": {"seed": 2}, "history": [
            {"epoch": 5, "training_seconds": 9., "validation": {"loss": .2, "accuracy": .97, "n": 10000},
             "test": {"loss": .3, "accuracy": .96, "n": 10000}, "stochastic_training_loss": .1}]}
        payload = exporter.prepare_payload(result)
        row = payload["rows"][0]
        self.assertEqual(row["validation/accuracy_pct"], 97.)
        self.assertEqual(row["val_accuracy"], 96.)
        self.assertNotIn("run_elapsed_seconds", row)
        result["history"][0]["validation"]["accuracy"] = 97.
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)

    def test_explicit_empty_preferred_not_replaced_by_old_history(self):
        result = example()
        result["history"] = result["telemetry_history"]
        result["telemetry_history"] = []
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)

    def test_actual_837_metric_row_preserves_all_finite_scalars(self):
        source = Path(__file__).resolve().parents[1]/"results/telemetry/telemetry-main-residual-s101-v1.json"
        if not source.exists():
            self.skipTest("Completed scientific result not installed")
        result = json.loads(source.read_text())
        actual = next(t for t in result["telemetry_history"] if t["epoch"] == 1)
        self.assertEqual(len(actual["metrics"]), 837)
        result["telemetry_history"] = [actual]
        payload = exporter.prepare_payload(result)
        row = payload["rows"][0]
        self.assertGreater(len(row), 1000)
        self.assertLessEqual(len(row), exporter.MAX_SCALAR_FIELDS)
        for key, value in actual["metrics"].items():
            self.assertEqual(row[key], value)
        self.assertEqual(row["training_seconds"], actual["training_seconds"])
        self.assertEqual(row["run_elapsed_seconds"], actual["run_elapsed_seconds"])
        self.assertEqual(row["diagnostics/probe/n"], 128)
        self.assertEqual(row["val_accuracy"], actual["metrics"]["test/accuracy_pct"])

    def test_oversize_rows_are_rejected_never_truncated(self):
        result = example()
        result["telemetry_history"][0]["metrics"] = {
            f"metric_{i}": float(i) for i in range(exporter.MAX_SCALAR_FIELDS + 1)}
        with self.assertRaisesRegex(ValueError, "no values were truncated"):
            exporter.prepare_payload(result)

    def test_historical_project_and_invalid_run_id_rejected(self):
        with self.assertRaises(ValueError):
            exporter.prepare_payload(example(), project="train_ciresan")
        result = example()
        result["run_id"] = "/local/path"
        with self.assertRaises(ValueError):
            exporter.prepare_payload(result)

    def test_deterministic_files_and_new_content_ids(self):
        payload = exporter.prepare_payload(example(), source_sha256="a"*64)
        self.assertTrue(payload["run_id"].startswith("telemetry-"))
        self.assertNotEqual(payload["run_id"], example()["run_id"])
        self.assertEqual(payload, exporter.prepare_payload(example(), source_sha256="a"*64))
        changed = example()
        changed["telemetry_history"][0]["metrics"]["validation/loss"] += .01
        self.assertNotEqual(payload["run_id"], exporter.prepare_payload(changed)["run_id"])
        with tempfile.TemporaryDirectory() as directory:
            files = exporter.write_jsonl(payload, directory)
            lines = Path(directory, files["jsonl_file"]).read_bytes()
            manifest = json.loads(Path(directory, files["manifest_file"]).read_text())
            self.assertEqual(manifest["jsonl_sha256"], hashlib.sha256(lines).hexdigest())
            self.assertEqual([json.loads(row) for row in lines.splitlines()], payload["rows"])
            self.assertNotIn(directory, json.dumps(manifest))

    def test_fake_sdk_upload_uses_new_run_and_real_axes(self):
        payload = exporter.prepare_payload(example())
        sdk = FakeSDK()
        with tempfile.TemporaryDirectory() as directory:
            status = exporter.publish(payload, mode="online", entity="test", sdk_directory=directory, wandb_module=sdk)
        self.assertEqual(sdk.kwargs["project"], "gradient-dissent")
        self.assertEqual(sdk.kwargs["resume"], "never")
        self.assertEqual(sdk.kwargs["reinit"], "create_new")
        self.assertTrue(sdk.kwargs["force"])
        self.assertFalse(sdk.kwargs["save_code"])
        self.assertFalse(sdk.kwargs["sync_tensorboard"])
        for setting in ("disable_code", "disable_git", "x_disable_meta", "x_disable_stats", "x_disable_machine_info"):
            self.assertTrue(sdk.settings[setting])
        self.assertEqual(sdk.settings["console"], "off")
        self.assertFalse(sdk.settings["x_save_requirements"])
        self.assertEqual(sdk.run.logged, [(row, {"step": i, "commit": True}) for i, row in enumerate(payload["rows"])])
        for name, definition in sdk.run.definitions:
            if name not in exporter.AXES:
                self.assertEqual(definition, {"step_metric": "training_seconds", "step_sync": False})
        self.assertTrue(sdk.run.finished)
        self.assertEqual(status["run_url"], sdk.run.url)
        self.assertNotIn("_runtime", json.dumps(sdk.run.logged))

    def test_fake_sdk_offline_does_not_require_login_or_claim_url(self):
        payload = exporter.prepare_payload(example())
        sdk = FakeSDK()
        with tempfile.TemporaryDirectory() as directory:
            status = exporter.publish(payload, mode="offline", sdk_directory=directory, axis="epoch", wandb_module=sdk)
        self.assertFalse(sdk.kwargs["force"])
        self.assertEqual(sdk.kwargs["mode"], "offline")
        self.assertIsNone(status["run_url"])


if __name__ == "__main__":
    unittest.main()
