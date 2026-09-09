"""Synthetic local filesystem qualification of offline provenance/persistence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import offline_curvature as offline


class SyntheticMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(3,4),nn.Linear(4,10)])
    def forward(self,x):
        return self.layers[1](self.layers[0](x).tanh())


class OfflineCurvatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="offline-curvature-synthetic-")
        self.root=Path(self.temporary.name)
        generator=torch.Generator().manual_seed(221)
        x=torch.randn(256,3,generator=generator)
        y=torch.arange(256)%10
        metadata={k:"a"*64 for k in ("training_data_sha256","train_indices_sha256","val_indices_sha256","split_sha256")}
        metadata["test_included"]=False
        self.data={"train_x":x,"train_y":y,"val_x":x[:10],"val_y":y[:10],"metadata":metadata}
        training={"run_id":"synthetic-source","runner":"controlled","cohort":"main","stage":"evaluate",
                  "seed":101,"recipe":"residual","epochs":100,"train_size":50000,
                  "source_sha256":{"model_data.py":offline.source_hashes()["model_data.py"]}}
        source_dir=self.root/"runs"/training["run_id"];source_dir.mkdir(parents=True)
        model=SyntheticMLP()
        initial=copy.deepcopy(model.state_dict())
        snapshots=[]
        initial_hash=offline._parameter_hash(model)
        for epoch in offline.EPOCHS:
            model.load_state_dict(initial)
            with torch.no_grad():
                for p in model.parameters():p.add_(epoch*1e-5)
            path=source_dir/f"snapshot-{epoch:03d}.pt"
            torch.save(model.state_dict(),path)
            snapshots.append({"epoch":epoch,"path":path.name,"sha256":offline._file_hash(path)})
        self.source={"run_id":training["run_id"],"spec":training,"diverged":False,
            "trained_final_parameters_sha256":offline._parameter_hash(model),
            "initial_parameters_sha256":initial_hash,"parameter_count":sum(p.numel() for p in model.parameters()),
            "telemetry_source_sha256":training["source_sha256"],"source_sha256":training["source_sha256"],
            "telemetry":{"enabled":True,"snapshots":snapshots,"probe_n":128,
                "probe_sha256":offline._raw_probe_hash(x[:128],y[:128])},"dataset":metadata,"telemetry_history":[]}
        self.source_path=source_dir/"result.json"
        self.write_source()
        self.spec={"run_id":"synthetic-offline","source_run_id":"synthetic-source","runner":"curvature",
                   "device":"cpu","curvature_source_sha256":offline.source_hashes()}
        self.model_patch=patch.object(offline,"CiresanMLP",SyntheticMLP)
        self.data_patch=patch.object(offline,"load_mnist",return_value=self.data)
        self.model_patch.start();self.loader=self.data_patch.start()

    def tearDown(self):
        self.model_patch.stop();self.data_patch.stop();self.temporary.cleanup()

    def write_source(self):
        self.source_path.write_text(json.dumps(self.source))

    def run_wrapper(self):
        commits=[]
        def commit():
            path=self.root/"runs"/self.spec["run_id"]/"result.json"
            commits.append(json.loads(path.read_text())["completed_snapshots"])
        result=offline.run(self.spec,self.root,progress_commit=commit)
        return result,commits

    def test_complete_fp64_seven_snapshots_and_incremental_commits(self):
        self.spec["source_result_sha256"]=offline._file_hash(self.source_path)
        result,commits=self.run_wrapper()
        self.assertTrue(result["passed"]);self.assertTrue(result["complete"])
        self.assertEqual([s["epoch"] for s in result["snapshots"]],list(offline.EPOCHS))
        self.assertEqual(commits,[0,1,2,3,4,5,6,7,7])
        self.assertTrue(result["probe"]["same_full_probe_as_recorder"])
        self.assertTrue(result["source_result_digest_pinned_in_spec"])
        self.loader.assert_called_once_with(self.root/"data",train_size=50000,include_test=False)
        for snapshot in result["snapshots"]:
            self.assertEqual(snapshot["curvature"]["derivative_dtype"],"torch.float64")
            self.assertTrue(snapshot["parameters_unchanged"])
            path=self.root/"runs"/self.spec["run_id"]/snapshot["artifact"]["file"]
            self.assertEqual(offline._file_hash(path),snapshot["artifact"]["sha256"])
        self.assertEqual(result,json.loads((self.root/"runs"/self.spec["run_id"]/"result.json").read_text()))
        with self.assertRaises(FileExistsError):self.run_wrapper()

    def test_snapshot_tamper_fails_before_diagnostics(self):
        path=self.source_path.parent/self.source["telemetry"]["snapshots"][3]["path"]
        with path.open("ab") as handle:handle.write(b"tamper")
        with self.assertRaisesRegex(AssertionError,"Snapshot SHA"):
            self.run_wrapper()
        record=json.loads((self.root/"runs"/self.spec["run_id"]/"result.json").read_text())
        self.assertEqual(record["status"],"failed");self.assertEqual(record["completed_snapshots"],0)
        self.assertFalse(record["passed"])

    def test_raw_recorder_probe_tamper_rejected(self):
        self.source["telemetry"]["probe_sha256"]="f"*64;self.write_source()
        with self.assertRaisesRegex(AssertionError,"probe raw bytes"):
            self.run_wrapper()

    def test_source_pin_and_frozen_training_pin_rejected(self):
        self.spec["curvature_source_sha256"]["telemetry/curvature.py"]="0"*64
        with self.assertRaisesRegex(AssertionError,"offline source"):
            self.run_wrapper()
        self.spec["curvature_source_sha256"]=offline.source_hashes()
        self.source["telemetry_source_sha256"]={"model_data.py":"1"*64};self.write_source()
        with self.assertRaisesRegex(AssertionError,"frozen source map"):
            self.run_wrapper()

    def test_work_budget_censors_without_changing_scientific_panel(self):
        self.spec["work_limit_seconds"]=1e-12
        result,commits=self.run_wrapper()
        self.assertEqual(result["status"],"budget_censored")
        self.assertFalse(result["complete"]);self.assertFalse(result["passed"])
        self.assertEqual(result["remaining_snapshot_epochs"],list(offline.EPOCHS))
        self.assertEqual(commits,[0,0])

    def test_noncanonical_snapshot_or_partial_panel_rejected(self):
        self.source["telemetry"]["snapshots"][0]["path"]="../snapshot-000.pt";self.write_source()
        with self.assertRaisesRegex(AssertionError,"manifest entry"):
            self.run_wrapper()


if __name__=="__main__":unittest.main()
