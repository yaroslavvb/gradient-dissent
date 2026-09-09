"""Verified offline checkpoint wrapper; no training or resource provisioning.

Called by telemetry.run.run(spec, root, progress_commit). Sources and snapshot
bytes are verified before diagnostics. Every completed snapshot is saved and
committed independently, including when the job becomes censored or fails.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time

import torch

try:
    from ..model_data import CiresanMLP, load_mnist
except ImportError:
    from model_data import CiresanMLP, load_mnist
from .curvature import analyze_curvature


HERE = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("telemetry/curvature.py", "telemetry/offline_curvature.py", "model_data.py")
EPOCHS = (0, 1, 5, 10, 20, 50, 100)
RECIPES = ("residual", "sd_constant", "sd_annealed")


def source_hashes():
    return {name: _file_hash(HERE/name) for name in SOURCE_FILES}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _raw_probe_hash(x, y):
    # Exact recorder convention: raw float32 image bytes then int64 label bytes.
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()+
                          y.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _parameter_hash(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Run IDs must be plain identifiers, not paths")
    return value


def _check_spec(spec):
    if spec.get("runner") != "curvature":
        raise ValueError("Expected runner='curvature'")
    _safe_id(spec.get("run_id")); _safe_id(spec.get("source_run_id"))
    if spec["run_id"] == spec["source_run_id"]:
        raise ValueError("Offline output must have a new run ID")
    if tuple(spec.get("snapshot_epochs", EPOCHS)) != EPOCHS:
        raise ValueError("Frozen offline panel is epochs 0,1,5,10,20,50,100")
    fixed = {"probe_n":128, "mode":"exact", "eig_max_dimension":1536,
             "include_directional":True, "precision":"fp64"}
    for name, value in fixed.items():
        if spec.get(name, value) != value:
            raise ValueError("Frozen curvature setting mismatch: "+name)
    expected = spec.get("curvature_source_sha256")
    if not isinstance(expected, dict) or set(expected) != set(SOURCE_FILES):
        raise ValueError("curvature_source_sha256 must independently pin all three offline source files")
    actual = source_hashes()
    if expected != actual:
        raise AssertionError("Frozen offline source mismatch")
    deadline = float(spec.get("work_limit_seconds", 420.))
    if not math.isfinite(deadline) or not 0 < deadline <= float(spec.get("timeout_seconds",480))-20:
        raise ValueError("Work limit must leave at least 20 seconds before invocation timeout")
    return actual, deadline


def _read_source(root, spec):
    source_dir = Path(root)/"runs"/spec["source_run_id"]
    source_path = source_dir/"result.json"
    actual_hash = _file_hash(source_path)
    if spec.get("source_result_sha256") is not None and spec["source_result_sha256"] != actual_hash:
        raise AssertionError("Exact remote training-result bytes do not match pinned SHA")
    source = json.loads(source_path.read_text())
    training = source.get("spec", {})
    if source.get("run_id") != spec["source_run_id"] or training.get("run_id") != spec["source_run_id"]:
        raise AssertionError("Training source run ID mismatch")
    if (training.get("runner") != "controlled" or training.get("cohort") != "main" or
        training.get("stage") != "evaluate" or training.get("seed") != 101 or
        training.get("recipe") not in RECIPES or training.get("epochs") != 100 or
        training.get("train_size",50000) != 50000 or source.get("diverged",False) or
        "error" in source):
        raise ValueError("Offline panel requires a completed central main-cohort seed101 controlled run")
    if not source.get("trained_final_parameters_sha256"):
        raise AssertionError("Training result is not finalized")
    frozen_training = training.get("source_sha256")
    recorded_training = source.get("telemetry_source_sha256")
    if not isinstance(frozen_training,dict) or not frozen_training or frozen_training != recorded_training:
        raise AssertionError("Training result does not match its separate frozen source map")
    if recorded_training.get("model_data.py") != source_hashes()["model_data.py"]:
        raise AssertionError("Model factory differs from the source training model")
    # The trainer's additional internal source inventory should agree wherever
    # it overlaps the independently pinned telemetry execution inventory.
    for name, digest in source.get("source_sha256",{}).items():
        if name in recorded_training and recorded_training[name] != digest:
            raise AssertionError("Training source inventories disagree: "+name)
    telemetry = source.get("telemetry",{})
    if telemetry.get("enabled") is not True:
        raise ValueError("Source telemetry and sparse checkpoint manifest are required")
    snapshots = telemetry.get("snapshots",[])
    if len(snapshots) != len(EPOCHS) or sorted(s.get("epoch") for s in snapshots) != list(EPOCHS):
        raise AssertionError("Source must record exactly the seven frozen snapshot epochs")
    by_epoch = {entry["epoch"]:entry for entry in snapshots}
    for epoch in EPOCHS:
        entry = by_epoch[epoch]
        expected_path = f"snapshot-{epoch:03d}.pt"
        if entry.get("path") != expected_path or not re.fullmatch(r"[0-9a-f]{64}",str(entry.get("sha256",""))):
            raise AssertionError("Invalid source snapshot manifest entry")
        if _file_hash(source_dir/expected_path) != entry["sha256"]:
            raise AssertionError(f"Snapshot SHA mismatch at epoch {epoch}")
    return source_dir, source, actual_hash, by_epoch


def _source_probe(source, epoch):
    matches = [r for r in source.get("telemetry_history",[]) if r.get("epoch")==epoch]
    if len(matches)>1:
        raise AssertionError("Duplicate source telemetry epoch")
    if not matches:
        return None
    row = matches[0]
    probe = row.get("diagnostics",{}).get("probe")
    return {"epoch":epoch,"training_seconds":row.get("training_seconds"),
            "run_elapsed_seconds":row.get("run_elapsed_seconds"),
            "dense_probe":None if probe is None else {k:probe.get(k) for k in ("loss","accuracy","n")}}


def run(spec, root, progress_commit=None):
    """Evaluate seven verified snapshots; persist honest partial/failure state.

    The CPU device override exists only for small synthetic local tests. Paid
    resource provisioning and the external invocation timeout belong to caller.
    """
    started = time.perf_counter()
    source_pins, limit = _check_spec(spec)
    root = Path(root)
    directory = root/"runs"/spec["run_id"]
    directory.mkdir(parents=True,exist_ok=True)
    # Refuse a duplicate invocation before overwriting any scientific artifact.
    with (directory/"claimed.json").open("x") as handle:
        json.dump({"spec":spec,"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},handle,allow_nan=False)
    result = {"schema_version":1,"run_id":spec["run_id"],"spec":spec,
              "source_run_id":spec["source_run_id"],"status":"running","passed":False,
              "complete":False,"snapshot_epochs":list(EPOCHS),"snapshots":[],
              "curvature_source_sha256":source_pins,"work_limit_seconds":limit,
              "timing":{"training_seconds":0.,"offline_curvature_seconds":0.,
                        "checkpoint_load_seconds":0.,"progress_commit_seconds":0.},
              "persistence_timing_note":"Elapsed/commit totals are recorded immediately before each result write. They include completed prior commits and exclude that write's own serialization/final commit; caller invocation/dispatch timing covers the full operation.",
              "interpretation":"Post-training fixed-probe FP64 diagnostics on existing weights. Exact affine-block diagonals and structured KFAC approximations; no exact full Fisher/GGN/Jacobian parameter matrix or full Hessian. No training or new test-set selection."}
    commits = 0
    commit_seconds = 0.
    def persist():
        nonlocal commits, commit_seconds
        result["completed_snapshots"] = len(result["snapshots"])
        result["remaining_snapshot_epochs"] = [e for e in EPOCHS if e not in {r["epoch"] for r in result["snapshots"]}]
        result["total_run_seconds"] = time.perf_counter()-started
        result["progress_commit_calls"] = commits
        result["timing"]["progress_commit_seconds"] = commit_seconds
        _atomic_json(directory/"result.json",result)
        if progress_commit:
            begin = time.perf_counter();progress_commit()
            commit_seconds += time.perf_counter()-begin
            commits += 1
    try:
        source_dir, source, source_hash, manifests = _read_source(root,spec)
        training = source["spec"]
        result.update({"recipe":training["recipe"],"seed":training["seed"],
                       "source_training_result_sha256":source_hash,
                       "source_result_digest_pinned_in_spec":spec.get("source_result_sha256") is not None,
                       "source_training_source_sha256":source["telemetry_source_sha256"],
                       "source_snapshot_manifest_sha256":_canonical_hash(source["telemetry"]["snapshots"]),
                       "source_training_seconds":source.get("training_seconds"),
                       "source_training_hardware":source.get("hardware"),
                       "verification":{"all_snapshot_bytes_preverified":True,
                                       "training_and_offline_source_maps_verified":True,
                                       "source_training_complete":True}})
        device = torch.device(spec.get("device","cuda"))
        if device.type not in ("cuda","cpu"):
            raise ValueError("Only explicit CUDA execution or local CPU qualification is supported")
        if device.type=="cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Offline curvature requires an already-present CUDA device")
            device = torch.device("cuda",torch.cuda.current_device() if device.index is None else device.index)
        torch.set_num_threads(2)
        result["hardware"] = {"device":str(device),"torch":str(torch.__version__),
            "python":platform.python_version(),"cuda_runtime":torch.version.cuda,
            "diagnostic_dtype":"torch.float64","training_parameter_dtype":"torch.float32",
            "cpu_threads":torch.get_num_threads(),"gpu":None}
        if device.type=="cuda":
            properties=torch.cuda.get_device_properties(device)
            result["hardware"].update({"gpu":properties.name,
                "compute_capability":[properties.major,properties.minor],
                "total_memory_bytes":properties.total_memory})
        data_start=time.perf_counter()
        data=load_mnist(root/"data",train_size=training.get("train_size",50000),include_test=False)
        meta=data["metadata"]
        for key in ("training_data_sha256","train_indices_sha256","val_indices_sha256","split_sha256"):
            if not meta.get(key) or meta[key] != source.get("dataset",{}).get(key):
                raise AssertionError("Original training data/split mismatch: "+key)
        if meta.get("test_included") is not False or any(key.startswith("test_") for key in data):
            raise AssertionError("Offline diagnostics must not load official test tensors")
        recorder_n=source["telemetry"].get("probe_n")
        if not isinstance(recorder_n,int) or not 128 <= recorder_n <= len(data["train_y"]):
            raise AssertionError("Recorded source probe does not contain the frozen 128-example prefix")
        if _raw_probe_hash(data["train_x"][:recorder_n],data["train_y"][:recorder_n]) != source["telemetry"].get("probe_sha256"):
            raise AssertionError("Recorded training-probe raw bytes mismatch")
        raw_hash=_raw_probe_hash(data["train_x"][:128],data["train_y"][:128])
        result["dataset"]=meta
        result["probe"]={"n":128,"definition":"First 128 examples of original fixed fitting-training pool, same order as recorder; no validation or official-test examples",
            "raw_pixels_labels_sha256":raw_hash,"source_recorder_probe_n":recorder_n,
            "source_recorder_probe_sha256":source["telemetry"]["probe_sha256"],
            "same_full_probe_as_recorder":recorder_n==128,
            "positions":list(range(128)),"source_input_dtype":str(data["train_x"].dtype),
            "diagnostic_input_dtype":"torch.float64"}
        x=data["train_x"][:128].to(device=device,dtype=torch.float64)
        y=data["train_y"][:128].to(device=device,dtype=torch.long)
        del data
        result["timing"]["data_preparation_seconds"]=time.perf_counter()-data_start
        result["verification"]["dataset_and_recorder_probe_verified"]=True
        persist()
        durations=[]
        for epoch in EPOCHS:
            elapsed=time.perf_counter()-started
            conservative_next=(max(durations)*1.25+5) if durations else 0.
            if elapsed+conservative_next>limit:
                result["status"]="budget_censored"
                result["censoring"]={"next_epoch":epoch,"elapsed_seconds":elapsed,
                    "estimated_next_snapshot_seconds":conservative_next,
                    "rule":"Stop before another snapshot when elapsed plus 1.25 times maximum observed snapshot duration plus five seconds exceeds the frozen work limit; never reduce scientific settings."}
                persist();return result
            snapshot_started=time.perf_counter()
            manifest=manifests[epoch]
            path=source_dir/manifest["path"]
            # Recheck immediately before load, not only at job preflight.
            if _file_hash(path)!=manifest["sha256"]:
                raise AssertionError("Snapshot changed after preverification")
            loading=time.perf_counter()
            model=CiresanMLP(recipe=training["recipe"],pmax=training.get("pmax",0.),
                input_scale=training.get("input_scale",1/255),output_relu=training.get("output_relu",False))
            state=torch.load(path,weights_only=True,map_location="cpu")
            if not isinstance(state,dict) or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 for v in state.values()):
                raise AssertionError("Expected the recorded FP32 Ciresan state dictionary")
            model.load_state_dict(state,strict=True);del state
            if sum(p.numel() for p in model.parameters()) != source.get("parameter_count"):
                raise AssertionError("Snapshot parameter count differs from source")
            before_conversion=_parameter_hash(model)
            if epoch==0 and before_conversion!=source.get("initial_parameters_sha256"):
                raise AssertionError("Epoch0 parameters differ from the original initialization")
            if epoch==100 and before_conversion!=source["trained_final_parameters_sha256"]:
                raise AssertionError("Epoch100 parameters differ from trained final parameters")
            model=model.to(device=device,dtype=torch.float64).eval()
            converted_hash=_parameter_hash(model)
            if device.type=="cuda":torch.cuda.synchronize(device)
            load_seconds=time.perf_counter()-loading
            result["timing"]["checkpoint_load_seconds"]+=load_seconds
            curvature=analyze_curvature(model,x,y,mode="exact",probe_id=raw_hash,
                eig_max_dimension=1536,include_directional=True,seed=int(spec.get("seed",20260909)))
            if curvature["probe_n"]!=128 or curvature["output_classes"]!=10 or curvature["derivative_dtype"]!="torch.float64":
                raise AssertionError("Offline curvature precision/panel differs from frozen specification")
            if any(not family["kfac"]["spectrum_exact"] for layer in curvature["layers"] for family in layer["families"].values()):
                raise AssertionError("Frozen N128 exact-output panel unexpectedly used approximate eigenspectra")
            if _parameter_hash(model)!=converted_hash:
                raise AssertionError("Diagnostic mutated checkpoint parameters")
            if source_hashes()!=source_pins:
                raise AssertionError("Offline source files changed during diagnostics")
            original_probe=_source_probe(source,epoch)
            comparison={"source":original_probe,
                "note":"Recorder used original FP32/high-matmul probe arithmetic; offline uses a separate FP64/highest-matmul model. Differences are descriptive, not a failed prediction-parity gate."}
            if original_probe and original_probe["dense_probe"] and recorder_n==128:
                p=original_probe["dense_probe"]
                comparison["fp64_minus_source_ce"]=None if p.get("loss") is None else curvature["dense_ce"]-p["loss"]
                comparison["fp64_minus_source_accuracy"]=None if p.get("accuracy") is None else curvature["dense_accuracy"]-p["accuracy"]
            record={"epoch":epoch,"checkpoint_file":manifest["path"],
                "checkpoint_sha256":manifest["sha256"],"source_fp32_parameters_sha256":before_conversion,
                "diagnostic_fp64_parameters_sha256":converted_hash,"load_seconds":load_seconds,
                "curvature":curvature,"source_probe_comparison":comparison,
                "parameters_unchanged":True,"snapshot_elapsed_seconds":time.perf_counter()-snapshot_started}
            artifact=directory/f"curvature-{epoch:03d}.json"
            _atomic_json(artifact,record)
            record["artifact"]={"file":artifact.name,"sha256":_file_hash(artifact),"bytes":artifact.stat().st_size}
            result["snapshots"].append(record)
            result["timing"]["offline_curvature_seconds"]+=curvature["elapsed_seconds"]
            durations.append(record["snapshot_elapsed_seconds"])
            del model
            if len(durations)==1:
                result["first_snapshot_budget_projection"]={"measured_snapshot_seconds":durations[0],
                    "projected_all_snapshots_seconds":(time.perf_counter()-started)+durations[0]*(len(EPOCHS)-1),
                    "meaning":"Runtime-only projection; no statistic selects checkpoints or settings"}
            print(json.dumps({"run_id":spec["run_id"],"epoch":epoch,"curvature_seconds":curvature["elapsed_seconds"],
                "largest_eigenproblem_dimension":curvature["largest_eigenproblem_dimension"],
                "completed_snapshots":len(durations)},allow_nan=False),flush=True)
            persist()
        if source_hashes()!=source_pins:
            raise AssertionError("Offline source files changed before finalization")
        result.update({"status":"completed","complete":True,"passed":True})
        result["verification"]["all_parameters_unchanged"]=True
        result["verification"]["all_sources_unchanged"]=True
        persist()
        return result
    except Exception as error:
        result.update({"status":"failed","complete":False,"passed":False,
            "error_type":type(error).__name__,"error":str(error)})
        persist()
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--spec",required=True);parser.add_argument("--root",required=True)
    arguments=parser.parse_args()
    print(json.dumps(run(json.loads(Path(arguments.spec).read_text()),Path(arguments.root)),allow_nan=False))
