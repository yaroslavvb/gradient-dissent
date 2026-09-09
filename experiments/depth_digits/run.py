#!/usr/bin/env python3
"""Independent depth robustness experiment on sklearn's 8x8 digits; CPU only.

Run from repo root: experiments/.venv/bin/python experiments/depth_digits/run.py
No auxiliary exits, test-time scaling, or classifier adaptation are used.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
DEPTH, WIDTH, STEPS, BATCH = 6, 64, 600, 128
LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01]
TUNE_SEEDS = [1701, 1702]
EVAL_SEEDS = [2701, 2702, 2703, 2704, 2705]
SPLIT_SEED = 913071
CONFIGS = {
    "dense6": {"depth": 6, "schedule": "none", "pmax": 0.},
    "constant_ild": {"depth": 6, "schedule": "constant", "pmax": .4},
    "decreasing_ild": {"depth": 6, "schedule": "decreasing", "pmax": .8},
    "dense3": {"depth": 3, "schedule": "none", "pmax": 0.},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ci(values):
    values = list(values)
    assert len(values) == 5
    mean = statistics.mean(values)
    radius = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
    return {"n": 5, "mean": mean, "low": mean-radius, "high": mean+radius,
            "std": statistics.stdev(values)}


def data():
    digits = load_digits()
    x = np.array(digits.data, dtype=np.float32)
    y = np.array(digits.target, dtype=np.int64)
    indices = np.arange(len(y))
    train, held = train_test_split(indices, test_size=.4, stratify=y, random_state=SPLIT_SEED)
    val, test = train_test_split(held, test_size=.5, stratify=y[held], random_state=SPLIT_SEED+1)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert sorted([*train, *val, *test]) == list(range(len(y)))
    mean = x[train].mean(0)
    std = x[train].std(0)
    std[std < 1e-6] = 1.
    normalized = (x - mean) / std
    assert np.max(np.abs(normalized[train].mean(0))) < 1e-5
    split_info = {name: {"n": len(ids), "indices": ids.tolist(),
                          "index_sha256": digest(ids.astype("<i8").tobytes()),
                          "label_counts": np.bincount(y[ids], minlength=10).tolist()}
                  for name, ids in [("train", train), ("validation", val), ("test", test)]}
    metadata = {"name": "scikit-learn load_digits: 8×8 handwritten digits (NOT MNIST)",
                "n": len(y), "split_seed": SPLIT_SEED,
                "split_protocol": "stratified 60/20/20; split seed 913071 then 913072; fixed across all seeds and treatments",
                "dataset_sha256": digest(x.astype("<f4").tobytes()+y.astype("<i8").tobytes()),
                "normalization": "per-pixel mean and population std fitted on training rows only; zero std replaced by1",
                "train_mean": mean.tolist(), "train_std": std.tolist(), "splits": split_info,
                "sources": ["https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                            "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"]}
    tensors = {name: (torch.tensor(normalized[ids]), torch.tensor(y[ids]))
               for name, ids in [("train", train), ("validation", val), ("test", test)]}
    return tensors, metadata


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH*2),
                                 nn.GELU(), nn.Linear(WIDTH*2, WIDTH))

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        # Build all six before slicing so dense3 shares initial embed/head/first3
        # with the six-block models for each seed.
        self.embed = nn.Linear(64, WIDTH)
        blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 10)
        self.blocks = nn.ModuleList(list(blocks)[:depth])
        self.depth = depth

    def forward(self, x, probabilities=None, mask_generator=None, keep=None, prefix=None):
        if not self.training:
            assert probabilities is None, "No training dropout or inverted scaling at inference"
        if prefix is not None:
            assert keep is None
            keep = [i < prefix for i in range(self.depth)]
        if keep is not None:
            assert len(keep) == self.depth
        h = self.embed(x)
        active = 0
        for i, block in enumerate(self.blocks):
            if keep is not None and not keep[i]:
                continue
            p = 0. if probabilities is None else float(probabilities[i])
            residual = block(h)
            if p:
                mask = torch.rand((len(x), 1), generator=mask_generator) >= p
                h = h + residual * mask / (1-p)
                active += int(mask.sum())
            else:
                h = h + residual
                active += len(x)
        return self.head(self.norm(h)), active


def probabilities(config, step):
    depth = config["depth"]
    factor = {"none": 0., "constant": 1., "decreasing": 1-step/(STEPS-1)}[config["schedule"]]
    return config["pmax"] * np.arange(depth) / (depth-1) * factor


def assertions(tensors):
    schedule_checks = {}
    for name, c in CONFIGS.items():
        ps = np.stack([probabilities(c, step) for step in range(STEPS)])
        expected = .2 if "ild" in name else 0.
        assert abs(ps.mean()-expected) < 1e-12
        assert np.all(ps[:, 0] == 0)
        if name == "decreasing_ild":
            assert np.max(ps[-1]) == 0. and ps[0, -1] == .8
        schedule_checks[name] = {"mean_dropout": float(ps.mean()),
                                 "expected_active_blocks_per_step": float((1-ps).sum(1).mean())}
    torch.manual_seed(991)
    model = Classifier()
    x = tensors["train"][0][:11].clone().requires_grad_(True)
    model.eval()
    full = model(x)[0]
    assert torch.equal(full, model(x, keep=[True]*6)[0])
    assert torch.equal(full, model(x, prefix=6)[0])
    assert torch.equal(full, model(x)[0]), "Evaluation must be deterministic"
    for k in range(1, 7):
        assert torch.equal(model(x, prefix=k)[0], model(x, keep=[i<k for i in range(6)])[0])
    assert not any(isinstance(m, nn.Dropout) for m in model.modules())
    full.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.train()
    no_drop = model(x, probabilities=np.zeros(6))[0]
    assert torch.equal(full, no_drop)
    torch.manual_seed(991)
    small = Classifier(3)
    assert torch.equal(small.embed.weight, model.embed.weight)
    assert torch.equal(small.head.weight, model.head.weight)
    for i in range(3):
        assert torch.equal(small.blocks[i].net[1].weight, model.blocks[i].net[1].weight)
    return {"passed": True, "checks": ["disjoint exhaustive data splits", "train-only normalization",
             "identical initial embed/head/first3 for same seed", "prefix/full/allkeep equivalence",
             "repeated eval exactly deterministic", "no nn.Dropout modules", "finite input and parameter gradients",
             "zero-dropout training equals inference", "schedule mean and first-layer retention"],
            "schedule": schedule_checks}


@torch.no_grad()
def score(model, x, y, **kwargs):
    model.eval()
    logits, _ = model(x, **kwargs)
    return {"ce": float(nn.functional.cross_entropy(logits, y)),
            "accuracy": float((logits.argmax(1) == y).float().mean())}


def train(name, seed, lr, tensors, phase):
    config = CONFIGS[name]
    torch.manual_seed(seed)
    model = Classifier(config["depth"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    x, y = tensors["train"]
    data_generator = torch.Generator().manual_seed(seed+100000)
    mask_generator = torch.Generator().manual_seed(seed+200000)
    batches = torch.randint(len(x), (STEPS, BATCH), generator=data_generator)
    batch_hash = digest(batches.numpy().astype("<i8").tobytes())
    active = 0
    expected_active = 0.
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(STEPS):
        decay = .1 + .9 * .5 * (1 + math.cos(math.pi*step/(STEPS-1)))
        for group in optimizer.param_groups:
            group["lr"] = lr*decay
        ps = probabilities(config, step)
        optimizer.zero_grad(set_to_none=True)
        pred, now_active = model(x[batches[step]], probabilities=ps, mask_generator=mask_generator)
        loss = nn.functional.cross_entropy(pred, y[batches[step]])
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        active += now_active
        expected_active += BATCH * float((1-ps).sum())
        if step == 0 or (step+1) % 100 == 0:
            trace.append({"phase": phase, "treatment": name, "seed": seed, "lr": lr,
                          "step": step+1, "minibatch_ce": float(loss.detach()),
                          "grad_norm_before_clip": float(grad_norm), "mean_dropout": float(ps.mean())})
    seconds = time.perf_counter() - started
    val = score(model, *tensors["validation"])
    row = {"phase": phase, "treatment": name, "seed": seed, "lr": lr,
           "depth": config["depth"], "steps": STEPS, "batch": BATCH,
           "parameter_count": sum(p.numel() for p in model.parameters()),
           "validation_ce": val["ce"], "validation_accuracy": val["accuracy"],
           "train_seconds": seconds, "batch_indices_sha256": batch_hash,
           "expected_active_example_blocks": expected_active,
           "realized_active_example_blocks": active,
           "executed_example_blocks": STEPS*BATCH*config["depth"],
           "last_minibatch_ce": trace[-1]["minibatch_ce"]}
    return model, row, trace


def masks(depth):
    # Complete finite population: no mask Monte Carlo error or mask seed needed.
    return [tuple(i in combo for i in range(depth))
            for k in range(1, depth+1) for combo in itertools.combinations(range(depth), k)]


def evaluate(model, name, seed, lr, tensors):
    x, y = tensors["test"]
    full = score(model, x, y)
    rows = []
    for keep in masks(model.depth):
        k = sum(keep)
        s = score(model, x, y, keep=keep)
        prefix = keep == tuple(i < k for i in range(model.depth))
        if prefix:
            prefix_score = score(model, x, y, prefix=k)
            assert prefix_score == s
        if k == model.depth:
            assert s == full
        rows.append({"treatment": name, "seed": seed, "lr": lr,
                     "model_depth": model.depth, "retained_depth": k,
                     "mask": "".join("1" if v else "0" for v in keep),
                     "is_prefix": prefix, "keeps_first": keep[0],
                     "is_alternating": keep in (tuple(i%2==0 for i in range(model.depth)),
                                                   tuple(i%2==1 for i in range(model.depth))),
                     "ce": s["ce"], "accuracy": s["accuracy"],
                     "excess_ce": s["ce"]-full["ce"],
                     "accuracy_change": s["accuracy"]-full["accuracy"],
                     "full_ce": full["ce"], "full_accuracy": full["accuracy"]})
    return rows


def aggregate(mask_rows):
    per_seed = []
    for name, c in CONFIGS.items():
        for seed in EVAL_SEEDS:
            for k in range(1, c["depth"]+1):
                for kind in ["prefix", "all_masks", "keeps_first", "drops_first", "alternating"]:
                    group = [r for r in mask_rows if r["treatment"] == name and r["seed"] == seed
                             and r["retained_depth"] == k
                             and (kind == "all_masks" or (kind == "prefix" and r["is_prefix"])
                                  or (kind == "keeps_first" and r["keeps_first"])
                                  or (kind == "drops_first" and not r["keeps_first"])
                                  or (kind == "alternating" and r["is_alternating"]))]
                    if not group:
                        continue
                    per_seed.append({"treatment": name, "seed": seed, "retained_depth": k, "mask_group": kind,
                                     "n_masks": len(group),
                                     **{metric: statistics.mean(r[metric] for r in group)
                                        for metric in ["ce", "accuracy", "excess_ce", "accuracy_change"]}})
    means = []
    keys = sorted({(r["treatment"], r["retained_depth"], r["mask_group"]) for r in per_seed})
    for name, k, kind in keys:
        rows = [r for r in per_seed if (r["treatment"], r["retained_depth"], r["mask_group"]) == (name, k, kind)]
        means.append({"treatment": name, "retained_depth": k, "mask_group": kind, "n_masks_per_seed": rows[0]["n_masks"],
                      **{metric: ci(r[metric] for r in rows) for metric in ["ce", "accuracy", "excess_ce", "accuracy_change"]}})
    paired = []
    for name, k, kind in keys:
        if name == "dense6":
            continue
        for metric in ["ce", "accuracy", "excess_ce", "accuracy_change"]:
            differences = []
            for seed in EVAL_SEEDS:
                treatment_rows = [r for r in per_seed if (r["treatment"],r["seed"],r["retained_depth"],r["mask_group"])
                                  == (name,seed,k,kind)]
                dense_rows = [r for r in per_seed if (r["treatment"],r["seed"],r["retained_depth"],r["mask_group"])
                              == ("dense6",seed,k,kind)]
                if treatment_rows and dense_rows:
                    differences.append(treatment_rows[0][metric]-dense_rows[0][metric])
            if differences:
                paired.append({"treatment": name, "reference": "dense6", "retained_depth": k, "mask_group": kind,
                               "metric": metric, "difference": ci(differences), "seed_differences": differences,
                               "interpretation": "Treatment minus dense6 at same retained depth; dense3 has a separately trained final classifier and a different full-depth baseline" if name=="dense3" else "Treatment minus dense6 at same retained depth"})
    return per_seed, means, paired


def main():
    began = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    tensors, dataset_info = data()
    checks = assertions(tensors)
    write_json("dataset.json", dataset_info)
    tuning, evaluation, traces, mask_rows = [], [], [], []
    selected = {}
    for name in CONFIGS:
        for lr in LRS:
            for seed in TUNE_SEEDS:
                _, row, trace = train(name, seed, lr, tensors, "tuning")
                tuning.append(row)
                traces.extend(trace)
                print(json.dumps({"phase": "tuning", "treatment": name, "lr": lr, "seed": seed,
                                  "validation_ce": row["validation_ce"], "seconds": row["train_seconds"]}), flush=True)
                write_csv("tuning.csv", tuning)
        scores = {lr: statistics.mean(r["validation_ce"] for r in tuning if r["treatment"] == name and r["lr"] == lr)
                  for lr in LRS}
        best = min(scores, key=lambda lr: (scores[lr], lr))
        selected[name] = {"lr": best, "mean_validation_ce_by_lr": scores,
                          "at_grid_boundary": best in [LRS[0], LRS[-1]]}
    # Test data is never scored until every treatment's full-depth validation-only LR is locked.
    write_json("selected_lrs.json", selected)
    for seed in EVAL_SEEDS:
        for name in CONFIGS:
            lr = selected[name]["lr"]
            model, row, trace = train(name, seed, lr, tensors, "evaluation")
            measured = evaluate(model, name, seed, lr, tensors)
            full = [r for r in measured if r["retained_depth"] == model.depth][0]
            row.update({"test_ce": full["ce"], "test_accuracy": full["accuracy"]})
            evaluation.append(row)
            traces.extend(trace)
            mask_rows.extend(measured)
            print(json.dumps({"phase": "evaluation", "treatment": name, "seed": seed,
                              "test_ce": row["test_ce"], "test_accuracy": row["test_accuracy"],
                              "seconds": row["train_seconds"]}), flush=True)
            write_csv("runs.csv", evaluation)
            write_csv("masks.csv", mask_rows)
    # Paired data streams and expected active budget checks across full6 methods.
    for seed in EVAL_SEEDS:
        paired_rows = [r for r in evaluation if r["seed"] == seed]
        assert len({r["batch_indices_sha256"] for r in paired_rows}) == 1
        for r in paired_rows:
            target = STEPS * BATCH * (4.8 if "ild" in r["treatment"] else CONFIGS[r["treatment"]]["depth"])
            assert abs(r["expected_active_example_blocks"]-target) < 1e-6
    per_seed, summary, paired = aggregate(mask_rows)
    write_csv("seed_summary.csv", per_seed)
    write_csv("training_trace.csv", traces)
    write_json("checks.json", checks)
    write_json("spending.json", {"currency": "USD", "experiment_budget": 50,
                                 "paid_services_invoked": [], "local_cpu_cost_charged": 0,
                                 "total_external_spend": 0,
                                 "note": "Local CPU only. Electricity/hardware amortization are not priced; no cloud, Modal, or paid API calls."})
    wall = time.perf_counter()-wall_start
    metadata = {"started_utc": began, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": wall, "training_seconds_sum": sum(r["train_seconds"] for r in tuning+evaluation),
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                "torch_threads": 1, "device": "cpu", "dtype": "float32",
                "source_sha256": digest(Path(__file__).read_bytes()),
                "git_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OUT, text=True).strip(),
                "dependency_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
    result = {"title": "Depth robustness on 8×8 handwritten digits", "dataset": {k:v for k,v in dataset_info.items() if k not in ["splits", "train_mean", "train_std"]},
              "metadata": metadata, "protocol": {
                  "architecture": "64-pixel linear embed→6 residual blocks (pre-LayerNorm→Linear64→128→GELU→Linear128→64)→LayerNorm→10-class linear head; dense3 has3 blocks. No auxiliary losses/heads. No residual scale.",
                  "width": WIDTH, "steps": STEPS, "batch": BATCH, "training_examples_seen": STEPS*BATCH,
                  "optimizer": "AdamW; weight decay .01; global grad norm clip1; cosine LR decay from lr to .1lr; default betas(0.9,0.999)",
                  "training_sampler": "128 train rows sampled with replacement at each optimizer step; identical batch streams across paired treatments",
                  "dropout": "independent per-example per-layer Bernoulli masks; one residual MLP branch per block; compute then mask; inverted scaling only during training",
                  "schedule": "p_l,t=pmax*l/(L−1)*f(t), l=0..L−1. constant f=1,pmax=.4; decreasing f=1−t/(T−1),pmax=.8. Both have mean dropout .2; layer0 always kept.",
                  "regime": "matched600 optimizer steps; equal data presentations; sparse ideal active work is80% of dense6, dense3 is50%. Executed training computes every branch: active budget is theoretical, not a measured FLOP or time saving.",
                  "evaluation": "deterministic evaluation at residual scale1 with SAME trained final norm/head, no adaptation, all63 nonempty masks for6 blocks and7 masks for3 blocks",
                  "mask_groups": "prefix; all masks uniform within retained depth; masks preserving first block; masks dropping first block; odd/even layer alternating subsets. Indexing begins at0.",
                  "tuning": "select LR by mean final full-depth validation CE across2 tuning seeds; independent5 evaluation seeds; no test or reduced-depth tuning; no early stopping. Equal LR grid extended downward from[.001,.003,.01] because all4 methods selected its lower boundary on validation; initial completed run is archived in initial_grid. No test metric informed the extension or LR selection.",
                  "learning_rate_grid": LRS, "tuning_seeds": TUNE_SEEDS, "evaluation_seeds": EVAL_SEEDS,
                  "primary_endpoint": "paired difference vs dense6 of prefix4 CE minus own full-depth CE; lower is better; full-depth CE/accuracy and raw prefix4 CE/accuracy also reported",
                  "inference": "95% Student t intervals df4 across5 independent initialization/batch/mask seeds on same fixed test split. For mask ensembles average masks within seed first. No multiple-comparison correction for secondary exploratory endpoints; intervals exclude dataset/split uncertainty.",
                  "limitations": "tiny supervised digits dataset and residual MLP, not MNIST, language modeling, a transformer, auxiliary early-exit decoding, or an LLM reproduction. Prefix deletion is structured and out of distribution relative to independent training masks. Layer0 dropping is never trained. Dense3 has fewer parameters and lower modeled active budget. Small LR grid can select boundary; full-depth validation target does not optimize robustness."},
              "configs": CONFIGS, "selected_lrs": selected, "counts": {"tuning_runs": len(tuning), "evaluation_runs": len(evaluation), "test_mask_evaluations": len(mask_rows)},
              "execution_audit": {"initial_grid_tuning_runs": 24, "initial_grid_evaluation_runs": 20,
                                  "initial_grid_archived": "initial_grid/", "total_training_executions_including_initial": 44+len(tuning)+len(evaluation),
                                  "note": "Final rerun uses identical seeds, so repeated initial-grid results are not additional independent replicates."},
              "checks": checks, "runs": evaluation, "curves": summary, "paired_comparisons": paired,
              "primary_comparisons": [r for r in paired if r["retained_depth"] ==4 and r["mask_group"] =="prefix" and r["metric"] =="excess_ce"],
              "spending": {"total_usd": 0, "budget_usd": 50, "compute": "local CPU only"}}
    write_json("result.json", result)
    print(json.dumps({"completed": True, "wall_seconds": wall, "primary": result["primary_comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
