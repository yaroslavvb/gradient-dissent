#!/usr/bin/env python3
"""Small falsification/measurement tests for arXiv:2609.05275, not an LLM replication.

Run from the repository root: experiments/.venv/bin/python experiments/run_experiments.py
All stochastic experiments have declared seeds. No GPU or energy inference is made.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "experiments" / "results"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values):
    """Student t interval; only used for the declared n=6 held-out seed means."""
    values = list(values)
    n = len(values)
    mean = statistics.mean(values)
    t_crit = {6: 2.5705818356}[n]
    sem = statistics.stdev(values) / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95_low": mean - t_crit * sem,
            "ci95_high": mean + t_crit * sem, "std": statistics.stdev(values)}


def exact_checks():
    """Exact enumeration over tiny networks: no Monte Carlo error."""
    x, a, b, target = 1.0, 0.5, 0.4, 0.7
    rows = []
    for p in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]:
        rho = 1 - p
        # One isolated residual update preserves the conditional mean given x.
        one = sum(pr * (x + m * a * x / rho) for m, pr in [(0, p), (1, rho)])
        # The recommended single shared mask wraps both residual sublayers (Eq. 6).
        outcomes = []
        expected_gradient = 0.0
        for m, pr in [(0, p), (1, rho)]:
            z = x + m * a * x / rho
            y = z + m * b * z / rho
            dy_da = m * x / rho * (1 + m * b / rho)
            expected_gradient += pr * (y - target) * dy_da
            outcomes.append((pr, y))
        y_dense = (1 + a) * (1 + b) * x
        mean = sum(pr * y for pr, y in outcomes)
        formula_bias = a * b * p / rho * x
        assert abs(one - (1 + a) * x) < 1e-12
        assert abs(mean - y_dense - formula_bias) < 1e-12
        # Independent masks remove this affine cross-term, but nonlinear composition
        # still does not commute with expectation. f_2(z) = b*z^2.
        nonlinear_mean = 0.0
        affine_independent_mean = 0.0
        for m1, m2 in itertools.product([0, 1], repeat=2):
            pr = (rho if m1 else p) * (rho if m2 else p)
            z = x + m1 * a * x / rho
            nonlinear_mean += pr * (z + m2 * b * z * z / rho)
            affine_independent_mean += pr * (z + m2 * b * z / rho)
        nonlinear_dense = (1 + a) * x + b * ((1 + a) * x) ** 2
        assert abs(affine_independent_mean - y_dense) < 1e-12
        assert abs(nonlinear_mean - nonlinear_dense - b * a*a*x*x*p/rho) < 1e-12
        # One-branch quadratic-loss objective: E[.5(y-t)^2] equals the
        # dense objective plus .5*p/rho*(a*x)^2, despite identical mean y.
        one_expected_loss = sum(pr * .5 * (x + m*a*x/rho - target)**2
                                for m, pr in [(0, p), (1, rho)])
        one_dense_loss = .5 * ((1+a)*x-target)**2
        penalty = .5*p/rho*(a*x)**2
        assert abs(one_expected_loss - one_dense_loss - penalty) < 1e-12
        one_expected_gradient = ((1+a)*x-target)*x + p/rho*a*x*x
        rows.append({"dropout_p": p, "density": rho,
                     "single_residual_mean": one, "single_residual_dense": (1+a)*x,
                     "shared_mask_dense_prediction": y_dense,
                     "shared_mask_expected_prediction": mean,
                     "shared_mask_prediction_bias": mean-y_dense,
                     "shared_mask_expected_half_mse": sum(pr*.5*(y-target)**2 for pr,y in outcomes),
                     "shared_mask_dense_half_mse": .5*(y_dense-target)**2,
                     "shared_mask_expected_gradient_da": expected_gradient,
                     "shared_mask_dense_gradient_da": (y_dense-target)*(1+b)*x,
                     "independent_linear_mean": affine_independent_mean,
                     "independent_nonlinear_dense": nonlinear_dense,
                     "independent_nonlinear_mean": nonlinear_mean,
                     "single_residual_loss_penalty": penalty,
                     "single_residual_expected_gradient_da": one_expected_gradient,
                     "single_residual_dense_gradient_da": ((1+a)*x-target)*x})
    # Check the paper's finite discrete schedule exactly, including endpoints.
    L, T, pmax = 6, 240, .8
    grid = pmax * np.arange(L)[:, None]/(L-1) * (1-np.arange(T)[None, :]/(T-1))
    assert abs(grid.mean()-pmax/4) < 1e-12
    # Skipping autograd is not automatically equivalent to materializing zero
    # gradients at the optimizer boundary. AdamW can decay/move a dropped block.
    optimizer_outcomes = {}
    for policy in ["zero_gradient", "absent_gradient"]:
        parameter = nn.Parameter(torch.tensor(1., dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()  # initialize identical nonzero momentum
        before_skip = float(parameter.detach())
        parameter.grad = torch.zeros_like(parameter) if policy == "zero_gradient" else None
        optimizer.step()
        optimizer_outcomes[policy] = {"before_skipped_step": before_skip,
                                      "after_skipped_step": float(parameter.detach())}
    assert optimizer_outcomes["zero_gradient"]["after_skipped_step"] != optimizer_outcomes["absent_gradient"]["after_skipped_step"]
    write_csv(RAW / "exact_checks.csv", rows)
    return {"protocol": "Exact mask enumeration, float64 arithmetic; scalar inputs x=1,a=0.5,b=0.4,target=0.7; half-MSE loss.",
            "claims": ["Inverted dropout preserves an isolated residual branch's conditional mean.",
                       "The Eq.6 shared mask biases even an affine two-sublayer block's mean by ab*p/(1-p)*x.",
                       "Independent masks remove that affine covariance, but nonlinear composition still creates mean bias.",
                       "Even one branch with an unbiased mean adds a dropout-dependent MSE penalty and changes its expected gradient."],
            "rows": rows, "schedule_check": {"layers": L, "steps": T, "pmax": pmax,
                                              "measured_mean_dropout": float(grid.mean()), "expected_pmax_over_4": pmax/4},
            "optimizer_skip_check": {"protocol": "Scalar float64 AdamW, parameter initialized1, lr.1, weight_decay.1, default betas. First gradient1 initializes identical momentum. Second dropped step passes either zero or absent gradient; no optimizer synchronization policy is imposed.",
                                     "outcomes": optimizer_outcomes,
                                     "interpretation": "Compute-then-mask and true skips need an explicit optimizer update policy to give the same training trajectory, even when forward outputs and mathematical gradients agree."},
            "all_assertions_passed": True}


class Block(nn.Module):
    def __init__(self, width, expansion=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width*expansion), nn.GELU(),
                                 nn.Linear(width*expansion, width))

    def forward(self, x):
        return self.net(x)


def execute_blocks(blocks, x, masks, density, mode):
    for i, block in enumerate(blocks):
        if mode == "dense":
            x = x + block(x)
        elif mode == "full_mask":
            x = x + block(x) * masks[i, :, None, None] / density
        elif mode == "gather":
            idx = torch.nonzero(masks[i], as_tuple=True)[0]
            if len(idx):
                x = x.index_add(0, idx, block(x.index_select(0, idx))/density)
        elif mode == "batch_skip":
            if masks[i, 0]:
                x = x + block(x)/density
        else:
            raise ValueError(mode)
    return x


def benchmark():
    seed, density, depth = 713, .5, 6
    torch.manual_seed(seed)
    equivalence = []
    # Forward AND backward check with shared masks and float64.
    for kind in ["sequence", "batch"]:
        blocks = nn.ModuleList([Block(8).double() for _ in range(3)])
        x = torch.randn(5, 4, 8, dtype=torch.float64, requires_grad=True)
        masks = torch.rand(3, 5) < density
        masks[0] = False  # exercise all-dropped layer
        masks[2] = True   # and all-kept layer
        if kind == "batch":
            masks = masks[:, :1].expand(-1, 5)
        results = []
        for mode in (["full_mask", "gather"] if kind == "sequence" else ["full_mask", "gather", "batch_skip"]):
            for param in blocks.parameters():
                param.grad = None
            x.grad = None
            y = execute_blocks(blocks, x, masks, density, mode)
            y.square().mean().backward()
            grads = torch.cat([x.grad.flatten()] + [(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                                                     for p in blocks.parameters()])
            results.append((mode, y.detach().clone(), grads.detach().clone()))
        for mode, y, g in results[1:]:
            ferr = float((y-results[0][1]).abs().max())
            gerr = float((g-results[0][2]).abs().max())
            assert ferr < 1e-10 and gerr < 1e-10
            equivalence.append({"mask_granularity": kind, "comparison": f"full_mask vs {mode}",
                                "max_forward_error": ferr, "max_gradient_error": gerr})

    rows = []
    rng = np.random.default_rng(seed)
    for label, B, S, D in [("small", 8, 16, 32), ("medium", 32, 32, 64)]:
        torch.manual_seed(seed)
        blocks = nn.ModuleList([Block(D) for _ in range(depth)])
        x = torch.randn(B, S, D)
        # 16 fixed masks used in every repetition. Exactly half of the work remains
        # for a fair matched-work benchmark; these balanced masks are NOT Bernoulli.
        seq_masks = []
        batch_masks = []
        for _ in range(16):
            m = torch.zeros(depth, B, dtype=torch.bool)
            for l in range(depth):
                m[l, torch.randperm(B)[:B//2]] = True
            seq_masks.append(m)
            bm = torch.zeros(depth, B, dtype=torch.bool)
            bm[torch.randperm(depth)[:depth//2]] = True
            batch_masks.append(bm)
        tasks = [("dense", "none"), ("full_mask", "sequence"), ("gather", "sequence"),
                 ("full_mask", "batch"), ("gather", "batch"), ("batch_skip", "batch")]
        for phase in ["forward", "forward_backward"]:
            def one(mode, granularity, i):
                masks = seq_masks[i] if granularity != "batch" else batch_masks[i]
                if phase == "forward":
                    with torch.no_grad():
                        execute_blocks(blocks, x, masks, density, mode)
                else:
                    for param in blocks.parameters():
                        param.grad = None
                    execute_blocks(blocks, x, masks, density, mode).square().mean().backward()
            for mode, granularity in tasks:
                for i in range(4):
                    one(mode, granularity, i)
            for rep in range(9):
                # randomized interleaving mitigates simple temperature/order drift.
                for idx in rng.permutation(len(tasks)):
                    mode, granularity = tasks[int(idx)]
                    start = time.perf_counter_ns()
                    for i in range(16):
                        one(mode, granularity, i)
                    elapsed = (time.perf_counter_ns()-start)/1e6/16
                    rows.append({"shape": label, "batch": B, "sequence": S, "width": D,
                                 "depth": depth, "phase": phase, "mode": mode,
                                 "mask_granularity": granularity, "repetition": rep,
                                 "ms_per_execution": elapsed,
                                 "nominal_active_fraction": 1 if mode == "dense" else density,
                                 "executed_block_matmul_fraction": 1 if mode in ["dense", "full_mask"] else density})
    summaries = []
    for label in ["small", "medium"]:
        for phase in ["forward", "forward_backward"]:
            dense_ms = statistics.median(r["ms_per_execution"] for r in rows
                                         if r["shape"] == label and r["phase"] == phase and r["mode"] == "dense")
            for mode, granularity in tasks:
                values = [r["ms_per_execution"] for r in rows if r["shape"] == label and r["phase"] == phase
                          and r["mode"] == mode and r["mask_granularity"] == granularity]
                median = statistics.median(values)
                summaries.append({"shape": label, "phase": phase, "mode": mode, "mask_granularity": granularity,
                                  "median_ms": median, "min_ms": min(values), "max_ms": max(values),
                                  "speedup_vs_dense": dense_ms/median, "repetitions": len(values),
                                  "executions_per_repetition": 16})
    write_csv(RAW / "timing_raw.csv", rows)
    write_csv(RAW / "timing_summary.csv", summaries)
    return {"protocol": "CPU, float32, 1 PyTorch thread, GELU MLP residual blocks; 4 warmups then 9 randomized-order repetitions of 16 executions. Fixed balanced masks retain exactly half of sequence-block or batch-block work; they isolate execution overhead, not Bernoulli variance. Masks are precomputed. Backward excludes optimizer update. Dense uses no dropout scaling. No compilation.",
            "seed": seed, "density": density, "equivalence_checks": equivalence, "summaries": summaries,
            "limitations": "This is a local CPU MLP measurement, not an attention, GPU, CS-3, energy, or distributed benchmark. Same semantic output/gradient is checked only between implementations using the same masks; sequence and batch masks define different stochastic models. Zero and absent gradients are mathematically equivalent for this check but can differ in optimizer state updates."}


class Regressor(nn.Module):
    def __init__(self, depth=6, width=24):
        super().__init__()
        self.depth = depth
        self.embed = nn.Linear(8, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, 1)
        self.scale = 1/math.sqrt(depth)

    def forward(self, x, probabilities=None, mask_generator=None, retained_depth=None):
        h = torch.tanh(self.embed(x))
        active = 0
        for l, block in enumerate(self.blocks):
            if retained_depth is not None and l >= retained_depth:
                continue
            p = 0 if probabilities is None else float(probabilities[l])
            if p:
                mask = torch.rand((len(x), 1), generator=mask_generator) >= p
                active += int(mask.sum())
                # Deliberately compute all rows: training's active-work budget is
                # a MODEL of ideal skipping and is never labelled measured FLOPs.
                h = h + self.scale*torch.tanh(block(h))*mask/(1-p)
            else:
                active += len(x)
                h = h + self.scale*torch.tanh(block(h))
        return self.head(h).squeeze(-1), active


def teacher(x):
    return (torch.sin(x[:, 0] + .5*x[:, 1]) + .5*torch.cos(x[:, 2]-x[:, 3])
            + .25*x[:, 4]*x[:, 5] + .1*x[:, 6]**2 - .2*x[:, 7])


CONFIGS = {
    "dense6": {"depth": 6, "schedule": "none", "pmax": 0., "mean_dropout": 0.},
    "constant_ild": {"depth": 6, "schedule": "constant", "pmax": .4, "mean_dropout": .2},
    "decreasing_ild": {"depth": 6, "schedule": "decreasing", "pmax": .8, "mean_dropout": .2},
    "increasing_ild": {"depth": 6, "schedule": "increasing", "pmax": .8, "mean_dropout": .2},
    "dense5": {"depth": 5, "schedule": "none", "pmax": 0., "mean_dropout": 0.},
}


def train_run(name, regime, lr, seed, base_steps=240, record_curve=False):
    c = CONFIGS[name]
    steps = base_steps if regime == "matched_steps" else round(base_steps*6/(c["depth"]*(1-c["mean_dropout"])))
    torch.manual_seed(seed)
    model = Regressor(depth=c["depth"])
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    data_gen = torch.Generator().manual_seed(seed + 100000)
    mask_gen = torch.Generator().manual_seed(seed + 200000)
    # One common prefix of fresh data for each paired seed, independent of masks.
    xs = torch.randn(steps, 64, 8, generator=data_gen)
    # Targets are noiseless: no duplicated finite training set and no label noise.
    ys = teacher(xs.reshape(-1, 8)).reshape(steps, 64)
    check_gen = torch.Generator().manual_seed(999)
    valx = torch.randn(1024, 8, generator=check_gen)
    testx = torch.randn(2048, 8, generator=check_gen)
    valy, testy = teacher(valx), teacher(testx)
    active_total, expected_total, curves = 0, 0., []
    start = time.perf_counter()
    for t in range(steps):
        decay = .1 + .9 * .5 * (1+math.cos(math.pi*t/(steps-1)))
        for group in optim.param_groups:
            group["lr"] = lr*decay
        factor = {"none": 0., "constant": 1., "decreasing": 1-t/(steps-1),
                  "increasing": t/(steps-1)}[c["schedule"]]
        ps = c["pmax"]*np.arange(c["depth"])/(c["depth"]-1)*factor
        optim.zero_grad(set_to_none=True)
        pred, active = model(xs[t], ps, mask_gen)
        loss = (pred-ys[t]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optim.step()
        active_total += active
        expected_total += float((1-ps).sum())*64
        if record_curve and (t == 0 or (t+1) % 24 == 0 or t == steps-1):
            with torch.no_grad():
                val_loss = float((model(valx)[0]-valy).square().mean())
            curves.append({"step": t+1, "dense_equivalent_active_steps": active_total/(6*64),
                           "validation_mse": val_loss, "mean_dropout": float(ps.mean())})
    seconds = time.perf_counter()-start
    with torch.no_grad():
        val_loss = float((model(valx)[0]-valy).square().mean())
        test_loss = float((model(testx)[0]-testy).square().mean())
        exit_loss = float((model(testx, retained_depth=4)[0]-testy).square().mean())
    row = {"config": name, "regime": regime, "seed": seed, "lr": lr, "steps": steps,
           "depth": c["depth"], "validation_mse": val_loss, "test_mse": test_loss,
           "early_exit_depth4_test_mse": exit_loss, "parameter_count": sum(p.numel() for p in model.parameters()),
           "realized_active_sequence_blocks": active_total,
           "expected_active_sequence_blocks": expected_total,
           "expected_budget_relative_to_dense6": expected_total/(base_steps*6*64),
           "executed_sequence_blocks": steps*64*c["depth"], "train_seconds": seconds}
    return row, curves


def training():
    learning_rates = [.001, .003, .01, .03, .1]
    tuning_seeds, evaluation_seeds = [10, 11], [100, 101, 102, 103, 104, 105]
    tuning_rows, evaluation_rows, selected_lrs, curves = [], [], [], []
    for regime in ["matched_steps", "matched_expected_active_budget"]:
        for name in CONFIGS:
            for lr in learning_rates:
                for seed in tuning_seeds:
                    row, _ = train_run(name, regime, lr, seed)
                    tuning_rows.append(row)
            best_lr = min(learning_rates, key=lambda lr: statistics.mean(r["validation_mse"] for r in tuning_rows
                          if r["config"] == name and r["regime"] == regime and r["lr"] == lr))
            selected_lrs.append({"config": name, "regime": regime, "selected_lr": best_lr})
            print(f"tuned {regime} {name}: lr={best_lr}", flush=True)
            for seed in evaluation_seeds:
                row, curve = train_run(name, regime, best_lr, seed, record_curve=True)
                evaluation_rows.append(row)
                curves.append({"config": name, "regime": regime, "seed": seed, "points": curve})
            write_csv(RAW / "training_tuning.csv", tuning_rows)
            write_csv(RAW / "training_evaluation.csv", evaluation_rows)
    summaries, pairs = [], []
    for regime in ["matched_steps", "matched_expected_active_budget"]:
        dense = {r["seed"]: r["test_mse"] for r in evaluation_rows if r["regime"] == regime and r["config"] == "dense6"}
        for name in CONFIGS:
            rows = [r for r in evaluation_rows if r["config"] == name and r["regime"] == regime]
            summaries.append({"config": name, "regime": regime, "test_mse": ci95(r["test_mse"] for r in rows),
                              "early_exit_depth4_test_mse": ci95(r["early_exit_depth4_test_mse"] for r in rows),
                              "expected_budget_relative_to_dense6": statistics.mean(r["expected_budget_relative_to_dense6"] for r in rows),
                              "steps": rows[0]["steps"], "selected_lr": rows[0]["lr"],
                              "median_train_seconds": statistics.median(r["train_seconds"] for r in rows)})
            if name != "dense6":
                pairs.append({"config": name, "regime": regime,
                              "paired_test_mse_difference_vs_dense6": ci95(r["test_mse"]-dense[r["seed"]] for r in rows)})
    write_json(RAW / "training_curves.json", curves)
    return {"protocol": {"task": "Online fresh-sample noiseless 8D regression; target sin(x0+.5*x1)+.5*cos(x2-x3)+.25*x4*x5+.1*x6^2-.2*x7, inputs iid N(0,1).",
                         "architecture": "Trainable 8→24 tanh embedding, 6 or 5 residual tanh-linear 24→24 blocks scaled by 1/sqrt(depth), linear scalar head; no attention or normalization.",
                         "optimizer": "AdamW(weight_decay=.01), cosine LR to 10% of initial LR, batch=64, base_steps=240; no gradient clipping or early stopping.",
                         "mask": "Independent Bernoulli per sample and residual block; inverted 1/(1-p) scaling in training, scale=1 at evaluation. Compute-then-mask implementation.",
                         "tuning": "Equal 5 learning rates × 2 tuning seeds independently for every config/regime; select by mean MSE on fixed 1,024-point validation set. Evaluation uses 6 disjoint seeds and a fixed separate 2,048-point test set. Initial 3-point pilot hit the upper boundary for every treatment; grid expanded equally to include .03 and .1 before interpreting final comparisons.",
                         "learning_rates": learning_rates, "tuning_seeds": tuning_seeds, "evaluation_seeds": evaluation_seeds,
                         "budget": "Matched steps = 240 for every architecture. Matched expected active sequence-block budget = 240*6*64; dropout configurations get 300 steps; dense5 gets 288. Excludes embedding, head, optimizer, and mask overhead. Actual execution computes every row; budget is conceptual, NOT measured FLOPs or elapsed time.",
                         "confidence": "Two-sided Student-t 95% intervals over 6 evaluation seeds (df=5); paired intervals use per-seed MSE differences. No multiple-comparison correction; these characterize this toy protocol only."},
            "configs": CONFIGS, "selected_lrs": selected_lrs, "summaries": summaries,
            "paired_differences": pairs, "curves": curves,
            "limitations": "A tanh residual MLP is not a transformer; seeds share evaluation data, intervals capture initialization/data-stream/mask variability only. LR grid is small and selection can land on its boundary. This does not test CompleteP transfer, language loss, token scaling, CS-3 speed, speculative decoding or universal optimal schedules. Sparse toy training uses compute-then-mask, so the budget is theoretical active work."}


def metadata():
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu", "precision": "float32 training/timing; float64 exact/equivalence checks",
            "source_paper": "https://arxiv.org/abs/2609.05275", "source_version": "v1, 2026-09-04"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["all", "exact", "timing", "training"], default="all")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    path = ROOT / "docs" / "data" / "experiment-results.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result["metadata"] = metadata()
    start = time.perf_counter()
    for key, fn in [("exact", exact_checks), ("timing", benchmark), ("training", training)]:
        if args.part in ["all", key]:
            result[key] = fn()
            write_json(path, result)
            write_json(RAW / f"{key}_results.json", result[key])
            print(f"completed {key}: {time.perf_counter()-start:.1f}s cumulative", flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
