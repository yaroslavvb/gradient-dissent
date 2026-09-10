#!/usr/bin/env python3
"""Independent CPU/float32 reference for the browser residual-MLP trainer.

No repository model, optimizer, training helper, or NumPy implementation is used.
Weight layout is PyTorch [out_features, in_features], flattened row-major.
Every mask is an ACTIVE mask: bit 0 controls the shallowest middle branch.
All four branches are eligible for 50% dropout in training; test gain is one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


WIDTHS = [3, 4, 3, 2, 2, 2, 2]
LR = 0.03
MOMENTUM = 0.9
X = torch.tensor([
    [0.25, -0.5, 0.75],
    [1.0, 0.2, -0.3],
    [-0.4, 0.8, 0.1],
    [0.6, 0.6, -0.2],
], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.int64)


def initialize():
    params = {}
    for layer, (n_in, n_out) in enumerate(zip(WIDTHS, WIDTHS[1:])):
        # Fixed rational values, no random generator and no implicit initializer.
        # Layer-specific patterns expose slot mix-ups between equal-shaped layers.
        weights = [[
            (((layer + 1) * 7 + row * 5 + column * 3) % 17 - 8) / 32
            + (0.125 if row == column else 0.0)
            for column in range(n_in)
        ] for row in range(n_out)]
        biases = [((layer * 3 + row * 2) % 7 + 2) / 32 for row in range(n_out)]
        params[f"layer_{layer}.weight"] = torch.tensor(
            weights, dtype=torch.float32, requires_grad=True)
        params[f"layer_{layer}.bias"] = torch.tensor(
            biases, dtype=torch.float32, requires_grad=True)
    return params


def forward(params, active_mask=15, training=True):
    def affine(value, layer):
        return F.linear(value, params[f"layer_{layer}.weight"],
                        params[f"layer_{layer}.bias"])

    h = F.relu(affine(X, 0))
    for branch in range(4):
        layer = branch + 1
        bypass = h[:, :WIDTHS[layer + 1]]
        if not training or (active_mask >> branch) & 1:
            h = F.relu(bypass + affine(h, layer) * (2.0 if training else 1.0))
        else:
            # Actual control-flow bypass: no affine call, no gradient connection.
            h = F.relu(bypass)
    return affine(h, 5)


def tensor_record(tensor):
    if tensor is None:
        return None
    value = tensor.detach().cpu()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def snapshot(mapping):
    return {name: tensor_record(value) for name, value in mapping.items()}


def gradients(params, logits):
    loss = F.cross_entropy(logits, Y, reduction="mean")
    values = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    grads = dict(zip(params, values))
    assert torch.isfinite(loss)
    assert all(g is None or torch.isfinite(g).all() for g in grads.values())
    return loss, grads


def branch_parameter_names(active_mask):
    return [f"layer_{branch + 1}.{kind}"
            for branch in range(4) if not (active_mask >> branch) & 1
            for kind in ("weight", "bias")]


def generate():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = initialize()
    training_cases = []
    for mask in range(16):
        params = initialize()
        logits = forward(params, mask, training=True)
        loss, grads = gradients(params, logits)
        skipped = branch_parameter_names(mask)
        assert {name for name, grad in grads.items() if grad is None} == set(skipped)
        training_cases.append({
            "active_mask": mask,
            "active_bits_shallow_to_deep": [(mask >> b) & 1 for b in range(4)],
            "logits": tensor_record(logits),
            "mean_cross_entropy": loss.item(),
            "gradients": snapshot(grads),
            "skipped_parameters": skipped,
        })

    params = initialize()
    logits = forward(params, training=False)
    loss, grads = gradients(params, logits)
    inference_case = {
        "active_mask": 15,
        "branch_gain": 1,
        "logits": tensor_record(logits),
        "mean_cross_entropy": loss.item(),
        "gradients": snapshot(grads),
    }

    params = initialize()
    velocity = {name: torch.zeros_like(p) for name, p in params.items()}
    steps = []
    frozen_checks = 0
    frozen_nonzero_velocity_checks = 0
    for step, mask in enumerate([15, 0, 5, 10, 15]):
        before_params = {name: p.detach().clone() for name, p in params.items()}
        before_velocity = {name: v.clone() for name, v in velocity.items()}
        logits = forward(params, mask, training=True)
        loss, grads = gradients(params, logits)
        skipped = branch_parameter_names(mask)
        with torch.no_grad():
            for name, p in params.items():
                grad = grads[name]
                if grad is None:
                    continue
                # Explicit parameter-keyed state, preserving PyTorch SGD's
                # skip-on-None semantics without invoking a library optimizer.
                next_velocity = velocity[name] * MOMENTUM + grad
                p.copy_(p - LR * next_velocity)
                velocity[name].copy_(next_velocity)
        for name in skipped:
            assert torch.equal(params[name], before_params[name])
            assert torch.equal(velocity[name], before_velocity[name])
            frozen_checks += 1
            frozen_nonzero_velocity_checks += int(torch.count_nonzero(velocity[name]) > 0)
        steps.append({
            "step": step + 1,
            "active_mask": mask,
            "logits_before_update": tensor_record(logits),
            "mean_cross_entropy_before_update": loss.item(),
            "gradients": snapshot(grads),
            "parameters_after_update": snapshot(params),
            "velocities_after_update": snapshot(velocity),
            "skipped_parameters": skipped,
            "skipped_parameters_and_velocities_exactly_frozen": True,
        })
    assert frozen_nonzero_velocity_checks > 0
    final_test_logits = forward(params, training=False)

    return {
        "schema": "gradient-dissent-browser-numerical-reference-v1",
        "provenance": {
            "generator": Path(__file__).name,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "independent_of_repository_model_and_optimizer": True,
        },
        "architecture": {
            "widths": WIDTHS,
            "affine_count": 6,
            "droppable_affine_indices": [1, 2, 3, 4],
            "weight_layout": "out_features,in_features; row-major flat values",
            "stem": "relu(affine_0(input))",
            "kept_middle_training": "relu(crop(h,out_features) + 2*affine_i(h))",
            "dropped_middle_training": "relu(crop(h,out_features))",
            "middle_evaluation": "relu(crop(h,out_features) + affine_i(h))",
            "head": "affine_5(h); linear logits",
            "mask_definition": "ACTIVE mask; bit 0 is affine_1, shallowest branch",
            "training_drop_probabilities": [0.5, 0.5, 0.5, 0.5],
            "evaluation_keep_probabilities": [1, 1, 1, 1],
            "cross_entropy_reduction": "mean across examples",
        },
        "inputs": tensor_record(X),
        "labels": tensor_record(Y),
        "parameter_order": list(initial),
        "initial_parameters": snapshot(initial),
        "training_cases": training_cases,
        "inference_case": inference_case,
        "momentum_sequence": {
            "learning_rate": LR,
            "momentum": MOMENTUM,
            "dampening": 0,
            "nesterov": False,
            "weight_decay": 0,
            "initial_velocities": snapshot({name: torch.zeros_like(p) for name, p in initial.items()}),
            "update_rule": "if gradient exists: v=0.9*v+g; parameter=parameter-0.03*v; otherwise freeze both",
            "steps": steps,
            "skipped_exact_freeze_checks": frozen_checks,
            "skipped_nonzero_velocity_freeze_checks": frozen_nonzero_velocity_checks,
            "full_depth_logits_after_sequence": tensor_record(final_test_logits),
            "full_depth_mean_cross_entropy_after_sequence": F.cross_entropy(final_test_logits, Y).item(),
        },
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("reference-fixture.json")
    document = generate()
    out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "fixture": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
        "cases": len(document["training_cases"]),
        "momentum_steps": len(document["momentum_sequence"]["steps"]),
        "skipped_exact_freeze_checks": document["momentum_sequence"]["skipped_exact_freeze_checks"],
        "skipped_nonzero_velocity_freeze_checks": document["momentum_sequence"]["skipped_nonzero_velocity_freeze_checks"],
    }, indent=2))
