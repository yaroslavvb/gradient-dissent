"""Bounded, post-ReLU mechanism probes for the four-branch residual MLP.

No fitting or parameter mutation. forward_fn(model, x, gains) must implement
deterministic inference with affine biases included in the branch gain.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .metrics import mobius_transform, spearman_stats
except ImportError:
    from metrics import mobius_transform, spearman_stats


EPSILONS = (0.001, 0.01)
GAUSSIAN_SEED = 20260911
CENTRAL_RECIPES = {"residual", "sd_constant", "sd_annealed"}


def _array(tensor):
    return tensor.detach().double().cpu().numpy()


def _center(value):
    return value - value.mean(axis=-1, keepdims=True)


def _ce(logits, labels):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return np.log(np.exp(shifted).sum(axis=-1)) - shifted[np.arange(len(labels)), labels]


def _rms_rows(values):
    return np.sqrt(np.mean(values ** 2, axis=-1))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "rms": None, "mean_absolute": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "mean_absolute": float(np.abs(values).mean())}


def multilinear_weights(gains):
    """Weights for corner interpolation (or extrapolation outside [0,1])."""
    if len(gains) != 4:
        raise ValueError("Expected four gains")
    return np.asarray([math.prod(gains[j] if mask & (1 << j) else 1-gains[j]
                                 for j in range(4)) for mask in range(16)])


def survival_probabilities(recipe, checkpoint_epoch):
    if not isinstance(checkpoint_epoch, int) or not 1 <= checkpoint_epoch <= 100:
        raise ValueError("checkpoint_epoch must be an integer in 1..100")
    if recipe == "residual":
        return [1.] * 4
    if recipe not in CENTRAL_RECIPES:
        raise ValueError("No central SD survival rule for this recipe")
    pmax = 0.4 if recipe == "sd_constant" else 0.8 * (1-(checkpoint_epoch-1)/99)
    return [1-pmax*(j+1)/4 for j in range(4)]


def _suffix(model, state, branch_index):
    """Continue immediately after eligible branch branch_index."""
    h = state
    for layer in list(model.layers)[branch_index+2:-1]:
        h = F.relu(h[..., :layer.out_features] + layer(h))
    logits = model.layers[-1](h)
    return F.relu(logits) if model.output_relu else logits


def _prediction_summary(predicted, actual, norms, angles, select):
    p, a = predicted[select], actual[select]
    error = p-a
    denom = float(np.sum(a*a))
    valid_angle = select & np.isfinite(angles)
    return {"n": int(select.sum()), "first_order_minus_actual": _summary(error),
            "first_order_r2_vs_zero_change": 1-float(np.sum(error*error))/denom if denom > 0 else None,
            "zero_change_squared_error_denominator": denom,
            "sign_agreement": float(np.mean(np.sign(p) == np.sign(a))) if len(a) else None,
            "gradient_vs_damage_spearman": spearman_stats(p, a),
            "relative_norm_vs_damage_spearman": spearman_stats(norms[select], actual[select]),
            "angle_vs_damage_spearman": spearman_stats(angles[valid_angle], actual[valid_angle])}


def _interpolation_comparison(actual, labels, gains, corners, corner_ce, logit_mobius, ce_mobius):
    centered = _center(actual)
    weights = multilinear_weights(gains)
    interpolated = np.tensordot(weights, corners, axes=(0, 0))
    interpolated_ce = weights @ corner_ce
    first_degree = logit_mobius[0].copy()
    first_degree_ce = ce_mobius[0].copy()
    for j in range(4):
        first_degree += gains[j]*logit_mobius[1 << j]
        first_degree_ce += gains[j]*ce_mobius[1 << j]
    actual_ce = _ce(actual, labels)
    records = {"actual_ce": actual_ce.tolist(),
               "centered_logit_multilinear_error_rms": _rms_rows(centered-interpolated).tolist(),
               "centered_logit_degree1_error_rms": _rms_rows(centered-first_degree).tolist(),
               "ce_minus_corner_ce_interpolant": (actual_ce-interpolated_ce).tolist(),
               "ce_minus_ce_of_interpolated_logits": (actual_ce-_ce(interpolated, labels)).tolist(),
               "ce_minus_degree1_corner_ce": (actual_ce-first_degree_ce).tolist()}
    return {"gains": list(gains), "per_example": records,
            "summary": {key: _summary(value) for key, value in records.items()}}


def _run(model, x, y, corner_logits, checkpoint_epoch, recipe, forward_fn):
    if not model.residual or len(model.layers) != 6:
        raise ValueError("Mechanistic probes require four eligible residual branches")
    n = len(x)
    if not 1 <= n <= 128 or tuple(corner_logits.shape[:2]) != (16, n):
        raise ValueError("Expected 1..128 examples and matching [16,N,C] corner logits")
    labels = _array(y).astype(np.int64)
    raw_corners = _array(corner_logits)
    if raw_corners.ndim != 3 or raw_corners.shape[2] < 2 or not np.isfinite(raw_corners).all():
        raise ValueError("Invalid corner logits")
    if labels.shape != (n,) or np.any(labels < 0) or np.any(labels >= raw_corners.shape[2]):
        raise ValueError("Invalid labels")
    corners = _center(raw_corners)
    corner_ce = np.asarray([_ce(row, labels) for row in raw_corners])
    dense_correct = raw_corners[15].argmax(-1) == labels
    rng = torch.Generator(device="cpu").manual_seed(GAUSSIAN_SEED)

    # A single backward traversal yields all four local downstream gradients.
    # Clone allows callers to supply tensors created inside inference_mode.
    x_grad, y_grad = x.detach().clone(), y.detach().clone()
    stem = F.relu(model.layers[0](x_grad.reshape(n, model.widths[0])*model.input_scale))
    h = stem.detach().requires_grad_(True)
    states, bypasses = [], []
    for layer in list(model.layers)[1:-1]:
        bypass = F.relu(h[..., :layer.out_features])
        h = F.relu(h[..., :layer.out_features]+layer(h))
        states.append(h)
        bypasses.append(bypass)
    logits = model.layers[-1](h)
    if model.output_relu:
        logits = F.relu(logits)
    # Fail on a different forward convention, rather than silently treating
    # numerical/precision mismatch as an ablation effect.
    torch.testing.assert_close(logits.detach(), corner_logits[15].to(logits), rtol=1e-4, atol=1e-5)
    gradients = torch.autograd.grad(F.cross_entropy(logits, y_grad, reduction="sum"), states)
    local_states = [state.detach() for state in states]
    local_deltas = [(bypass-state).detach() for bypass, state in zip(bypasses, states)]
    first_orders = [_array((grad*delta).sum(-1)) for grad, delta in zip(gradients, local_deltas)]
    del states, bypasses, gradients, logits, h, stem

    blocks = []
    with torch.no_grad():
        for j, (state, delta, first_order) in enumerate(zip(local_states, local_deltas, first_orders)):
            state_np, delta_np = _array(state), _array(delta)
            bypass_np = state_np + delta_np
            state_norm = np.linalg.norm(state_np, axis=-1)
            bypass_norm = np.linalg.norm(bypass_np, axis=-1)
            delta_norm = np.linalg.norm(delta_np, axis=-1)
            relative_norm = delta_norm/np.maximum(state_norm, 1e-12)
            valid_angle = (state_norm > 0) & (bypass_norm > 0)
            angles = np.full(n, np.nan)
            angles[valid_angle] = np.arccos(np.clip(np.sum(state_np[valid_angle]*bypass_np[valid_angle], axis=-1)
                                             /(state_norm[valid_angle]*bypass_norm[valid_angle]), -1, 1))
            actual = corner_ce[15 ^ (1 << j)] - corner_ce[15]
            random = torch.randn(state.shape, generator=rng, dtype=torch.float32, device="cpu").to(state)
            random = random/random.norm(dim=-1, keepdim=True).clamp_min(1e-12)*delta.norm(dim=-1, keepdim=True)
            direction_records, response_vectors = {}, {}
            for direction_name, direction in (("deletion", delta), ("gaussian", random)):
                # One stacked suffix forward for both epsilon scales and signs.
                perturbed = torch.cat([state+sign*eps*direction for eps in EPSILONS for sign in (1, -1)], dim=0)
                shifted = _center(_array(_suffix(model, perturbed, j))).reshape(2, 2, n, -1)
                derivatives = [(shifted[k, 0]-shifted[k, 1])/(2*eps) for k, eps in enumerate(EPSILONS)]
                response_vectors[direction_name] = derivatives
                responses = {str(eps): np.linalg.norm(derivatives[k], axis=-1).tolist()
                             for k, eps in enumerate(EPSILONS)}
                disagreement = np.linalg.norm(derivatives[0]-derivatives[1], axis=-1)
                direction_records[direction_name] = {
                    "centered_logit_directional_response_l2": responses,
                    "two_epsilon_disagreement_l2": disagreement.tolist(),
                    "summary_response_l2": {key: _summary(value) for key, value in responses.items()},
                    "summary_disagreement_l2": _summary(disagreement)}
            delta_vs_gaussian = {
                str(eps): _summary(np.linalg.norm(response_vectors["deletion"][k], axis=-1)
                                  -np.linalg.norm(response_vectors["gaussian"][k], axis=-1))
                for k, eps in enumerate(EPSILONS)}
            blocks.append({"branch_index": j, "deleted_mask_id": 15 ^ (1 << j),
                           "post_relu_width": int(state.shape[-1]),
                           "per_example": {"delta_norm": delta_norm.tolist(), "full_state_norm": state_norm.tolist(),
                                           "bypass_norm": bypass_norm.tolist(), "relative_delta_norm": relative_norm.tolist(),
                                           "angle_radians": [float(a) if valid else None for a, valid in zip(angles, valid_angle)],
                                           "zero_full_state": (state_norm == 0).tolist(),
                                           "zero_bypass_state": (bypass_norm == 0).tolist(),
                                           "gradient_dot_delta": first_order.tolist(), "actual_deletion_excess_ce": actual.tolist()},
                           "all_examples": _prediction_summary(first_order, actual, relative_norm, angles, np.ones(n, bool)),
                           "full_correct_examples": _prediction_summary(first_order, actual, relative_norm, angles, dense_correct),
                           "finite_difference": direction_records,
                           "deletion_minus_gaussian_response_l2": delta_vs_gaussian})

        logit_mobius, ce_mobius = mobius_transform(corners), mobius_transform(corner_ce)
        interpolation = []
        for kind, branch in [("single_branch", j) for j in range(4)] + [("all_branch_diagonal", None)]:
            for alpha in (0., .25, .5, .75, 1.):
                gains = [alpha]*4 if branch is None else [alpha if j == branch else 1. for j in range(4)]
                if alpha in (0., 1.):
                    mask = sum((1 << j) for j, gain in enumerate(gains) if gain == 1)
                    actual_logits = raw_corners[mask]
                else:
                    actual_logits = _array(forward_fn(model, x, gains))
                interpolation.append({"path": kind, "branch_index": branch, "alpha": alpha,
                                      **_interpolation_comparison(actual_logits, labels, gains, corners, corner_ce,
                                                                  logit_mobius, ce_mobius)})

        survival = survival_probabilities(recipe, checkpoint_epoch)
        mean_bias = []
        if recipe.startswith("sd_"):
            for j, rho in enumerate(survival):
                gains = [1.]*4
                gains[j] = 1/rho
                retained = raw_corners[15] if rho == 1 else _array(forward_fn(model, x, gains))
                removed = raw_corners[15 ^ (1 << j)]
                expected_logits = rho*retained+(1-rho)*removed
                expected_ce = rho*_ce(retained, labels)+(1-rho)*_ce(removed, labels)
                ce_of_expected_logits = _ce(expected_logits, labels)
                records = {"centered_logit_mean_bias_rms": _rms_rows(_center(expected_logits)-corners[15]).tolist(),
                           "expected_ce_minus_dense_ce": (expected_ce-corner_ce[15]).tolist(),
                           "ce_of_expected_logits_minus_dense_ce": (ce_of_expected_logits-corner_ce[15]).tolist(),
                           "ce_jensen_gap": (expected_ce-ce_of_expected_logits).tolist()}
                mean_bias.append({"branch_index": j, "survival_probability": rho, "retained_gain": 1/rho,
                                  "per_example": records, "summary": {key: _summary(value) for key, value in records.items()}})

    return {"status": "complete", "recipe": recipe, "checkpoint_epoch": checkpoint_epoch,
            "n_examples": n, "full_correct_count": int(dense_correct.sum()),
            "full_correct": dense_correct.tolist(), "labels": labels.tolist(),
            "probe_order": "Caller-supplied fixed policy-fit prefix; records preserve its order",
            "gradient_convention": "CE sum gives per-example downstream gradients; delta is detached post-ReLU bypass minus full state",
            "finite_difference_convention": "Centered logits under post-block state +/- epsilon*direction; two steps; Gaussian direction norm matches delta",
            "gaussian_seed": GAUSSIAN_SEED, "epsilons": list(EPSILONS),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "autocast": False,
            "survival_mean_bias_convention": "One branch sampled at its checkpoint-epoch survival probability, all other branch gains fixed at one; not the joint training-mask expectation",
            "first_order_score_convention": "1-SSE(prediction)/SSE(zero change), not R-squared relative to a fitted mean",
            "inference": "Descriptive within-checkpoint probes; examples and layers are not independent trained-model replicates",
            "blocks": blocks, "fractional_gate_interpolation": interpolation,
            "survival_mean_bias": mean_bias}


def run_mechanisms(model, x, y, corner_logits, checkpoint_epoch, recipe, forward_fn):
    if recipe not in CENTRAL_RECIPES:
        return {"status": "excluded", "recipe": recipe,
                "reason": "Continuous-gain mechanism panel is restricted to the three central residual/SD recipes"}
    # Override an outer inference-mode context without changing model flags,
    # parameter values, RNG state, or accumulated parameter gradients.
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        result = _run(model, x, y, corner_logits, checkpoint_epoch, recipe, forward_fn)
    # Catch NaN/Inf before an expensive remote artifact is accepted.
    import json
    json.dumps(result, allow_nan=False)
    return result
