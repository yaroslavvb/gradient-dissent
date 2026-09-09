"""Deterministic algebraic counterexamples; standard-library CPU only.

Run from any directory:
    python experiments/ciresan_stochastic_depth/layerdrop_hypotheses/counterexamples.py

These constructed examples refute universal claims. They do not estimate how
often the corresponding failure modes occur in trained MNIST or language models.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def softplus(x: float) -> float:
    """Stable log(1 + exp(x)) for binary CE with true label one."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def mobius(values: list[float]) -> list[float]:
    """Presence-bit coefficients, indexed by integer subset bitmask."""
    count = len(values)
    if not count or count & (count - 1):
        raise ValueError("Expected a nonempty power-of-two table")
    coefficients = list(values)
    for bit in range(count.bit_length() - 1):
        for mask in range(count):
            if mask & (1 << bit):
                coefficients[mask] -= coefficients[mask ^ (1 << bit)]
    return coefficients


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def residual_reencoding() -> dict:
    """A 90-degree rotation is exactly one unrestricted residual block."""
    rotation = [[0.0, -1.0], [1.0, 0.0]]
    residual_matrix = [[-1.0, -1.0], [1.0, -1.0]]
    rows = []
    max_error = 0.0
    for point in itertools.product((-1.0, 1.0), repeat=2):
        rotated = [sum(a * x for a, x in zip(row, point)) for row in rotation]
        increment = [sum(a * x for a, x in zip(row, point)) for row in residual_matrix]
        full = [x + dx for x, dx in zip(point, increment)]
        target = int(-point[1] > 0)
        full_prediction, skipped_prediction = int(full[0] > 0), int(point[0] > 0)
        max_error = max(max_error, *(abs(a - b) for a, b in zip(full, rotated)))
        rows.append({"input": list(point), "target": target, "residual_increment": increment,
                     "full_state": full, "rotated_state": rotated,
                     "full_prediction": full_prediction, "skip_prediction": skipped_prediction})
    full_accuracy = sum(row["full_prediction"] == row["target"] for row in rows) / len(rows)
    skipped_accuracy = sum(row["skip_prediction"] == row["target"] for row in rows) / len(rows)
    require(max_error == 0, "Residual parameterization must equal the rotation")
    require(full_accuracy == 1 and skipped_accuracy == 0.5, "Unexpected corner classification")
    return {"claim_refuted": "Residual blocks cannot re-encode the coordinate system",
            "rotation_A": rotation, "residual_matrix_A_minus_I": residual_matrix,
            "classifier": "class 1 iff first output coordinate > 0",
            "target_rule": "class 1 iff second input coordinate < 0",
            "examples": rows, "max_absolute_identity_error": max_error,
            "full_accuracy": full_accuracy, "skipped_accuracy": skipped_accuracy,
            "accuracy_denominator": 4, "units": "accuracy is a fraction"}


def degree_one_damage() -> dict:
    """A degree-one loss in four presence bits can have unbounded damage."""
    cases = []
    for magnitude in (20.0, 200.0):
        logits = [magnitude * (2 * int(bool(mask & 1)) - 1) for mask in range(16)]
        losses = [softplus(-logit) for logit in logits]
        coefficients = mobius(losses)
        high_order_max = max(abs(value) for mask, value in enumerate(coefficients)
                             if mask.bit_count() >= 2)
        damage = losses[14] - losses[15]  # Remove only branch 0 from the full model.
        require(high_order_max == 0, "Loss must have no degree >= 2 terms")
        require(math.isclose(damage, magnitude, rel_tol=1e-14), "CE damage must equal magnitude")
        cases.append({"magnitude": magnitude, "logit_formula": "C * (2*m_0 - 1)",
                      "true_label": 1, "mask_losses": losses, "mobius_coefficients": coefficients,
                      "max_absolute_coefficient_degree_ge_2": high_order_max,
                      "full_ce": losses[15], "drop_branch_0_ce": losses[14],
                      "ce_damage": damage, "full_prediction_correct": logits[15] > 0,
                      "drop_branch_0_prediction_correct": logits[14] > 0})
    return {"claim_refuted": "Low interaction degree alone guarantees small deletion damage",
            "mask_convention": "integer mask; bit i equals presence of branch i; 15 is full",
            "number_of_presence_variables": 4, "cases": cases,
            "interpretation": "The same degree-one structure permits arbitrarily large positive C; "
                              "a coefficient magnitude or influence condition is additionally needed."}


def nonlinear_loss_interaction() -> dict:
    logits = [-2.0 + 2.0 * int(bool(mask & 1)) + 2.0 * int(bool(mask & 2))
              for mask in range(4)]
    losses = [softplus(-logit) for logit in logits]
    logit_coefficients, loss_coefficients = mobius(logits), mobius(losses)
    require(logit_coefficients[3] == 0, "Logits must be exactly additive in the gates")
    require(loss_coefficients[3] > 0.8, "Nonlinear CE must induce a pair interaction")
    return {"claim_refuted": "A high-order loss interaction necessarily means a serial network circuit",
            "logit_formula": "-2 + 2*m_0 + 2*m_1", "true_label": 1,
            "mask_convention": "integer mask; bit i equals presence of branch i; 3 is full",
            "mask_logits": logits, "mask_losses": losses,
            "logit_mobius_coefficients": logit_coefficients,
            "loss_mobius_coefficients": loss_coefficients,
            "pair_interaction_logit": logit_coefficients[3],
            "pair_interaction_ce": loss_coefficients[3],
            "interpretation": "CE curvature alone creates a nonzero second-order loss term."}


def euler_stability() -> dict:
    step, steps = 0.5, 20
    cases = []
    for label, eigenvalue in (("positive_real", complex(1, 0)), ("pure_imaginary", complex(0, 1))):
        factor = 1 + step * eigenvalue
        wrong_test = step * abs(eigenvalue) <= 2
        euler_absolute_stability = abs(factor) <= 1
        numerical_state = complex(1, 0)
        for _ in range(steps):
            numerical_state *= factor
        exact_norm = math.exp(eigenvalue.real * step * steps)
        require(wrong_test and not euler_absolute_stability, "The magnitude-only criterion must fail")
        require(abs(numerical_state) > 1, "Euler norm must grow")
        if label == "pure_imaginary":
            require(exact_norm == 1, "The imaginary-eigenvalue exact flow must preserve norm")
        cases.append({"case": label, "lambda_real": eigenvalue.real, "lambda_imag": eigenvalue.imag,
                      "step_size": step, "steps": steps,
                      "tau_abs_lambda": step * abs(eigenvalue),
                      "passes_incomplete_tau_abs_lambda_le_2_test": wrong_test,
                      "abs_one_plus_tau_lambda": abs(factor),
                      "satisfies_euler_absolute_stability": euler_absolute_stability,
                      "final_euler_norm": abs(numerical_state), "final_exact_norm": exact_norm})
    return {"claim_refuted": "Forward Euler is stable whenever tau*abs(lambda) <= 2",
            "correct_scalar_condition": "abs(1 + tau*lambda) <= 1", "cases": cases,
            "interpretation": "The positive-real exact ODE also grows, so that case only exposes "
                              "the omitted sign/phase condition. The imaginary case demonstrates "
                              "numerical growth for an exactly norm-preserving flow."}


def deletion_changes_horizon() -> dict:
    """Constant vector field makes the horizon distinction exact, with no truncation error."""
    original_steps, retained_steps, step = 10, 5, 0.1
    full = math.fsum([step] * original_steps)
    deleted = math.fsum([step] * retained_steps)
    coarsened_step = original_steps * step / retained_steps
    coarsened = math.fsum([coarsened_step] * retained_steps)
    require(full == 1 and deleted == 0.5 and coarsened == 1, "Unexpected integration endpoint")
    return {"claim_refuted": "Deleting Euler updates is automatically a coarser integration of the same horizon",
            "ode": "y'=1; y(0)=0", "original_steps": original_steps, "retained_steps": retained_steps,
            "original_step_size": step, "fixed_horizon_coarsened_step_size": coarsened_step,
            "full_endpoint": full, "deleted_unscaled_endpoint": deleted,
            "fixed_horizon_coarsened_endpoint": coarsened,
            "full_horizon": original_steps * step, "deleted_unscaled_horizon": retained_steps * step,
            "interpretation": "Unscaled deletions omit half the accumulated flow. Keeping the original "
                              "horizon requires doubling retained step sizes in this example."}


def build_results() -> dict:
    return {"schema_version": 1,
            "purpose": "Constructed counterexamples to universal theoretical claims, not trained-model estimates",
            "execution": "Python standard library; CPU binary64 arithmetic; no randomness or training",
            "external_spend_usd": 0,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "all_checks_passed": True,
            "residual_reencoding": residual_reencoding(),
            "degree_one_damage": degree_one_damage(),
            "additive_logits_nonlinear_ce": nonlinear_loss_interaction(),
            "euler_stability": euler_stability(),
            "deletion_changes_horizon": deletion_changes_horizon()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "hypotheses" / "theory-counterexamples.json")
    args = parser.parse_args()
    results = build_results()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_checks_passed": results["all_checks_passed"],
                      "cases": 5, "external_spend_usd": 0}))


if __name__ == "__main__":
    main()
