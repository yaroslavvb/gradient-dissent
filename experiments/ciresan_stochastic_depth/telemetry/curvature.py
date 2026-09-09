"""Offline affine-block Fisher / CE-GGN / Jacobian diagnostics.

No training, jobs, logging, or parameter-square curvature matrices. Intended for
small fixed probes of example-independent dense MLPs at saved checkpoints.
All model derivatives use the model's dtype; moment/spectrum reductions use
float64. autograd.grad never writes parameter .grad or graph-owned buffers.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


def _number(value):
    value = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    return value if math.isfinite(value) else None


def _ratio(numerator, denominator):
    a, b = _number(numerator), _number(denominator)
    return None if a is None or b is None or b <= 0 else _number(a / b)


def _summary(values):
    values = values.detach().double().flatten()
    finite = torch.isfinite(values)
    kept = values[finite]
    if not len(kept):
        return {"n": len(values), "valid_n": 0, "mean": None, "rms": None,
                "min": None, "median": None, "max": None}
    return {"n": len(values), "valid_n": int(finite.sum()),
            "mean": _number(kept.mean()), "rms": _number(kept.square().mean().sqrt()),
            "min": _number(kept.min()), "median": _number(kept.median()),
            "max": _number(kept.max())}


def _array_hash(value):
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stable_output_statistics(logits, targets):
    """Explicit cancellation-resistant CE derivative and PSD Hessian factor.

    R[q,c] = sqrt(p[q]) * (1[q=c] - p[c]); R.T @ R is CE's
    logit Hessian. Compute diagonal complements from the OTHER probabilities,
    rather than 1-p, to retain tiny positive curvature at high confidence.
    """
    probabilities = logits.detach().double().softmax(-1)
    n, classes = probabilities.shape
    one_hot = F.one_hot(targets, classes).bool()
    gradient = probabilities.clone()
    gradient[one_hot] = -(probabilities.masked_fill(one_hot, 0).sum(-1))
    root = -probabilities[:, None, :].expand(n, classes, classes).clone()
    for q in range(classes):
        root[:, q, q] = probabilities[:, torch.arange(classes, device=logits.device) != q].sum(-1)
    root *= probabilities.sqrt()[:, :, None]
    return probabilities, gradient, root


def _spectrum(factor, *, eig_max_dimension, power_iterations=32):
    """Spectrum of factor.T @ factor via its smaller covariance/Gram matrix.

    Exact trace/Frobenius norm in both modes. Above the configured eigenproblem
    limit, only the top eigenvalue is estimated by deterministic power iteration;
    no truncated list is represented as a complete exact eigenspectrum.
    """
    factor = factor.double()
    rows, columns = factor.shape
    left = rows <= columns
    gram = factor @ factor.T if left else factor.T @ factor
    gram = (gram + gram.T) * .5
    trace = gram.diagonal().sum()
    squared_fro = gram.square().sum()
    dimension = len(gram)
    info = {"factor_rows": rows, "factor_columns": columns,
            "eigenproblem_dimension": dimension, "matrix_dimension": columns,
            "trace": _number(trace), "frobenius_norm": _number(squared_fro.sqrt()),
            "frobenius_norm_squared": _number(squared_fro),
            "implicit_zero_eigenvalue_count": max(0, columns - dimension)}
    if dimension <= eig_max_dimension:
        raw = torch.linalg.eigvalsh(gram).flip(0)
        scale = raw.abs().max() if len(raw) else torch.zeros((), device=gram.device)
        tolerance = scale * max(rows, columns) * torch.finfo(torch.float64).eps * 8
        if bool((raw < -tolerance).any()):
            raise FloatingPointError("Factor Gram has a negative eigenvalue beyond roundoff")
        values = raw.clamp_min(0)
        top = values[0]
        info.update({"method": "exact_smaller_gram_eigvalsh", "eigenvalues": values.cpu().tolist(),
                     "negative_roundoff_eigenvalues": int((raw < 0).sum()),
                     "rank_tolerance": _number(tolerance),
                     "numerical_rank": int((values > tolerance).sum()),
                     "top_eigenvalue_residual": None})
    else:
        # Multiple deterministic starts avoid assuming that an all-ones vector
        # overlaps the leading eigenspace. Still an estimate, not a certificate.
        gen = torch.Generator(device="cpu").manual_seed(20260909)
        vectors = torch.randn((dimension, min(4, dimension)), generator=gen,
                              dtype=torch.float64).to(gram.device)
        vectors = torch.linalg.qr(vectors, mode="reduced").Q
        for _ in range(power_iterations):
            vectors = torch.linalg.qr(gram @ vectors, mode="reduced").Q
        small = vectors.T @ gram @ vectors
        evals, evecs = torch.linalg.eigh((small + small.T) * .5)
        top = evals[-1].clamp_min(0)
        vector = vectors @ evecs[:, -1]
        residual = torch.linalg.vector_norm(gram @ vector - top * vector)
        info.update({"method": "approximate_top_subspace_iteration", "eigenvalues": None,
                     "power_iterations": power_iterations, "numerical_rank": None,
                     "rank_tolerance": None, "negative_roundoff_eigenvalues": None,
                     "top_eigenvalue_residual": _number(residual)})
    info.update({"top_eigenvalue": _number(top), "effective_rank": _ratio(trace, top),
                 "stable_rank": _ratio(squared_fro, top.square())})
    return info


def _matrix_summary(diagonal):
    diagonal = diagonal.double()
    return {"diag_max": _number(diagonal.max()),
            "diag_frobenius_norm": _number(torch.linalg.vector_norm(diagonal)),
            "diag_trace": _number(diagonal.sum()),
            "diag_mean": _number(diagonal.mean())}


def _family_summary(activations, backprops, *, normalization, a_spectrum,
                    fisher_backprops, eig_max_dimension, include_directional):
    """backprops is [directions, examples, affine output dimensions]."""
    a = activations.double()
    b = backprops.double()
    n, inputs = a.shape
    q, n_b, outputs = b.shape
    if n_b != n:
        raise ValueError("Backprop and activation example counts differ")
    xa = a / math.sqrt(n)
    xb = b.reshape(q * n, outputs) / math.sqrt(n * normalization)
    bsquare = b.square().sum(0) / normalization
    diagonal = bsquare.T @ a.square() / n
    bias_diagonal = bsquare.mean(0)
    a_diagonal = a.square().mean(0)
    b_diagonal = xb.square().sum(0)
    kfac_diagonal = b_diagonal[:, None] * a_diagonal[None, :]
    b_spectrum = _spectrum(xb, eig_max_dimension=eig_max_dimension)
    af, bf = a_spectrum, b_spectrum
    exact_diagonal = _matrix_summary(diagonal)
    kfac = {"trace": af["trace"] * bf["trace"],
            "frobenius_norm": af["frobenius_norm"] * bf["frobenius_norm"],
            "top_eigenvalue": af["top_eigenvalue"] * bf["top_eigenvalue"],
            "effective_rank": None if af["effective_rank"] is None or bf["effective_rank"] is None
                              else af["effective_rank"] * bf["effective_rank"],
            "stable_rank": None if af["stable_rank"] is None or bf["stable_rank"] is None
                           else af["stable_rank"] * bf["stable_rank"],
            "spectrum_exact": af["eigenvalues"] is not None and bf["eigenvalues"] is not None,
            "diagonal_relative_l1_error": _ratio((kfac_diagonal-diagonal).abs().sum(), diagonal.abs().sum()),
            "normalization": "E[AA^T] tensor-product E[sum_direction BB^T]"}
    result = {"directions": q, "direction_average_divisor": normalization,
              "weight_parameter_count": inputs * outputs,
              "weight_diagonal": exact_diagonal, "bias_diagonal": _matrix_summary(bias_diagonal),
              "kfac": kfac, "backprop_factor": b_spectrum}
    if include_directional:
        f = fisher_backprops.double()
        g = f.T @ a / n
        gnorm2 = g.square().sum()
        # <g, BB*g*AA> without constructing a parameter-square Kronecker matrix.
        gkg = (xb @ (g @ xa.T)).square().sum()
        a_metric = (a @ xa.T).square().sum(-1)
        b_metric = (f @ xb.T).square().sum(-1)
        individual_quadratic = a_metric * b_metric
        individual_norm2 = a.square().sum(-1) * f.square().sum(-1)
        quotients = torch.where(individual_norm2 > 0,
                                individual_quadratic / individual_norm2,
                                torch.full_like(individual_norm2, float("nan")))
        xf = f / math.sqrt(n)
        kfac_fisher_trace_product = af["frobenius_norm_squared"] * (xb @ xf.T).square().sum()
        centered_exact = individual_quadratic.mean() - gkg
        centered_kfac = kfac_fisher_trace_product - gkg
        result["gradient_directions"] = {
            "mean_gradient_squared_norm": _number(gnorm2),
            "mean_gradient_quadratic_form": _number(gkg),
            "mean_gradient_rayleigh_quotient": _ratio(gkg, gnorm2),
            "per_example_rayleigh_quotient": _summary(quotients),
            "per_example_quadratic_form": _summary(individual_quadratic),
            "per_example_curvature_weighted_norm": _summary(individual_quadratic.clamp_min(0).sqrt()),
            "empirical_gradient_covariance_curvature_trace": _number(centered_exact),
            "kfac_gradient_second_moment_curvature_trace": _number(kfac_fisher_trace_product),
            "kfac_gradient_covariance_curvature_trace": _number(centered_kfac),
            "empirical_noise_to_mean_gradient_quadratic_ratio": _ratio(centered_exact, gkg),
            "kfac_noise_to_mean_gradient_quadratic_ratio": _ratio(centered_kfac, gkg),
            "note": "Curvature operator is this family's KFAC block; gradient vectors are actual per-example CE weight gradients. Empirical covariance contraction retains A/B dependence, whereas the KFAC second-moment proxy factors it. Centered approximate quantities are not clipped."}
    return result


def _output_mismatch(probabilities, gradient, root, *, rcond=1e-10):
    """Output-logit H/F mismatch, with explicit support projection.

    H is mean CE logit Hessian; F is mean outer product of observed-label
    CE logit gradients. Solve H L + L H = 2 F only on H's retained support.
    """
    n, classes = probabilities.shape
    h = torch.einsum("nqi,nqj->ij", root, root) / n
    f = gradient.T @ gradient / n
    evals, vectors = torch.linalg.eigh((h+h.T)*.5)
    evals = evals.clamp_min(0)
    cutoff = evals.max() * rcond
    keep = evals > cutoff
    v, lam = vectors[:, keep], evals[keep]
    projected = v.T @ f @ v
    projection = v @ projected @ v.T
    discarded = torch.linalg.vector_norm(f-projection)
    output = {"n": n, "classes": classes, "relative_eigenvalue_cutoff": rcond,
              "absolute_eigenvalue_cutoff": _number(cutoff), "retained_hessian_rank": int(keep.sum()),
              "mean_logit_hessian": h.cpu().tolist(), "empirical_logit_fisher": f.cpu().tolist(),
              "hessian_eigenvalues": evals.flip(0).cpu().tolist(),
              "fisher_eigenvalues": torch.linalg.eigvalsh((f+f.T)*.5).clamp_min(0).flip(0).cpu().tolist(),
              "fisher_outside_retained_hessian_support_fro": _number(discarded),
              "fisher_outside_support_relative_fro": _ratio(discarded, torch.linalg.vector_norm(f)),
              "note": "Output logits only, not a parameter-space global noise bound. Small Hessian directions are projected out at the stated relative cutoff; discarded Fisher mass is reported. Ambient-dimension rho retains the original dimension convention, including the softmax constant-shift null direction."}
    if not len(lam):
        output.update({"lyapunov_eigenvalues": [], "preconditioned_fisher_eigenvalues": [],
                       "lyapunov_effective_rank": None, "preconditioned_fisher_effective_rank": None,
                       "rho_ambient": None, "rho_support": None, "rho_cheap_ambient": None,
                       "lyapunov_to_preconditioned_effective_rank_ratio": None,
                       "lyapunov_residual_fro": 0.})
        return output
    l = 2*projected/(lam[:,None]+lam[None,:])
    cheap = projected / (lam.sqrt()[:,None]*lam.sqrt()[None,:])
    le = torch.linalg.eigvalsh((l+l.T)*.5).clamp_min(0).flip(0)
    ce = torch.linalg.eigvalsh((cheap+cheap.T)*.5).clamp_min(0).flip(0)
    lerank, cerank = _ratio(le.sum(), le[0]), _ratio(ce.sum(), ce[0])
    residual = lam[:,None]*l+l*lam[None,:]-2*projected
    output.update({"lyapunov_eigenvalues": le.cpu().tolist(),
                   "preconditioned_fisher_eigenvalues": ce.cpu().tolist(),
                   "lyapunov_effective_rank": lerank, "preconditioned_fisher_effective_rank": cerank,
                   "rho_ambient": None if not lerank else classes/lerank,
                   "rho_support": None if not lerank else len(lam)/lerank,
                   "rho_cheap_ambient": None if not cerank else classes/cerank,
                   "lyapunov_to_preconditioned_effective_rank_ratio": None if not cerank or lerank is None else lerank/cerank,
                   "lyapunov_residual_fro": _number(torch.linalg.vector_norm(residual))})
    return output


def analyze_curvature(model, x, y, *, mode="exact", samples=1, seed=20260909,
                      probe_id=None, eig_max_dimension=1536, include_directional=True):
    """Return JSON-safe offline diagnostics without modifying model/RNG state.

    mode='exact': C unit logit directions and C CE-Hessian-factor directions.
    mode='sampled': independent fixed-seed Rademacher combinations, averaged
    over samples; unbiased diagonal/factor estimators, nonlinear summaries are
    descriptive estimates. The observed-label Fisher is always exact on probe.
    This requires an example-independent MLP: each nn.Linear executes once,
    has 2D input/output, and logits have shape [N,C]. No cross-example layers.
    """
    if mode not in {"exact", "sampled"} or isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError("Expected mode exact/sampled and a positive integer sample count")
    if isinstance(eig_max_dimension, bool) or int(eig_max_dimension) != eig_max_dimension or eig_max_dimension < 1:
        raise ValueError("eig_max_dimension must be a positive integer")
    parameters = list(model.parameters())
    if not parameters or any(not p.requires_grad for p in parameters):
        raise ValueError("Model parameters must require gradients; use an unfrozen diagnostic model copy")
    device = parameters[0].device
    if x.device != device or y.device != device or y.ndim != 1 or len(x) != len(y) or len(y) < 1:
        raise ValueError("Probe and targets must be nonempty and on the model device")
    if y.dtype != torch.long:
        raise ValueError("Targets must be torch.long")
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not layers:
        raise ValueError("Expected affine layers")
    _sync(device)
    started = time.perf_counter()
    modes = [(module, module.training) for module in model.modules()]
    old_precision = torch.get_float32_matmul_precision()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    captures, handles = {}, []
    def hook(name):
        def capture(module, args, output):
            if name in captures or len(args) != 1 or args[0].ndim != 2 or output.ndim != 2:
                raise ValueError("Each Linear must execute once with [examples,features] tensors")
            captures[name] = (args[0].detach(), output)
        return capture
    try:
        torch.set_float32_matmul_precision("highest")
        model.eval()
        for name, layer in layers:
            handles.append(layer.register_forward_hook(hook(name)))
        with torch.enable_grad():
            logits = model(x)
            for handle in handles:
                handle.remove()
            handles.clear()
            if logits.ndim != 2 or len(logits) != len(y) or set(captures) != {name for name,_ in layers}:
                raise ValueError("Expected [N,C] logits and every affine layer exactly once")
            if not bool(torch.isfinite(logits).all()) or bool((y < 0).any()) or bool((y >= logits.shape[-1]).any()):
                raise FloatingPointError("Invalid logits or targets")
            probabilities, output_gradient, root = _stable_output_statistics(logits, y)
            outputs = [captures[name][1] for name,_ in layers]
            def backward(direction):
                return [b.detach() for b in torch.autograd.grad(logits, outputs,
                        grad_outputs=direction.to(logits.dtype), retain_graph=True,
                        create_graph=False, allow_unused=False)]
            fisher = backward(output_gradient)
            n, classes = logits.shape
            if mode == "exact":
                jac_directions = torch.eye(classes, dtype=torch.float64, device=device)[:,None,:].expand(classes,n,classes)
                ggn_directions = root.transpose(0,1)
                divisor = 1
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                def rademacher():
                    return (torch.randint(0,2,(samples,n,classes),generator=generator,dtype=torch.int64)*2-1).to(device=device,dtype=torch.float64)
                jac_directions = rademacher()
                ggn_directions = torch.einsum("snq,nqc->snc",rademacher(),root)
                divisor = samples
            backprops = {"ce_ggn": [[] for _ in layers], "jacobian_gram": [[] for _ in layers]}
            for family,directions in [("ce_ggn",ggn_directions),("jacobian_gram",jac_directions)]:
                for direction in directions:
                    for index,b in enumerate(backward(direction)):
                        backprops[family][index].append(b)
            # Drop graph references before the dense float64 summaries.
            ce = F.cross_entropy(logits.detach().double(),y)
            accuracy = (logits.detach().argmax(-1)==y).double().mean()
            del outputs
            capture_inputs = {name:a for name,(a,_) in captures.items()}
            captures.clear()
            del logits
        output = {"schema_version":1, "mode":mode, "samples":samples if mode=="sampled" else None,
                  "seed":seed, "probe_id":probe_id, "probe_n":n, "output_classes":classes,
                  "probe_x_sha256":_array_hash(x), "probe_y_sha256":_array_hash(y),
                  "model_dtype":str(parameters[0].dtype), "derivative_dtype":str(parameters[0].dtype),
                  "reduction_dtype":"torch.float64", "matmul_precision":"highest",
                  "output_derivative_convention":"Probabilities in float64; CE true-label derivative and Hessian-root diagonal use sums of other probabilities instead of subtracting from one; directions cast to model-logit dtype for backward.",
                  "output_probability_zero_count":int((probabilities==0).sum()),
                  "dense_ce":_number(ce), "dense_accuracy":_number(accuracy),
                  "eig_max_dimension":eig_max_dimension, "include_directional":include_directional,
                  "layers":[], "output_mismatch":_output_mismatch(probabilities,output_gradient,root),
                  "definitions":{
                      "ce_ggn":"Affine weight block of generalized Gauss-Newton for mean CE; omits cross-layer blocks and true-Hessian model-second-derivative terms.",
                      "jacobian_gram":"Mean per-example J_logits.T @ J_logits affine weight block, summed over output classes, not divided by number of classes.",
                      "empirical_fisher":"Mean outer product of observed-label per-example CE gradients; not generally equal to the Hessian or CE-GGN.",
                      "kfac":"Product of separately averaged activation and backprop factors; cross-factor dependence is discarded.",
                      "rank":"effective_rank = trace/top eigenvalue; stable_rank = Frobenius norm squared/top eigenvalue squared.",
                      "zero_denominators":"Undefined ratios are null; exact zeros are not represented as an informative finite ratio.",
                      "sampling":"Exact observed-label Fisher on the fixed probe. Sampled GGN/Jacobian use independent per-example Rademacher probes averaged over samples; ratios/ranks of sampled matrices are not unbiased estimates of exact ratios/ranks.",
                      "scope":"Dense eval-mode probe; no masked training gradients and no full parameter-square matrix."},
                  "omitted_legacy_fields":["buggy duplicated eigenvalue histograms", "unnormalized kfac_fro", "misnamed mean_norm", "unwritten lyapunov sums / curv_ratio", "unverified offset-regret formulas", "heuristic lr1-lr4 and Jain/Bottou/Yaida prescriptions"]}
        largest = 0
        for i,(name,layer) in enumerate(layers):
            a = capture_inputs[name].double()
            fb = fisher[i].double()
            a_spectrum = _spectrum(a/math.sqrt(n),eig_max_dimension=eig_max_dimension)
            mean_gradient = fb.T @ a / n
            norm2 = a.square().sum(-1)*fb.square().sum(-1)
            mean_norm2 = mean_gradient.square().sum()
            family_results = {}
            for family in ["empirical_fisher","ce_ggn","jacobian_gram"]:
                b = fb[None] if family=="empirical_fisher" else torch.stack(backprops[family][i])
                family_results[family] = _family_summary(a,b,
                    normalization=1 if family=="empirical_fisher" else divisor,
                    a_spectrum=a_spectrum,fisher_backprops=fb,
                    eig_max_dimension=eig_max_dimension,include_directional=include_directional)
                if layer.bias is None:
                    family_results[family]["bias_diagonal"] = None
                largest=max(largest,family_results[family]["backprop_factor"]["eigenproblem_dimension"])
            largest=max(largest,a_spectrum["eigenproblem_dimension"])
            output["layers"].append({"layer_index":i,"module_name":name,
                "input_features":layer.in_features,"output_features":layer.out_features,
                "bias_present":layer.bias is not None,"activation_factor":a_spectrum,
                "gradient_moments":{"mean_gradient_l2":_number(mean_norm2.sqrt()),
                    "mean_gradient_squared_norm":_number(mean_norm2),
                    "per_example_gradient_norm":_summary(norm2.sqrt()),
                    "per_example_gradient_squared_norm_mean":_number(norm2.mean()),
                    "gradient_covariance_trace":_number(norm2.mean()-mean_norm2),
                    "gradient_diversity":_ratio(norm2.mean(),mean_norm2)},
                "families":family_results})
        output["largest_eigenproblem_dimension"] = largest
        _sync(device)
        output["elapsed_seconds"] = time.perf_counter()-started
        return output
    finally:
        for handle in handles:
            handle.remove()
        for module,training in modes:
            module.training = training
        torch.set_float32_matmul_precision(old_precision)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng,device)
