"""Exact four-branch mask analysis, with no fitting, sampling, or paid calls.

Mask bit i=1 means branch i is present. Arrays use mask order 0..15; logits
have shape [16, examples, classes]. Walsh characters use (2*presence-1).
Möbius coefficients instead use products of 0/1 presence indicators. These
bases have different fixed-cardinality expectations and only Walsh is
orthogonal under the uniform distribution on all 16 masks.
"""
from itertools import combinations
import math

import numpy as np

N_BRANCHES = 4
N_MASKS = 16
FULL_MASK = 15


def _values(values):
    value = np.asarray(values, dtype=np.float64)
    if value.ndim < 1 or value.shape[0] != N_MASKS or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("Expected finite values with first dimension 16")
    return value


def canonical_logits(logits, mask_ids=None):
    value = _values(logits)
    if value.ndim != 3 or value.shape[1] < 1 or value.shape[2] < 2:
        raise ValueError("logits must have shape [16,N,C], N>=1, C>=2")
    if mask_ids is not None:
        ids = list(mask_ids)
        if len(ids) != 16 or any(not isinstance(i, (int, np.integer)) for i in ids) or set(ids) != set(range(16)):
            raise ValueError("mask_ids must be a permutation of integers 0..15")
        value = value[np.argsort(ids)]
    return value


def _labels(labels, n, classes):
    value = np.asarray(labels)
    if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("labels must be an integer vector of length N")
    if np.any(value < 0) or np.any(value >= classes):
        raise ValueError("label outside class range")
    return value.astype(np.int64, copy=False)


def mask_costs(branch_costs, mandatory_cost=0.0):
    """Nominal cost from actually present branches, not measured runtime."""
    costs = np.asarray(branch_costs, dtype=np.float64)
    mandatory_cost = float(mandatory_cost)
    if costs.shape != (4,) or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Need four finite nonnegative branch costs")
    if not math.isfinite(mandatory_cost) or mandatory_cost < 0 or mandatory_cost+costs.sum() <= 0:
        raise ValueError("Mandatory cost must be nonnegative; full cost positive")
    full = mandatory_cost + float(costs.sum())
    return [{"mask_id": m, "active": [bool(m & (1 << i)) for i in range(4)],
             "retained_branches": m.bit_count(),
             "cost": mandatory_cost + sum(float(costs[i]) for i in range(4) if m & (1 << i)),
             "fraction_of_full_cost": (mandatory_cost + sum(float(costs[i]) for i in range(4) if m & (1 << i)))/full}
            for m in range(16)]


def _choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def uniform_cardinality_retention(n=4):
    """[k,d] probability a fixed d-element monomial survives retaining k/n.

    This is C(n-d,k-d)/C(n,k), not (k/n)**d. It is NOT the expectation of a
    Walsh character. Degree zero survives even when no branches are retained.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return np.asarray([[_choose(n-d, k-d)/math.comb(n, k) for d in range(n+1)] for k in range(n+1)])


def uniform_cardinality_walsh_expectation(n=4):
    """[k,d] E[product_i(2*presence_i-1) | exactly k branches present]."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return np.asarray([[sum((-1)**(d-j)*_choose(d,j)*_choose(n-d,k-j) for j in range(d+1))/math.comb(n,k)
                        for d in range(n+1)] for k in range(n+1)])


def _walsh_matrix():
    return np.asarray([[(-1.)**(term.bit_count()-(mask & term).bit_count())
                        for mask in range(16)] for term in range(16)])


def walsh_transform(values):
    """Coefficients E_uniform_masks[f(mask)*character_term(mask)]."""
    value = _values(values)
    return np.tensordot(_walsh_matrix()/16, value, axes=(1, 0))


def inverse_walsh_transform(coefficients):
    return np.tensordot(_walsh_matrix().T, _values(coefficients), axes=(1, 0))


def mobius_transform(values):
    """Anchored-at-empty multilinear coefficients: f(S)=sum_{T subset S} a_T."""
    result = _values(values).copy()
    for i in range(4):
        for mask in range(16):
            if mask & (1 << i):
                result[mask] -= result[mask ^ (1 << i)]
    return result


def inverse_mobius_transform(coefficients):
    result = _values(coefficients).copy()
    for i in range(4):
        for mask in range(16):
            if mask & (1 << i):
                result[mask] += result[mask ^ (1 << i)]
    return result


def centered_dense_deltas(logits):
    """Class-centered logit perturbations; dense row exactly zero.

    A separate common scalar may be added to every example/mask's classes
    without changing this quantity. Subtracting dense affects only Walsh
    degree zero; class centering also removes class-independent interactions.
    """
    value = canonical_logits(logits)
    centered = value-value.mean(axis=-1, keepdims=True)
    return centered-centered[FULL_MASK:FULL_MASK+1]


def walsh_spectrum(values):
    """Orthogonal energy by degree, averaged over all non-mask dimensions."""
    value = _values(values)
    coefficients = walsh_transform(value)
    term_energy = np.mean(coefficients.reshape(16, -1)**2, axis=1)
    degree_energy = np.asarray([sum(term_energy[m] for m in range(16) if m.bit_count()==d) for d in range(5)])
    energy = float(np.mean(value**2)); spectral = float(degree_energy.sum())
    np.testing.assert_allclose(inverse_walsh_transform(coefficients), value, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(spectral, energy, rtol=1e-11, atol=1e-12)
    nonconstant = float(degree_energy[1:].sum())
    return {"degrees": list(range(5)), "degree_energy": degree_energy.tolist(),
            "term_energy_by_mask_id": term_energy.tolist(), "total_energy": energy,
            "parseval_spectral_energy": spectral, "parseval_absolute_error": abs(spectral-energy),
            "degree_energy_fraction": (degree_energy/energy).tolist() if energy > 0 else [None]*5,
            "degree_ge2_fraction_of_nonconstant_energy": float(degree_energy[2:].sum()/nonconstant) if nonconstant > 0 else None,
            "measure": "Mean squared function value over uniform16 masks and remaining array coordinates; no seed-based uncertainty"}


def interaction_summary(values):
    """Möbius interactions and context-sensitive two-branch differences.

    A mixed difference with the other two branches present includes higher
    interactions containing the pair. It need not equal the pair coefficient
    anchored where all other branches are absent.
    """
    value = _values(values); coefficients = mobius_transform(value)
    np.testing.assert_allclose(inverse_mobius_transform(coefficients), value, rtol=1e-11, atol=1e-12)
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a)**2)))
    pairs = []
    for i,j in combinations(range(4), 2):
        pair = (1 << i) | (1 << j)
        context = FULL_MASK ^ pair
        empty = value[pair]-value[1 << i]-value[1 << j]+value[0]
        full = value[FULL_MASK]-value[FULL_MASK ^ (1 << i)]-value[FULL_MASK ^ (1 << j)]+value[context]
        np.testing.assert_allclose(empty, coefficients[pair], rtol=1e-11, atol=1e-12)
        pairs.append({"branches":[i,j], "empty_context_rms":rms(empty), "full_context_rms":rms(full),
                      "context_change_rms":rms(full-empty)})
    return {"term_coefficient_rms_by_mask_id":[rms(c) for c in coefficients],
            "pair_mixed_differences":pairs,
            "basis":"0/1 presence monomials, anchored at all-removed mask0",
            "warning":"Möbius coefficient magnitudes are not an orthogonal energy decomposition; context changes can contain degree3/4 contributions."}


def cardinality_means(values):
    value = _values(values)
    return np.asarray([value[[m for m in range(16) if m.bit_count()==k]].mean(axis=0) for k in range(5)])


def cross_entropy_by_example(logits, labels):
    value = canonical_logits(logits); y = _labels(labels, value.shape[1], value.shape[2])
    shifted = value-value.max(axis=-1,keepdims=True)
    return np.log(np.exp(shifted).sum(axis=-1))-shifted[:,np.arange(len(y)),y]


def dense_margin(logits):
    """Top1 minus top2 dense logit margin; never uses true labels."""
    full = canonical_logits(logits)[FULL_MASK]
    top = np.partition(full, -2, axis=-1)[:,-2:]
    return top[:,1]-top[:,0]


def _ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start+end-1)/2+1
        start=end
    return ranks


def spearman_stats(x, y):
    """Descriptive Spearman rho with average ties; no invalid IID p-value."""
    x = np.asarray(x,dtype=np.float64); y = np.asarray(y,dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need equally sized finite vectors")
    n=len(x)
    if n < 2:
        return {"n":n,"rho":None,"undefined_reason":"fewer_than_two_examples"}
    a=_ranks(x); b=_ranks(y); a-=a.mean(); b-=b.mean()
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    if denominator==0:
        return {"n":n,"rho":None,"undefined_reason":"constant_rank_vector"}
    return {"n":n,"rho":float(np.clip(np.dot(a,b)/denominator,-1,1)),
            "undefined_reason":None,"inference":"descriptive only; no p-value or confidence interval"}


def classification_robustness(logits, labels):
    value=canonical_logits(logits); y=_labels(labels,value.shape[1],value.shape[2])
    predictions=value.argmax(axis=-1); correct=predictions==y[None,:]
    full_correct=correct[FULL_MASK]; n=len(y); full_n=int(full_correct.sum())
    summarize=lambda masks: {
        "accuracy":float(correct[masks].mean()),
        "dense_prediction_agreement":float((predictions[masks]==predictions[FULL_MASK]).mean()),
        "accuracy_given_dense_correct":float(correct[masks][:,full_correct].mean()) if full_n else None,
        "harm_probability":float((~correct[masks] & full_correct).mean()),
        "repair_probability":float((correct[masks] & ~full_correct).mean())}
    by_mask=[{"mask_id":m,**summarize([m])} for m in range(16)]
    by_k=[]; counts=[]
    for k in range(5):
        masks=[m for m in range(16) if m.bit_count()==k]
        by_k.append({"retained_branches":k,"mask_count":len(masks),**summarize(masks)})
        counts.append(correct[masks].sum(axis=0).tolist())
    proper=correct[1:15]
    proper_fraction=proper.mean(axis=0)
    return {"n_examples":n,"dense_accuracy":float(full_correct.mean()),"dense_correct_examples":full_n,
            "by_mask":by_mask,"by_retained_cardinality":by_k,
            "per_example":{"dense_correct":full_correct.tolist(),"dense_margin":dense_margin(value).tolist(),
                           "correct_mask_counts_by_cardinality":counts,
                           "fraction_correct_over14_nonempty_pruned_masks":proper_fraction.tolist(),
                           "all14_nonempty_pruned_masks_correct":proper.all(axis=0).tolist()},
            "spearman_dense_margin_vs_pruned_accuracy":spearman_stats(dense_margin(value),proper_fraction),
            "spearman_dense_margin_vs_retention_given_dense_correct":spearman_stats(dense_margin(value)[full_correct],proper_fraction[full_correct]),
            "aggregation":"Uniform masks within each cardinality, then uniform examples; conditional retention restricts to examples correct at mask15. No mask is a separate seed replicate."}


def risk_coverage(logits, labels, coverages=(1.0,.9,.75,.5,.25)):
    """Selective risk using ONLY dense margin to rank examples.

    Ceil(coverage*N) examples, ties resolved by original example index. All
    mask outcomes use the same selected examples; labels enter risk scoring
    only, never ranking. Requires the dense prediction and is therefore a
    diagnostic, not a free pre-inference cost-saving gate.
    """
    value=canonical_logits(logits); y=_labels(labels,value.shape[1],value.shape[2]); n=len(y)
    order=np.argsort(-dense_margin(value),kind="stable")
    correctness=value.argmax(axis=-1)==y[None,:]; ce=cross_entropy_by_example(value,y)
    rows=[]
    for coverage in coverages:
        coverage=float(coverage)
        if not math.isfinite(coverage) or not 0 < coverage <= 1:
            raise ValueError("coverage must lie in (0,1]")
        count=math.ceil(coverage*n); ids=order[:count]; dense_correct=correctness[15,ids]
        by_k=[]
        for k in range(5):
            masks=[m for m in range(16) if m.bit_count()==k]
            selected=correctness[np.ix_(masks,ids)]
            by_k.append({"retained_branches":k,"mask_count":len(masks),
                         "risk":float(1-selected.mean()),"ce":float(ce[np.ix_(masks,ids)].mean()),
                         "risk_given_dense_correct":float(1-selected[:,dense_correct].mean()) if dense_correct.any() else None})
        rows.append({"requested_coverage":coverage,"actual_coverage":count/n,"selected_count":count,
                     "dense_correct_selected_count":int(dense_correct.sum()),
                     "minimum_selected_dense_margin":float(dense_margin(value)[ids[-1]]),
                     "by_retained_cardinality":by_k})
    return {"selection_rule":"Descending dense top1-top2 logit margin, stable original-index tie break; no labels or pruned logits in ranking",
            "dense_ranking_example_indices":order.tolist(),"rows":rows,
            "limitation":"Dense logits are required, so this is a diagnostic of predictable robustness, not an inference-compute saving policy."}


def analyze_masks(logits, labels, *, mask_ids=None, branch_costs=(5000000,3000000,1500000,500000),
                  mandatory_cost=1965000, coverages=(1.0,.9,.75,.5,.25)):
    """Return a JSON-safe report from an exact 16-mask prediction panel."""
    value=canonical_logits(logits,mask_ids); y=_labels(labels,value.shape[1],value.shape[2])
    delta=centered_dense_deltas(value); ce=cross_entropy_by_example(value,y)
    correct=(value.argmax(axis=-1)==y[None,:]).astype(float)
    # Explicitly test both different basis/cardinality identities.
    monomial=mobius_transform(delta); walsh=walsh_transform(delta)
    by_degree_m=np.asarray([sum((monomial[m] for m in range(16) if m.bit_count()==d)) for d in range(5)])
    by_degree_w=np.asarray([sum((walsh[m] for m in range(16) if m.bit_count()==d)) for d in range(5)])
    exact=cardinality_means(delta)
    np.testing.assert_allclose(np.tensordot(uniform_cardinality_retention(),by_degree_m,axes=(1,0)),exact,rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(np.tensordot(uniform_cardinality_walsh_expectation(),by_degree_w,axes=(1,0)),exact,rtol=1e-10,atol=1e-11)
    return {"mask_order":"Integer0..15; bit i=present branch i; mask15=dense, mask0=all four body branches removed",
            "n_examples":len(y),"n_classes":value.shape[2],"mask_costs":mask_costs(branch_costs,mandatory_cost),
            "uniform_cardinality":{"monomial_term_retention_k_by_degree":uniform_cardinality_retention().tolist(),
                                    "walsh_character_expectation_k_by_degree":uniform_cardinality_walsh_expectation().tolist(),
                                    "both_basis_reconstructions_verified":True},
            "logit_perturbation_definition":"Class-center each mask/example logit vector and subtract centered dense logits; dense perturbation is exactly zero; invariant to common per-mask/example logit shifts",
            "centered_logit_delta_walsh":walsh_spectrum(delta),
            "centered_logit_delta_mobius":interaction_summary(delta),
            "ce_delta_walsh":walsh_spectrum(ce-ce[FULL_MASK]),
            "correctness_delta_walsh":walsh_spectrum(correct-correct[FULL_MASK]),
            "nonlinearity_warning":"CE and argmax can create high-order mask interactions even when centered logits are exactly additive. Loss/correctness degree is not a direct measure of serial computational path length.",
            "classification":classification_robustness(value,y),"risk_coverage":risk_coverage(value,y,coverages)}
