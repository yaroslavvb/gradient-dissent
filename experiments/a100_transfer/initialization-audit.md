# ConvNeXt initialization audit

The same ConvNeXt seed produced two different SHA-256 state hashes across Modal CPU hosts. Full-tensor comparison shows that these are **roundoff-scale differences, not bitwise-identical initializations**. The first differing operation in a controlled initialization probe is CPU `erfinv_`. Training code, random seeds, and reported hashes remain unchanged.

## What was observed

The [nine-worker probe](results/initialization-worker-probes.json) reconstructed seed 1000 using frozen `vision.py` and PyTorch `2.8.0+cu128`. Eight workers reporting PyTorch CPU capability `AVX512` produced `0b281e0b…d43939`; one reporting `AVX2` produced `14ceace6…370af8`. These reproduce both hashes observed in the training records. All 59 randomized Conv/Linear weight tensors differed in their byte hashes; all 123 constant tensors, including LayerScale, matched. All workers had the same post-initialization CPU RNG state. A matching RNG state alone establishes neither matching tensor values nor matching arithmetic.

The [kernel-stage probe](results/initialization-kernel-probes.json) then separated uniform sampling, inverse-error-function transformation, and scaling on a one-million-element float32 tensor:

| CPU host / process setting | Uniform hash | `erfinv` hash | Full model hash |
| --- | --- | --- | --- |
| AVX512 capability, default | `1a981190…173d2` | `c0dc1b63…9ce46` | `0b281e0b…d43939` |
| AVX2 capability, default | `1a981190…173d2` | `c6b5437e…b6774` | `14ceace6…370af8` |
| AVX512 capability, `MKL_ENABLE_INSTRUCTIONS=AVX2` | `1a981190…173d2` | `d23e8d50…55d3f7` | `f122b694…3f4da93` |

MKL was available in all three processes. The uniform inputs matched exactly, and the first observed divergence was at `erfinv`. Changing the MKL instruction setting on the same host changed that result, but **did not reproduce the other host**. The third model hash is a diagnostic result, not a trained-run initialization and not an accepted comparison variant. Earlier changes to `ATEN_CPU_CAPABILITY` alone had not reproduced the other host either.

## Why this can happen

PyTorch 2.8 implements [`trunc_normal_`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L78-L116) as uniform sampling, `erfinv_`, multiplication by the scale, addition of the mean, and clamping. Its [CPU unary kernel](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L660-L807) calls the VML implementation; the [MKL-enabled Linux float specialization](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/cpu/vml.h#L89-L133) uses `vmsErfInv` in high-accuracy mode. Intel documents [CPU-dependent instruction dispatch and its separate MKL override](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/instruction-set-specific-dispatch-on-intel-archs.html).

The observed conclusion is CPU `erfinv` numerical variation. The source path and successful MKL intervention support a CPU math-library dispatch explanation. They do **not** identify the precise internal kernel, establish that the two hosts differ only in AVX width, or show that an AVX2 override fully emulates the other host. PyTorch's reported CPU capability is a capability label, not a trace of MKL's selected instructions.

## Full-state numerical bounds

Both default-host variants were reconstructed for the tuning seed and all three final seeds. The [comparison script](compare_initialization_variants.py) checks tensor names, shapes and dtypes, hashes the complete reconstructed states, and computes differences and norms in float64. The [numerical audit](results/initialization-numerical-audit.json) contains every tensor's statistics; the [manifests](results/initialization-variant-manifests.json) contain the full state hashes, RNG hashes, source hash and saved-state locations.

The executed diagnostic drivers are archived for [worker reconstruction](diagnostics/probe_workers_executed.py), [saving the two variants](diagnostics/save_variants_executed.py), and [kernel-stage comparison](diagnostics/probe_kernel_executed.py). Their [README](diagnostics/README.md) explains the historical container references and CPU-only execution.

| Seed | Changed elements / 27,897,028 | Maximum absolute difference | Relative L2 difference, whole state |
| --- | ---: | ---: | ---: |
| 1000 | 183,867 (0.6591%) | 7.4506 × 10⁻⁹ | 5.3559 × 10⁻⁹ |
| 2000 | 183,638 (0.6583%) | 7.4506 × 10⁻⁹ | 5.3528 × 10⁻⁹ |
| 2001 | 183,534 (0.6579%) | 7.4506 × 10⁻⁹ | 5.3624 × 10⁻⁹ |
| 2002 | 183,059 (0.6562%) | 7.4506 × 10⁻⁹ | 5.3609 × 10⁻⁹ |

Relative L2 means `||state_B − state_A||₂ / ||state_A||₂`. Constant normalization weights contribute to that denominator. Restricting it to the 59 randomized tensors gives 7.0447–7.0573 × 10⁻⁹; the largest individual-tensor relative L2 difference is below 8.82 × 10⁻⁹. Thus the small aggregate error is not an artifact of including constant weights. Equal initial BF16 validation loss is not used as evidence of weight equality.

## Qualification and limits

This exception was developed after the hash mismatch, independently of final-test analysis. It is a **post-hoc provenance audit**, not a criterion preregistered in the original training protocol. The diagnostic comparison uses reconstructed initial states only: no trained weights, test outcomes, recipe rankings, or seed selection. It ran on CPUs in existing containers; no additional GPU training jobs or changes to the frozen model/training code were required. The diagnostic processes shared capped CPU resources with the first final-training wave, so those wall times may include diagnostic contention. The MKL override applied only to a separate diagnostic child process; the training processes and their RNG states were unchanged.

The numerical qualification requires maximum absolute difference ≤ 10⁻⁷ and whole-state relative L2 difference ≤ 10⁻⁶. All four comparisons pass. Only the two explicitly compared full hashes for each seed in the numerical audit qualify; the third MKL-override hash and any other unrecognized hash do not. Raw mismatching hashes are preserved rather than replaced by a tolerance-based hash.

The correct inferential description is **seed-paired training with a disclosed roundoff-scale initialization exception**, not identical paired initialization. These bounds do not prove identical training trajectories or negligible effects on final scores: nonlinear optimization can amplify small differences. The audit resolves the magnitude and location of the initialization discrepancy; it does not measure training sensitivity to it. No runs or seeds are discarded because of their outcomes.
