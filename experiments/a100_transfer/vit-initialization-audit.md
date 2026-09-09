# ViT initialization audit

A later initialization check found that ViT seed 2001's dense run did not share the ILD runs' exact initial-state hash. Independent reconstruction from frozen `vision.py` on two CPU hosts reproduced both hashes. Full comparison now bounds the difference at **7.4506 × 10⁻⁹ per parameter** for all four declared seeds. ViT therefore joins ConvNeXt as a specifically audited exception to byte-identical initialization; the recorded GPT initializations remain byte-identical within seed.

## Observation and reconstruction

For seed 2001, the [dense run](results/eval-vit_cifar100-dense-s2001.json) recorded `3532faee…9285c5d`; the [constant](results/eval-vit_cifar100-constant_ild-s2001.json) and [decreasing](results/eval-vit_cifar100-decreasing_ild-s2001.json) ILD runs recorded `78b9cccc…8c7420c`. CPU reconstruction reproduced these on hosts reporting AVX2 and AVX512 capability, respectively, using PyTorch `2.8.0+cu128` and the same frozen vision source SHA-256. The [variant manifests](results/vit-initialization-variant-manifests.json) contain complete hashes for seeds 1000, 2000, 2001 and 2002; post-initialization CPU RNG hashes match across hosts for every seed. The [CPU probes](results/vit-initialization-cpu-probes.json) record the workers' capabilities and MKL availability.

The initial, **pretraining validation** CE also differed: dense recorded `4.705471875000001`, versus `4.7054406250000005` for both ILD runs, a difference of about 3.125 × 10⁻⁵ nats per image. These values come only from `validation_before`; final validation and test outcomes were not inputs to this audit. Equal RNG state clearly does not guarantee equal evaluated outputs. Because those evaluations ran on different workers, this comparison does not isolate the contribution of weight differences from other numerical execution differences.

## Full-state numerical evidence

The [comparison](compare_vit_initialization_variants.py) checks names, shapes and dtypes for all 151 tensors, containing 85,219,684 parameters, and computes differences/norms in float64. All 51 randomized tensors differ in byte hash, including the positional embedding; all 100 constant tensors match. The [full numerical evidence](results/vit-initialization-numerical-audit.json) records per-tensor statistics and reconstructed state hashes, which match the manifests.

| Seed | Changed elements / 85,219,684 | Maximum absolute difference | Whole-state relative L2 difference |
| --- | ---: | ---: | ---: |
| 1000 | 559,735 (0.6568%) | 7.4506 × 10⁻⁹ | 5.6167 × 10⁻⁹ |
| 2000 | 560,246 (0.6574%) | 7.4506 × 10⁻⁹ | 5.6292 × 10⁻⁹ |
| 2001 | 559,424 (0.6564%) | 7.4506 × 10⁻⁹ | 5.6200 × 10⁻⁹ |
| 2002 | 560,663 (0.6579%) | 7.4506 × 10⁻⁹ | 5.6353 × 10⁻⁹ |

Relative L2 is `||state_B − state_A||₂ / ||state_A||₂`. Excluding constant tensors from the norm gives 7.0235–7.0474 × 10⁻⁹, so the conclusion does not depend on normalization constants diluting the denominator. The largest individual-tensor relative L2 difference is below 7.41 × 10⁻⁹. Aggregate counts, maxima, norms and manifest correspondence were independently checked from the saved evidence.

The acceptance bounds—maximum absolute difference ≤ 10⁻⁷ and whole-state relative L2 ≤ 10⁻⁶—were fixed before inspecting these full-state differences. Every comparison passes. Only the two explicitly compared **ViT** hashes for each corresponding seed qualify. This is not permission to accept arbitrary numerically close or previously unseen hashes, and it does not reuse ConvNeXt's hash list.

## Mechanism: evidence and inference

ViT uses the same [`trunc_normal_` initialization](vision.py#L100-L120) as the earlier [ConvNeXt diagnosis](initialization-audit.md). PyTorch 2.8's [implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/init.py#L78-L116) transforms uniform draws through `erfinv_` before scaling. In an MKL-enabled Linux build, the [CPU specialization](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/cpu/vml.h#L89-L133) uses MKL's inverse-error-function routine; Intel documents [CPU-dependent math-library dispatch](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/instruction-set-specific-dispatch-on-intel-archs.html).

The earlier [standalone kernel probe](results/initialization-kernel-probes.json) directly observed matching uniform samples followed by differing CPU `erfinv` outputs. A separate MKL instruction override changed the result without reproducing the other host. This supports the same arithmetic explanation for ViT's closely matching reconstructed states, but the ViT constructor was not separately instrumented at every initialization operation. Neither the precise internal kernel nor a complete emulation of one CPU host by another has been established.

## Execution and interpretation limits

The [worker probe](diagnostics/probe_vit_workers.py), [state reconstruction](diagnostics/save_vit_variants.py), and [comparison driver](diagnostics/compare_vit_executed.py) are archived. The first comparison encountered an older mounted-volume view. The retry streamed the committed files through the volume API into temporary local files, without reloading the mount used by ongoing training. It did not repeat training or alter the evidence states.

All diagnostics ran as separate CPU processes in existing capped containers, with **no new GPU function calls**. They did not change the frozen training/model sources, training processes' RNG states, recipes, seeds or masks. Sharing CPU resources with ongoing training can nevertheless add contention to overlapping runs' wall times; these timings are not controlled speed comparisons. The earlier MKL environment override applied only to its diagnostic child process.

This is a **post-hoc verification adjustment independent of final outcomes**, prompted by provenance metadata. It preserves seed-paired analysis with a disclosed numerical initialization exception. Tiny initial differences and identical RNG states do not imply identical training trajectories or negligible final-score effects. No training-sensitivity experiment was performed, and no final result or seed was selected or discarded to make this qualification pass.
