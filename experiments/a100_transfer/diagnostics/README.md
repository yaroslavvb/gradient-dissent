# Initialization diagnosis: exact executed orchestration

These scripts preserve the CPU-only commands used to diagnose the ConvNeXt initialization mismatch while the declared A100 sweep was already running. Their container IDs are historical, not reusable infrastructure. They do not create GPUs, applications, or paid function calls. All CPU work ran within the existing containers' capped resources and recorded invocation budgets.

1. `probe_workers_executed.py` reconstructs seed1000 on nine existing workers, saving state/tensor/RNG hashes and a temporary CPU state file on each worker.
2. `save_variants_executed.py` reconstructs seeds1000/2000/2001/2002 on one worker of each observed variant, writes their CPU state dictionaries to the existing volume, and commits them.
3. The root-level `compare_initialization_variants.py` compares every parameter in those saved states. The executed wrapper first reloaded the volume, then ran the script as a CPU process in the existing AVX2 worker. The caller saved JSON stdout as `results/initialization-numerical-audit.json`.
4. `probe_kernel_executed.py` compares uniform sampling, inverse-erf and scaling on both hosts, plus an MKL instruction override on the AVX512 host. The override produces an additional diagnostic variant; it is not an accepted training initialization.

No trained checkpoint or test score is an input to these diagnostics. The exact numerical thresholds were chosen before inspecting the full-state differences. The training code, schedule, masks, learning rates, seeds, and outcomes were unchanged. The resulting bounded exception to exact initialization-hash equality is a disclosed post hoc verification adjustment, described in [the initialization audit](../initialization-audit.md).

To reproduce the numerical issue on fresh CPU workers, use the frozen vision source and PyTorch2.8.0 Linux build on the two recorded CPU/library configurations. Exact kernel outputs can depend on the host and math-library dispatch; a CPU-capability label alone does not guarantee an identical result. Use an explicit new output history rather than overwriting this evidence. The approximately900MB of untrained CPU state dictionaries remain in the experiment volume, outside Git.

## Later ViT diagnosis

`probe_vit_workers.py` recorded CPU capabilities on the seven remaining workers. `save_vit_variants.py` reconstructed the same four seeds for the native-32 ViT on two of those workers. The eight CPU state dictionaries occupy approximately 2.73 GB in `/vit-initialization-audit` in the same volume; they are not Git blobs.

`compare_vit_executed.py` is the successful comparison orchestration, using `compare_vit_initialization_variants.py`. An initial attempt could not see the newly committed files through the older worker's mount. The successful path streamed committed states through the volume API into `/tmp`, avoiding a reload of the training mount. A transfer attempt was interrupted and resumed from complete local files after its required duration became clear. The final comparison verifies the complete tensor contents and state hashes; a cached file's size alone does not qualify any initialization.

These diagnostics also used CPU processes within existing capped workers, with no new GPU calls or repeated training. The separate [ViT audit](../vit-initialization-audit.md) records full-state bounds, provenance, and the limitation that numerical closeness does not establish identical training trajectories. The historical container IDs and temporary paths are not instructions to restart the study.
