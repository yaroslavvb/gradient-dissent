# Depth robustness in a tiny causal transformer

Qualitative reproduction of the layer-dropout depth-robustness mechanism from [Don't Drop Dropout](https://arxiv.org/pdf/2609.05275v1), not its LLM scale, training recipe or efficiency figures.

## Reproduce

From the repository root:

```bash
python3 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -r experiments/requirements.txt
experiments/.venv/bin/python experiments/depth_lm/fetch_data.py
experiments/.venv/bin/python experiments/depth_lm/run.py --part all
```

Local CPU only: four isolated worker processes, one PyTorch thread each. No Modal or paid service is invoked. The full experiment has 50 tuning runs and 25 final runs. `--part tune` and `--part evaluate` run those stages separately. `--part pilot` is a separate 120-step engineering check. Keep final raw outputs if comparing a rerun; they are overwritten by a new run.

The exact Tiny Shakespeare revision and SHA-256 are pinned in the code. Downloads are stored in ignored `data/`. The character vocabulary is derived from training text only. A disjoint contiguous 80/10/10 split is constructed before windows. Validation and test use 64 and 128 evenly spaced nonoverlapping 64-character target windows, respectively; they are identical across methods and seeds. This is a small literary corpus, not the paper's modern token stream.

## Protocol

Six pre-LayerNorm transformer blocks, width 64, four causal attention heads, FFN width256 with GELU, learned absolute positions, untied character embedding and final head. The model has 312,448 parameters. The smaller dense control has four blocks. All run 800 steps of batch 16×context 64:819,200 character targets per run. AdamW has a 20-step warmup, cosine cooldown, matrix-only weight decay 0.01, and global gradient clipping at 1.

Treatments:

- Dense six-block baseline.
- Constant increasing layer dropout, maximum 0.4.
- Decreasing increasing-layer dropout, maximum 0.8 at the start and zero at the final step.
- Constant alternating dropout: only original blocks 2, 4, 6 are dropped, each with probability 0.4.
- Dense four-block control, with the same training tokens and fewer parameters/blocks.

All dropout treatments omit 20% of sequence-block work **in expectation**. Training computes each branch and multiplies its output by its mask, so this is an objective experiment, not measured sparse training acceleration. The transformer attention and FFN share one per-sequence mask; each residual branch separately gets the inverse-survival multiplier from Eq. 6. At inference, kept blocks use scale 1 and the original final normalization/readout. No auxiliary losses, adapters, calibration, pruning search, or fine-tuning are used. This implementation does not use CompleteP, ALiBi, squared ReLU or the paper's tokenizer.

Each treatment is selected by full-depth validation CE from five learning rates × two tuning seeds. An initial partial three-rate validation sweep favored its upper boundary; every treatment's grid was extended equally to [.001,.003,.01,.02,.03] before any final evaluation. The earlier partial sweep is archived. Five distinct final seeds give paired comparisons of initialization, sampled training windows and masks. Different tasks use different seed sets.

At evaluation, all 63 nonempty subsets are enumerated for six-block models and all 15 for the four-block model. The primary endpoint is prefix4 CE−full CE, paired against dense6; raw full and prefix CE must accompany it. Secondary summaries average exact subset losses **within a seed** before constructing 95% Student-t intervals across five seeds. These intervals are not adjusted for multiple comparisons and do not cover dataset/split uncertainty. Keeping or deleting original block 1 is distinguished because ILD never drops it during training. Exact subset averages are not ensemble prediction losses.

## Artifacts

`results/protocol.json`, `dataset.json`, `checks.json`, `tuning.csv`, `selected_lrs.json`, `evaluation_runs.csv`, `pruning.csv`, `curves.json`, and `result.json` preserve the executed protocol and measurements. `checkpoints/` contains small local model state dictionaries and is ignored in Git. Run the script to recreate them. The public report is generated from saved outputs, without rerunning training.

Cloud spending is **$0**. Local electricity, hardware ownership and human time are not priced. Per-run training seconds are recorded, but concurrent CPU execution is not a controlled performance benchmark.
