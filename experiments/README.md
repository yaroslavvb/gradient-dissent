# Local falsification and measurement suite

These experiments support a critical review of [Don't Drop Dropout, arXiv:2609.05275v1](https://arxiv.org/abs/2609.05275). They test specific mathematical and implementation claims. They do **not** reproduce LLM pretraining, Cerebras throughput, GPU throughput, energy consumption, or speculative decoding.

## Reproduce

From the repository root:

```bash
python3 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -r experiments/requirements.txt
experiments/.venv/bin/python experiments/run_experiments.py
```

The original run used Python 3.14.4, PyTorch 2.14.0, NumPy 2.5.3, macOS on an Apple M5 Max, CPU execution, and one PyTorch thread. Complete machine metadata is in `docs/data/experiment-results.json`. The virtual environment is ignored by Git. Wall-clock results will change with hardware, software, system load, and timing noise.

Use `--part exact`, `--part timing`, or `--part training` for a single stage. Run stages sequentially because they update the same combined report JSON. The complete suite takes tens of seconds on the original machine. No external dataset, credential, or network access is needed after dependency installation.

## Protocol

1. **Exact expectation checks.** Enumerate all masks for a scalar single residual branch, two affine sublayers with the paper's recommended shared mask, and two nonlinear branches with independent masks. Verify conditional expectation, full-block mean, half-MSE objective, gradients, and the finite discrete ILD+DTS mean dropout. No sampling error. An additional deterministic AdamW check shows why zero versus absent gradients require an explicit policy when implementing real skips.
2. **Execution benchmark.** Six GELU MLP residual blocks, two tensor sizes, 50% retained work. Compare dense execution, full compute followed by masking, gather/compute/scatter of active sequences, and true whole-batch skipping. Forward and input/parameter-gradient equivalence are checked for the same masks in float64, including all-kept and all-dropped cases. Timing uses float32, precomputed balanced masks, four warmups, nine randomized-order repetitions, and sixteen executions per repetition. Balanced masks isolate implementation overhead; they are deliberately not independent Bernoulli draws. Forward+backward timings exclude the optimizer.
3. **Training experiment.** Fresh Gaussian inputs at each step and a fixed analytic nonlinear regression teacher. Compare dense six-block, dense five-block, constant ILD, decreasing ILD, and increasing ILD residual MLPs. Constant ILD uses maximum dropout .4; both ramps use maximum .8, yielding mean dropout .2 in each dropout treatment. Matched-step runs use 240 steps. Matched expected active sequence-block budgets use 240 steps for dense6, 300 for dropout, and 288 for dense5. All methods independently receive the same five-rate learning-rate search with two tuning seeds. Final results use six separate evaluation seeds, a distinct test set, and paired Student-t intervals. The initial three-rate pilot placed all winners at the upper boundary; the grid was expanded equally before final interpretation. All final choices are the interior rate .03.

**The training implementation computes all rows before masking. Its expected active sequence-block budget models ideal skipping; it is not measured FLOPs, elapsed time, or energy.** The execution benchmark separately tests real skipping. Embedding, readout, optimizer, normalization, attention, and communication costs are absent from the budget model. The toy architecture contains no attention and does not implement CompleteP.

The final sweep consists of 100 tuning runs and 60 evaluation runs. Dense6 is deterministically repeated in the two regimes, so these are optimizer-run counts, not counts of statistically independent samples. The six evaluation seeds, not individual test examples or timing repetitions, are the units used for training intervals. No multiple-comparison correction is applied.

## Outputs

- `results/exact_checks.csv`: exact predictions, objectives, and gradients for six dropout probabilities.
- `results/exact_results.json`: checks and the AdamW skip-policy example.
- `results/timing_raw.csv`: every timing repetition.
- `results/timing_summary.csv`: per-configuration medians, ranges, and speedups.
- `results/timing_results.json`: timing summaries and equivalence checks.
- `results/training_tuning.csv`: all learning-rate search outcomes.
- `results/training_evaluation.csv`: all held-out seed outcomes and actual/expected active work.
- `results/training_curves.json`: per-seed validation curves against steps and realized active work.
- `results/training_results.json`: protocol, confidence intervals, selected rates, and paired differences.
- `results/verification.json`: source SHA-256, raw/report consistency checks, and exact repeatability checks on dense6 and decreasing-ILD evaluation seed 100.
- `../docs/data/experiment-results.json`: combined source for the interactive report.
- `../research/experiment-findings.md`: interpretation and limitations.

The assertions in the script verify the exact identities, numerical forward/backward equivalence, schedule accounting, and finite training losses. Training uses fixed independent generators for initialization, data, and masks. Evaluation is deterministic with all layers enabled, except for the explicitly labelled depth-four early-exit test.
