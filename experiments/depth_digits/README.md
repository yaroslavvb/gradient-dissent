# Depth robustness on 8×8 handwritten digits

**Independent mechanism experiment, 9 September 2026.** Training with layer dropout made this small residual classifier substantially less sensitive to removing its last two blocks. The result concerns cross-entropy robustness; neither an accuracy gain nor a full-depth quality gain was resolved by five training seeds. A separately trained three-block model was better than retaining three blocks of either dropout-trained six-block model.

This uses **scikit-learn's 8×8 digits, not MNIST**. It is not a reproduction of the paper's language-model runs, transformer architecture, speculative decoding, or accelerator performance.

## Main result

Each cell below is a mean over the same five evaluation seeds. CE is cross-entropy in natural-log units per image; lower is better. Prefix inference keeps the first *k* residual blocks and uses the **same final trained LayerNorm and classifier**, with no extra head, auxiliary loss, calibration, fine-tuning, or inference scaling.

| Treatment | Full CE | Full accuracy | Prefix-4 CE | Prefix-4 accuracy | Prefix-4 CE − full CE | Prefix-3 CE |
|---|---:|---:|---:|---:|---:|---:|
| Dense 6 | 0.07996 | 98.33% | 0.15629 | 97.67% | 0.07633 | 0.32074 |
| Constant ILD, pmax 0.4 | 0.07837 | 98.28% | 0.08880 | 98.22% | 0.01044 | 0.11875 |
| Decreasing ILD, pmax 0.8 | 0.07996 | 98.28% | 0.08858 | 98.28% | 0.00862 | 0.10872 |
| Dense 3, separately trained | 0.08279 | 98.33% | — | — | — | 0.08279 |

The declared primary endpoint is the difference from dense-6 in **prefix-4 CE minus the model's own full-depth CE**. Negative values favor ILD. Paired 95% Student-t intervals across five seeds are:

| Comparison with dense 6 | Difference in excess CE | Paired 95% interval |
|---|---:|---:|
| Constant ILD | −0.06589 | [−0.08439, −0.04739] |
| Decreasing ILD | −0.06771 | [−0.08579, −0.04962] |

This improvement is not an artifact of making the full-depth anchor much worse: full-depth CE differences are −0.00159 [−0.00916, 0.00597] for constant ILD and −0.000004 [−0.00691, 0.00690] for decreasing ILD. However, a confidence interval containing zero does not establish equivalence. Prefix-4 accuracy differences are +0.556 percentage points [−0.253, 1.364] and +0.611 points [−0.175, 1.398]; these intervals include zero.

The smaller-model control matters. At three retained blocks, separately trained dense-3 reaches CE 0.08279, versus 0.11875 for constant ILD and 0.10872 for decreasing ILD. Dense-3 uses 55,050 parameters; the six-block models have 105,162. This result supports reusable subnetworks inside the deeper model, while leaving a clear advantage for separately training a model at the desired smaller depth on this task.

## All masks, including a failure boundary

We evaluate **all 63 nonempty layer subsets**, in original order, for every six-block model, and all seven subsets for dense-3. There is no mask sampling error. Within each retained depth we average masks **within each training seed**, then compute intervals across the five seeds. Masks and test examples are never counted as independent training replicates.

At retained depth four, averaged over all masks:

| Treatment | CE, all 15 masks | CE, 10 masks keeping block 0 | CE, 5 masks dropping block 0 |
|---|---:|---:|---:|
| Dense 6 | 0.23639 | 0.19253 | 0.32411 |
| Constant ILD | 0.15271 | 0.09574 | 0.26665 |
| Decreasing ILD | 0.15649 | 0.09840 | 0.27267 |

The schedule always retains block 0 during training. Dropping it at inference is outside the training mask distribution and sharply increases loss. ILD therefore provides incomplete robustness to arbitrary layer deletion. A prefix, which preserves block 0, is substantially easier than an unrestricted four-block subset. Odd/even alternating masks are separately tagged in the raw data and aggregate JSON.

## Protocol

- **Data:** the 1,797-row dataset returned by `sklearn.datasets.load_digits`; fixed stratified splits of 1,078 training, 359 validation, and 360 test rows. The split seeds are 913071 and 913072. Per-pixel normalization is fitted only on the training rows. Raw-array hash, split indices, class counts, and normalization parameters are in [dataset.json](dataset.json).
- **Dataset scope:** scikit-learn documents that these images are a copy of UCI's original test set. Our new random row splits are **not UCI's original writer-disjoint split**. Writer-level or split-level generalization is not measured. See the [official scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) and [UCI dataset description](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
- **Model:** 64-pixel linear embedding to width 64; six residual blocks, each `LayerNorm → Linear(64,128) → GELU → Linear(128,64)`; final LayerNorm and ten-class linear head. Each block has one residual branch. Dense-3 uses its first three blocks, initialized identically to the first three of the six-block models for the same seed. Embedding and head initialization are also shared.
- **Training:** 600 optimizer steps, batches of 128 training rows sampled with replacement (76,800 image presentations). AdamW, weight decay 0.01, default betas, global gradient norm clipped to 1. The learning rate follows cosine decay from its selected value to 10% of that value. No early stopping. Paired treatments use identical initialization and minibatch streams and a separate seeded mask generator.
- **ILD:** independent Bernoulli masks per image and residual block, with training-only inverted scaling `1/(1-p)`. For zero-based layer `l`, `p(l,t) = pmax × l/(L−1) × f(t)`. Constant ILD uses `pmax=0.4`, `f(t)=1`; decreasing ILD uses `pmax=0.8`, `f(t)=1−t/(T−1)`. Both average exactly 20% dropout over layers and steps. The first block is never dropped.
- **Budget comparison:** all treatments have equal optimizer steps and data presentations. Expected active residual work is 2,880 layer steps for ILD, 3,600 for dense-6, and 1,800 for dense-3. This implementation computes every branch before masking; active-work accounting models ideal skipping and is **not a measured FLOP, elapsed-time, or energy saving**. Embedding, classifier, and optimizer work are not included in those layer counts.
- **Tuning:** every treatment receives the same five learning rates `{0.0001, 0.0003, 0.001, 0.003, 0.01}` and two tuning seeds, 1701 and 1702. Select by mean final **full-depth validation CE**, never test or reduced-depth metrics. All treatments select the interior value 0.0003. Evaluation seeds are 2701–2705.
- **Inference:** eval mode, retained branches at scale 1, original block order, same trained final normalization and classifier. Full depth, every prefix, all nonempty masks, masks preserving/dropping block 0, and alternating subsets are reported.
- **Statistics:** paired t intervals have df=4. They describe training-seed variation conditional on this fixed data split and selected hyperparameters. They do not include uncertainty from dataset choice, split choice, LR selection, or architectural choice. Secondary comparisons are exploratory and are not adjusted for multiplicity.

## Tuning sensitivity and execution audit

The initial equal LR grid `{0.001, 0.003, 0.01}` selected its lowest value for all four methods. That validation boundary triggered the same downward extension for every treatment. The completed initial run, including its test outputs, remains archived in [initial_grid](initial_grid/). Test metrics did not determine the extension or the selected LR; both selections use only full-depth validation CE.

The initial grid's primary excess-CE differences were −0.01640 for constant ILD and −0.01417 for decreasing ILD, also favoring ILD. The sign survives the extension, but the effect size changes substantially. Dense-6's mean validation CE is nearly tied at LR 0.0003 (0.16393) and LR 0.001 (0.16496), making LR-selection uncertainty a material limitation beyond the plotted seed intervals.

There were 44 initial and 60 final training executions, **104 total**, all local CPU. Repeated configurations reuse the same seeds and are not additional independent evidence. The final run contains 40 tuning runs, 20 evaluation runs, and 980 test-mask evaluations. Measured training time summed to approximately 47 seconds in the final run; final experiment wall time was 48 seconds, after imports. The initial run took 37 seconds. Environment versions and source hash are recorded in [result.json](result.json).

**External experiment spend: $0 of the $50 cap.** No Modal, cloud GPU, or paid API was used. [spending.json](spending.json) records the ledger; local electricity and hardware amortization are not priced.

## Reproduce and inspect

From the repository root:

```sh
experiments/.venv/bin/python -m pip install -r experiments/depth_digits/requirements.txt
experiments/.venv/bin/python experiments/depth_digits/run.py
experiments/.venv/bin/python experiments/depth_digits/verify.py
```

The script asserts disjoint exhaustive splits, train-only normalization, shared initial parameters, prefix/allkeep/full equivalence, deterministic dropout-free inference, finite input/parameter gradients, exact schedule means, paired data-stream hashes, and theoretical active-work totals. The independent persistence audit passed for source hash, all 980 mask rows, full-depth anchors, selected interior LRs, and primary paired confidence intervals.

Artifacts: [training code](run.py), [independent audit](verify.py), [aggregate result](result.json), [per-run records](runs.csv), [all mask scores](masks.csv), [seed-level mask averages](seed_summary.csv), [all tuning records](tuning.csv), [selected LRs](selected_lrs.json), and [training trace](training_trace.csv).
