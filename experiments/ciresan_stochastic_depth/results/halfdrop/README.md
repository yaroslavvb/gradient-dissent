# Fixed 50% body-branch dropout study

This is a new fixed-protocol training study, separate from the earlier depth-pruning and inference-dropout experiments. `main-manifest.json` freezes all 16 eligibility subsets of four residual body branches, each at seeds 201, 202, and 203: 48 runs. An eligible branch is independently dropped with probability 0.5 per minibatch during training; retained eligible branches receive gain 2. All four branches run with gain 1 for validation and official-test inference. The learned input stem and classifier remain present.

Every run uses the same Ciresan-width residual MLP, normalized inputs, linear logits, learning rate 0.01, momentum 0.9, batch size 64, and 100 epochs. The fixed split uses 50,000 fitting examples and 10,000 validation examples from the official training set. Each epoch presents 49,984 fitting examples because the final incomplete training batch is omitted. The primary checkpoint is the earliest minimum validation cross-entropy at the fixed epoch 1, 5, 10, …, 100 cadence; epoch 100 is secondary. The official 10,000-example test set is opened only after training and checkpoint selection. This design does not retune learning rates for individual subsets.

## Frozen inputs and qualification

- `main-source-freeze.json` binds the 48-run manifest, exact GPU-executed sources, CPU qualification, and the completed pilot receipt. `root-review.json` approves these exact hashes before dispatch.
- `qualification-manifest.json` covers one bounded GPU invocation containing two independent two-epoch pilots: no eligible body branches and all four eligible branches. Neither pilot loaded or scored official-test data.
- Eight trainer CPU tests, three launcher CPU tests, and six CUDA graph tests passed before the 48 main runs. The main recipe was specified before the pilots; pilots checked implementation and runtime feasibility.
- `failure_watch.py` is an additional CPU-only status/provenance watchdog. It checks completion, the exact source/specification, all 100 epochs, and same-seed initialization, data order, raw mask draws, and dataset hashes. It does not evaluate test metrics. It cancels the main app if any of these checks fail.

## Results and transport

All 48 main runs completed their full 100 epochs without errors. The independent transport audit verified 288 declared downloaded files (264,532,393 bytes), including all 96 prediction NPZ files, plus 48 authoritative server result files (135,876,987 bytes). All three same-seed groups had exact initialization, data-order, and raw mask-draw provenance matches. The three additional transport tests passed, including changed-payload rejection and safe resumption.

Each flat `<run_id>.json` contains the complete result returned by the remote trainer plus one explicitly local field, `local_dispatch_elapsed_seconds`. Each flat `<run_id>-validation.json` contains the validation-only view committed before test access. The analyzer freezes its ordering using these validation-only records before analyzing new test outcomes.

`raw/<run_id>/` preserves the trainer's authoritative `result.json`, `validation.json`, `telemetry.json`, progress and claim records, and selected/final official-test NPZ files. The NPZ files contain all 10,000 examples in their original order: logits, probabilities, predictions, labels, cross-entropy, margins, confidence, and indices. Copying or hashing these files does not make a new checkpoint or select a result.

`<run_id>-download.json` records every trainer-declared artifact's SHA-256, byte count, and Modal Volume path. The large `best.pt`, `final.pt`, and byte-identical selected alias `checkpoint.pt` remain on volume `gradient-dissent-ciresan-20260909`, environment `gradient-dissent-ciresan`, under `/runs/<run_id>/`. These weight files were not downloaded by the transport task; their hashes and sizes remain in each raw result and receipt.

Reproduce the local integrity audit from the repository root:

```sh
uv run python experiments/ciresan_stochastic_depth/halfdrop/verify_transport.py
```

The optional `--download-authoritative` flag retrieves only the 48 original server `result.json` files using CPU-only Modal Volume reads. It does not start GPU jobs. `authoritative-result-download.json` records these downloads. `download-verification.json` recomputes all local hashes and sizes, verifies exact server/local result equality after removing the local dispatch timer, and checks paired provenance. Prediction arrays are not evaluated by this transport audit; the independent scientific analyzer performs that separate step.

## Resources, timing, and billing

Each call is explicitly limited to A100-40GB, two physical CPU cores, 8 GiB host RAM, 180 seconds execution, 90 seconds startup, zero automatic retries, and a two-second container scale-down window. At most four containers run concurrently. The shared telemetry budget ledger is append-only for this study; no earlier reservation was reset or released. The cumulative reserved upper bound after the qualification and 48 main calls is $19.95333965, below the $24 stop guard and the user's $30 shared cap.

`preparation-opening.json` records the verified September 9, 2026 [Modal prices](https://modal.com/pricing), initial environment meter, and stopped preexisting apps. `main-app-state-snapshot.json` records a mid-study check that only this study's main app was active. `budget-closing.json` records the final before/after environment meter, app-stop verification, completed cohort, and timing totals. Environment meter differences are provider snapshots, not an itemized invoice for exactly these calls; delayed posting or unobserved concurrent activity can affect attribution.

The closing meter was $7.16520530, compared with $5.08899130 before qualification: a $2.07621400 environment increment for qualification and main training. The main-only meter increment was $2.05237325. Both study apps were verified stopped with zero tasks; every other returned app was also stopped. First submission through the last local result took 840.94 seconds (14.02 minutes). Summed synchronized training execution was 2,584.34 seconds across the overlapping 48 runs.

`training_seconds` measures synchronized training execution and excludes setup, telemetry, and evaluation. `total_run_seconds` includes the trainer's setup, training, telemetry, evaluation, and test export through its final timer; final artifact hashing, final commit, and network transport occur after that timer. `local_dispatch_elapsed_seconds` additionally includes dispatch, container startup, and returning the result. Summing call times double-counts concurrent intervals and is not study wall time or billed time. `download-verification.json` preserves these totals separately from first-submission-to-last-result elapsed time.
