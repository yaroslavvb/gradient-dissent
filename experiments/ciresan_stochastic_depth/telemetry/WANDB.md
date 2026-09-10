# W&B export and chart integration

Checked 9 September 2026. The exporter is ready for completed rerun results. Following the explicit request for W&B run links, W&B SDK 0.30.0 was installed in the experiment virtual environment and its offline export path was tested. No online upload, project creation, login, or paid compute was performed during preparation. The JSONL path works without W&B and does not block the public Pages report.

## Availability and historical provenance

Initial presence-only checks found no installed `wandb` SDK and no configured W&B API-key environment variable, usual settings/credentials file, or netrc login. Read-only Modal secret-name inventories found no W&B-named secret in either the experiment environment or the default environment. The SDK was subsequently installed; the last preparation-time authentication recheck still found no configured login. Secret values were never printed or retained in repository artifacts. This is an availability snapshot, not a claim about every possible account or machine.

The existing [history inventory](../history/README.md), `run-metadata.json`, `selected-histories.json`, and `source-audit.json` already preserve the useful sanitized evidence from [yaroslavvb/train_ciresan](https://wandb.ai/yaroslavvb/train_ciresan). They contain all 327 run summaries and selected extrema/evaluation records, not complete per-layer diagnostic curves. No maintained local history-retrieval script or W&B connector was found. The earlier inventory used public GraphQL `sampledHistory` with row-count checks; sampled summaries must not be drawn as a reconstructed dense history.

The historical source's `val_accuracy` is **official MNIST test accuracy in percent**; `val_loss` is official-test mean cross-entropy. Historical `_runtime` includes evaluation, diagnostics and logging, with unverified GPU hardware. It is unsuitable as an isolated GPU-training clock. Modern controlled runs have a real validation split, which remains `validation/accuracy_pct` and `validation/loss`. The baseline speed experiment monitors official test and is separately labeled. See the [metric inventory](METRIC_PLAN.md) for the original statistics and known source bugs.

## Export contract

`wandb_export.py` consumes the recorder's `telemetry_history`: measured `epoch`, `training_seconds`, optional `run_elapsed_seconds`/`optimizer_step`, flat scalar `metrics`, and nested `diagnostics`. It preserves recorded clocks and finite metrics, flattens scalar diagnostics, and counts omitted null, text, or array diagnostics. Full arrays and explanations remain in the original result file. It accepts the recorder's `/loss` names as well as explicit `/ce` fields. Original `val_*` aliases are emitted only from explicit official-test metrics; ambiguous input aliases require a schema declaration and are checked for conflicts. True validation never acquires a historical test alias.

The default command writes deterministic JSONL plus a SHA-256 manifest. It requires no network, credentials, or SDK:

```sh
python experiments/ciresan_stochastic_depth/telemetry/wandb_export.py \
  path/to/completed/result.json \
  --output-dir experiments/ciresan_stochastic_depth/results/telemetry/wandb-export
```

Only `--mode offline` or `--mode online` imports the optional SDK. Offline creates native W&B files after training without login. Online requires an existing configured login and creates a new content-derived `telemetry-*` run under `gradient-dissent-ciresan`, grouped by `ciresan-telemetry-20260909`. The importer uses `resume="never"` and `reinit="create_new"`; it does not resume, overwrite or finish an unrelated active run, and explicitly rejects the historical `train_ciresan` project. Existing-ID failures are intentional protection against accidental duplicate imports. W&B documents these run-initialization semantics in its [init reference](https://docs.wandb.ai/models/ref/python/functions/init).

Native offline files can later be uploaded with [`wandb sync`](https://docs.wandb.ai/models/ref/cli/wandb-sync). **The SDK-free JSONL is not itself a native W&B offline file**: replay the completed result through the exporter's offline/online SDK mode first. Do not replay online and then sync another offline copy with the same run ID.

The SDK import disables console capture, code/git capture, machine metadata, system metrics and requirements collection. It receives whitelisted scientific configuration, scalar metrics and source hashes; not checkpoints, files, environment dumps or credentials. Some privacy settings use W&B's documented `x_` internal options, which can change between SDK releases; an incompatible SDK should fail rather than silently remove these controls. Bounded initialization/finalization waits affect only the importer, never recorded training time. See the current [Settings reference](https://docs.wandb.ai/models/ref/python/experiments/settings).

## Timing and chart semantics

W&B sends data asynchronously, but local logging, conversion, serialization and final synchronization still cost time. Its documentation recommends batching related metrics and using offline logging when appropriate. This implementation adds **no W&B calls to training**: it replays each saved epoch boundary afterward, one scalar dictionary per row. The rerun separately measures its actual diagnostic overhead. [Asynchronous logging](https://docs.wandb.ai/support/models/articles/does-logging-block-my-training), [performance guidance](https://docs.wandb.ai/models/track/limits).

Every scalar is registered with `run.define_metric(..., step_metric="training_seconds", step_sync=False)`, and the measured clock is logged in the same row. `--axis epoch` or `--axis run_elapsed_seconds` selects another measured default; unavailable axes are rejected. W&B's own `_step` is the importer row index and `_runtime` is importer duration. Neither is relabeled as training. The interface can select the explicitly recorded axes using [custom log axes](https://docs.wandb.ai/models/track/log/customize-logging-axes).

Recommended main charts are: train and validation loss/accuracy against measured training seconds, a separate elapsed-time view, per-layer dense-probe gradient norms and zero fractions, parameter displacement between snapshots, and declared stochastic-depth probabilities. Official-test speed curves and validation-controlled learning curves belong in separately titled panels. Use unsmoothed seed traces; aggregate intervals should retain their small-seed qualification. Expensive curvature remains a separately labeled post-training diagnostic.

W&B offers programmatic report and workspace creation through the separate `wandb-workspaces` package, currently described as Public Preview. A new `wandb_workspaces.reports.v2.Report` can contain `PanelGrid`/`LinePlot(x="training_seconds", y=[...])` blocks and be saved; workspaces expose `Workspace` and `Section` objects. A future authenticated integration should create a **new** report/workspace in the new project, with small curated sections instead of hundreds of automatic panels. No report/workspace mutation was attempted here. [Reports API](https://docs.wandb.ai/models/ref/wandb_workspaces/reports), [Workspaces API](https://docs.wandb.ai/models/ref/wandb_workspaces/workspaces).

## Verification

Fifteen exporter CPU tests pass using a fake SDK, with no network calls. They cover source immutability, clock preservation, absent-axis refusal, units and test/validation aliases, finite values, safe configuration, new run IDs, JSONL checksums, historical-project rejection, metadata controls and custom-axis logging. The actual recorded 837-metric row also passes: the sanity bound is 2,048 total fields to include its additional scalar diagnostics, with no truncation. All 18 completed reruns were exported as JSONL, and every original finite metric, recorded clock, source hash and JSONL hash was checked. A real SDK offline smoke test produced a native W&B file from a complete 22-row run in a temporary directory; it uploaded nothing and the temporary files were removed. Online behavior still awaits authentication. Run the tests with:

```sh
python -m unittest experiments.ciresan_stochastic_depth.telemetry.test_wandb_export -v
```

## Verified run-link publication

`upload_wandb.py` prepares `results/telemetry/wandb-upload-manifest.json`, an 18-entry registry whose `verified_wandb_url` fields start as null. It also copies frozen validation-selected and final test endpoints, target-reaching timestamps where available, and measured training/run/telemetry times into an explicit scientific summary. These additions do not change the existing history hashes or content-derived IDs. There is no new checkpoint or hyperparameter selection.

After the user has authenticated locally, the upload command is:

```sh
experiments/.venv/bin/python experiments/ciresan_stochastic_depth/telemetry/upload_wandb.py --upload
```

The driver verifies existing runs before writing. A complete matching run is reused; a conflicting run is refused; an active partial run is left alone. A non-active partial run can resume only after its config and contiguous metric/clock prefix match the exact source. Each worker has a 180-second wall limit and each source gets at most three attempts per invocation. Uncertain writes are read back under the same ID before retries. An interrupted driver lock requires inspection of the process before removal; do not run competing upload drivers.

The registry exposes the SDK-returned URL only after the public API retrieves the actual finished run, every expected scalar/clock cell, all source hashes, and frozen endpoint summaries. `scan_history` is used without key intersection or cached history so sparse metrics are retained. Eight additional CPU tests exercise prefix conflicts, completion and URL gating, endpoint units, and the exact 18-source registry. No model weights or checkpoints are uploaded. [Public API](https://docs.wandb.ai/models/ref/python/public-api/api), [run-history interface](https://docs.wandb.ai/models/ref/python/public-api/run).
