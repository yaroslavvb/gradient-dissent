# Curated W&B report

`build_wandb_report.py` creates one identifiable report in the user-requested
`yaroslavvb/gradient-dissent` project. It requires all 18 exact source runs to be
verified in `results/telemetry/wandb-upload-manifest.json` before publishing.
It does not train, upload run histories, move runs, change visibility, or touch
unrelated reports.

From the repository root:

```sh
uv run python experiments/ciresan_stochastic_depth/telemetry/build_wandb_report.py
uv run python -m unittest discover -s experiments/ciresan_stochastic_depth/telemetry -p test_build_wandb_report.py
uv run python experiments/ciresan_stochastic_depth/telemetry/build_wandb_report.py --publish
```

The first command writes a local plan without network access. Publication checks
for an exact title plus ownership marker and reuses a unique matching report.
If a save is interrupted, rerunning performs this lookup before saving, rather
than immediately creating another report. Ambiguous matches stop the process.

The eight panels show controlled train/validation cross-entropy and accuracy,
two fixed-probe gradient statistics, and two official-test baseline target views.
Seven axes use saved `training_seconds`; the last uses saved
`run_elapsed_seconds`, including setup and telemetry. The 15 controlled runs
and three baseline replays have separate exact-ID run filters. Each seed is an
individual trace; there is no smoothing, seed aggregation or confidence band.
The baseline threshold is a stated official-test stopping goal, not evidence
from an independently held-out test protocol.

The script reloads the saved report through the SDK and compares all eight
panel axes/metrics, canonical filter trees, aggregation and smoothing settings.
Only then does `results/telemetry/wandb-report-manifest.json` receive the actual
verified URL. It also records a minimal read-only project visibility check;
null is reported as unavailable. Authentication success or an HTTP page alone
is not treated as proof of anonymous access.

The implementation uses W&B's official
[Reports API](https://docs.wandb.ai/models/ref/wandb_workspaces/reports) and
[report editing workflow](https://docs.wandb.ai/models/reports/edit-a-report),
with `wandb-workspaces` locked in the root uv environment. The API is documented
as public preview; the offline tests exercise the installed SDK's serialization
round trip to catch configuration changes.
