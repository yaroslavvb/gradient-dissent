# Six-affine inference follow-up

The successful panel is `manifest-v2.json` / `source-freeze-v2.json`: three
seeds, five original recipes, selected and final checkpoints, and all 64 masks.
The 30 states use the original graph-evaluation checkpoints, not the later
telemetry replays. No training, fitting, checkpoint selection or W&B mutation
occurred. The gallery indices were frozen from the existing page before these
new interventions in `gallery-freeze.json`.

The experiment protocol is
[FULL_DEPTH_PROTOCOL.md](../../layerdrop_hypotheses/FULL_DEPTH_PROTOCOL.md).
The absent stem becomes zero-padded normalized pixels; absent body branches
use the existing crop/ReLU bypass. An absent classifier means abstention and
immediate exit before any affine operation: coverage and correct-output rate
are zero, while classification accuracy and CE are undefined. The expanded
endpoint surgery is an exploratory follow-up on the previously used test set.

The initial v1 wrapper failed to import a sibling module during Modal function
hydration, before the evaluator ran. Its app was explicitly stopped. The
failed results, original manifest/freeze, and reservations are retained; the
original wrapper and test source are archived in `attempt-v1/`. The wrapper
import path was corrected and checked by an isolated remote-style import
smoke test. V2 received new run IDs and a separate source freeze/reservation.
The evaluator and scientific protocol were unchanged. Thirteen CPU tests
qualified the evaluator and launcher before dispatch.

All three v2 jobs completed on NVIDIA A100-SXM4-40GB GPUs, with 180-second
function and 90-second startup limits, two CPU cores, 8 GiB RAM limits,
three maximum containers, no configured input retries, and a two-second
scale-down window. The shared telemetry ledger retains both attempts:
$1.16614560 in new conservative reservations, raising its cumulative upper
estimate to $10.42981725, below the $24 guard and $30 shared cap.

The closing provider meter was **$5.08899130**, versus **$5.00375173** before
this follow-up: **$0.08523957** additional metered cost, including v1.
Both experiment apps were confirmed stopped with zero tasks; all apps
returned by the environment's running/deployed/recently-stopped inventory
were stopped. Provider metering can lag final shutdown, so the conservative
reservation is retained. See `budget-closing.json`; published rates were
checked against [Modal pricing](https://modal.com/pricing), and the explicit
40 GB request follows [Modal's GPU documentation](https://modal.com/docs/guide/gpu).

`raw/` preserves 69 downloaded files (93,438,599 bytes), including 30 NPZ
per-example panels. Every declared remote artifact SHA-256 and file size
matched; `download-manifest.json` records the checks. The three top-level
`full-depth-s*-v2.json` files contain complete compact model summaries for
the website; their only addition to the remote result is the separately
named local dispatch clock.

Independent local recomputation verified all **1,920 mask summary rows**,
**1,408 gallery records**, and **4,800,000 original-mask predictions**.
Every original prediction and original-mask mean CE matched exactly.
Recomputing remote summary means with the local NumPy version produced at
most 1.78e-15 reduction roundoff; the audit records a 1e-12 local summation
tolerance, without changing the frozen evaluator parity criterion.
See `independent-verification.json`. NPZ predictions use -1 for abstention;
only the corresponding CE rows contain NaN. Public JSON uses null instead.

Reproduce all downloaded-file hash checks and the independent numeric audit
locally, with no cloud calls:

```sh
uv run python experiments/ciresan_stochastic_depth/verify_full_depth.py
```
