# Instrumented Ciresan reruns

Status: **complete_verified**. 18 of 18 declared reruns verified; 0 pending. All declared and additional attempts are retained.

| Run | Status | Last epoch | Original scientific history | Training s | Telemetry s | Telemetry / training |
|---|---|---:|---|---:|---:|---:|
| telemetry-main-plain-s101-v1 | verified | 100 | exact | 62.372 | 1.324 | 2.12% |
| telemetry-main-plain-s102-v1 | verified | 100 | exact | 62.358 | 1.328 | 2.13% |
| telemetry-main-plain-s103-v1 | verified | 100 | exact | 61.660 | 1.721 | 2.79% |
| telemetry-main-residual-s101-v1 | verified | 100 | exact | 66.499 | 3.947 | 5.94% |
| telemetry-main-residual-s102-v1 | verified | 100 | exact | 66.082 | 0.862 | 1.30% |
| telemetry-main-residual-s103-v1 | verified | 100 | exact | 66.073 | 0.887 | 1.34% |
| telemetry-main-sd_constant-s101-v1 | verified | 100 | exact | 56.800 | 3.165 | 5.57% |
| telemetry-main-sd_constant-s102-v1 | verified | 100 | exact | 56.839 | 1.236 | 2.18% |
| telemetry-main-sd_constant-s103-v1 | verified | 100 | exact | 56.909 | 0.857 | 1.51% |
| telemetry-main-sd_annealed-s101-v1 | verified | 100 | exact | 56.993 | 2.581 | 4.53% |
| telemetry-main-sd_annealed-s102-v1 | verified | 100 | exact | 56.757 | 1.151 | 2.03% |
| telemetry-main-sd_annealed-s103-v1 | verified | 100 | exact | 56.828 | 1.152 | 2.03% |
| telemetry-main-residual_unit_dropout-s101-v1 | verified | 100 | exact | 68.936 | 0.853 | 1.24% |
| telemetry-main-residual_unit_dropout-s102-v1 | verified | 100 | exact | 68.685 | 0.899 | 1.31% |
| telemetry-main-residual_unit_dropout-s103-v1 | verified | 100 | exact | 68.527 | 1.550 | 2.26% |
| telemetry-baseline-s101-v1 | verified | 24 | exact | 6.048 | 1.121 | 18.54% |
| telemetry-baseline-s102-v1 | verified | 23 | exact | 5.806 | 0.821 | 14.13% |
| telemetry-baseline-s103-v1 | verified | 21 | exact | 5.293 | 0.796 | 15.05% |

## Instrumentation qualification

Status: verified.

- telemetry-qualification-v1: scientific parity verified; 10% overhead gate not passed.
  - baseline-s101: exact final parameter equality=True; exact scientific history=True; telemetry 0.738 s (9.77% of training).
  - controlled-s101: exact final parameter equality=True; exact scientific history=True; telemetry 0.765 s (11.55% of training).
  - controlled-s102: exact final parameter equality=True; exact scientific history=True; telemetry 0.414 s (6.24% of training).
- telemetry-qualification-v2: scientific parity verified; 10% overhead gate passed.
  - baseline-s101: exact final parameter equality=True; exact scientific history=True; telemetry 0.320 s (4.22% of training).
  - controlled-s101: exact final parameter equality=True; exact scientific history=True; telemetry 0.483 s (3.63% of training).
  - controlled-s102: exact final parameter equality=True; exact scientific history=True; telemetry 0.259 s (1.95% of training).

## Verification limits

- Exact recorded curves and initial tensors do not prove exact final weights when the original run did not save a final parameter hash. New trained-final and reloaded-selected hashes remain separate.
- Qualification on/off pairs test instrumentation with byte-identical end states; they are not additional independent experimental seeds.
- Telemetry evaluation reduces CE differently from legacy summed minibatch evaluation. Replay equality is checked on the unchanged legacy scientific history; the diagnostic curve remains explicitly separate.
- Three repeated seeds reuse the same MNIST data. The baseline recipe was selected using official-test outcomes; these reruns add measurements, not independent generalization evidence.
- Training seconds exclude evaluation/telemetry/checkpoint I/O. Invocation seconds include those costs; driver time also includes dispatch/startup. Historical W&B runtime has a different unknown-hardware/instrumentation scope.
- Historical data: All count-verified public evaluation rows (91 retained observations).
