# Ciresan-width MNIST stochastic-depth experiment

Protocol drafted 2026-09-09 before any official test evaluation. Pilot runs may
choose the common training duration or correct implementation bugs; their scores
are exploratory. The tuning and final manifests will freeze the executed study.

## Question and identifying comparison

Does stochastic depth improve optimization or generalization in the tapered,
11,972,510-parameter MNIST MLP in `train_ciresan_new.py`? The source is not a
ResNet. Its widths are 784 → 2500 → 2000 → 1500 → 1000 → 500 → 10. Keep the stem
and classifier compulsory and add a fixed prefix-crop bypass to the four middle
transitions. No new trainable parameters are introduced. For a transition from
width a to b, P(h)=h[:b] and the adapted block is ReLU(P(h)+M/(1−p)·(Wh+b)).
The crop discards coordinates: it is not an identity shortcut on the wider space.

Compare plain versus residual-dense to measure architectural effects; compare
residual-dense versus residual-SD to identify the effect of dropping branches.
Dropping a branch genuinely omits its affine operation, not just its output.
Inference uses all branches at scale one. Gradient/momentum updates for a skipped
branch are absent in that minibatch. The expectation statement applies to the
pre-ReLU branch contribution, not to the nonlinear network output.

## Main controlled adaptation

Pixels divided by 255; linear output logits; default PyTorch Linear initialization;
ReLU hidden activations; no augmentation, normalization layers, or weight decay.
These input/head choices are explicit changes from the source, investigated in
separate historical-fidelity checks. This is not a reproduction of Ciresan's
original augmentation-heavy record-setting system.

Five arms, all with the same parameter count:

1. Plain MLP, no dropout.
2. Residual-dense, no dropout.
3. Residual + constant SD: p=.4 × [1,2,3,4]/4.
4. Residual + decreasing SD: p=.8 × [1,2,3,4]/4 × (1−epoch/(E−1)).
5. Residual + independent unit dropout, p=.2 after each of the five hidden ReLUs.

The two SD arms have the same expected integrated number of omitted branches.
Constant .4 omits one of five hidden affine layers per minibatch in expectation;
decreasing .8 starts at two and finishes at zero. Both average 20% omission by
hidden-layer count, but only 14.626% of nominal affine multiply-accumulates because
the widest, most expensive middle transition is dropped least. Unit dropout is
an ordinary regularization comparator, not an exposure-matched SD intervention.
The residual-plus-unit-dropout arm also drops shortcut activations.

Use SGD with momentum .9, batch64, constant LR, FP32 parameters and TF32 matrix
multiplication. Train data are resident on GPU; each epoch shuffles all training
examples, then drops the last incomplete minibatch, as the source does. All arms
share initial tensors and shuffled indices for each seed; masking has a separate
RNG stream. No curvature instrumentation or expensive per-step validation.

## Data and selection

One fixed random permutation (seed20260909) splits original MNIST train60,000
into train50,000 and validation10,000. Official test10,000 is separate. No test
files are opened during pilots/tuning. Record download hashes and split hashes.

Plan equal LR tuning for each main arm over {.01,.03,.1}, with seed1 and the same
full training horizon as final runs (target100epochs, confirmed from pilot speed).
Select LR by lowest validation cross entropy across the scheduled checkpoints.
Evaluate each selected arm with five fresh paired seeds101–105. Select each run's
checkpoint by validation cross entropy. This validation-selected checkpoint is
the primary endpoint; the last fixed-epoch model is the secondary endpoint.
Test accuracy never selects recipes, LRs, epochs, or follow-up interventions.
Both endpoints are reported, not whichever makes SD look better.

Historical fidelity controls use raw0–255 pixels, lr.001, momentum.9, batch64,
and direct parameter shrinkage p←(1−.00002)p each step. Frozen fidelity variants are
plain/ReLU-head, plain/linear-head, residual/ReLU-head, and constant-SD/ReLU-head (pmax=.4). Three fresh seeds
share initialization and data order. These controls are descriptive and are not
part of the equally tuned five-arm main comparison. The 50k split differs from
the historical60k training runs, so historical W&B accuracy is context only.

## Measurements and uncertainty

Log validation loss/accuracy, deterministic dense-inference training-probe
loss/accuracy, stochastic training loss, sparse per-layer gradient norms, zero
logit fraction, predicted-class histogram, actual skip counts, and synchronized
training time. Evaluate the full training set at final/selected checkpoints.
Report held-out errors out of10,000, means and sample SD across seeds, paired
accuracy differences relative to the residual-dense control, and 95% Student-t
intervals over seed-paired differences. These intervals measure seed uncertainty
on one fixed data split/test set; they are not population confidence intervals.
Do not treat five uses of the same test set as50,000 independent test examples.
Save error indices to inspect paired disagreements without choosing new runs.

Training time includes optimizer and diagnostics in the training loop; excludes
data loading, evaluation, and checkpointing. Total run time also includes those
costs. Modal billed time can include startup/idle/build costs. Distinguish nominal
MAC reduction, measured training speed, and actual invoice-style metering.

## Compute limit and stopping

New isolated Modal environment `gradient-dissent-ciresan`, no premium region or
nonpreemptible multiplier. A100 rate .000583/s, max2physicalCPU cores at.0000131/s
and max8GiB RAM at.00000222/GiB/s, checked2026-09-09 at modal.com/pricing.
Reserve each invocation's entire timeout plus130s startup/idle allowance BEFORE
dispatch. Never release old reservations or automatically retry a failed run.
A local locked ledger rejects aggregate reservations above$24, leaving$6 of the
user's$30 cap for image builds, billing lag, transfers, and interruptions.
An independent metering monitor cancels outstanding calls at$24. No deployment
or scheduled training. At most six A100 jobs run concurrently; scaledown2seconds.
Stop all apps after completion and report observed metered spend.

## Pilot decision and freeze

All six five-epoch pilots completed, without official test access. Measured
training time was6.4–13.4seconds for five epochs. Freeze100epochs for every
tuning/final run, timeout420seconds per invocation. Source residualization
produced all-zero logits and chance validation accuracy at seed1; retain it and
a raw-output-ReLU SD arm as explicit failure controls. Main methods use the
normalized, linear-logit adaptation specified above. No early stopping or
method-specific extension. Five arms×three learning rates=15tuning runs; five
selected arms×five new seeds=25main final runs; four fidelity arms×three seeds=12
fidelity final runs. Planned52full runs+six pilots reserve approximately$19.12
including CPU preparation, below the$24 dispatch ceiling and$30 absolute budget.


## Resumed protocol, frozen after baseline optimization

On September 9, the user explicitly prioritized optimizing the baseline to
98.63% before continuing this study. The first Torch2.8 tuning batch was stopped;
its completed and abandoned runs remain archived and charged to the same ledger.
No old tuning outcome is used to select the resumed experiment. The separate
baseline optimization used official test feedback and is reported as such.
Thus the test set is no longer untouched globally, even though the main SD
comparison below remains fixed and selects learning rates/checkpoints solely
by its independent validation split. Treat its results as exploratory.

The resumed study uses PyTorch2.14/CUDA13 and the optimized CUDA Graph runner
for every arm. An A100 qualification passed all six tests, including the
16 mask topologies, switching masks/scales, state restoration, and fresh
unit-dropout RNG. Replaced scalar division with FP32 reciprocal multiplication
agrees numerically within tolerance; bitwise equivalence with the old runner
is not claimed. A guard-only validation change after qualification enforces
the five recipe/probability combinations. Model and kernel code are unchanged.

Keep the already chosen100epochs, FP32/TF32, B64, fixed50k/10k split, five main
recipes, three-LR grid and constant learning rate. Do not transfer the separately
test-tuned BF16/B256/dropout/LR-schedule recipe into this causal comparison.
Masks genuinely skip compute and skipped-branch momentum updates. Graph capture
and epoch preparation are timed separately from training.

To keep the complete optimization plus SD work under the same$30 cap, use
three fresh final seeds101–103, not five. The fidelity cohort has three arms
(plain raw/ReLU, residual raw/ReLU, SD-constant raw/ReLU), each at three seeds.
The dropped fourth fidelity arm, plain raw/linear, is investigated separately
by the explicitly test-monitored60k baseline head diagnostic; it is not a
fourth matched50k fidelity control. There are15 new tuning runs,15 main final
runs, and9 fidelity final runs. Timeout180s per call; no automatic retries.
Original reservations remain counted. The new manifests and source hashes
are frozen before resumed tuning; LR selection/final manifests are frozen
before any resumed final test evaluation.
