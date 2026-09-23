# E5 Contract Note — offset exclusion, caught and reversed pre-run

**Status:** note, **not an amendment.** Nothing had run; the contract as written was
restored rather than changed. Recorded because the loader briefly deviated and the
record should show it.

---

## 1. What happened

The generality contract (§2 E5) specifies the HAR shift set as *"(a) fixed
per-channel gain/**offset** miscalibration, (b) axis rotations, (c) channel
reordering."*

The first implementation of `src/data/har_shift.py` **excluded offsets**, on the
reasoning that the locked adapter spec (§4) is bias-free — `nn.Linear(d, d,
bias=False)` — and a linear map without bias cannot invert an additive constant.
The exclusion was documented in the module docstring at the time, flagged for
ruling before any run, and **reverted**.

## 2. Why the reversal was right

1. **Direction of drift.** The deviation made the test *easier for our own method*
   by guaranteeing every shift was exactly invertible by our adapter. That is the
   one direction of deviation this project's rules do not permit. Reverting needs
   no justification; it was the spec.
2. **Track record on non-invertible shifts.** Every time a non-invertible shift
   entered the package it paid: Rotated MNIST killed the wrong theory and broadened
   the claim, and E3's corruptions were valued for exactly this property.
3. **Realism.** DC bias is arguably the most realistic sensor-drift component in the
   set — accelerometer zero-g offset and gyroscope zero-rate offset are textbook
   failure modes. Removing it traded realism for a guarantee our own results say we
   do not need.

## 3. What is now under test rather than engineered

The adapter **cannot** fully invert tasks 1 and 4. The standing expectation, on
record before the runs: the LSTM's input weights and gates can absorb a constant
offset internally — the encoder is trainable — so the ON arm should lose little.
That is now a **measured** prediction rather than a structural certainty.

## 4. A second correction, found by measurement

The first offset-restored implementation applied the shift to the **raw** signal and
standardized afterwards with fixed task-0 constants. Measurement rejected it:

| task | shift | bias-free recovery residual |
|---|---|---|
| 1 | gain + offset | 0.159 |
| 2 | rotation only | **0.146** |
| 3 | permutation only | **0.265** |

Tasks 2 and 3 specify **no offset**, yet became non-invertible — and task 3's
residual *exceeded* task 1's. Cause: `total_acc` carries gravity (channel mean
0.80 g), so any map that moves the channel means induces its own constant term once
fixed task-0 constants are subtracted. The largest non-invertible component sat in
the task that specifies no offset at all, and the offset manipulation would have
been confounded by gravity-times-rotation — a variable the contract never names.

**Fix:** the shift is applied to the **calibrated** stream,
`z_shifted = P R (G z + b/σ)`. This also matches the cited phenomenon more exactly —
bias *drift* is by definition the component calibration does not catch, so injecting
it after calibration is the faithful placement, not merely the convenient one.

Resulting structure, which is the designed contrast:

| task | shift | bias-free | affine (contrast) |
|---|---|---|---|
| 0 | identity | 0.000 | 0.000 |
| 1 | gain + offset | **0.158** | 0.000 |
| 2 | rotation | 0.000 | 0.000 |
| 3 | permutation | 0.000 | 0.000 |
| 4 | all combined | **0.145** | 0.000 |

Tasks 2–3 are exactly invertible by the locked adapter; tasks 1 and 4 are
non-invertible **only** because of the contracted offsets. The affine column
confirms the offset is precisely the irreducible part.

## 5. Adapter shape — contract-compliant, no ruling needed

§2 E5 pre-registers *"per-timestep linear 9→9 **or** a small full linear on the
flattened window, chosen by the identity-init rule, recorded pre-run."* Per-timestep
9→9 is the recorded choice: **81 parameters/task, 324 bytes at fp32.**

The design principle, visible only once a second modality forced the choice:

> **Match the adapter to the space where the shift lives.**

On MNIST the shift lived in pixel space (784×784); on HAR it lives in channel space
applied uniformly over time (9×9). This is why the `O(T·d²)` objection is
**regime-dependent rather than fundamental**: adapter cost tracks the shift's
intrinsic dimensionality, not the input's. The entire 5-task adapter set is smaller
than one raw 128-timestep window.

The flattened variant (1152×1152 = 1.33M params/task) is **not run**: 16,000× larger
with no additional expressivity for this shift class. Footnote, not experiment.

## 6. Frozen record

`runs/e5/FROZEN_CONFIG.json` — spec fingerprint `3de66e205eb7`, gain/offset/rotation/
permutation parameters, offset magnitudes in native units **with cited justification**
and their standardized-unit equivalents, per-task condition numbers, and the
bias-free recovery residuals above.

Offset magnitudes are drawn from documented consumer-grade MEMS specifications
(accelerometer zero-g offset in the tens of mg, reaching ~±90 mg before temperature
drift; gyroscope zero-rate offset of a few deg/s), not chosen arbitrarily — the
realism of this experiment is its entire marginal value over E1.

**Recorded simplification:** bias is applied independently per channel. Physically a
`total_acc` bias would propagate into `body_acc` through the gravity filter. This is
a simulated shift, not a derived sensor model, and is not claimed otherwise — which
restates the contract's own pre-named caveat: the supported claim is *"the mechanism
handles realistic shift classes on real sensor data,"* not *"validated on deployed
drift."*
