# Pre-Registration: E5c — Shift-Dimensionality Test

**Status:** locked. Written before any E5c code exists.
**Scheduling:** runs **after** E2's calibration. E2 carries the gate and the
deadline; E5c carries the explanation.

---

## 1. Question

E5's diagnostics showed the HAR adapters never engaged. On task 3 the adapter needed
to travel **1.414** to reach an ideal it can represent **exactly**, and moved
**0.067** — a travel fraction of **0.047**.

**Proposed explanation:** input-path adaptation is taken only when it is *cheaper
than encoder accommodation*. A 9-dimensional channel map is trivially absorbed inside
the encoder's input weights; a 784-dimensional pixel permutation is not, because
absorbing it would wreck the encoder's learned features.

If that is right, **raising the shift's dimensionality on the same dataset should
make the adapters engage.** Same data, same backbone, same protocol — only the
dimensionality of the shift changes. That isolates the proposed variable.

This is worth running precisely because E5b established the two facts that make it a
puzzle: the shifts **genuinely caused** the forgetting (`f_attr = 0.982`), and the
adapters **demonstrably sat at the identity** anyway.

## 2. Design

UCI HAR, same loader, same 5-task protocol, same LSTM, same locks as E5.
**One change:** the shift is a full **1152×1152 permutation of the flattened
(9×128) window**, drawn per task and frozen in config before any run.

Adapter matched to the shift space: **flattened 1152×1152**, identity-init, frozen
per task. Contract §2 E5 already permits the flattened form, so this is a selection
among pre-registered options, not an amendment.

Arms: ON / OFF. **Seed 42 only at first.**

**Storage, recorded rather than discovered later:** 1152² = **1,327,104
params/task ≈ 5.3 MB at fp32**, against 81 params (324 bytes) for the per-timestep
form. This is the `O(T·d²)` problem reappearing *in the sensor regime*, and it
belongs in the memo — it is the direct cost of the thing being tested, and it
qualifies the "storage frontier is absurd in the sensor regime" claim from the
per-step design note.

## 3. Precondition check — gates everything

This is the exact failure E5 hit, so it is checked **first** and independently.

Compute **row 0** of the OFF-arm accuracy matrix: a task-0-only model evaluated on
all five tasks.

- **P-pass:** mean row-0 retention on tasks 1–4 **≤ 0.30** → the shift is genuinely
  damaging; the test is valid.
- **P-fail:** **> 0.30** → the test is **VOID**, same defect as E5.
  **Report and stop. Do not interpret the ON/OFF delta.**

Reference points: Fashion-permuted retained **12.0%**, Fashion-rotated **15.9%**,
E5's HAR shifts retained **64.9%** (tasks 1–2 at 0.907/0.856 against a 0.916 ceiling —
nothing for the adapters to repair).

## 4. Hypotheses — evaluated only if P-pass

- **H-E5c-1 (engagement):** adapter travel fraction
  `‖A_k − I‖ / ‖M_k⁻¹ − I‖ ≥ 0.50`, mean over tasks 1–4.
  *Derived, not chosen for convenience:* HAR's measured value was **0.047**. This
  bar demands an **order-of-magnitude increase**, not a nudge.
- **H-E5c-2 (benefit):** `ON − OFF ≥ +15pp` AVG — **the same bar as H-G1b,
  deliberately unchanged.** A restored claim on a friendlier threshold would be
  goalpost-moving.

## 5. Branches — tie-breaks resolve DOWNWARD

| branch | condition | reading |
|---|---|---|
| **(A)** | P-pass ∧ H1 ∧ H2 | Engagement condition **confirmed**: the mechanism engages when shift dimensionality makes encoder accommodation expensive. E5 converts from a null into a **positive result about the mechanism's operating condition**. **Promote to n=3 before any claim enters the paper.** |
| **(B)** | P-pass ∧ H1 ∧ ¬H2 | Adapters engage but do not help. Genuinely puzzling; report as an **open finding**, no claim. |
| **(C)** | P-pass ∧ ¬H1 | The dimensionality explanation is **wrong**. E5's null stands unexplained; **report the failed prediction at full prominence.** |
| **(D)** | P-fail | **Void.** Report and stop, as §3. |

## 6. Locks

- Permutations frozen in config **with fingerprint** before any training.
- Seed 42 first; **n=3 only if (A) fires**, per the standing rule that a new claim
  cannot rest on n=1.
- Mechanical verification per the standing definition. **Per catch 19: independent
  relaunch, not mtime checking**, for anything preempted — and the reproduction
  floor applies here too. Any delta below the measured floor is reported as null
  rather than signed.
- **Timebox: 90 minutes total.** If blocked past it, report state and stop.

## 7. Predictions on record (before running)

- **P-pass: ~80%**
- **H-E5c-1 (engagement): ~65%**
- **H-E5c-2 (benefit): ~55%**

## 8. Reporting

Row-0 table, per-task travel fractions, ON/OFF accuracies, and the branch taken.
Storage cost stated. Failed predictions reported at the same prominence as passes.
