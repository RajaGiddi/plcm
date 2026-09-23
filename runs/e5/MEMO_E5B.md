# E5b — No-Shift Control: Result Memo

**Status:** Complete, 9/9 verified, n=3. **BRANCH (A) FIRES.**
Pre-registration: `docs/E5B_control_prereg.md` (branches locked before the run).

**The expected outcome did not happen.** Branch (B) — failed benchmark construction —
was the anticipated result on both sides of the review, at roughly 75% confidence.
It did not fire. The construction worked.

---

## 1. Verdict

Five **identical** tasks (identity map everywhere), otherwise identical to E5.

| quantity | value |
|---|---|
| OFF forgetting, **no shift** | **0.0067 ± 0.0013** |
| OFF forgetting, E5 shifts (frozen) | 0.3736 |
| **shift-attributable fraction `f_attr`** | **0.9821** |

    branch (A)  f_attr >= 0.67   FIRES
    branch (B)  f_attr <= 0.33   does not fire

**The shifts caused ~98% of E5's forgetting.** With no shifts, forgetting is
essentially zero (0.0067) — the pipeline is sane and five identical tasks are not
forgotten, exactly as predicted on record.

## 2. What this overturns

The emerging reading — *"HAR's forgetting is dataset-intrinsic, our constructed
tasks never created the phenomenon, E5 belongs in the appendix as a failed benchmark
construction"* — is **refuted**. The precondition the mechanism needs
(substantial input-shift-induced forgetting) **was established**. E5 is a **valid
test**, and the mechanism failed on it.

This must be held together with the row-0 severity audit, which remains true: HAR
tasks 1 and 2 were nearly free (0.907, 0.856 against a 0.916 ceiling). Both facts
coexist — **two of the four shifted tasks posed little challenge, and the shift set
as a whole still produced 0.374 forgetting against a 0.007 floor.** Tasks 3 and 4
carried it. "Mild shifts" was true task-by-task and wrong in aggregate.

The row-0 table stays in the paper as an honest characterization of *which* tasks
did the work. It no longer supports "the benchmark failed to pose the question."

## 3. The adapter's cost is shift-independent — the cleanest number here

| condition | ON | OFF | delta |
|---|---|---|---|
| **no shift** (nothing to adapt to) | 0.9055 | 0.9148 | **−0.93pp** |
| **E5 shifts** | 0.6017 | 0.6111 | **−0.95pp** |

The adapters cost the *same* amount whether or not there is a shift to correct.
Combined with the inversion diagnostic — adapters travelled 0.067 of a required
1.414 and stayed at the identity — the account is coherent and now triangulated:

> **The adapters never engaged with the shift at all.** Their entire measured effect
> is a small constant overhead, present even when the input is untouched.

## 4. Reproducibility — and a correction to catch 19

The OFF-arm relaunch (`e5recheck_*`, no preemption reported for any of them):

| seed | E5 | relaunch | |
|---|---|---|---|
| 42 | 0.6395 | 0.6395 | identical |
| 1337 | 0.5745 | 0.5724 | −0.21pp |
| **2024** | 0.6194 | 0.6706 | **+5.12pp** |

**Catch 19's original diagnosis was wrong.** ON/2024's +5.50pp divergence was
attributed to its preemption. But OFF/2024 diverges +5.12pp **with no preemption**,
while seed 42 is bit-identical in both arms. The pattern is **seed-correlated, not
preemption-correlated** — seed 2024 is chaotic, seed 42 is stable. Preemption was a
coincidence lying on top of ordinary run-to-run nondeterminism.

**The consequence is more serious than the misdiagnosis.** Reproduction noise reaches
**5.5pp**; the ON−OFF delta being interpreted is **0.95pp**. *The E5 delta sits below
its own noise floor and was never resolvable* — and seeding does not fix this, since
the noise is within-configuration, not across seeds.

- **H-G1b's delta is uninterpretable at this power.** Report as "no measurable
  effect," with the noise floor stated.
- **H-G1a is unaffected**: 0.6576 against a 0.90 bar is a 24.2pp miss, far outside a
  5.5pp floor. That verdict stands.

## 5. Consequences for the manuscript

1. **E5 stays in the results, not the appendix.** It is a valid test that the
   mechanism failed.
2. **The story-changing threshold stands and the reframe fires.** Since E5 is valid,
   its 0.6576 belongs in the span: 31.6pp against a 5pp trigger. The invariance
   framing is **not** restored — that possibility was flagged as pending on E5b and
   is now closed against restoration.
3. **The "modality boundary" reading is available again but must be stated with the
   optimization mechanism, not as a data-domain law.** The measured claim is: the
   adapter path engages only when it is the *cheap* path. A 9-dim channel map is
   cheaper for the encoder to absorb than for the adapter to invert; a 784-dim
   permutation is not. That is a statement about shift dimensionality relative to
   encoder capacity — testable, and not "sensor data is different."
4. **E2 remains the gate.** Unchanged by E5b.

## 6. Deviations

None. Seeds, arms, epochs, thresholds, and the frozen `F_shift = 0.3736` as
pre-registered; branches were locked before the run and the tie-break was never
invoked (`f_attr = 0.9821` is not near a boundary).

Control validated before launch: all five tasks bit-identical to each other, and
task 0 bit-identical to E5's task 0.
