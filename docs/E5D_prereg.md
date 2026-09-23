# Pre-Registration: E5d — Forced Adapter Recruitment (Method v2)

**Status:** **v2, locked.** All six review catches applied (§8). Written before any
E5d code existed.

This is a **METHOD CHANGE**, creating method v2. **The v1 gate verdict (failed —
E5 and E2 both) stands and is not reopened.** v2 is evaluated under this contract
and, if it survives, earns a new generality package (v2) with its own gate.

**Diagnosis being acted on (measured, not assumed):**
- HAR forgetting is real and shift-caused — E5b: 0.0067 with no shifts vs 0.3736
  with them (`f_attr` = 0.982).
- The locked adapter **can** solve tasks 2/3 exactly; the closed-form inverse exists.
- The adapter never engaged: travel **0.1042** of the needed distance over tasks
  {2,3} (4.7% on task 3 alone).
- Proposed cause: for a low-dimensional shift, **encoder accommodation is the
  locally cheaper gradient path**. The adapter is recruited only when it is the
  cheap path.

**The fix under test — adapter-first warmup.** At each task boundary for tasks
**1…T−1**, train **only the new adapter** for W warmup epochs with the encoder
**and the shared readout frozen**; then unfreeze and continue normally. During
warmup the adapter is the only loss-reducing path, forcing recruitment.
W = 1 (primary); W = 2 (secondary, only if W=1 partially engages).

---

## 1. Two metrics, because one of them is gameable

**This structure exists because of a counterexample already sitting in v1's own
table.** Task 1's adapter has travel fraction **0.8288** — it would comfortably
pass a ≥0.50 recruitment bar — while its residual-to-ideal is **1.2351**, meaning
it ended up **further from `M⁻¹` than the identity is** (identity scores exactly
1.0). "Moved a lot" is not "moved toward the answer," and a single-metric contract
cannot tell them apart.

| metric | definition | identity | perfect |
|---|---|---|---|
| **travel** | `‖A_k − I‖ / ‖M_k⁻¹ − I‖` | 0.0 | 1.0 |
| **residual** | `‖A_k − M_k⁻¹‖ / ‖M_k⁻¹ − I‖` | **1.0** | **0.0** |

v1 measured, n=3:

| task | travel | residual |
|---|---|---|
| 1 (gain+offset) | 0.8288 | **1.2351** ← moved away |
| 2 (rotation, exactly invertible) | 0.1613 | 0.9775 |
| 3 (permutation, exactly invertible) | 0.0471 | 1.0079 |
| 4 (combined) | 0.0428 | 1.0054 |
| **mean over {2,3}** | **0.1042** | **0.9927** |

**Per-task residuals are reported in the memo regardless of verdict.** The
trajectory of that number under warmup is the mechanism evidence, whichever branch
fires.

## 2. Experiments

**E5d-HAR** — method v2 on the frozen E5 config, fingerprint `3de66e205eb7`
verified byte-identical before launch. Arms:

| arm | warmup | what trains during warmup | status |
|---|---|---|---|
| **v2-ON** | yes | **adapter only** (encoder + readout frozen) | new, 3 seeds |
| **v2-control** | yes | **readout only** (encoder + adapter frozen) | new, 3 seeds |
| v1-ON | no | — | existing (`e5diag_*`) |
| OFF | no | — | existing (`e5_har_noadapt_*`) |

**E5d-MNIST (consistency check)** — method v2 on Permuted MNIST, 3 seeds, plus
**v1-ON MNIST re-run concurrently** (§catch 6). One recipe across all datasets; no
per-dataset schedules. Prediction: neutral, since adapters already engage there. If
warmup costs accuracy, the cost is reported and weighed, not hidden.

## 3. Hypotheses (fixed before any run)

- **H-D1 (recruitment — movement):** mean **travel** over tasks {2,3} ≥ **0.50**.
  *(v1 = 0.1042 → demands 4.8×.)*
- **H-D1b (recruitment — direction):** mean **residual** over tasks {2,3} ≤ **0.50**,
  i.e. at least half the gap to `M⁻¹` closed. *(v1 = 0.9927, i.e. no better than
  identity.)* **H-D1 without H-D1b certifies nothing** — see §1.
- **H-D2 (forgetting collapses):** v2-ON HAR forgetting ≤ **0.10**.
  *(v1-**ON** = 0.3920; OFF = 0.3736; no-shift floor = 0.0067.)*
- **H-D3 (benefit):** v2-ON − OFF ≥ **+15pp** AVG on HAR — the unmoved v1 bar.
- **H-D4 (no regression):** v2-ON MNIST vs **concurrently re-run** v1-ON MNIST,
  evaluated as a paired same-infrastructure delta **with its uncertainty reported**,
  against the 0.9331 ± 0.0099 historical band as context only.

### 3a. Attribution rule — what H-D2 and H-D3 now MEAN

**H-D2 and H-D3 are evaluated as v2-ON vs v2-control deltas, with OFF as the floor
reference.** Warmup necessarily gives the encoder fewer training epochs (10−W vs
10), and this project has already measured that encoder freezing alone reduces
forgetting (+4.08pp, Option A). Absolute bars alone would re-run the
freeze-misattribution error from the other direction.

> **If v2-ON ≈ v2-control, warmup's benefit is encoder-rest, not recruitment.**
> That outcome routes to **(D)-adjacent even if H-D2's absolute bar passes.**

## 4. Branches (downward tie-breaks)

| branch | condition | reading |
|---|---|---|
| **(A)** | H-D1 ∧ **H-D1b** ∧ H-D2 ∧ H-D3 ∧ H-D4, **and v2-ON > v2-control** | Recruitment is a **controllable design variable, not a fixed boundary.** E5-v1 becomes the motivating ablation. Generality package v2 drafted (§6). |
| **(B)** | H-D1 ∧ H-D1b ∧ H-D2 ∧ ¬H-D3 | Shift-forgetting fixed, delta small because OFF also partially copes. Claim scoped to the delta actually measured. |
| **(C)** | H-D1 ∧ H-D1b ∧ ¬H-D2 | Adapter **provably** learns the inverse and forgetting persists → the drift story is incomplete. Characterize residual forgetting (per-task columns; offset tasks 1/4 vs invertible 2/3) before any further design iteration. **No method v3 without a new diagnosis.** |
| **(D)** | ¬H-D1 **∨ ¬H-D1b** | Warmup fails to recruit even when it is the only path → the optimization problem is deeper than path-cheapness. Report, stop, v1 paper to TMLR unchanged. |
| **(D)-adj** | any branch where **v2-ON ≈ v2-control** | Benefit is encoder-rest, not recruitment. Treated as (D) for method purposes regardless of absolute bars. |
| **any** | ¬H-D4 | The MNIST cost is weighed openly. **A v2 that trades MNIST points for HAR points is a tradeoff to report, not a win to declare.** |

## 5. Predictions on record

H-D1 ~85% · **H-D1b ~70%** · H-D2 ~60% · H-D3 ~55% · H-D4 ~80% · joint (A) ~35%.

*Calibration note:* this project's mechanism predictions have run **directionally
right, magnitude wrong**; the 75%-confidence E5 miss is the reference point for
discounting these. H-D1b is priced below H-D1 precisely because §1's counterexample
shows movement is the easier of the two to achieve.

## 6. Competence precondition for package v2 (if (A) fires)

HAR + one more LSTM-native dataset, chosen by criteria **verified BEFORE task
construction**:
1. single-task **DIAG ≥ 0.85**, and
2. an **E5b-style no-shift control** showing forgetting is shift-caused.

*This is E2/E5's lesson encoded.* CIFAR would have failed criterion 1 (DIAG 0.51);
HAR passes at 0.915. ICLR 2027 becomes the target **iff** package v2 passes its own
gate in time; else ICLR 2028 / TMLR with the stronger method.

## 7. Locks

- **One recipe, all datasets.** W ∈ {1, 2} only; no other schedule search.
- Warmup applies to **tasks 1…T−1 only**. Task 0 trains normally end-to-end — an
  adapter fitted against a randomly initialized encoder would bake noise into the
  foundation.
- During v2-ON warmup: encoder frozen **and shared classifier frozen**. The
  mechanism *is* "the only loss-reducing path runs through the adapter"; a trainable
  readout is a bypass that would sometimes work well enough to blur every downstream
  number.
- Frozen E5 shift config reused byte-identical (fingerprint checked at launch).
- **Bias-free linear adapter unchanged from v1** — offset tasks (1/4) remain
  non-invertible BY DESIGN; their per-task columns are the within-experiment
  invertibility contrast.
- `--save-checkpoints` required; H-D1/H-D1b are unmeasurable without them.
- Mechanical verification + **measured reproduction floor** (relaunch one arm-pair,
  both arms) before any delta is interpreted. **No claim at n=1.**
- Timebox: **2 days.** Blocked past that → report state, stop, v1 to TMLR.

## 8. Review log — six catches, all sustained

1. **H-D1 was gameable (blocking).** `‖A−I‖` measures distance travelled, not
   distance travelled *toward* anything; task 1 passes it at 0.8288 while sitting
   further from the ideal than the identity. **H-D1b added**; (A) and (C) require both.
2. **Warmup confounded recruitment with reduced encoder training (blocking).**
   **v2-control arm added**; H-D2/H-D3 re-defined as against-control deltas (§3a).
3. **Task 0 warmup unspecified.** Resolved: tasks 1…T−1 only.
4. **"(+ new head)" does not exist** — `--mafc-arm lambda0` disables per-task heads.
   Resolved: shared classifier explicitly frozen during warmup. *(Pleasant echo: this
   is structurally the inverse of E4b's lemma setting — during warmup all gradient
   signal is forced through the input path, the cleanest possible test of whether
   that path can learn the correction at all.)*
5. **Two baselines mislabeled.** H-D1's "0.047" was task 3 alone; the {2,3} mean is
   **0.1042** (bar = 4.8×, not >10×). H-D2's 0.3736 was the **OFF** arm; v1-ON is
   **0.3920**.
6. **H-D4's ±1pp ≈ its own noise** (reference spread ±0.99pp). Resolved by running
   v1-ON MNIST **concurrently**, converting a noisy historical comparison into a
   clean contemporaneous paired delta.

## 9. Reporting

Memo with: travel **and residual** per task (v1 vs v2 vs v2-control), forgetting per
task, all five hypothesis verdicts, the v2-ON/v2-control attribution, branch taken,
MNIST consistency table, and — if (A) — the generality-package-v2 contract drafted
**before** any further runs. **Claim-ledger rule: the one-liner gets its edit first,
analysis second.**
