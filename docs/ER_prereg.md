# Pre-Registration: Experience Replay (ER) baseline

**Status:** Locked before ER code exists. ER is the comparison a reviewer
reaches for first, because the paper's mechanism is now identified as gradient
isolation rather than memory — and replay is the canonical memory method.

---

## 1. Why this baseline is load-bearing

The paper claims per-task input adapters deliver +49.3pp over vanilla with **no
stored data**. ER is the standard rehearsal baseline and *does* store raw data.
If ER at a small buffer matches the adapters, the contribution is
recontextualized ("matches replay without storing data" — still a claim, a
narrower one). If adapters win, the claim strengthens.

**Asymmetry to state plainly in the paper:** ER stores raw examples; adapters
store parameters. These are different privacy/storage regimes, so the comparison
is informative but not apples-to-apples on constraints. ER's buffer is an
*advantage* in information the adapter method does not have.

## 2. Design (single variable against the measured no-adapter control)

Base = the **no-adapter control** already measured at **0.4304 ± 0.0109**:
PLCM backbone, unfrozen encoder, shared readout, **no adapters**, 5 tasks × 10
epochs, task-incremental eval. ER adds *only* a replay buffer + replay loss.

- **Buffer: 100 examples per task** (reservoir-free: uniform sample taken at
  task end), so ≤400 stored examples when training task 4.
- **Replay loss:** cross-entropy on a replayed batch drawn uniformly over all
  buffered tasks, added to the task loss with weight 1.0.
- **Replay batch = 128** (matches the training batch, so replay is not
  under-weighted relative to current-task gradient).
- Seeds {42, 1337, 2024}, mean ± std.

## 3. Pre-named branches (fixed before the number exists)

| branch | criterion (AVG, n=3) | reading |
|---|---|---|
| **ER-dominant** | ≥ 0.9331 (adapter arm) | replay matches/beats adapters. Contribution narrows to *"adapters match replay with zero stored data and no task-boundary buffer management"*. Central-figure framing changes. |
| **ER-partial** | 0.55 – 0.93 | adapters win by the margin measured. Paper's claim stands, ER reported as the strong baseline it is. **Expected.** |
| **ER-weak** | < 0.55 | ER underperforms at 100/task. Do **not** claim adapters beat replay generally — report buffer size as the binding constraint and, if this branch fires, run ONE larger-buffer arm (500/task) before any comparative claim. |

**Anti-strawman commitment:** if ER-weak fires, the honest reading is that the
buffer was too small, not that replay fails. The 500/task follow-up is
pre-authorized for that branch only.

## 4. Pre-stated risks

- **R1:** Permuted MNIST is unusually replay-friendly — old tasks' pixel
  statistics are recoverable from few examples. ER may look better here than on
  harder shifts. Scope any comparative claim to this benchmark.
- **R2:** shared readout means replayed examples directly correct the classifier;
  with per-task heads ER would look different. Held constant with the control by
  design, but note it.
- **R3:** buffer sampling at task end (not reservoir) slightly favours ER by
  guaranteeing exactly 100 clean samples/task. Acceptable — it biases *against*
  our own method, which is the safe direction.

## 5. Reporting

Memo: both arms n=3, mean ± std, branch taken, deviations. Row joins the central
figure as the rehearsal comparison.
