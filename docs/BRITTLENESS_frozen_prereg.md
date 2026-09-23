# Pre-Registration: Frozen-Feature Brittleness (LSTM vs MLP)

**Status:** Locked, awaiting launch. **Blocking** — this claim is already
asserted in the paper without evidence.

**Claim under test:**

> *Frozen recurrent features fail across permutations because sequential
> processing bakes the input arrangement into temporal dynamics; feedforward
> features transfer better.*

---

## 1. Design (no adapters anywhere)

For each backbone ∈ {LSTM, MLP} and each seed ∈ {42, 1337, 2024}:

1. Train encoder + head on **task 0** (identity permutation), standard protocol.
2. **Freeze the encoder.** It is never updated again.
3. For **each** task k ∈ {0..4}: train a **fresh linear head** on the frozen
   features of task k's data. Report test accuracy per task.

Plus a **raw-pixel floor**: a fresh linear head trained directly on raw 784-dim
pixels, per task, with no encoder at all.

**Metric — frozen transfer:** mean accuracy over tasks 1–4 (task 0 is
in-distribution and reported separately as a sanity check).

**Nothing is shared across tasks except the frozen encoder.** No adapters, no
memory, no replay. This isolates exactly one question: *how much does a frozen
representation retain when the input arrangement changes?*

## 2. Thresholds (fixed before the number exists)

Gap = (MLP frozen transfer) − (LSTM frozen transfer).

| branch | criterion | reading |
|---|---|---|
| **stands** | gap ≥ +15pp | recurrent features are brittle to input rearrangement in a way feedforward features are not. Claim enters the paper as stated. |
| **weak** | +5 to +15pp | directional but small. Claim must be softened to the measured gap and cannot carry the "because sequential processing" mechanism story on its own. |
| **dies** | gap ≤ +5pp | the brittleness claim is unsupported and must be **removed** from the paper. Frozen features transfer comparably regardless of architecture. |

**Sign convention:** a *negative* gap (LSTM transfers better) also lands in
"dies" and is reported as such.

## 3. The raw-pixel reference (AMENDED pre-launch — category error corrected)

**The raw-pixel head is a permutation-invariant reference, not a floor:** a
per-task linear map on raw inputs absorbs any permutation by construction
(~0.90, uniform across tasks). If either frozen arm falls below it, that arm's
encoder has traded permutation-robustness for task-0 specialization. If **both**
arms fall below it, report the joint finding with the same prominence as the
pre-registered comparison; it supersedes the architecture attribution: the
failure is representational commitment generally, not recurrence specifically.

*Amendment provenance:* v1 framed this line as a floor ("worse than this means
actively destructive"). A 1-epoch mechanics smoke showed both arms ~28pp below
it and the reference sitting at exactly 0.90 on every task — revealing that a
fresh per-task linear head *inverts the permutation by construction*, so the
line measures permutation-invariance, not a performance minimum. Corrected
before launch; the smoke result is not treated as evidence.

### 3a. Interpretive note held ready (if the joint branch fires)

If frozen features of *any* architecture are destructive under permutation while
a per-task linear input map is invariant by construction, then *"put the
per-task linear map at the input and let the encoder stay general"* is not
merely empirically good — it is the design the reference line **prescribes**.
The adapter is the raw-pixel head's trick, installed in front of a deep network.
The paper's brittleness *sentence* would change; its *architecture* would gain a
cleaner motivation than the recurrence story it replaces.

## 4. Why this must run BEFORE the MLP-attribution control

The other contract (docs/BRITTLENESS_prereg.md) ports the adapter attribution
design to an MLP backbone. **Its branch (c) is uninterpretable without this
run's arm B.** If frozen MLP features already transfer well, then a small
adapter delta on an MLP is *expected* — the backbone barely forgets, so there is
little left for adapters to buy. Without this number, "mechanism isn't general"
and "mechanism isn't needed here" are indistinguishable. This run supplies the
discriminator.

## 5. Locks

- Encoder training on task 0: identical to the main runs (10 epochs, Adam
  lr 1e-3, batch 128).
- Head training: fresh linear head, fixed recipe, identical across arms and
  tasks; never reused between tasks.
- Backbone parameter counts, recorded pre-run, not tuned:
  LSTM 292,864 / MLP 266,752 (within 9%).
- Seeds {42, 1337, 2024}, mean ± std. No claim at n=1.
- Real Permuted MNIST only.

## 6. Reporting

Memo: per-arm per-task accuracies, frozen-transfer means ± std, the gap, the
floor comparison, branch taken, deviations. If "dies", the claim is struck from
the paper and the deletion is noted in the process appendix.
