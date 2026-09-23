# Pre-Registration: E15 — ViT Tier-0, Prototypes-Only
## Can stored class prototypes realign the reader's input where features barely moved?

**Status: DRAFT for sign-off.** Analysis-only — no training. Existing E12 base-arm
era checkpoints. **Half-day timebox**; blocked or degenerate → report state, stop.

**This is a discriminator, not a confirmation.** E13 demoted tier-0 to a weak
baseline in both regimes it has been measured in (HAR 0.075, MNIST 0.120), so the
cure section currently rests on E10's C3 alone. E15 asks whether the *surviving*
fragment of the theory holds where the conditions most favour it:

> **If features stayed put (E12 measured F_enc = 0.06), do era prototypes still
> span the right subspace — so projecting the deployed feature onto that span
> recovers a real fraction of the reader gap?**

Fires → the cure family gains a **scope law**: *reader repair works where features
stayed put, in proportion to how put they stayed.* Fails → tier-0 is weak
universally, the theory dies cleanly, and the cure section is **C3, full stop** —
stated, not mourned.

---

## 0. Why this is prototypes-only — a degeneracy settled in the contract

Tier-0 has two components. **On the ViT the era-head component is degenerate by
construction**, and running it would be catch-31-shaped: a comparison whose two
sides are the same object.

In E12's task-IL deployment, head *k* is created at task *k*, **frozen at its
boundary**, and P3b verified by module identity that it is what already scores
task *k*. So "apply the stored era head" reproduces `acc_orig` exactly. It would
have surfaced as a mysterious ρ ≈ 0 and been misread as evidence about the
theory.

**And the collapse is itself a measurement we already hold.** On the ViT the
era-head component was *deployed all along* — and forgetting is 0.72 anyway, at
F_enc = 0.06. **Stored era heads alone recover approximately nothing when the
trunk moves under them.** That falsifies the era-head clause of the prior
prediction *before any run*, from E12's own table.

**Consequence:** the era-head row is printed as `acc_orig` **with the degeneracy
note attached, never as a measurement**. The component table has one measured
row. That is the honest consequence of the analysis, not a thin design.

---

## 1. Setup

- **Checkpoints:** E12 **base arm** era checkpoints (`ckpt_e12_base_seed*`),
  fp16, 20 tasks, **5 seeds** — the same population H-V1 was measured on, so the
  ρ denominators are the same cells.
- **Old tasks:** 0..18 at θ_T (task 19).
- **Environment:** Modal — where the checkpoints live and their data was built.
- **Setting:** task-IL, deployed path, `head_routing_by_hint` on (as E12 ran).

### The one cure

**`proj`** — project the deployed θ_T feature onto the **stored era-prototype
span**, then read with the deployed (frozen, task-*k*) head.

Stored per task: **class prototypes only** — per-class means of the deployed
feature at that task's own boundary. **5 × 768 floats ≈ 15 KB per task at fp32.**
No raw data, no era head (degenerate, §0), no generative step.

`span_of` is **imported** from `scripts/transport_estimate.py`, which owns it.
Prototypes are built from era-checkpoint features through the **audited loader**
(catch 32); no parallel extraction path.

---

## 2. Preconditions — the full inherited block

1. **P3a** — `head(captured_feature) == deployed_logits`, max |Δ| = 0, every cell.
2. **Catch-32 alignment** — features and labels through the audited loader;
   position-vs-dataset bitwise check printed **PASS**, shuffled-loader positive
   control **FIRED**. No probe number read without this line.
3. **Identity** — `F_enc + F_read − R = F_total` to machine precision, per cell.
4. **Instrument bias** — pooled |R| ≤ 0.05 on the arm.
5. **Per-cell guard** — `RHO_MIN_DENOM = 0.02` applied per cell, never
   post-pooling; **forced-inclusion printed beside every guarded value** (EV1).
6. **Reproduction floor quoted** — E12's: aggregate ~0.4pp, **per-cell ~15pp**;
   pooled-with-CI is primary, per-seed shown and never leaned on.

### 2a. RANK CONTROL — the confound this design must not walk into

Task *k* has **5 classes**, so the prototype span is **rank 5 in a 768-dim
feature space**. HAR projected rank-6-of-256 and MNIST rank-10-of-256; **the ViT
discards proportionally far more.** A null could therefore mean *features moved*
(the theory) **or** *rank-5-of-768 is too lossy to read through* (geometry), and
those are not the same finding.

**Control, run before the cure is scored:** apply the identical projection **at
the era checkpoint**, where the features definitionally match the prototypes that
were built from them, and read with the era head.

- Era-projected accuracy ≈ era ceiling → the projection is **information-
  preserving**; a null on θ_T is about drift, and the cure's verdict is readable.
- Era-projected accuracy **materially below** the era ceiling → the projection is
  lossy **by geometry**, the θ_T null is uninformative about the theory, and the
  pre-registered verdict is **"tier-0 projection unreadable at this rank on this
  architecture"** — reported, never read as evidence against the theory.

Bar: era-projected within **0.05** of the era ceiling. *(Derivation: the same
0.05 the program uses for instrument bias — this is an instrument question, not
a hypothesis one.)*

---

## 3. Measurements

Per seed × old task *k*, reusing E12's definitions unchanged:

- `acc_orig` — deployed at θ_T (**= the era-head component, §0**)
- `acc_refit` — labeled convex refit on the frozen θ_T feature (E12: 0.888 mean)
- `R` — instrument bias, `acc_refit_ceiling − acc_ceiling`
- `acc_proj` — the cure
- `acc_proj_era` — the §2a rank control

ρ against the **bias-adjusted** denominator (era-head family, per the E11
ruling): `ρ = (acc_proj − acc_orig) / (acc_refit − R − acc_orig)`.
Mean-of-ratios, binomial-propagated CIs, guarded and forced columns.

---

## 4. Hypotheses

- **H-P1 (PRIMARY).** Pooled ρ ≥ **0.30** on the ViT base arm, guarded, with the
  forced column agreeing in sign.
  *Derivation:* E8's H-R4 bar, reused unchanged — the same bar H-S2 used on
  MNIST, so the three regimes are compared on one scale.
- **H-P2 (descriptive).** Report ρ against HAR's 0.075 and MNIST's 0.120. Whether
  recovery scales with feature stability (F_enc 0.06 vs the LSTM range) is the
  scope law's evidence — **no bar**, because no prior measurement constrains it
  (the dynamic-range rule).

---

## 5. Branches

- **(G) any precondition fails, or the §2a rank control fails** → report which;
  no verdict on the theory.
- **(A) H-P1 MET** → the scope law lives: reader repair works where features
  stayed put. Cure section regains a second result, with tier-0 scoped by feature
  stability rather than by dataset.
- **(B) ¬H-P1, rank control PASSED** → **PRE-COMMITTED READING, written before the
  number exists.** This null is *specifically informative* given F_enc = 0.06, and
  it is **not** "the theory dies". The refit proves linearly recoverable
  information **persists** (0.888 at θ_T); a projection null on top of that means
  the discriminative directions have **rotated out of the era-prototype span**.
  So:

  > **Feature stability and span stability are different quantities, and reader
  > repair from stored prototypes tracks the second, not the first.**

  The theory is **refined, not killed**: tier-0 requires *span* stability, which
  none of the program's three regimes provides. That is a scope law even in
  failure — and a stronger sentence than "tier-0 is weak everywhere, full stop",
  which is what this branch would otherwise have been improvised into. The cure
  section still narrows to C3 as its load-bearing result.
- **(C) ¬H-P1, rank control FAILED** → unreadable at this rank; the theory is
  untested here and the ledger says so. **Not** evidence against it.

---

## 6. Predictions on record

**H-P1 (ρ ≥ 0.30 on the ViT): ~45%.** Revised down from the earlier ~65%: the
era-head clause is **withdrawn** — it was placed on a component whose null was
already sitting in E12's own table — and tier-0's universal weakness (0.075,
0.120) is now the base rate. What holds it above that base rate at all is the
F_enc = 0.06 argument, and nothing else.

Rank control passes: **~70%**. H-P2 shows recovery ordered with feature
stability: ~50%.

*Calibration:* six misses on record, the most recent being a prediction anchored
to a frozen literal that turned out to be fiction — *a frozen, unsourced number
is not a prior*. Treat the interval, not the point.

---

## 7. Locks

- Analysis only; E12 base-arm era checkpoints exclusively; **audit by loading**
  before any cell.
- `span_of` and the audited loader imported, never re-implemented (catch 32).
- Era-head row printed as `acc_orig` with the degeneracy note; **never scored**.
- Rank control run and printed **before** the cure is scored.
- Verdicts computed from the printed arrays (catch 22).
- Provenance: memo numbers from recorded script output only.
- Ledger edit drafted in the memo first, applied second.
- **Timebox: half a day.** If the projection is degenerate on the ViT path for any
  reason not foreseen here, **stop at the report**.
