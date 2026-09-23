# Pre-Registration: E10 — Subject-Disjoint HAR (Benchmark Repair + Falsifiable Cure Screen), v2

> **PROVENANCE HEADER — read before any number below is cited.**
> v1 supplied as message text from the drafting environment; six amendment
> rulings quoted from the recorded review exchange; P3 v3 re-implemented in this
> repo from the recorded derivation and re-validated on all three controls before
> its live reading — the prior session's reported results serve as reproduction
> targets only; assembly postdates training, predates the screen's licensed read.
> Every clause in this header is checkable against artifacts in this repository.
> **This document is not a pre-registration in the strict sense and does not
> claim to be.** P1/P2/P3 thresholds and the certificate arc were fixed in the
> recorded exchange before the training launch; the assembly, the denominator
> amendment, and the per-cell recompute postdate it. See §9, which enumerates
> every instrument decision made with live values known.

**Status:** Assembled post-training. Benchmark repair + cure screen.
**Shift spec fingerprint:** `3de66e205eb7` (frozen E5 set, reused verbatim)
**Subject partition fingerprint:** `5d047e4213d1`

---

## 0. Why this experiment exists

E8's C3 (pseudo-refit) reached ρ ≈ 1.0 — apparent full recovery of the reader
channel — but was excluded from the branch because HAR-as-constructed makes it
unfalsifiable: every task is the same 2947 windows under different channel
coordinates, so pseudo-old data generated via the known `M_k` reconstructs the
literal old data (max err 1e-6). The cure could not be distinguished from replay.

E10 repairs the benchmark: **tasks differ in CONTENT (disjoint subjects) and
PRESENTATION (calibration shifts)**, so pseudo-old data is genuinely novel
content in old calibration — generative, not resurrective.

---

## 1. Benchmark construction — AMENDED (supersedes v1 §1)

v1 specified 21 subjects (the canonical train split), 5 tasks × ~4 subjects.
**Catch 5 ruling:** *"sustained; the canonical 21/9 boundary serves no purpose
once we re-split, and the data recovery directly funds catches 3 and 4."*
**Catch 1 ruling:** *"pool 30 subjects, 6 per task, 1 held-out test subject each."*

Implemented in `src/data/har_subject.py`, verified against the trained runs:

- **Both official splits pooled** — 30 subjects, 10299 windows (vs 7352 from the
  train split alone: **+40.1% data recovery**, not the 45% quoted in the ruling;
  the measured figure governs).
- 30 subjects → **5 groups of 6**, greedy-balanced by window count.
- Within each group: **5 subjects TRAIN, 1 subject HELD OUT** as that task's test.
- Consequence: task train sets are subject-disjoint from each other, **and** each
  task's test subject is unseen in its own training data. A within-subject test
  split would have inflated DIAG and deleted the cross-subject generalisation
  difficulty the benchmark exists to pose.

| task | train subjects | test subject | n_train | n_test |
|---|---|---|---|---|
| 0 | 1, 5, 11, 13, 16 | 25 | 1658 | 409 |
| 1 | 8, 12, 15, 17, 21 | 29 | 1705 | 344 |
| 2 | 3, 7, 10, 20, 23 | 26 | 1669 | 392 |
| 3 | 2, 4, 6, 18, 27 | 30 | 1684 | 383 |
| 4 | 9, 14, 19, 22, 24 | 28 | 1673 | 382 |

Task *k* = subject-group *k*'s data under calibration shift `M_k`.

---

## 2. Preconditions (all gate the screen)

### P1 — competence: single-task DIAG ≥ 0.80

**Catch 1 annotation, adopted:** *"cross-subject DIAG will land lower than E5's
within-distribution 0.91 — that's the benchmark being itself, not a regression,
and P1's 0.80 bar should be annotated so a 0.82 doesn't get misread as decline."*

**Catch 3 ruling:** *"the h-escape fires only at n=3, per the standing rule it
nearly violated."* One capacity step to h=512 permitted, at n=3, recorded,
applied to all arms identically.

**MEASURED — PASS.** All nine cells clear the bar; no capacity escape needed.

| arm | DIAG range (n=3) |
|---|---|
| OFF | 0.8139 – 0.8569 |
| ON | 0.8173 – 0.8713 |

Cross-subject DIAG ≈ 0.84 vs E5's within-distribution 0.91 — the benchmark being
itself, exactly as annotated.

### P2 — shift-caused forgetting ≥ 60% of total

No-shift subject-disjoint control, OFF arm.

**MEASURED — PASS.** no-shift forgetting **0.02507**, shifted-OFF **0.44937** →
**94.4% shift-attributable** against the 60% bar. Subject change alone costs
almost nothing; the calibration shifts carry the damage.

### P3 — generative-honesty certificate, v3

**Catch 2 ruling:** *"sustained... The scale-free version is adopted — pseudo→real
NN distances must be no smaller than real→real within-task NN distances. And
validate the certificate against a known-dishonest case: run P3 on E8's original
construction... The certificate must FAIL there."* New standing rule: *"every
certificate/gate ships with a positive control — a case where it is known it must
fail — demonstrated before its first live use."*

v1's threshold (100× the E8 reconstruction error scale) is **withdrawn**: 100 ×
1e-6 = 1e-4 sits six orders of magnitude below the distance between any two
distinct windows, so it could not fail by construction.

**Operative definition (v3), in `scripts/generative_certificate.py`:**

- Denominator `D` = **pooled** median within-task real→real NN distance across
  all tasks. (Pooling is v1's fix: a per-task denominator scores group tightness,
  a nuisance variable, not honesty. v1's failure stands as recorded.)
- **Distribution bar** (v2, unchanged): per-task median(pseudo→real NN) ≥ 0.5·D.
- **Contamination bar** (v3): fraction of individual pseudo points below
  `τ = sqrt(recon_scale · honest_scale) · D` must be **< 1.00%**.
  `recon_scale` from Control A, `honest_scale` from Control C — **τ is derived
  from the controls, never from the live data it certifies.**

**Controls — all three must behave before any live reading:**

| control | construction | requirement | measured |
|---|---|---|---|
| A | E8 shared-window relayout | **must FAIL** | median 0.0007·D, 100% < τ → **FAILS** ✓ |
| B | 10% real task-k windows injected | contamination bar **must fire** | **9.98%** < τ → **FIRES** ✓ |
| C | foreign subject group → task-k layout | **must PASS** | median 0.97–3.06·D, 0.00% < τ → **PASSES** ✓ |

**MEASURED — PASS.** τ = **0.030151·D**; control separation **2,128×**; live
medians **1.156–1.349·D** with **0.00%** below τ across all four tasks — a **43×
verdict margin**.

---

## 3. Arms and screen

**Training (10 runs, complete and mechanically verified):** OFF and ON (adapters),
3 seeds each; no-shift control ×3; floor pair (both arms, per the standing rule).

**Screen (analysis-only):** `scripts/cure_screen.py --arms e10`. Cures C0, C1,
C0+C1, C2 (oracle, tier-2, flagged), C3 (pseudo-refit, tier-1).

### 3.1 Generator gate — AMENDED

E8's sec-0 pre-gate compared relayout output against **real** task-k windows and
required 0.0000. That comparison is **undefined** on subject-disjoint data: task
4's subjects are different people, so there is no row correspondence. Carrying it
over would `sys.exit` before any cure was scored; loosening its threshold would
delete the only algebraic check on the generator.

**Replaced by a round-trip identity gate:** `relayout(relayout(x,4,k),k,4) == x`,
scale-free (normalised by the data's own magnitude), and able to fail — v1's
`x@M_k.T` spec fails it. **MEASURED: worst 4.829e-08 ≤ 1e-4 → PASS.** Honesty is
explicitly *not* asserted here; P3 decides it.

Note: E8's pre-gate and P3's Control A are the same measurement read in opposite
directions. The pre-gate passes at err ≈ 0 *because generation is reconstruction*
— which is exactly the known-dishonest case Control A must reject.

### 3.2 Denominator assignment — AMENDED (see §9.3)

ρ = (acc_cure − acc_orig) / denominator. The bias adjustment subtracts `R` — the
convex probe's advantage over the era model's deployed head — charging the recipe
advantage against the cure. That is correct **only for cures that do not share
the ceiling's fitting recipe**:

- **era-head cures** (C0, C1, C0+C1, C2) read through the stored era head and get
  no recipe benefit → **bias-adjusted** denominator.
- **recipe-matched cures** (C3) fit their head with the same convex procedure as
  the ceiling → the recipe gap cancels by construction → **raw** denominator.

Rationale is **recipe symmetry alone**. No mechanism story about why `R` is large
is needed, and none is available — see §9.4.

### 3.3 Per-cell guard — AMENDED (catch 26)

**Catch 4 ruling:** *"report every ρ with its binomial-propagated CI, evaluate
H-X3 pooled across seeds and tasks, and pre-commit that if the propagated CI on
the pooled estimate exceeds half of H-X3's ±0.15 band, H-X3 demotes to
descriptive — reported, not gated."*

ρ is computed **mean-of-ratios over valid cells**, not ratio-of-pooled-means.
`RHO_MIN_DENOM = 0.02` applies **per cell**. Per-cell denominators are printed
and exclusions counted in the table. Implemented in `scripts/rho_percell.py`.

E10's per-cell denominators span **0.0026 to 0.556 (200×)**, with **4 of 24
cells** at or below the guard — none of which tripped it under pooling.

**Instrument gate H-C3′:** |R| ≤ 0.05 pooled per arm. **MEASURED — FAILS** in
both arms (OFF **+0.0945**, ON **+0.1375**). Era-head cure readings therefore
remain **gated**; C3 is unaffected (raw denominator, 0 exclusions).

**Known limitation, recorded not fixed (deferred deliberately):** the 0.02 guard
still admits cells at D ≈ 0.03–0.09 that produce ρ of ±3. Tightening it mid-read
with values known is an instrument change no live verdict requires — C3 uses raw
denominators and keeps all 12 cells.

---

## 4. Hypotheses

- **H-X1 (the claim):** C3 pooled ρ ≥ **0.80** on subject-disjoint HAR/OFF under
  P3's certificate. (0.80 = the E8 H-R3 bar, unchanged.)
- **H-X2 (forgetting-range conversion):** end-to-end forgetting with the best
  storage-honest cure ≤ 0.10. **Catch 6 ruling:** *"cure-corrected final row,
  then the standard forgetting formula — stated in the metric definitions."*
- **H-X3 (stack replication):** tier-0 stack ρ within ±0.15 of E8's 0.347.
  Demotes to descriptive if the propagated CI exceeds ±0.075 (catch 4).
- **H-X4 (adapter engagement, free column):** adapters remain inert (travel
  fraction < 0.15).

---

## 5. Audit

| requirement | source | status |
|---|---|---|
| 30 subjects pooled, 5×6 groups, 5+1 split | `src/data/har_subject.py` | ✓ verified against trained runs |
| frozen shift config | `3de66e205eb7` | ✓ printed by screen |
| subject partition frozen | `5d047e4213d1` | ✓ printed by screen |
| 10 runs complete | `scripts/verify_runs.py` | ✓ 9/9 + floor pair 2/2 |
| era snapshots on ALL arms | `--save-checkpoints` | ✓ `runs/ckpt_e10_*` |
| screen instrument (convex, feature-locked) | E6b v3 | ✓ |
| P3 controls validated in THIS repo | `scripts/generative_certificate.py` | ✓ A/B/C all behaved |

---

## 6. Branches (gates first)

- **(G) any precondition fails:** report which and STOP.
- **(A) H-X1 ∧ H-X2:** the 80–90% claim earned on real sensor data.
- **(B) H-X1 ∧ ¬H-X2:** channel recovered, end-to-end forgetting above 0.10.
- **(C) ¬H-X1, C3 ρ ≥ 0.50:** generation honest but harder than reconstruction —
  the gap between reconstruction and generation is itself the finding: how much
  of C3's E8 score was the benchmark's gift.
- **(D) C3 ρ < 0.50:** pseudo-refit does not survive honest generation.

---

## 7. Predictions on record (from v1, unchanged)

P1 ~70%, P2 ~65%, P3 ~90%, H-X1 ~60%, H-X2 ~50%, H-X3 ~70%, H-X4 ~85%,
joint (A) ~45%.

---

## 8. Locks

- Partition and shift config frozen with fingerprints before any run; no
  re-partitioning after results exist.
- One capacity decision (P1's h=512 escape) permitted, at n=3, applied to all
  arms identically. **Not exercised.**
- Verdicts computed from printed arrays (catch 22).
- Reproduction floor measured before any delta is interpreted (catch 19).
- Provenance: memo numbers only from recorded output.

---

## 9. DISCLOSURES — instrument decisions made with live values known

Every item below postdates the training launch. They are listed so a reader can
discount them appropriately; none is hidden in the body.

**9.1 Assembly provenance.** This document was assembled after training from v1
(message text, drafting environment) plus six amendment rulings (quoted from the
recorded review exchange). v1 and the prior P3 execution outputs have **no
artifact in this repository**. The prior session's P3 results are reproduction
targets, never the pass.

**9.2 C3 known before P3.** The cure screen was run as a wiring validation
*before* P3 v3 was implemented, so C3's value was known when the certificate was
built. τ derives from control-measured scales and no C3-related quantity, but the
ordering is a fact about how this was produced and is recorded rather than
reasoned away. Precedent: P3-v2's own docstring disclosed that live numbers were
known when v2 was specified.

**9.3 Denominator assignment amendment.** §3.2's per-cure assignment was adopted
knowing the live values. Its derivation is value-blind (recipe symmetry: what a
cure's read procedure shares with the ceiling's), and its effect was
**value-adverse** — it moved our own headline **down** in every arm of both
benchmarks. Value-aware in timing, value-blind in derivation, value-adverse in
effect is about as clean as a post-hoc instrument change can be. **It is still
one.**

**9.4 Per-cell recompute, and a falsified diagnosis.** §3.3 postdates the screen
and is rule-mandated (catch 26), not discretionary. The mechanism originally
predicted for `R` — *"the era head underfits on the smaller benchmark"* — is
**falsified three ways**: E10's per-task train counts vary **2.8%** (1658–1705)
while `R` varies **6×**; `R` vs n_train correlates **+0.58**, the wrong sign; and
the E8-vs-E10 level gap is confounded across data size, subject-disjointness and
test construction simultaneously. Recorded as a falsified diagnosis. `R` instead
tracks **era index k** (r = **+0.939** OFF, **+0.938** ON) — held as *suggestive
and not a claim* at n=3 with per-task sd up to 0.10; it belongs in discussion as
the deposition/erosion account's fourth appearance, not in the claims.

**9.5 τ divergence from the recorded derivation.** `honest_scale` measured
**1.391·D** here against the recorded **0.966·D**, giving τ = **0.030151·D**
against the recorded **0.0251·D**. The divergence is anchoring: `honest_scale` is
taken from **Control C** — a foreign subject group, data the certificate will
never certify — rather than from anything nearer the live set. That is the more
independent anchor; the recorded derivation was looser. The verdict is
insensitive: with a **2,128×** control separation and a **43×** margin between τ
and the live medians, no threshold in that range changes the outcome.

---

## 10. Supersession — E8

The per-cell recompute and the denominator assignment together re-grade E8's
headline. Decomposition:

| | (a) pooled/adj | (b) pooled/raw | (c) per-cell/raw | assignment | mean-of-ratios |
|---|---|---|---|---|---|
| E8 C3/OFF | +1.033 | +0.823 | **+0.798** | −0.210 | −0.025 |
| E10 C3/OFF | +0.937 | +0.683 | **+0.615** | −0.255 | −0.068 |

E8's published 1.037 reproduces at 1.033. **The supersession traces primarily to
an amendment adopted knowing live values, not to a rule-mandated correction** —
the assignment term is ~8× the mean-of-ratios term. Both corrections moved the
headline down.

**E8's H-R3 demotes to descriptive.** ρ = 0.798, CI [0.768, 0.828] straddles the
pre-registered 0.80 bar: not a pass, not a clean fail — indeterminate against it.
The narrative sentence *"full recovery of the reader channel"* is superseded
everywhere it appears by *"recovery of ~0.80 of the reader channel (CI straddling
the pre-registered bar; indeterminate against it)."* E8's load-bearing structure
survives — C3 was excluded from E8's branch for unfalsifiability regardless.

**E8 degeneracy audit: 0/24 cells excluded.** 2000-window test sets give
uniformly large denominators; the 200× spread is E10's small per-task cells.

**Frozen:** the tier-0 stack's **0.347** is a hardcoded literal
(`scripts/transport_estimate.py:51`) and a registered table entry
(`docs/E9_prereg.md:24`) with **no computing artifact in this repository**. It is
an era-head cure, so the assignment leaves its denominator unchanged, but it is
owed the same per-cell re-derivation. Until then the ledger's repair clause reads
*"partially repairable (magnitude under re-audit)."* H-X3, which is defined
against 0.347, inherits the freeze.
