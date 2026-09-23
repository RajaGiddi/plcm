# E8 — Reader-Repair Cure Screen: Result Memo

**Status:** Complete, n=3, analysis-only. **BRANCH (C).**
Pre-registration: `docs/E8_prereg.md` (v2).

## 0. Ledger — a RETRACTION

The prior edit claimed the reader channel is *"repairable at read time from stored
parameters alone (pseudo-refit reaches the labeled-refit ceiling)."* **That sentence
is withdrawn.** Corrected:

> *...and where the input path fails to engage, 75–90% of the forgetting is
> reader–encoder mismatch — a channel that is **rotational in character** and
> **recoverable in principle** (~58% under oracle-estimated alignment), but for
> which **no storage-honest read-time repair has yet been demonstrated**: the one
> cure that reached ceiling did so by reconstructing old data the benchmark makes
> recoverable, and **exact input correction is actively harmful**.*

---

## 1. HEADLINE — H-R0 fails, and it closes the E5 arc

**The analytic adapter makes things worse.** `A_k := M_k⁻¹`, written in by hand,
exact by construction:

| arm | floor (deployed) | **C0 (perfect inverse)** | ρ |
|---|---|---|---|
| HAR/OFF | 0.5087 | **0.4738** | **−0.151** |
| HAR/v1-ON | 0.5513 | **0.4980** | **−0.216** |

    H-R0  C0 (1 - acc) = 0.5262 <= 0.20   ->  FAIL

**The mechanism is in the artifact column.** C0 maps task-k input to the **base**
layout, which is what the locked adapter spec was always aiming at. But θ₄'s
encoder has drifted to **task-4's** layout, and mapping there instead scores
**0.8035** (HAR/OFF) — a labelled ceiling artifact, not a cure, since on HAR that
transformation reproduces task 4's own test set.

> **The exact input inverse is not a solution the optimizer missed — it is a
> solution the deployed system has moved past.**

Three experiments asked why gradient descent would not find `M_k⁻¹`. **The answer
is that finding it would have hurt.** One mechanism now reconciles: E5's null,
E5d's *away-from-ideal* travel (task 1 residual 1.2351 → 1.4333 under forced
recruitment), E7's frozen-early failure, and C0+C1's cancellation here.

## 2. The screen

| cure | tier | HAR/OFF acc | ρ (adj) | HAR/v1-ON acc | ρ (adj) |
|---|---|---|---|---|---|
| floor | — | 0.5087 | — | 0.5513 | — |
| **C0** analytic adapter | 1 | 0.4738 | **−0.151** | 0.4980 | **−0.216** |
| **C1** moment matching | **0** | 0.5524 | +0.189 | 0.5740 | +0.092 |
| **C0+C1** | 1 | 0.5101 | +0.006 | 0.5235 | −0.113 |
| **C2** Procrustes | **2 (oracle)** | 0.6420 | **+0.577** | 0.6846 | **+0.542** |
| **C3** pseudo-refit | 1 | 0.7485 | +1.037 | 0.7901 | +0.971 |
| *C0deg* | *artifact* | *0.8035* | — | *0.8672* | — |

    H-R1  best non-oracle rho +1.033 >= 0.50    PASS  (on C3 -- see sec-3)
    H-R2  C2 - C1 = +0.354 >= 0.15              PASS
    H-R3  C3 rho +1.033 >= 0.80                 PASS  (scope-limited, sec-3)
    H-R5  C0+C1 vs best single -0.191 >= 0.10   FAIL

**H-R2 is the constructive result.** Procrustes beats moment matching by **+0.354**,
so the mismatch is **substantially rotational, not affine** — consistent with E6's
48–77° principal angles, and it is what designs E9.

**H-R5's failure follows from H-R0.** Composing a harmful correction (C0) with a
helpful one (C1) nets out worse than C1 alone: +0.006 vs +0.189.

## 3. C3 is excluded from the branch — and why exclusion is not enough

C3 reaches ceiling. It is nonetheless **excluded from the branch determination**,
and the reason is stronger than confounding:

**C3's ρ is unfalsifiable on this benchmark.** The generator reconstructs task-k's
inputs from task-4's **exactly** — verified on the sequential test split, max err
**0.000001** across all four old tasks. Every HAR task is the same 2947 windows in
different channel coordinates (which is exactly what E5b's no-shift control found),
and permuted MNIST shares the property. *Any* method that reconstructs old inputs
exactly will match the labeled-refit ceiling, **because that is what the ceiling
is.** So C3's 1.03 measures **the benchmark's invertibility, not the cure's power.**

> **C3 achieves ceiling but is, on permutation-family benchmarks, replay with a
> zero-cost buffer. Its result carries no evidence about datasets where old content
> is not recoverable from current content.**

Reporting it as a cure would be the storage-frontier error in a new costume: a claim
that looks like a win because the accounting hides what is being retained.

### 3a. A methodological finding worth its own line

**Permuted-MNIST-family benchmarks cannot distinguish generative repair from
replay.** Any coordinate-permutation task family is exactly invertible, so a
"generative" method that regenerates old inputs is doing replay with the storage
cost hidden in the task construction. This is a caution the continual-learning
literature would benefit from, and it is not specific to this project.

## 4. Branch (C), as the contract anticipated

> *All non-oracle ρ < 0.20 while C2-oracle ρ ≥ 0.50: the mismatch is rotational AND
> its estimation is the hard part — the cure exists but needs old-data-free rotation
> estimation; that estimation problem becomes the named open question.*

Excluding C3: best non-oracle is **C1 at +0.189** (below 0.20); **C2-oracle at
+0.577** clears 0.50. Branch (C) exactly.

**No E9 cure contract yet.** What E9 must solve is now precisely posed:

- **Target:** match C2-oracle's **0.577** without oracle access.
- **Problem:** data-free rotation estimation in the readout subspace.
- **Constraint 1:** the adapter route is **dead on HAR** — `A_k ≈ I` (E5d travel
  0.1042), so "current inputs through the old adapter" yields current inputs.
- **Constraint 2:** C3-style generation is **disqualified as evidence** on
  permutation-family benchmarks; a new generator or a genuinely non-invertible
  benchmark is required.

## 5. Provenance

Numbers from `scripts/cure_screen.py` → `runs/e8/cures.json`. Generator pre-gate
passed at 0.0000 (v1's spec erred by **19.19**); had it not been caught in review,
C3 would have received doubly-shifted pseudo-data and its failure would have read as
"pseudo-refit does not work" — a false negative on the screen's strongest candidate.

**Catch 22 logged.** The first version of §3's scope check printed *"EXACT
reconstruction … recovered for free"* directly above its own numbers reading **17.5**
and **1.0**. The conclusion was right but had been written from the expected result
rather than computed from the measurement, and the check as shown did not establish
it — the train loaders are shuffled, so it compared different row orderings. The
sequential test split proved the point afterwards. Second instance of its class
(E4's `drift_probe.py` fall-through verdict was the first), now a standing rule:
**a printed verdict must be derived from the values printed beside it; when a
printed claim and a printed number disagree, the number wins.**
