# Pre-Registration: E8 — Reader-Repair Cure Screen (Analysis-Only)

**Status:** **v2, locked.** Seven review catches applied (§9). No training.
One-day timebox. Immune to the 11.36pp training-path anomaly, which gets its own
bounded investigation in parallel and does not gate this.

**Question:** E6b proved a matched-late reader recovers old-task accuracy to within
`F_enc` of ceiling — but did it **with old-task labels** (replay in a head-sized
costume). Which *storage-honest* read-time correction recovers how much of
`F_read`? The screen grades cure families against a fixed floor and ceiling; the
winner (if any) earns a live-training validation contract (E9).

**Program context:** reader–encoder mismatch carries **75.5–88.2%** of HAR's
forgetting (bias-charged bound). E7 falsified the frozen-early cure — forgetting
*worsened* +10.8pp, because a head frozen at θ_{k+1} is **more** mismatched to θ₄'s
encoder, not less.

---

## 0. Pre-gate: generator correctness (run before any cure is scored)

C3 and the C0 family depend on constructing task-k-shaped data. The v1 spec's
generator was **wrong**: task-k data is `M_k Z + b_k` and task-4 data is
`M_4 Z + b_4`, so mapping current inputs *through* `M_k` yields `M_k M_4 Z`.

    contract v1:  x4 @ M_k.T                    max err vs true task-1 data = 5.1985  WRONG
    corrected  :  M_k M_4^-1 (x4 - b_4) + b_k   max err vs true task-1 data = 0.0000  OK

**The affine composition `M_k ∘ M_4⁻¹`, offsets carried, is locked.** The check
above re-runs at launch and **must print 0.0000 before any cure is scored.**
Uncaught, it would have handed C3 doubly-shifted pseudo-data and we would have read
the failure as "pseudo-refit doesn't work" — a false negative on the screen's
strongest candidate.

## 1. Cure candidates, with honesty tiers

Every cure = (write-time stored object) + (read-time procedure). Tiers declare what
each needs at read time. **No tier-0 claims from tier-2 measurements.**

| id | cure | tier | write-time storage (HAR) |
|---|---|---|---|
| **C0** | **analytic adapter** `A_k := M_k⁻¹` (affine) | 1 | none — `M_k` is the known calibration |
| **C1** | moment matching + era head | 0 | 512 (moments) + 1542 (era head) = **2054 floats ≈ 8KB/task** |
| **C0+C1** | both channels at once | 1 | as C1 |
| **C2** | Procrustes-in-S, oracle-estimated | 2 | C×256 prototypes/task |
| **C3** | pseudo-refit | 1 | era snapshot (parameters only) |

### C0 — the analytic adapter (FIRST in the read order)

Three experiments asked *why gradient descent will not find `M_k⁻¹`* (E5's null,
E5d's travel 0.1042, E7's frozen-head failure). **None wrote the answer in by
hand.** C0 does: set the adapter to the closed-form inverse and evaluate.

**Adapter class, stated rather than quietly upgraded:** tasks 1 and 4 carry
offsets, so the exact inverse is **affine** — `A_k(x) = M_k⁻¹(x − b_k)`. This is a
**different adapter class than the locked bias-free spec.** C0 is explicitly a
read-time oracle construction, not a claim that the locked adapter could have
learned this.

**Why either outcome is decisive:**
- Forgetting **collapses** → the input channel was always sufficient and *only the
  learning of it* ever failed. Every E5-arc conclusion sharpens retroactively.
- Forgetting **persists** → the input path, *solved exactly*, does not close the
  gap. Combined with E7's frozen-early failure this triangulates the reader
  mismatch as load-bearing **from both directions**, beyond argument.

**Reported alongside — the degenerate upper bound.** `A_k := M_4 ∘ M_k⁻¹` maps old
tasks into the *current* layout. On HAR all five tasks share identical underlying
windows and labels, so this reproduces task-4's test set exactly and hits DIAG by
construction. It is printed as a **labelled ceiling artifact, not a cure**; the gap
between it and C0 measures how far the encoder's preferred layout has moved.

### C1 — moment matching (tier 0, deployment-ready)

Write: per-dimension mean/std of task-k's deployed-feature distribution at θ_{k+1}.
Read: standardize the eval batch's own θ₄ features (**batch size 256, fixed
protocol parameter**) to the stored moments; read with the stored **era head**.
The head is not moved; the features are shifted back into its coordinate frame.

### C2 — Procrustes-in-S (tier 2, oracle)

Closed-form orthogonal solution (SVD), no iteration. θ₄-prototype estimation uses
task-k TRAIN inputs — **oracle**. The screen measures what rotation-correction
*can* recover; data-free estimation is E9's problem.

### C3 — pseudo-refit (tier 1, given known shift class)

Pseudo-old inputs via the §0 generator, labelled by the era snapshot's own
predictions (LwF-style: parameters, not data), convex head fit on their θ₄
features — the same recipe as the ceiling.

## 2. References (fixed per cell before any cure is scored)

- **Floor:** `acc_orig` — deployed accuracy. A cure must beat it to count.
- **Ceiling:** `acc_refit_θ₄` — E6b's labelled convex refit, same amended
  instrument (lbfgs, deployed feature, deployed pathway incl. adapters).
- **Recovery fraction (PRIMARY):** `ρ = (acc_c − acc_orig) / (acc_refit − R − acc_orig)`
  — the **bias-adjusted** ceiling. Charging the instrument's bias against the cure
  keeps the arm rather than discarding it, and is the conservative direction (E6b
  precedent). HAR/OFF's denominator is **0.2330**; HAR/v1-ON's **0.2456**.
- **Secondary column:** ρ against the raw ceiling, printed alongside.
- Cells with adjusted denominator ≤ 0.02 print **N/A**, never a ratio.

## 3. Arms

Primary **HAR/OFF** and **HAR/v1-ON**. Secondary **MNIST/OFF** for cross-regime
evidence. Seeds {42, 1337, 2024}.

*(v1 excluded HAR/OFF via a gate that would have made H-R1/H-R3 unevaluable on the
very arm they name — its pooled R is +0.0589. §2's adjusted denominator replaces
that exclusion.)*

## 4. Hypotheses

- **H-R0 (input channel, solved exactly):** C0 forgetting on HAR/OFF **≤ 0.20**
  vs the 0.3736 baseline (**arm: HAR/OFF `mafc_off`, forgetting** — named per
  Wave 1's rule: "baseline" is fine for a method and never for an arm).
- **H-R1 (repairable storage-honestly):** at least one of {C0, C1, C3} — the
  non-oracle tiers — reaches pooled **ρ ≥ 0.50** on HAR/OFF. *(E6b proved ρ=1.0 is
  achievable with labels; half the gap without them is the minimum worth a
  live-training contract.)*
- **H-R2 (rotation content):** C2's ρ exceeds C1's by **≥ 0.15** pooled. If moment
  matching matches Procrustes, the drift is affine and the cheap cure suffices —
  either answer designs E9.
- **H-R3 (approach to ceiling):** C3 pooled **ρ ≥ 0.80** on HAR/OFF.
- **H-R4 (cross-regime):** best HAR cure reaches **ρ ≥ 0.30** on MNIST/OFF.
  *(Runnable: MNIST/OFF is PERMUTED MNIST and permutations are known and exactly
  invertible, so the generator is `P_k P_4⁻¹`. The "no known generator" caveat
  belongs to ROTATED MNIST's lossy resampling, which is not this arm.)*
- **H-R5 (the composition):** C0+C1 pooled ρ exceeds the better of C0, C1 alone by
  **≥ 0.10** — both measured channels attacked at once, at no extra compute. This
  is the closest the screen comes to "the full cure, storage-honest."

## 5. Branches (ties downward)

| branch | condition | reading |
|---|---|---|
| **(A)** | H-R1 | Reader channel repairable at read time from stored parameters/statistics. **E9 drafted before any training run.** Ledger gains: *"...and that mismatch is repairable at read time without stored data."* |
| **(B)** | ¬H-R1 ∧ best non-oracle ρ ≥ 0.20 | Partially repairable; name the residual from the per-task tables; **no E9 until the residual is named.** |
| **(C)** | all non-oracle ρ < 0.20 ∧ C2-oracle ρ ≥ 0.50 | Mismatch is rotational **and its estimation is the hard part**. The cure exists; data-free rotation estimation becomes the named open question. |
| **(D)** | everything < 0.20 incl. oracle C2 | Not linearly correctable at read time. **Repairability claim dropped**; framework reports measurement without remedy. Full prominence. |

**Independent of branch:** H-R0's answer is reported at full prominence either way
— it closes the program's oldest open question.

## 6. Audit — artifacts and premises

| requirement | source | status |
|---|---|---|
| θ₄ checkpoints, HAR OFF / v1-ON, 3 seeds | E6 fresh runs / E5 | ✅ |
| Era snapshots θ_{k+1}, all k, both arms | 50-ckpt/run saves | ✅ |
| MNIST OFF θ₄ + era snapshots, 3 seeds | `e4_off_*` | ✅ |
| Frozen shift config (`M_k` known) | `3de66e205eb7`, re-checked at run | ✅ |
| Era classifier heads inside snapshots | full `state_dict` saves | ✅ |
| MNIST permutations reconstructible | `PermutedMNISTBenchmark(seed=run_seed)` | ✅ |
| Convex-refit instrument (amended, feature-locked) | E6b v3 | ✅ |
| **HAR adapters are at identity** — so the "current inputs through `A_k`" generator is DEAD on HAR | E5d travel 0.1042, residual 0.9927 | ⚠️ **E9 design constraint** |

## 7. Locks

- Analysis only; §6 is the complete inventory; absent cells print **ABSENT**.
- **§0 generator check must print 0.0000 before any cure is scored.**
- Instrument: E6b-amended convex refit verbatim; **per-seed R printed**; ρ primary
  against the bias-adjusted ceiling.
- No per-cell tuning. C1's standardization is batch-level with no learned
  parameters (**batch 256**); C2's Procrustes is closed-form SVD; C3's head fit is
  the ceiling's recipe.
- Read order: audit → **§0 generator check** → references → **C0** → C1 → C0+C1 →
  C2 → C3 → hypotheses → branch.
- Provenance: memo numbers only from the script's recorded output.
- Timebox: one day. Blocked → report state, stop.

## 8. Predictions on record

C0 forgetting 0.15–0.25 vs the 0.3736 HAR/OFF `mafc_off` forgetting baseline (~55%, wide — *this is exactly where the
program keeps surprising us*) · C1 pooled ρ 0.20–0.40 (~60%) · H-R2 (~50%) ·
H-R3 (~65%) · H-R1 (~70%) · H-R4 (~55%) · H-R5 (~55%) · joint (A) ~70%.

*Calibration:* two 75% misses (E5, E5b) and one 15%-branch fire (E7).

## 9. Review log — seven catches

1. **BLOCKING — §7's gate excluded HAR/OFF**, the arm H-R1/H-R3 name (R = +0.0589).
   → bias-adjusted ceiling as primary denominator; exclusion only if it goes ≤0.02.
2. **BLOCKING — C3's generator was algebraically wrong** (5.1985 error). → affine
   `M_k ∘ M_4⁻¹`, verified 0.0000, promoted to a pre-gate.
3. **BLOCKING — C3's known-`M_k` assumption licensed a cheaper untested
   intervention.** → **C0 added, first in the read order.** Three experiments asked
   why SGD won't find `M_k⁻¹`; none wrote it in by hand.
4. **The adapter route is dead on HAR** (adapters at identity) → recorded as an E9
   design constraint, not a usable generator.
5. **H-R4's premise was wrong for its arm** — permuted MNIST *is* invertible. →
   caveat moved to rotated MNIST.
6. **C1's storage accounting omitted the era head** → full 8KB/task in the tier table.
7. **C1's batch statistics are a protocol parameter** → batch 256 fixed and recorded.
