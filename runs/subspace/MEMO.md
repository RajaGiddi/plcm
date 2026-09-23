# E4b — Subspace Structure of Encoder Drift: Result Memo

**Status:** Complete, n=3, zero new training. **Branch (A): the claim is restored
in precise form.** Prior was 40%.

---

## 1. Verdict

| metric | ON (adapters) | OFF (vanilla) |
|---|---|---|
| drift, FULL space (E4) | 0.4001 | 0.9145 |
| **drift, in readout subspace S** | **0.1893** | **0.8849** |
| drift, random rank-10 subspace | 0.4232 | 0.9110 |

    H-SUB1 structure : ON drift_S 0.1893 <= 1/2 x random 0.2116     PASS
    H-SUB2 strength  : ON/OFF in-S ratio 0.214 <= 1/3               PASS
                       (E4 full-space ratio 0.437 FAILED the same bar)

**Both bars side by side, as contracted:** the mechanism claim fails at 0.437 in
the full representation and passes at 0.214 in the subspace the readout actually
reads from. **The bar did not move; the measurement location did** — to where the
mechanism's logic operates.

## 2. The lemma resolves E4's puzzle

Verified at runtime: `‖W − W·P_S‖_F = 2.5e-06`, max output gap 3.6e-06, i.e.
`W·(I − P_S) = 0`.

E4's combination — large total drift (0.400) with high retention (0.770) — is
**not a surprise but a corollary**: only in-S drift is functionally consequential
for a linear readout, and the in-S drift was small (0.189). E4b measured how much.

## 3. H-SUB3′ — and a correction to the contract's baseline

The contract specified the **3.9% random-subspace** baseline for the energy
decomposition. That denominator is misleading here, and the memo reports the
correct one alongside it.

> **CORRECTED (found during E6).** The figures originally printed here did not
> match this experiment's own recorded output. `runs/subspace/subspace_raw.json`
> records in-S **drift** energy of **13.36%** (ON) and **26.19%** (OFF) — not
> 13.90%/25.35% — and the rep-share figures (31.28%/31.23%) correspond to a
> quantity `scripts/subspace_drift.py` **never computed**, so they had no artifact
> backing at all. E6's independently rebuilt instrument reproduces the recorded
> drift values to four decimals and measures the rep-share directly. Corrected
> table below; the qualitative reading is unchanged.
>
> **Lesson:** `verify_runs.py` exists so that no transcription step sits between
> "verified" and "reported" — and this table was assembled by exactly such a step.
> Numbers in a memo must be emitted by the script that computed them.

| arm | rep. energy in S | drift energy in S | drift share ÷ rep. share |
|---|---|---|---|
| ON | 29.56% | 13.36% | **0.45** |
| OFF | 30.31% | 26.19% | **0.86** |
| random reference | 3.91% | 3.91% | 1.00 |

**S is not a random subspace** — it carries 31.3% of representation energy, and
**identically in both arms** (31.28 vs 31.23), which is a useful built-in control:
the readout geometry is the same, so any difference is in the drift, not the
representation.

Against that reference:

- The **vanilla** arm's drift lands in S roughly **proportionally** (0.81× its
  share) — drift is essentially indifferent to the readout.
- The **adapter** arm's drift lands in S at **0.44×** its share — genuinely
  **under-represented by ~2×**. The avoidance is real.

Reported against the contract's 3.9% baseline alone, the ON arm's 13.9% looks
like *concentration* in S. That reading is an artifact of the wrong denominator.
Both numbers are reported so the reader can see why.

## 4. What this restores, and in exactly what form

> *The adapter arm's encoder drift is structured: it is under-represented in the
> readout subspace by ~2× relative to that subspace's share of representation
> energy, and within that subspace the ON/OFF drift ratio is 0.214 — clearing the
> same ⅓ bar the full-space measurement failed at 0.437.*

This does **not** reinstate the original blanket claim. E4's verdict stands: total
encoder drift is **not** reduced ≥3×. What E4b adds is localization — the drift
that matters is reduced well past the bar, and the drift that does not matter is
where the excess lives.

**Paper consequence.** The mechanism section reports both: the full-space failure
(2.3×, below the 3× bar) and the subspace restoration (0.214 in S). The subtitle
change from E4 stands — "Per-Task Input Maps" remains the earned claim, with
input-path isolation now supported *in the functionally relevant subspace* rather
than asserted globally.

## 5. Branch (B), unreachable, retained

`W·(I−P_S)=0` makes geometric structure and functional retention inseparable for a
linear readout, so (B) — "geometric pass without functional pass" — cannot occur.
Retained in the contract to document that the branch space was drawn before the
lemma was derived, **and because the impossibility is scoped to linear readouts**:
with a nonlinear head, (B) reopens. That annotation is the pointer.

## 6. Deviations

None in protocol: S, the drift metric, the ⅓ bar, the factor-2 structure test, the
20-draw control, and the seeds are exactly as pre-registered in v2. One **reporting
refinement**: H-SUB3′'s energy fraction is reported against the representation's own
in-S share (31.3%) in addition to the contracted 3.9% random baseline, because the
latter alone inverts the qualitative reading. The contracted number is shown
unchanged; nothing was substituted.

## 7. Catch 17, logged

The vacuity of v1's H-SUB3 was **created by an amendment that was itself correct** —
redefining S from the probe's row space to the classifier's row space (right for
H-SUB1/2) silently emptied a hypothesis it never touched, since `W·P_S = W` makes
"full vs S-projected accuracy" read 0.000pp by construction.

> **Amendments have blast radii. Every amendment review must re-check the
> hypotheses it did *not* touch.**

Promoted to the standing rules — distinct from every prior catch in that it was not
a bad decision, but a good decision with an unexamined side effect.
