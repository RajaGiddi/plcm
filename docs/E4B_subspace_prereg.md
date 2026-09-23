# Pre-Registration: E4b — Subspace Structure of Encoder Drift

**Status:** **v2, locked.** Zero training; runs on E4's existing checkpoints.

**Motivation.** E4 measured drift in the full 256-dim representation and H-G2
failed (ON/OFF ratio 0.437 vs a ⅓ bar). But the mechanism claim was never about
*total* drift — it is about whether the encoder still supports the old readout.
E4b asks whether the ON arm's drift is **structured**: does it avoid the subspace
the readout actually reads from?

---

## 1. Definitions (locked)

- **S** ≡ row space of the **task-0 shared classifier** `W` (rank 10), computed
  **per arm**. It is the readout the model actually used — no auxiliary training,
  no free parameter, and "each arm's drift judged against its own readout" is the
  semantically correct form of the mechanism claim.
  `P_S = Vᵀ V` from the SVD `W = U Σ Vᵀ` (top-10 right singular vectors).
- **drift-in-S** ≡ `1 − cos(P_S f_θ0, P_S f_θ4)` — the same functional form as E4,
  so the ⅓ bars are **literally comparable**. That comparability is the point.
- **Random control** ≡ 20 draws of a uniformly random rank-10 subspace; mean ± std.
- Task-0 test inputs; θ₀ = end of task 0; θ₄ = end of task 4; seeds {42, 1337, 2024}.

### 1a. LEMMA — for a linear readout, only in-subspace drift is functionally consequential

Let `W ∈ ℝ^{10×256}` be the readout and `P_S` the projector onto its row space.
Writing `W = U Σ Vᵀ` and `P_S = V Vᵀ`:

    W · P_S = U Σ Vᵀ V Vᵀ = U Σ Vᵀ = W          (V has orthonormal columns)
    therefore  W · (I − P_S) = 0

**Verified numerically:** `‖W − W·P_S‖_F = 2.5e-06` (float noise), and
`max |Wx − W(P_S x)| = 4.8e-06` over random x.

**Consequences.**
1. Out-of-S drift is **functionally invisible** to a linear readout, by algebra —
   not as an empirical finding.
2. Geometric structure and functional retention are therefore **inseparable** at
   this resolution: if drift avoids S, retention must hold; if retention falls,
   in-S drift must be responsible.
3. **E4's puzzle becomes a corollary.** Large total drift (0.400) with high
   retention (0.770) is not a surprise: the drift was large, but its in-S
   component evidently was not. E4b measures how much.

This lemma belongs in the paper's appendix regardless of which branch fires.

---

## 2. Hypotheses (fixed before any decomposition number exists)

- **H-SUB1 (structure):** ON-arm `drift-in-S ≤ ½ × drift-in-random-subspace`.
  The factor 2 is **the contract's one free choice, declared in plain sight.**
  *Why the control is the spine:* at d=256 a random rank-10 subspace captures
  **10/256 = 3.9%** of a random vector's energy. Drift "avoiding" *any* rank-10
  subspace is cheap geometry. Every structure claim is therefore made **relative
  to the random baseline**, never in absolute terms.
- **H-SUB2 (strength, in the subspace where the mechanism operates):** ON/OFF
  `drift-in-S` ratio **≤ ⅓** — deliberately the **same bar the full-space claim
  just failed at 0.437.** A restored claim on a friendlier threshold would be
  goalpost-moving with extra steps. Both bars are reported side by side so the
  reader sees exactly what failed and what passed.
- **H-SUB3′ (decomposition, reported not gated):** in-S drift energy fraction
  `‖P_S δ‖² / ‖δ‖²` per arm, against the 3.9% random baseline, where
  `δ = f_θ4 − f_θ0`. *(v1's H-SUB3 was a pass/fail guard; see §5.)*

## 3. Branches (no gaps; ties default DOWNWARD)

| branch | condition | reading |
|---|---|---|
| **(A)** | H-SUB1 ∧ H-SUB2 | Claim restored in precise form: drift is structured to avoid the readout subspace **and** meets the original strength bar there. |
| **(B)** | *unreachable* | **Unreachable under a linear readout:** `W·(I−P_S)=0` makes geometric structure and functional retention inseparable (§1a). **Retained** to document that the branch space was drawn before the lemma was derived — and because the impossibility is **scoped to linear readouts**. With a nonlinear head, (B) reopens; this annotation is the pointer that says so. |
| **(C)** | H-SUB1 ∧ ¬H-SUB2 | Structure without strength: drift avoids S more than chance, but not by the pre-registered factor. Measured-mechanism framing from E4 stands, now with a sharper localization. |
| **(D)** | ¬H-SUB1 | Retention despite **unstructured** drift. Deepens the mystery rather than resolving it: the readout survives without the encoder protecting its subspace. Reported as a genuine open problem. |

**Predictions on record (v2 structure):** (A) ~40%, (B) 0% by construction,
(C) ~45% (absorbing (B)'s prior mass — near-misses land here), (D) ~15%.

## 4. Locks

Seeds {42, 1337, 2024}; mean ± std; no claim at n=1. No new training. Metrics,
S-definition, and thresholds as above; no post-hoc bands (E4's lesson). Ties
resolve to the *lower* branch.

## 5. Amendment log

- **v1:** S ≡ row space of the **E4 probe**; H-SUB3 a pass/fail guard (S-projected
  vs full retention within 3pp); four live branches.
- **v2 (all pre-run, blind to any E4b decomposition number — none exists yet):**
  1. **S redefined: probe row space → task-0 classifier row space.** The probe is
     an auxiliary object trained for measurement; the classifier is the readout
     the model actually used. No extra training, no free parameter.
  2. **H-SUB3 → H-SUB3′.** The S-source change (correct for H-SUB1/2) **silently
     emptied H-SUB3**: under classifier-defined S, `W·P_S = W`, so "full vs
     S-projected accuracy" is **0.000pp by construction** and would print a pass
     every time. Replaced by the decomposition, which carries information.
  3. **Branch (B) annotated as unreachable**, not deleted (see table).
  4. **Random control 5 → 20 draws** (thrift without a reason; 20 costs nothing).

### Catch 17 and its lineage — a new failure mode

The vacuity was **created by an amendment that was itself correct.** Fixing S for
H-SUB1/2 emptied a hypothesis the amendment never touched.

> **Amendments have blast radii. Every amendment review must re-check the
> hypotheses it did *not* touch.**

Promoted to the standing rules. This is distinct from every prior catch: it was
not a bad decision, but a good decision with an unexamined side effect.

## 6. Reporting

One memo: per-arm drift-in-S vs random-subspace baseline, the ON/OFF in-S ratio
against the ⅓ bar (with E4's full-space 0.437 alongside), the H-SUB3′ energy
decomposition against 3.9%, the branch taken, and deviations. Supersessions visible.
