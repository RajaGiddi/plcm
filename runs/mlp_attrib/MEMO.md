# MLP-Attribution Control — Result Memo

**Status:** Complete, n=3, single variable (`adapters.enabled`), backbone = MLP.
**Pre-registered branch: (b) attenuated.** The mechanism reading is stronger
than the branch label — see §2.

---

## 1. Results

| arm | AVG | RET | DIAG | forgetting |
|---|---|---|---|---|
| MLP, adapters **ON** | 0.9198 ± 0.0236 | 0.9071 | 0.9739 | 0.0676 |
| MLP, adapters **OFF** | 0.6690 ± 0.0103 | 0.5924 | 0.9776 | 0.3858 |

    DELTA = +25.09pp +/- 1.88   (per-seed +24.7, +27.5, +23.0)
    LSTM reference delta = +50.3pp
    -> branch (b), attenuated (15-40pp band)

## 2. The decomposition — the mechanism did not degrade

| condition | OFF | ON | delta |
|---|---|---|---|
| Permuted / LSTM (ref) | 0.4304 | 0.9331 | +50.3pp |
| Rotated / LSTM | 0.6446 | 0.9366 | +29.2pp |
| **Permuted / MLP** | **0.6690** | **0.9198** | **+25.1pp** |

**Change vs the LSTM/Permuted reference:**

    Rotated : OFF +21.4pp    ON  +0.4pp
    MLP     : OFF +23.9pp    ON  -1.3pp

The delta fell from 50.3 → 25.1 because the MLP's no-adapter baseline is
**23.9pp easier**, not because adapters work less well (**−1.3pp**). The §4
hedge fired exactly as pre-registered, for the second time in two experiments.

## 3. The invariance, now across two axes

| condition | adapter arm |
|---|---|
| Permuted / LSTM | 0.9331 |
| Rotated / LSTM | 0.9366 |
| Permuted / MLP | 0.9198 |
| **spread** | **1.7pp** |

Two architectures × two shift types; baselines spanning 43–67%; shifts differing
in linear invertibility. The endpoint moves 1.7pp.

**Every delta reduction measured in this project is baseline relief, never
mechanism degradation.** Twice, along independent axes. This is what converts a
within-benchmark result into a generality claim: reviewers cannot attribute the
effect to permutation-specific invertibility (refuted by Rotated) or to
recurrence (refuted here), because the endpoint does not move under either.

## 4. The pre-registered prediction chain held

The frozen-feature control was sequenced first specifically to supply this
run's discriminator, and its prediction was correct end to end:

1. Frozen MLP features measured **somewhat** less committed than LSTM
   (−15.7pp vs −19.3pp below the permutation-invariant reference).
2. → predicted: MLP baseline higher, but still forgetting enough that adapters
   remain necessary; prior shifted toward branch (a)/(b), away from (c).
3. → observed: OFF = 0.6690 with forgetting 0.3858; adapters buy +25.1pp.

The §3a reinterpretation clause (written for a possible branch (c),
*"adapters aren't needed where the backbone has a free input layer"*) is
**not** invoked: the MLP's free input layer did not rescue its frozen features,
and the MLP still forgets 0.39 without adapters.

## 5. Consequences

**Title scope was under-claimed.** The mechanism holds on a feedforward
backbone, so "…in Trainable Recurrent Encoders" is too narrow. Adopted:

> *Adapt the Input, Not the Weights: Input-Path Isolation Prevents Catastrophic
> Forgetting in Trainable Encoders*

LSTM remains the primary setting; the MLP arm is the generality evidence.

**Central figure gains a decomposition panel.** The three-condition
OFF/ON/delta table (with the change-vs-reference columns) is more persuasive
than any single number, because it makes the invariance visible rather than
asserted. It also retroactively strengthens Rotated: what first looked like a
gating law, and was corrected to baseline relief, now has a second instance
confirming the correction.

## 6. Caveat for the writing

The MLP ON arm carries the **largest seed spread in the project** (±0.0236,
roughly 2× the LSTM's ±0.0099). Nothing is at risk — the effect is 25pp against
a ±1.9pp delta spread — but the honest table shows it. Footnote, not a fix.

## 7. Deviations

None. Design, thresholds, seeds, and the parameter-count lock as pre-registered
(LSTM backbone 292,864 / MLP 266,752, recorded pre-run, not tuned).
