# E4 — Encoder-Drift Probe: Result Memo

**Status:** Complete, n=3, matched arms differing in `adapters.enabled` only.

## VERDICT: H-G2 **FAILED** — both pre-registered thresholds.

    ON/OFF drift ratio = 0.437   (H-G2 required <= 0.333)   FAIL
    ON probe retention = 0.770   (H-G2 required >= 0.80)    FAIL

The verdict is not optional and is recorded as a failure. What follows is the
*characterization*, which the numbers carry.

---

## 1. Measurements

| metric | ON (adapters) | OFF (vanilla) | ON advantage |
|---|---|---|---|
| drift = 1 − cos(f_θ0, f_θ4) | **0.4001 ± 0.0229** | **0.9145 ± 0.0103** | **2.3× less drift** |
| CKA (higher = stabler) | 0.8119 | 0.2814 | **2.9× higher** |
| probe retention (θ₀ probe on θ₄ reps) | **0.7699 ± 0.0415** | 0.1481 | **5.2× better** |

Per-seed: ON drift 0.368 / 0.417 / 0.416; OFF drift 0.925 / 0.918 / 0.900.
ON probe retention 0.811 / 0.713 / 0.785.

**Sentence for the paper's mechanism section:**

> *Input-path isolation reduces encoder drift 2.3× and readout degradation 5.2×,
> but does not meet the ≥3× isolation threshold this work pre-registered.*

## 2. The contract failed before the mechanism did

This must be stated plainly, because both alternative readings are wrong.

The contract enumerated **two** outcomes — clean pass, clean falsification — for a
measurement that was always going to land on a continuum, and the ⅓ threshold was a
round number **with no derivation behind it**. The data landed between the branches
because *the branches were badly drawn*, not because the result is ambiguous.

**The result is crisp: large, measurable, and insufficient by a bar we invented.**

- The reading *"the mechanism barely does anything"* is **false** — 2.3×/2.9×/5.2×.
- The reading *"the thresholds don't count"* is **worse** — the contract binds.

Neither the pre-registered falsification language (*"the protective mechanism is not
reduced drift"* — drift **is** reduced 2.3×) nor the script's original fall-through text
(*"representations do not support the old readout"* — they retain 77% vs 14.8%)
described this outcome correctly. A post-hoc middle band was considered and
**rejected**: a band drawn after the numbers exist is not a weaker pre-registration,
it is not one at all.

## 3. What this settles for the paper

**Subtitle changes.** "Input-Path Isolation Prevents Catastrophic Forgetting"
over-claims a mechanism no longer statable at full strength. Adopted:

> *Adapt the Input, Not the Weights: Per-Task Input Maps Prevent Catastrophic
> Forgetting in Trainable Encoders*

The mechanism name survives **inside** the paper as the *hypothesis the drift probe
tested*, with its measured partial support — more honest, and more interesting, than
the assertion was.

**The mechanism section becomes a measurement section.** What isolation demonstrably
does (drift 2.3×, CKA 2.9×, readout 5.2×) presented alongside what it does **not** do
(render the encoder still; fully protect the old readout).

**Empirical claims are untouched.** 93.3% AVG, the +50.3pp attribution, and the 1.7pp
ON-arm invariance across four conditions stand unchanged. Nothing downstream of E4 in
the generality package is affected; E1/E5/E2 proceed as contracted.

## 4. n=1 optimism: now 6 of 7

The seed-42 preview read probe retention **0.811** (above the 0.80 bar); the n=3 mean
is **0.770** (below it). Seed 1337 came in at 0.713. Had the verdict been called at
n=1 it would have been recorded as a partial pass.

| configuration | n=1 | n=3 | shift |
|---|---|---|---|
| adapters, unfrozen | 0.9470 | 0.9331 | −1.4pp |
| vanilla LSTM | 0.4756 | 0.4399 | −3.6pp |
| EWC λ=200 | 0.4950 | 0.4330 | −6.2pp |
| low-rank r32 | 0.7047 | 0.6877 | −1.7pp |
| low-rank r64 | 0.8388 | 0.8719 | **+3.3pp** |
| **E4 probe retention** | **0.811** | **0.770** | **−4.1pp** |

Six of seven configurations first measured at n=1 moved on seeding; five optimistic,
one pessimistic. Seeding is a correction of unknown sign.

## 5. For the discussion — a finding about the finding

This is the **fourth** mechanism explanation this project has advanced and had to
retract or weaken:

1. GGC composition / persistent memory — measured at ~0 across four tests.
2. "The adapter represents `P_k⁻¹`" — falsified by Rotated (works where no exact
   inverse exists) and by low-rank (better accuracy, *worse* reconstruction).
3. Variance collapse as isolation evidence — retracted (ER is tighter *without*
   isolation).
4. Input-path gradient isolation — **partially supported**: 2.3×–5.2×, against a 3×
   bar set blind.

**The effect is far more robust than any explanation offered for it.** What is new here
is that we have now *quantified* how much of the best explanation survives, rather than
replacing one story with another.

## 6. Deviations

None in protocol. One **defect fixed post-hoc in reporting code** (not in method): the
probe script's fall-through verdict string mis-described a failing result as
"representations do not support the old readout." Corrected to print the pass/fail
verdict *and* the measured ratios. Metrics, thresholds, seeds, and arms are exactly as
pre-registered in `docs/GENERALITY_prereg.md` v2.
