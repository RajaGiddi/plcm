# E1 — Fashion-MNIST Generality: Result Memo

**Status:** Complete, 12/12 runs verified mechanically, n=3.
**Both contracted hypotheses PASS. One pre-registered prediction MISSED.**
Pre-registration: `docs/GENERALITY_prereg.md` (v2, §2 E1, §3).

Scope, as the contract itself states: this escapes MNIST *digits*, not the 28×28
grayscale regime. Supporting evidence, not the answer to the reviewer objection.
E5 (UCI HAR) is the one that answers it.

---

## 1. Verdict

LSTM, 5 tasks × 10 epochs, seeds {42, 1337, 2024}, ON = per-task input adapters.

| shift | arm | AVG | normalized retention (AVG/DIAG) |
|---|---|---|---|
| permuted | **ON** | **0.8168 ± 0.0344** | **0.9238 ± 0.0383** |
| permuted | OFF | 0.4654 ± 0.0107 | 0.5338 ± 0.0121 |
| rotated | **ON** | **0.8193 ± 0.0173** | **0.9284 ± 0.0183** |
| rotated | OFF | 0.3831 ± 0.0134 | 0.4311 ± 0.0153 |

    H-G1a  ON normalized retention >= 0.90  : E1 pooled 0.9261 +/- 0.0301   PASS
    H-G1b  ON - OFF >= +15pp AVG, EVERY pair: permuted +35.15pp             PASS
                                              rotated  +43.62pp             PASS

**Story-changing threshold (span > 5pp → reframe):** MNIST reference 0.954,
Fashion-permuted 0.9238, Fashion-rotated 0.9284 → **span 3.02pp**. The invariance
framing holds. **PROVISIONAL** — the contract evaluates this across *all* tested
benchmark-shift pairs, and E5/E2 are not in yet. E5 can still fire this branch.

## 2. The prediction that missed — reported first, not buried

The contract's §5 prediction for E1 was: *"ON arm ~90–94% raw AVG on both shifts."*

**Measured: 81.7% and 81.9%.** The prediction missed low by roughly 8–12pp.

The reason is checkable rather than post-hoc: **the ceiling moved.** Fashion-MNIST
DIAG is 0.884 against MNIST's 0.978 — the dataset is simply harder, so the same
retention quality lands at a lower raw number. That is precisely why the contract
made **normalized retention** the gating metric and pre-registered DIAG as its
denominator (v2 Finding 5), and the normalized metric passes comfortably.

Recording this anyway, because "the contracted hypothesis passed" and "our stated
prediction was right" are different claims and only the first one is true here. The
prediction was written in raw-accuracy units that do not transfer across datasets —
a small, real forecasting error worth not papering over.

## 3. Seed 2024, permuted — the n=3 rule earning its keep again

| seed | AVG | NR | forgetting |
|---|---|---|---|
| 42 | 0.8391 | 0.9488 | 0.0566 |
| 1337 | 0.8431 | 0.9530 | 0.0520 |
| **2024** | **0.7683** | **0.8697** | **0.1439** |

Seed 2024 is a genuine outlier: forgetting nearly 3× the other two, and **the only
one of six ON runs that falls below the 0.90 bar individually.**

> **Had permuted-Fashion been run at n=1 on seed 2024, H-G1a would have FAILED
> (0.8697 < 0.90).** At n=3 it passes at 0.9238.

This is the standing rule working in the pessimistic direction — the fifth instance
where single-seed measurement would have produced a wrong headline, and the second
where the error was pessimistic rather than optimistic. It is also why the ON-arm
std (±0.0383 permuted) is ~2× the rotated arm's: one seed carries it.

No hyperparameter was selected here, so the "selection at n=1" hazard does not
apply; this is purely a measurement-noise case.

## 4. What E1 adds, stated at its actual weight

The adapter mechanism transfers off MNIST digits intact: **+35.1pp and +43.6pp AVG**
over the no-adapter arm, with normalized retention within 3.02pp of the MNIST
reference across three benchmark-shift pairs.

It does **not** escape the 28×28 grayscale regime, and the contract said so before
the runs. The OFF arm's collapse is if anything *worse* on Fashion than on MNIST
(0.43–0.53 normalized retention), so the delta grows mainly because the baseline
degrades further — the Rotated/MLP hedge, applying again exactly as pre-named.

## 5. Deviations

None. Pipeline, seeds, arms, epochs, and both thresholds are as pre-registered.
Verification was mechanical (`scripts/verify_runs.py`, catch-18 definition): 12/12
complete matrices, full task histories, all mtimes post-launch.

## 6. Standing on the venue rule

The contract's ICLR gate needs H-G1 on **≥1 non-MNIST dataset (E2 or E5)**.
E1 is not that dataset — it is the MNIST family. **E5 remains the gate**, and it is
running.
