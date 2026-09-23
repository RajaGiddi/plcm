# E6 — Drift Anatomy on HAR: Result Memo

**Status:** Complete, n=3, 3 runs + analysis. **BRANCH (C).**
H-B1 **PASS**, H-B2 **FAIL**, H-B3′ **PASS**, H-B4 **PASS (with a caveat that
matters)**. Pre-registration: `docs/E6_prereg.md` (v2).

---

## 0. Instrument check and reproduction spread — read before anything

**Instrument validated.** On MNIST the rebuilt anatomy returns `drift_S` ON
**0.1893** / OFF **0.8849**, ratio **0.214** — E4b's recorded values to four
decimals, under the new per-dataset geometry (rank 6 HAR / rank 10 MNIST).

**§0a contingency did not fire.** All three regenerated OFF runs reproduce the
originals **bit-identically** (final-row hashes `34b4519c`, `0e213acd`,
`323439bb`). The OFF anatomy therefore sits on trajectories whose endpoints are
the published E5 numbers — no interpretation ambiguity at all.

## 1. The two-dataset anatomy (M5)

| dataset/arm | rank | drift_full | **drift_S** | drift_rand | drift energy in S | S-share | **aim** |
|---|---|---|---|---|---|---|---|
| HAR / v1-ON | 6 | 0.5687 | **0.4327** | 0.6092 | 5.38% | 7.76% | 0.70 |
| HAR / OFF | 6 | 0.6136 | **0.4459** | 0.6365 | 4.46% | 7.50% | 0.59 |
| HAR / v2-ON | 6 | 0.5839 | 0.4523 | 0.6008 | 5.64% | 7.91% | 0.72 |
| HAR / v2-control | 6 | 0.4591 | **0.3152** | 0.5147 | 4.39% | 7.76% | 0.58 |
| MNIST / ON | 10 | 0.4001 | **0.1893** | 0.4232 | 13.36% | 29.56% | 0.45 |
| MNIST / OFF | 10 | 0.9145 | **0.8849** | 0.9110 | 26.19% | 30.31% | 0.86 |

## 2. Hypotheses

    H-B1  HAR ON/OFF drift_S ratio  0.970 >= 0.43        PASS
    H-B2  HAR ON in-S energy        0.700x share >= 1.0  FAIL
    H-B3' mean{3,4} 0.4085 >= 2x mean{1,2} 0.1435        PASS (2.85x)
    H-B4  verdict identical under theta_4-defined S      PASS (see 5)

### H-B1 passes by a margin that changes its meaning

The bar was 0.43 (2× MNIST's 0.214). HAR returns **0.970**.

| dataset | ON `drift_S` | OFF `drift_S` | ratio | adapters cut in-S drift by |
|---|---|---|---|---|
| MNIST | 0.1893 | 0.8849 | 0.214 | **4.7×** |
| HAR | 0.4327 | 0.4459 | **0.970** | **1.03× — i.e. not at all** |

H-B1 was written to detect drift that is *less steered* than MNIST's. What it
found is drift that is **not steered**. On HAR the adapter and vanilla arms drift
inside the readout subspace by the same amount to within 1.3%.

**This is not a new mechanism — it is the same finding E5d reached from the other
end.** The adapters never left the identity (travel 0.1042, residual 0.9927), so
the ON arm *is* functionally the OFF arm, and an ON/OFF ratio of ~1.0 is what an
inert adapter must produce. Two independent instruments, one conclusion.

### H-B2 fails, and the direction is the informative part

Predicted: HAR's drift **over**-represented in S (≥1.0× its share), opposite in
sign to MNIST's 0.44× under-representation. Measured: **0.70×** — still
under-represented, **same sign as MNIST**, intermediate in magnitude.

So the drift is *not aimed at* the readout subspace. Note the within-HAR ordering
though: ON 0.70 vs OFF 0.59 — the adapter arm's drift is slightly **more**
S-directed than vanilla's, the opposite of the protective story.

### H-B3′ passes, and the fingerprint is exact

In-S drift increment per task transition (OFF arm, n=3), against row-0 damage:

| transition | in-S increment | row-0 damage |
|---|---|---|
| → task 1 (gain+offset) | 0.1010 ± 0.0519 | 0.010 |
| → task 2 (rotation) | 0.1859 ± 0.0522 | 0.066 |
| → task 3 (permutation) | **0.5631 ± 0.0288** | **0.688** |
| → task 4 (combined) | 0.2538 ± 0.0157 | 0.643 |

Grouped: mean{3,4} = 0.4085 vs 2× mean{1,2} = 0.2869 → **PASS at 2.85×**.

**Spearman ρ = 1.000** (descriptive, as contracted). The increments reproduce the
damage ordering *exactly* — including placing task 3 above task 4, the near-tie
(0.688 vs 0.643) that made the v1 rank test a coin flip. The grouped test was the
right remedy and the descriptive ρ turns out to agree with it emphatically.

**The absorption fingerprint is present.** Where the shift hurts, the encoder
moves inside the readout subspace, proportionally.

## 3. Branch (C) — elevated in-S drift, but not dominant

Disease B **exists** in the sense H-B1 tests (drift entirely unsteered relative to
MNIST) and carries the absorption signature H-B3′ tests. But H-B2 fails: the drift
is not concentrated in S — only 5.4% of drift energy lands in a subspace holding
7.8% of representation energy.

**Sizing both parts, as (C) requires.** In-S drift is real, tracks shift severity
exactly, and is completely unmitigated by adapters — but it is *not* where most of
the drift energy goes. Something else carries part of HAR's forgetting, and this
instrument does not see it. **No cure contract.** E7 is not drafted; a cure aimed
at in-S drift would be targeting a channel we have shown is unsteered but have
*not* shown is dominant.

## 4. v2-control: the cheapest-possible-intervention reference

| arm | drift_S | vs OFF | reader angle |
|---|---|---|---|
| OFF | 0.4459 | — | 50.3° |
| v2-control | **0.3152** | **−29%** | **76.5°** |

Merely **resting the encoder for one epoch per task boundary — with no adapter
recruitment whatsoever — cuts in-S drift by 29%**, more than the adapters achieve
on this dataset (0%). Its reader walks furthest (76.5° vs 50.3°), which is the
trade: during warmup only the readout trains, so absorption is pushed into the
readout instead of the encoder.

That is a three-way accounting of where a shift can be absorbed — adapter,
encoder, readout — and on HAR the adapter's share is approximately zero under
every intervention tried.

## 5. H-B4 passes, but not for the reason the hypothesis assumed

Principal angle between S(θ₀) and S(θ₄):

| arm | angle |
|---|---|
| MNIST / ON | 39.7° ± 0.8 |
| HAR / v1-ON | 48.1° ± 2.6 |
| HAR / OFF | 50.3° ± 4.4 |
| MNIST / OFF | 70.9° ± 3.5 |
| HAR / v2-control | 76.5° ± 4.8 |

The verdict is identical under θ₄-defined S (ratio 0.929 vs 0.970, both far above
0.43), so **H-B4 passes as written**.

**But it passes because H-B1's margin is enormous, not because S is stable.** A
48° rotation of the readout row space is a substantial reader-walk. Had H-B1
landed near its bar, this angle could easily have flipped it. Reported here rather
than absorbed into the drift story: **classifier co-adaptation is real in
magnitude on HAR; it simply cannot rescue a ratio sitting at 0.97.**

Note also that the MNIST ON/OFF angle gap (39.7° vs 70.9°) mirrors its drift_S
gap — adapters stabilise the reader too — while HAR's ON/OFF gap (48.1 vs 50.3)
is nil, again consistent with inert adapters.

## 6. Predictions vs outcome

| hypothesis | predicted | outcome |
|---|---|---|
| H-B1 | ~65% | PASS |
| H-B2 | ~55% | **FAIL** |
| H-B3′ | ~60% | PASS |
| H-B4 | ~75% | PASS (caveated) |
| joint (A) | ~35% | **(C)** |

The pre-run projection was that OFF would land ~0.88 (MNIST-OFF-like, → ratio
0.49) or ~0.55 (→ ratio 0.79). **OFF landed at 0.4459 — below both** — and the
ratio came in at 0.970, **above both**, because ON and OFF converged rather than
OFF rising. The anticipated framing was *steering degraded*; the measurement says
*steering absent*.

## 7. Deviations

None. Per-dataset geometry, the grouped H-B3′, `_apply_adapter` routing with a
shape assertion, and E4b's `ON_DIRS` mapping all as locked in v2. The §0 audit
table was completed before sign-off and drove the 3 added runs.

**Separate correction, logged in `runs/subspace/MEMO.md`:** E4b's memo reported
in-S drift energy 13.90%/25.35% and rep-share 31.28%/31.23%. The recorded artifact
(`runs/subspace/subspace_raw.json`) says **13.36%/26.19%**, and the rep-share
figures correspond to a quantity `subspace_drift.py` never computed. E6's
instrument reproduces the *recorded* values exactly; the memo's table was the
error.
