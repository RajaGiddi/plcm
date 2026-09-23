# E28 — equivariance-training pilot: the measurement

**Date:** 2026-09-22 · **Arms:** B0 (no aug), B1 (aug only), B2 (aug + aux perm loss) at
µ ∈ {p=0.5, p=0.25} · **Seeds:** 42, 1337, 2024 · **Budget:** 30 epochs/task, all arms
**Primary set:** tasks 3 and 4 — the tasks whose frame carries a permutation, fixed by
construction, not chosen after the data.
**Artifacts:** `runs/e28/measure_{arm}_seed{s}.json` (12), `runs/e28_row.json`,
`runs/e28/b0_mustfail.json`, `runs/e28/heldout.json` (fp `ccbeb4f4ecf1`, 20 permutations,
disjoint from the training family's exactly two components — `[0..8]` and
`[2,0,1,5,3,4,7,8,6]`), `runs/e28/regression.json`. Every measurement artifact cites
`ccbeb4f4ecf1`, so the swaps scored are the swaps withheld.

---

## 1. The three numbers that decide the method

Primary set, mean ± t95 over 3 seeds. Each held-out swap is unseen in training by exact
set membership, not by distance.

| arm | Δ_refit | deployed drop under swap, **no repair** | channel identity readable (chance 0.05) |
|---|---|---|---|
| B0 (no aug) | **+0.1806** ± 0.0095 | **+0.3902** ± 0.0333 | 0.637 ± 0.035 |
| B1 (aug only) | **+0.0155** ± 0.0114 | **−0.0302** ± 0.0645 | 0.549 ± 0.052 |
| B2 p=0.5 | **+0.0442** ± 0.0620 | **+0.0372** ± 0.0236 | **0.836** ± 0.048 |
| B2 p=0.25 | **+0.0402** ± 0.0479 | **−0.0121** ± 0.0171 | 0.730 ± 0.053 |

**The must-fail passes and is concentrated exactly where it should be.** B0 loses
38.4pp (task 3) and 39.6pp (task 4) under an unseen swap, against 0.9–20.7pp on tasks
0–2, which carry no permutation in their frame. There was something for the method to fix.

**Against the contract's own decision rule for B2** — *swapped below clean, refit
recovering nearly all of it, and the auxiliary head naming the held-out swaps*:

* swapped below clean — **holds, and only for B2 p=0.5.** It is the one arm whose swap
  drop interval excludes zero (+0.0372 ± 0.0236).
* channel identity readable — **holds** (0.836, 17× chance), and it is the highest of
  any arm.
* refit recovering nearly all of it — **fails.** Δ_refit is what survives the refit, and
  B2's +0.0442 is *above* B1's +0.0155, the arm with no equivariance machinery at all.

**So the aux loss did its stated job and the job did not buy the property.** It preserved
channel identity that plain augmentation erodes (0.836 vs B1's 0.549) and the
representation that retained more identity was *more* expensive to repair, not less.
B2 vs B1 on Δ_refit is not resolved (intervals overlap: ±0.0620 vs ±0.0114), so the
honest statement is that **the method-specific component is indistinguishable from plain
augmentation on the primary quantity, with the point estimate going the wrong way.**

**The augmentation is what worked.** B0 → B1 moves Δ_refit from 0.1806 to 0.0155, an 11×
reduction, and removes a 39pp deployed loss entirely.

---

## 2. Competence, both aggregations

Sec 5 says "within 5pp of B0's, per task" without naming how seeds combine. The
contracted verdict is **per task per seed, as executed**. The per-task mean over seeds is
recorded beside it, as an ambiguity found after the data — not as a verdict.

| arm | per-seed worst (contracted) | worst per-task mean | the mean's own t95 |
|---|---|---|---|
| B0 | +0.000 — **cannot vary** | +0.000 — **cannot vary** | ±0.0000 |
| B1 | −0.235 **FAIL** | −0.151 **FAIL** | ±0.1818 |
| B2 p=0.5 | −0.175 **FAIL** | −0.049 "PASS" | **±0.2995** |
| B2 p=0.25 | −0.199 **FAIL** | −0.077 **FAIL** | ±0.2648 |

**The two readings do not in fact give opposite verdicts.** B2 p=0.5's mean-over-seeds
"PASS" clears the −0.05 bar by **0.0013 against a resolution of ±0.2995** — a margin 230×
finer than the measurement that produced it. By R3's rule (*a gate finer than the
measurement's own resolution is not a gate, it detects noise*) that cell is
**unresolvable at the bar**, not a pass. Both aggregations are consistent with every
augmented arm failing competence. The contracted verdict stands and the alternative
reading does not contradict it.

**B0's row is a statistic that cannot vary** (catch 35(a)): it is the reference
subtracted from itself, so it is 0.000 ± 0.0000 by arithmetic and its "PASS" carries no
information. Recorded, not read.

**Consequence:** every augmented arm fails the contracted competence bar, so §1's method
reading is **descriptive**, per sec 5's own rule. It is a statement about what the
representations do, not a licensed comparison of methods.

---

## 3. Three defects in the measurement, found on reading it

**(a) ‖A−I‖ is scale-contaminated and no cross-arm comparison of it is licensed.**
`Affine.M_raw()` returns `diag(1/sd) M_std diag(sd)` ([d1_fit.py:90](../scripts/d1_fit.py#L90)),
so entry *(i,j)* carries a factor `sd[j]/sd[i]`: the quantity varies with **how
heterogeneous the feature dimensions' scales are**, which is not part of what it is
supposed to test. B2 reads 27.9–98.8 against B0/B1's 1.9–3.0, a 35× gap in a number that
is meant to measure geometry. This is the catch-25 normalizer corollary — *what else does
this vary with, and is that thing part of what I am testing?*
**Nothing rests on it:** the property classifier uses only the swap drop and the swap-ID
probe, and the one prediction citing ‖A−I‖ missed on its *other* conjunct (see §4). The
scale-free replacement is `‖A(Z_s) − Z_s‖ / ‖Z_s‖` on test; it needs a re-run and is not
worth one on its own.

**(b) The equivariance residual δ cannot be read at n=3 on the B2 arms.** B2 p=0.5 gives
0.947 ± 0.981 (per-seed 1.302 / 1.017 / 0.522); B2 p=0.25 gives 1.064 ± 1.899 (0.627 /
0.619 / 1.947). The interval is wider than the mean. A residual above 1 means the affine
map fits worse than predicting zero. Any prediction scored on this quantity is scored on
noise, which is how it should be read in §4.

**(c) The third contracted number was unmeasurable and was substituted.** The contract
asks for "the auxiliary head's accuracy on held-out swaps." **That head is not in any
checkpoint**: `_aux_perm_loss` builds it on the trainer
([trainer.py:926](../src/training/trainer.py#L926)) and checkpoints save
`model.state_dict()`, so the trained head was gone before the measurement ran. Substituted
with a **fresh 20-way linear probe on the deployed feature**, which answers the question
the contract gives for it ("whether channel identity is really in the tensor"), applies to
B0/B1/B2 alike rather than to B2 alone, and does not conflate what the tensor contains
with how well one particular head was fit. **Fixed in the trainer** — the head is now
saved into the era payload, and *only when it exists*, so flag-off checkpoint payloads are
byte-unchanged and the bit-identity that admits the existing S72 OFF checkpoints as B0 is
not spent. Verified both branches, including a save/reload/`load_state_dict` round-trip.

**A fourth observation, catch-24 shaped.** The substituted number shows that **B0 already
carries channel identity at 0.637, 12.7× chance, with no training for it.** "Channel
identity is in the tensor" is not something the aux loss creates; it is the ordinary
state of an un-augmented encoder. What the aux loss does is *prevent augmentation from
eroding it* (0.836 vs B1's 0.549). Had the trivial case been priced first, the third
number would have been specified as a B1↔B2 contrast from the start.

---

## 4. Predictions (sec 7), scored on the primary set

| prediction | odds | outcome |
|---|---|---|
| B0 must-fail passes on the primary set | 90% | **fired** (+0.1806) |
| B2's Δ_refit below B0's by more than floor | 65% | **fired** (0.1806 → 0.0442, floor 0.0620) |
| B2's Δ_refit below 0.02 | 20% | did not fire (+0.0442) |
| B1 swapped within 2pp of clean | 60% | **MISS** (drop −0.0302) |
| B1 passes competence | 65% | **MISS** (worst −0.235) |
| B2 passes competence at some µ | 75% | **MISS** (worst −0.175) |
| B2 shows ‖A−I‖ larger than B1's while δ comparable | 50% | **MISS** (δ 0.466 vs 0.947) |

Scored 7: **2 fired, 4 missed, 1 did not fire.**

Two of the misses need their direction stated, without re-grading either:

* **"B1 swapped within 2pp of clean"** missed because |−0.0302| > 0.02 — but the sign is
  that **swapped beat clean**, which is the opposite of the failure the bar was guarding
  against, and the interval (−0.095, +0.034) contains the whole pass region. The scored
  outcome stands as MISS; the data does not distinguish it from a pass, and it is not
  evidence against invariance.
* **"‖A−I‖ larger while δ comparable"** missed on the δ conjunct, which §3(b) shows is
  unresolvable at n=3. The miss is uninformative rather than a finding.

The competence misses are real and are the substantive ones: augmentation at 30
epochs/task costs more than 5pp of within-task accuracy on this benchmark, at every µ tried.

---

## 5. Reading, and what it means for Phase 3

**On HAR channel permutations, invariance suffices and the method is not needed.** B1
survives an unseen swap with no repair at all (−0.030 ± 0.065, interval containing zero)
and repairs to within 1.6pp when refit. B2 buys a small real swap cost (+0.037 ± 0.024)
and higher identity retention (0.836) and pays for it with a *larger* Δ_refit. There is no
room on this family for equivariance to beat invariance, because the family is small
enough — nine channels, a finite group — that a network can simply learn to ignore it.

**Recommendation for Phase 3: move to rotations, and do not stay on channel permutations.**
The contract's plan clause was "if Δ_refit confirms B2, the build stays on channel
permutations." Δ_refit confirms **augmentation** (B0 → B1, 11×) and does **not** confirm
B2 over B1 — the method-specific component is unresolved with its point estimate on the
wrong side. The condition for staying is not met.
The result also says *why* it was not met, which is the useful part: permutations are a
family invariance can absorb whole. A continuous family cannot be absorbed that way
without destroying the signal the task needs, which is the regime where equivariance has
to be used rather than chosen. **This is a scoping result for paper 2, not a failure:
the pilot was the cheap way to discover that the family, not the method, was wrong — and
it cost one benchmark rather than a paper's worth of rotation infrastructure.**

**Standing consequence adopted:** every contract from here states its aggregation over
seeds explicitly in the clause that names the bar.
