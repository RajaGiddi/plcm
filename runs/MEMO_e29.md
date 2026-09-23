# E29 — recovering an unknown format change from input statistics

**Date:** 2026-09-22 · **Contract:** v3, signed · **Partition:** `1104af185c87` (Modal), the
partition B0's checkpoints were trained on · **All CPU, analysis only, no training.**
**Artifacts:** `runs/e29/{pools,rot_bias,reads,controls,controls_rot}.json`,
`runs/e29/{tolerance,downstream}_seed{42,1337,2024}.json`, `runs/e29_row.json`.
Scripts `e29_common.py` (the single audited implementation), `e29_pools.py`,
`e29_rot_bias.py`, `e29_tolerance.py`, `e29_downstream.py`, `e29_reads.py`, `e29_row.py`.

**Predictions: 9 scored — 5 fired, 2 missed, 2 did not fire — and 1 unmeasurable (§5).**

---

## 0. §0a re-measured on Modal: the laptop reads reproduce

The contract wrote a branch for each way Modal's partition could disagree with the
laptop's. **It agrees on both directions**, so Part A proceeded as written.

| read | laptop (`5d047e4213d1`) | Modal (`1104af185c87`) |
|---|---|---|
| mixed pool, n=16 | 91% | 98.5% |
| mixed pool, n≥64 | 100% | **100%** |
| single-activity, n=256 | 0% (all six) | **0% (all six)** |
| rotation, matched subjects | 0.00° | **0.0003°** |
| rotation, disjoint subjects | 8.72° | 21.33° |

The rotation figure differing was **predicted and is not a disagreement**: the laptop's
8.72° was one pairing on another partition. Part B measures the distribution, and that
is the number reported.

---

## 1. Part A — what the query pool has to contain

n = 256, subject-disjoint. **Aggregation as contracted:** a target recovers if exact in
≥9 of 10 draws; a pool succeeds if ≥20 of 22 targets recover. Pooled rate beside, never
in place.

| pool | succeeds | targets recovering | pooled exact rate |
|---|---|---|---|
| all six activities | **YES** | 22/22 | 1.000 |
| all six, skewed 80/20 | **YES** | 22/22 | 1.000 |
| one static + one dynamic | **YES** | 22/22 | 0.995 |
| two dynamic activities | no | **0/22** | 0.000 |
| three static activities | no | **0/22** | 0.000 |
| contiguous session, stride 1 | no | **0/22** | 0.486 |
| contiguous session, stride 2 | — | **unmeasurable at n=256** (§5) | — |

**The finding is contrast, not count.** A two-activity pool suffices — *if the two
activities contrast*. `static+dynamic` recovers all 22 targets at n=256 while
`two_dynamic` recovers none, both with two activities and the same n. Three static
activities also recover none. What a calibration period needs is not more activities but
**at least one static and one dynamic**: variety in kind, not in count. The per-activity
effective rank says why — the static activities are nearly degenerate (laying 1.62,
sitting 2.73 of 9) while the reference mix reads 6.57.

**Stride 1 is the cleanest vindication of the aggregation rule the contract adopted.**
Its pooled exact rate is **0.486** — which, reported alone, reads as "works about half
the time". Per target, **0 of 22 recover**. A pooled rate hiding total per-cell failure
is exactly catch 35(b), and the rule caught it on its first live use.

---

## 2. Part B — the rotation estimator's transfer bias

Twenty ordered subject pairings, every group against every other.

| n | median | IQR | min | max | max equivariance spread |
|---|---|---|---|---|---|
| 16 | 23.29° | [12.7, 39.0] | 6.96° | 179.01° | 4.8e-06° |
| 64 | 15.18° | [8.7, 22.8] | 4.60° | 176.38° | 4.1e-06° |
| **256** | **12.21°** | [9.3, 21.7] | 3.52° | 33.86° | 2.4e-06° |
| 1024 | 13.36° | [6.8, 21.8] | 5.00° | 28.25° | 1.7e-06° |

**The equivariance identity holds at scale.** §0 proved the objective is independent of
the true rotation; across 20 pairings × 4 sample sizes the measured spread over three
different true rotations never exceeds **4.8e-06 degrees**. The error is a
subject-transfer bias and nothing else.

**It is a bias, not a variance** — it stops falling by n=256 and does not improve at
n=1024. More windows from the same wrong subjects buy nothing.

**Composition hurts rotations worse than it hurts permutations.** A two-dynamic pool at
n=64 and n=256 returns **170°** — not a degraded estimate but the wrong basin, the
spurious near-180° minimum the restart guard exists for.

---

## 3. Tolerance, and one task that cannot carry it

| θ | task 4 accuracy | drop | task 2 accuracy | drop |
|---|---|---|---|---|
| 0° | 0.7740 | — | 0.3878 | — |
| 2° | 0.7717 | +0.0023 | 0.3836 | +0.0041 |
| 5° | 0.7714 | +0.0026 | 0.3610 | +0.0268 |
| 9° | 0.7731 | +0.0009 | 0.3698 | +0.0180 |
| 15° | 0.7721 | +0.0019 | 0.3676 | +0.0202 |
| 30° | 0.7309 | +0.0431 | 0.3417 | +0.0460 |

θ=0 reproduces the clean number exactly on every cell (plumbing check). **Task 4 absorbs
15° for under 0.2pp** and costs 4.3pp at 30°.

**Task 2 is excluded from the reading, and the downstream arm shows why.** B0's recorded
final-row accuracy on task 2 is **0.480**; on the probe subset it reads 0.3878. In the
downstream arm a **random rotation scores 0.4721 against 0.3878 clean** — destroying the
input *beats* it. A tolerance curve measures how much error a working readout absorbs,
and task 2 has no working readout, so its flat curve reports absence of signal as
robustness. Catch 24's shape: price the trivial case first.

---

## 4. Downstream — estimated vs exact vs no repair

| family | clean | no repair | exact | **estimated** | est − exact |
|---|---|---|---|---|---|
| permutations | 0.7673 | 0.3550 | 0.7673 | 0.7340 | −0.0333 |
| rotations | 0.5809 | 0.4303 | 0.5809 | 0.5776 | −0.0033 |

Per task:

| arm | clean | no repair | exact | estimated | detail |
|---|---|---|---|---|---|
| perm, task 3 | 0.7607 | 0.3625 | 0.7607 | 0.6941 | recovered **2/18** |
| perm, task 4 | 0.7740 | 0.3474 | 0.7740 | **0.7740** | recovered **18/18** |
| rot, task 2 | 0.3878 | 0.4721 | 0.3878 | 0.3773 | median err 25.5° |
| rot, task 4 | 0.7740 | 0.3886 | 0.7740 | **0.7779** | median err 18.5° |

**Where the estimate succeeds it is exactly as good as knowing the map** — task 4's
permutation arm recovers all 18 swaps and the estimated re-layout equals the exact one to
four decimals. Permutation recovery is binary, so the downstream number is really "how
often does recovery succeed", and that is a property of the subject pair: task 3 reads
0/6, 1/6, 1/6 across seeds with the true permutation ranked 4th–7th.

**A structural caution about those three seeds.** They share one partition and one
reference/query split, so the seed varies the *model*, not the statistics; only the query
subsample differs. The per-task recovery counts are therefore three draws from **one**
subject pair, not three subject pairs. The pair-to-pair variance lives in §2's twenty
folds, and it is large.

---

## 5. Predictions, and one that could not be scored

| # | prediction | odds | outcome |
|---|---|---|---|
| A1 | two dynamic activities, balanced, succeeds | ~55% | **MISS** (0/22) |
| A2 | one static + one dynamic, balanced, succeeds | ~60% | **fired** (22/22) |
| A3 | three static activities only, succeeds | ~25% | did not fire (0/22) |
| A4 | all six skewed 80/20, succeeds | ~50% | **fired** (22/22) |
| A5 | contiguous session, stride 2, succeeds | ~45% | **unmeasurable** |
| A6 | contiguous session, stride 1, succeeds | ~35% | did not fire (0/22) |
| B1 | median error across pairings at n=256 below 10° | ~55% | **MISS** (12.21°) |
| B2 | error plateaus by n=256 | ~65% | **fired** (12.21 → 13.36, rel 0.09) |
| B3 | deployed accuracy at 9° within 2pp of exact | ~50% | **fired** (+0.0094) |
| B4 | estimated re-layout within 2pp of exact, rotations | ~45% | **fired** (−0.0033) |

**A5 has no measurement and is not scored.** Stride 2 at n=256 needs 512 consecutive
source windows; HAR subjects hold ~300 (subject 4: 317). The arm is unbuildable at the
contracted n **for every subject**, so the isolation the review flag asked for — stride 2
against random isolates time order, stride 1 against stride 2 isolates overlap — is only
available at n ≤ 64. **The CLAUSE → JOB → ARTIFACT table checked that the clause had a
job; nothing checked that the job's cell could exist.** That is a new gap in a check that
has caught four other things: catch 33 audits the *existence* of a job, not the
*feasibility* of the cell it will run. The contiguous arms should have been priced against
the subjects' window counts before the n grid was fixed.

**B2's definition was supplied after the fact.** The contract said "plateaus" without
saying what plateau means; scored here as the n=1024 median within 20% of the n=256
median, stated in the script and printed beside the verdict. Registered ambiguity, not a
chosen-to-fit bar.

---

## 6. Defects found, and one wrong verdict caught

**(a) A control that measured self-consistency, not identifiability.** Part B's isotropic
must-fail first read **0.0° six times out of six** — impossible, since isotropic
covariance identifies no rotation. Cause: reference and query were drawn from the **same
array**, so the query's covariance was the reference's conjugated exactly, and no finite
sample is ever exactly isotropic (diagonal 6.98–7.53, off-diagonal to 0.27 at n=2048).
The estimator locked onto that realization's own sampling noise and recovered the rotation
to 1.9e-07. **The control would have passed an estimator that could not generalise at
all.** Fixed by drawing reference and query independently from one population; it then
reads 6/6 inside the uniform-rotation band.

**(b) A bar repriced after the data, recorded as such.** Fixing (a) reintroduced genuine
finite-sample error, so the positive synthetic's "< 1°" bar — priced for a self-match —
no longer described anything. It was repriced on a principle rather than on the observed
2.86°: the error must *fall with n*, and at the largest n must sit below the 5th
percentile of the uniform-rotation distribution. It reads 3.27° → 2.20° → 1.63°, passing
both.

**(c) The equivariance control is seed-dependent and now scans.** Starved to 1–2 restarts
it fails on only **3 of 8 seeds** (max spread 156.8°); on a single unlucky seed it would
have looked vacuous. It now scans eight seeds and asserts the failure *exists*. At
`n_start=8` the spread is 7.6e-07°.

**(d) A printed verdict that disagreed with the table above it.** The row's first reading
keyed only on `two_dynamic` and printed *"only the full activity mix suffices"* while
`static_plus_dynamic` sat in the same table succeeding 22/22. Catch 22's exact shape,
caught by reading the numbers rather than the verdict. **The contract's readings table
carried the same gap** — it offered "a two-activity pool suffices" or "only the full mix
suffices", treating "two-activity" as one category, when the data says composition within
the pair decides it.

---

## 7. Readings

**A — a two-activity pool of a few hundred windows suffices, but only if the two
contrast.** P5's requirement is not "how many activities" but "at least one static and one
dynamic". A calibration period can be short and passive; it must be varied in kind.

**B — the median transfer bias (12.2°) sits inside the model's tolerance (15° on task 4),
so rotations are recoverable well enough from covariance on a typical subject pair. The
spread is what the median hides:** the upper quartile is 21.7° and the worst pairing
33.9°, both outside tolerance. **P5 extends to rotations for most deployments and fails
for a minority, and P6's equivariant encoder is what would have to absorb that tail.**

**On the storage-honest frontier:** this method stores 9 means, a 9×9 covariance and a
9×9 lag-1 cross-covariance per task — **171 floats, 684 bytes at fp32** — and needs no
labels at repair time. Against the per-task snapshot anchor at ~1MB/task it is three
orders of magnitude cheaper, and where recovery succeeds it reaches the known-map ceiling
exactly.
