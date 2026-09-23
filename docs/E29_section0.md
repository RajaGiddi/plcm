# E29 §0 — filled from the code, before the contract is drafted

**Date:** 2026-09-22. Every number below is measured on this machine, not recalled.
**Standing caveat, applies to every quantity here:** this laptop builds
`partition_fingerprint = 5d047e4213d1`; Modal/x86 builds `1104af185c87` (recorded).
E29 is *entirely* a statement about subject-pool statistics, so a different partition
means different groups, different cross-subject drift, and different numbers. **The
structural findings below are partition-independent; every quantitative one must be
re-measured on x86 before it enters a contract.**

---

## 1. Mechanisms — what the functions do

### The rotation is block-diagonal with ONE shared matrix

[`har_shift.py:channel_affine`](../src/data/har_shift.py) builds
```python
R = np.zeros((9, 9)); r = _rot_z(spec["rot_deg"])
for blk in range(3):                    # acc, gyro, total_acc triples
    R[3*blk:3*blk+3, 3*blk:3*blk+3] = r
```
**The same `r` on every triple.** Part B's conditional ("if the spec rotates every
triple by the same matrix") is **TRUE** — write Part B unconditionally.

Two corrections to how the contract describes it:

* **It is not a physical re-mounting.** [`har_subject.py:126-129`](../src/data/har_subject.py#L126)
  standardizes *first*, then maps: `Z = (X - mu)/sd; Z @ M.T + b/sd`. So `R` is
  orthogonal in **standardized** coordinates, with anisotropic per-channel scaling
  (sd ranges 0.103–0.405) sitting between it and the sensor. The estimation math is
  unaffected — `R` really is in SO(3) in the space E29 works in — but "as a physical
  re-mounting would" is not what this spec does, and the paper should not say it.
* **The spec contains exactly one rotation**, `ROT_DEG = 30.0` about z, on tasks 2 and 4.
  It is not a family. Part B's twenty random SO(3) targets are E29's own construction,
  which is fine, but they are not "the spec's family."

### The permutation family has exactly two elements

`SHIFT_SPEC` puts a `"perm"` key only on tasks 3 and 4, both `PERM = [2,0,1,5,3,4,7,8,6]`.
So "unseen" is exact set membership against `{identity, PERM}` — the same basis E28 used.
Convention: `P[i, perm[i]] = 1`, i.e. output channel `i` takes input channel `perm[i]`,
matching `e28_heldout.apply_perm`. Verified by the search recovering `PERM` exactly.

### The extraction path is clean

[`raw_windows(task_id, train)`](../src/data/har_subject.py#L131) returns the shifted
windows with **no DataLoader and no shuffle**. Catch 32's label-alignment hazard does
not arise here; use this, not a loader comprehension.

### Subject split

[`partition_subjects`](../src/data/har_subject.py#L69): greedy balance by window count,
**deterministic, no seed**. 30 subjects → 5 groups of 6; within each group 5 train and
**1 held-out test subject**. Reference/query subject-disjointness is therefore free —
any two distinct task indices are subject-disjoint by construction.

---

## 2. The finding that rewrites Part B

**Total acceleration is among the nine channels** (`CHANNELS` = body\_acc xyz, body\_gyro
xyz, total\_acc xyz), and gravity sits on **total\_acc\_x at mean +0.8422 g** in raw units.
So the contract's premise "total acceleration is present" holds.

**But the mean E29 would read is zero.** Standardization subtracts it *before* the
rotation is applied, and the calibration constants are computed from task-0's own train
windows ([har_subject.py:109-111](../src/data/har_subject.py#L109)), so on the reference
pool:

```
mu_ref = [0, 0, 0, 0, 0, -0, 0.00011, -0, 0]      ||mu_ref||_2 = 1.14e-04
diag(Sigma_ref) = [1.0002 1.0003 1.0003 1.0003 1.0003 1.0003 1.0001 1.0 1.0]
```

**Consequences, all fatal to Part B as drafted:**

1. **Wahba's problem has no input.** There are no mean directions to align. The entire
   tilt/spin identifiability argument — gravity fixes tilt, leaves a one-dimensional
   spin ambiguity — is **void**. Gravity survives in the *covariance* (its per-window
   direction varies with posture), never in the mean.
2. **The two predictions built on it are unanswerable as written**: "tilt error below 5°
   *if gravity is present*" and "spin error larger than tilt by 3×". There is no
   tilt/spin decomposition to measure.
3. **The spec's rotation is about z, and gravity is on x.** Even had the mean survived,
   the drafted prediction points the wrong way: the ambiguous component under
   gravity-only information is spin *about the gravity axis* (x), and a 30° rotation
   about z is perpendicular to it — the most identifiable case, not the least.

**Measured instead — and Part B works, cleanly:**

| query pool | truth | geodesic error | residual |
|---|---|---|---|
| same subjects (reference pool) | identity | **0.00°** | 0.0000 |
| same subjects | spec 30° z | **0.00°** | 0.0000 |
| same subjects | random SO(3) | **0.00°** | 0.0000 |
| disjoint subjects (task-2 pool) | identity | 8.72° | 0.1487 |
| disjoint subjects | spec 30° z | 8.72° | 0.1487 |
| disjoint subjects | random SO(3) | 8.72° | 0.1487 |

Rotation is **fully identifiable from covariance alone** — no ambiguity at all, because
the three triples share one `R` and the cross-triple blocks pin it.

**The identical 8.72° is a property, not a defect,** and the algebra says why. Writing
`B = R·A`, and using that Frobenius norm is orthogonal-invariant:
```
‖blk(R·A) Σ blk(R·A)ᵀ − blk(R) Σ_Q blk(R)ᵀ‖ = ‖blk(A) Σ blk(A)ᵀ − Σ_Q‖
```
The objective **does not depend on R**. The estimator is exactly equivariant, so its
error is a *fixed subject-transfer bias* independent of the rotation being recovered.
That is the quantity Part B should measure: **how the transfer bias varies across
subject pairs and with n**, not a tilt/spin split that does not exist. (Bias axis here
is `[0.661, 0.715, 0.229]` — not aligned with gravity's `[1,0,0]`.)

*This triggered catch 35's check — five truths giving one number — and survived it.
The contract should state the equivariance identity as the reason the constant is
expected, so the next reader does not re-flag it.*

---

## 3. The finding that rewrites Part A

### Part A as drafted is already solved

Reference = task-0 train subjects; query = task-3 subjects (**disjoint**); targets =
identity, `PERM`, and E28's twenty held-out permutations (`ccbeb4f4ecf1`); full
exhaustive 9! = 362,880 search:

| n | exact recovery | mean channels correct |
|---|---|---|
| 16 | 91% | 8.64/9 |
| **64** | **100%** | **9.00/9** |
| 256 | 100% | 9.00/9 |
| 1024 | 100% | 9.00/9 |

**Three of the seven drafted predictions are already answered**, so they cannot be
registered at the drafted odds:

* "exact recovery of `PERM` at n = 256: ~70%" → **100%, measured**
* "at least eight of nine channels correct at n = 64: ~60%" → **9.00/9, measured**
* "predicted ambiguity set larger than one element: ~50%" → **single element at n ≥ 64**

### A correction I made mid-§0, recorded because the reasoning was wrong

I first measured `S(identity)` under cross-subject drift at **0.14–0.67** against the
reference's self-symmetry margin of **0.0297** (identity 0.000 vs runner-up
gyro\_x↔gyro\_y at 0.0297) and concluded Part A would fail. **That was wrong.** Drift
perturbs *every* candidate's score alike, so it largely cancels in the **ranking**;
the absolute score of the truth rises to ~0.15 and so does every competitor's. Absolute
score was a cheap signal standing in for the expensive one, and running the actual
search settled it in 0.55 s. **Do not put an absolute-score threshold in the contract —
the ambiguity set must be defined on ranking, not on a score bar.**

### Where it actually breaks — and this is the real experiment

**Small n** (subject-disjoint, 22 targets × 5 draws):

| n | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| exact | 4% | 18% | 55% | 89% | 98% |
| channels | 2.53/9 | 4.16/9 | 6.66/9 | 8.41/9 | 8.89/9 |

**Single-activity query pools — total failure at any n:**

| activity | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| exact (n = 256) | 0% | 0% | 0% | 0% | 0% | 0% |
| channels | 3.00/9 | 1.00/9 | 3.00/9 | 4.00/9 | **0.73/9** | 1.59/9 |

A mixed pool recovers perfectly at n = 64; a **single-activity** pool of n = 256 recovers
**nothing** — activity 4 at 0.73/9 is below the ~1/9 a random permutation would score.
HAR channel covariance is dominated by *which activity is being performed*, and a query
pool whose activity mixture differs from the reference's has that mismatch swamp the
permutation signal.

**This is the practically decisive limitation and nobody has measured it.** A fielded
device after a firmware revision sees whatever the user happens to be doing. If the
first 256 windows are all sitting, the estimate is garbage.

**Suggested reframing:** Part A's question is not "does statistical recovery work"
(answered: yes, trivially, in the drafted construction) but **"what does the query pool
have to look like?"** — how many distinct activities, in what proportion, at what n.
That is well-posed, unmeasured, cheap, and is the number a deployment engineer needs.

---

## 4. Provenance and artifacts

**No new `arm_provenance` fields are needed** — E29 trains nothing. The existing `arm`
block on `e10off_ec_seed42` records `benchmark='har_subject'`, `era_checkpoints=True`,
`epochs_per_task=10`, `use_input_adapters=False`, `use_task_heads=False`,
`mafc_lambda=0.0`, plus thread counts. E29's *own* artifacts should record:
`partition_fingerprint`, `spec_fingerprint`, the held-out `fingerprint`, the score
variant, `n`, the draw seed, **and the platform**, since the partition differs between
this laptop and Modal.

**E28 held-out set:** `runs/e28/heldout.json`, fingerprint `ccbeb4f4ecf1`, 20
permutations, disjoint from `{identity, PERM}`.

**B0 checkpoints — catch-20 audit, arm × seed × exists, verified not assumed:**

| arm | seed 42 | seed 1337 | seed 2024 |
|---|---|---|---|
| `ckpt_e10off_ec_*/mafc_seed*_fp32` | 5 era files | 5 era files | 5 era files |

And they **load**, not merely exist: `runs/e28/b0_mustfail.json` holds 15 rows covering
seeds {42, 1337, 2024} × tasks {0,1,2,3,4}, all through `channel_decomp.load`.
*(Local checkout has only seed 42 partially pulled — audit the volume, not the laptop.)*

---

## 5. Cost — measured, not estimated

| item | measured |
|---|---|
| Part A: 9! conjugation precompute | **0.4 s**, 224 MB × 2 |
| Part A: one scored draw (362,880 candidates) | **0.11 s** |
| Part A: full grid, 22 targets × 4 n × 10 draws | **~97 s** |
| Part B: one 8-restart SO(3) fit | **0.14 s** |

The drafted "under an hour" is safe by roughly 20×. Both parts are CPU and laptop-scale;
only the **downstream check** needs Modal, because that is where the checkpoints are.

---

## 6. The count line — NOT quotable yet

`docs/appendix.tex` `\label{app:predictions}` currently holds **68 scored rows: 39 fired,
15 did not fire, 14 misses.** Last entries are E23 and W1.

**E25, E26, E27 and E28 are absent from the table** — four contracts' worth of scored
predictions, including E28's seven from today. Per the standing rule the table is brought
current *before* the quote, not after, so **E29 cannot quote a count line until those
are added.** That is a prerequisite task, not a footnote.

---

## 7. What I could not verify

* **Le Roux et al., 2007** (Part C's spectral-seriation citation) — not checked; Part C
  is deferred anyway, but the contract already marks it "to be verified before citing"
  and that flag should stay.
* **Every number here is on partition `5d047e4213d1`.** The downstream check runs on
  Modal's `1104af185c87`. The structural findings (zero mean, the equivariance identity,
  single-activity failure) do not depend on the partition; **8.72°, the 0.0297 margin,
  and the n-curves do.** Re-measure on x86 before any of them enters a table.
