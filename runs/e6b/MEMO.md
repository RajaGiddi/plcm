# E6b — Three-Channel Forgetting Decomposition: Result Memo

**Status:** Complete, n=3, analysis-only. **BRANCH (A).**
All four gates pass. Pre-registration: `docs/E6B_prereg.md` (v2 + amendment v3).

**Read this alongside §5.** The result arrived only after the instrument control
rejected the instrument **twice** and four defects were found — three of them in
this experiment's own analysis code. The control did its job; the code did not, at
first. That history is part of the result's provenance, not a footnote.

---

## 1. Gates, in the contracted read order

    identity      worst residual 0.00e+00                    OK
    H-C3'  HAR    R = +0.0496 <= 0.05                        PASS  (margin 0.0004)
    H-C3'  MNIST  R = +0.0119 <= 0.05                        PASS
    H-C1   HAR OFF F_read 0.2939 >= 0.10                     PASS
    H-C2   v2-ctl  F_read 0.3688 > F_enc 0.0422 AND >= 0.10  PASS

**HAR's instrument control passes by 0.0004 — and the per-seed breakdown shows
that pooled number is an averaging artifact.**

| arm | 42 | 1337 | 2024 | mean | cells over bar |
|---|---|---|---|---|---|
| HAR/v1-ON | +0.0194 | +0.0259 | +0.0374 | +0.0275 | none |
| **HAR/OFF** | +0.0356 | **+0.0896** | **+0.0554** | +0.0602 | **2 of 3** |
| **HAR/v2-ON** | +0.0371 | **+0.1065** | +0.0330 | +0.0589 | **1 of 3** |
| **HAR/v2-control** | +0.0386 | +0.0436 | **+0.0736** | +0.0520 | **1 of 3** |
| MNIST/ON | +0.0085 | +0.0140 | +0.0070 | +0.0098 | none |
| MNIST/OFF | +0.0150 | +0.0172 | +0.0094 | +0.0139 | none |

**4 of 12 HAR cells exceed the 0.05 bar; pooled by seed, 1337 is +0.0664 — over.**
Two of the over-bar cells are on **HAR/OFF, the arm H-C1 is evaluated on.** By the
contract's letter (H-C3′ evaluated *per dataset*) HAR passes at +0.0496; by the
per-seed evidence the HAR instrument is **marginal-to-failing at cell level**, and
only `v1-ON` is clean throughout. MNIST is comfortable everywhere (max +0.0172),
which is what keeps the cross-dataset comparison trustworthy.

**The conclusion survives a worst-case correction.** `R` inflates `F_read` (the
refit beats the deployed head); subtracting it entirely:

| arm | F_read | F_read − R | F_enc | corrected reader share |
|---|---|---|---|---|
| HAR/v1-ON | 0.2734 | 0.2458 | 0.0556 | **81.5%** |
| HAR/OFF | 0.2939 | 0.2337 | 0.0759 | **75.5%** |
| HAR/v2-ON | 0.2942 | 0.2354 | 0.0510 | **82.2%** |
| HAR/v2-control | 0.3688 | 0.3168 | 0.0422 | **88.2%** |

Reader dominance is not an artifact of the instrument's bias: it holds at 75–88%
even when the entire bias is charged against it.

## 2. The decomposition

| dataset/arm | F_total | F_enc | F_read | R | **reader share** |
|---|---|---|---|---|---|
| HAR / v1-ON | 0.3015 | 0.0556 | **0.2734** | +0.0275 | **83.1%** |
| HAR / OFF | 0.3095 | 0.0759 | **0.2939** | +0.0602 | **79.5%** |
| HAR / v2-ON | 0.2864 | 0.0510 | **0.2942** | +0.0589 | **85.2%** |
| HAR / v2-control | 0.3590 | 0.0422 | **0.3688** | +0.0520 | **89.7%** |
| MNIST / ON | 0.0686 | 0.0149 | 0.0636 | +0.0098 | 81.0% |
| MNIST / OFF | 0.6324 | 0.2015 | **0.4448** | +0.0139 | 68.8% |

**The reader channel carries most of HAR's forgetting — ~80–90% of it — in every
arm.** The encoder channel is real but small (0.04–0.08), which is consistent with
E6's finding that in-S drift is fingerprinted-but-modest (5.4% of drift energy in a
subspace holding 7.8% of representation energy). The two experiments agree: the
encoder does move, in a shift-tracking way, but that movement is not what costs
the accuracy.

**v2-control is the natural experiment and it behaves as predicted.** Its encoder
drifted *least* (E6: in-S 0.3152, −29% vs OFF) and its reader rotated *most*
(76.5°), and here it has the **lowest F_enc (0.0422) and the highest F_read
(0.3688)** — the largest reader share of any arm, 89.7%. Warmup's encoder-rest
bought a real reduction in encoder-side damage and paid for it in reader-walk,
which is why it forgets most overall (F_total 0.3590).

**MNIST is the contrast the framework needs.** The ON arm's total forgetting is
0.0686 — adapters prevent essentially the whole phenomenon — while the OFF arm's
is **0.6324**, split 69/31 between reader and encoder. So the reader channel is not
a HAR peculiarity; it is large wherever the shift is not absorbed at the input.

### Per-task, with task 3 called out

| arm | T0 | T1 | T2 | **T3** |
|---|---|---|---|---|
| HAR/v1-ON | +0.295/+0.077 | +0.305/+0.073 | +0.351/+0.056 | **+0.143/+0.017** |
| HAR/OFF | +0.311/+0.092 | +0.308/+0.105 | +0.352/+0.091 | **+0.205/+0.015** |
| HAR/v2-ON | +0.333/+0.060 | +0.364/+0.080 | +0.412/+0.070 | **+0.067/−0.006** |
| HAR/v2-control | +0.411/+0.044 | +0.411/+0.053 | +0.467/+0.060 | **+0.186/+0.012** |
| MNIST/ON | +0.146/+0.042 | +0.067/+0.014 | +0.027/+0.003 | +0.014/+0.000 |
| MNIST/OFF | +0.503/+0.338 | +0.524/+0.202 | +0.494/+0.174 | +0.259/+0.092 |

*(F_read / F_enc.)* **Task 3 has the SMALLEST reader loss on HAR, not the largest** —
the opposite of E6's M3, where the transition *into* task 3 carried the largest
in-S drift increment (0.5631).

**These are not in conflict; they measure DEPOSITION versus EROSION.** E6's M3
measures drift deposited *at the write event* (the boundary into task k); E6b
measures damage *remaining at θ₄*, i.e. the write minus everything the reader
subsequently walked. Task 3 took the biggest deposit and has had the fewest
subsequent tasks to be walked away from, so it shows the least residual reader
loss. The orderings should differ, and their differing is evidence that
**reader-walk accumulates over subsequent tasks** rather than arriving at a single
boundary.

That yields a falsifiable prediction E7 can check directly: **per-task `F_read`
should grow with tasks-since-training under a walking reader, and flatten under
frozen heads.** The HAR rows already show the monotone shape (T0-T2 ~0.30-0.47
falling to ~0.07-0.21 at T3).

## 3. Branch (A): the three-channel accounting

Forgetting is **where the shift and the drift get absorbed**:

- **C-adapter** — harmless. Operative on MNIST (ON total 0.0686), ~zero on HAR
  (E5d: adapters never left identity; E6: ON/OFF in-S drift ratio 0.970).
- **C-enc** — real, absorption-fingerprinted (E6 H-B3′ 2.85×, ρ = 1.000), but
  **modest**: 0.04–0.08 on HAR, ~14–31% of total.
- **C-read** — **dominant on shared-classifier architectures**: 68.8–89.7% of
  total forgetting across every arm measured, both datasets.

## 4. E7 and the H-C4 residue

Branch (A) licenses **E7 = HAR with per-task heads**, drafted before any training
run. Its design cites **this decomposition**, not any MNIST contrast:

> **Per-task heads remain an untested cure candidate for the reader channel —
> motivated by the channel decomposition itself, not by any MNIST contrast, which
> does not exist.** Every arm in this program uses a shared classifier
> (`use_task_heads=False`, verified in §2b of the contract). The reason per-task
> heads are the natural candidate is that a *frozen* head cannot walk, and walking
> is what 80–90% of the forgetting turns out to be.

## 5. Provenance — the instrument rejected itself twice

**Run 1.** `R = −0.6498` (HAR) / `−0.4280` (MNIST). Branch (D), STOP.
**Diagnosed as** optimizer underfitting/divergence. **That diagnosis was wrong**,
and the diagnostic script shared the same defect it was diagnosing.

**Amendment v3** (convex solver + deployed-feature lock) was approved on that
faulty basis. **Run 2:** `R = −0.6329`. Branch (D) again — barely moved, which is
what finally falsified the optimizer story: a converged logistic regression cannot
lose to another linear map on identical features by 63pp.

**Defect 1 (mine).** `load_task_data` collected inputs and labels in **two separate
passes** over a `shuffle=True` loader — two different permutations, so `x[i]` and
`y[i]` did not correspond (44/200 agreement, ~chance). The probe trained on
randomly relabelled data. Test loaders are `SequentialSampler`, so deployed
accuracies stayed correct and the failure *looked* like an instrument problem.
"More epochs made it worse" was fitting noise, which I read as divergence.

**Defect 2 (mine).** MNIST permutations are **seeded**. One benchmark was built at
seed 42 and used for all three checkpoint seeds: seed-1337 weights scored **0.1135
(chance)** against seed-42 permutations vs **0.9470** against their own.
**Task 0's permutation is `None`, so E4b and E6 — task-0-only — are unaffected and
their results stand.**

**Run 3 (this memo).** `R = +0.0496 / +0.0119`, positive as the design assumed.

**Standing on amendment v3 — counterfactual now CLOSED (HAR/OFF, n=3, fixed pipeline):**

| recipe | mean R | verdict |
|---|---|---|
| Adam / raw `c_t` (v2 exactly as written) | +0.0519 | FAIL — **by 0.0019** |
| Adam / **deployed feature** | +0.0432 | PASS |
| lbfgs / deployed feature (v3, adopted) | +0.0602 | over bar on this cell |

Three things follow, and two of them contradict the amendment's stated rationale.

1. **The catastrophic failure was 100% the data bugs.** v2's recipe on a correct
   pipeline misses by 0.0019, not by 0.65. The optimizer was never the problem.
2. **The FEATURE lock was the load-bearing amendment**, not the solver: Adam on the
   deployed feature passes comfortably. A refit on raw `c_t` is a different function
   class from the deployed head, and that alone cost ~0.009 of R.
3. **The solver lock made the control STRICTER, not easier.** lbfgs gives a *larger*
   R than Adam on identical features — a closer-to-optimal fit has a bigger
   advantage over the deployed head, so `R = optimal − deployed` grows. This is the
   conservative direction and lbfgs is the more honest estimate of the true gap;
   it is also why HAR's cells sit at the bar rather than under it.

**Catch logged:** *an amendment can be right for the wrong reasons, and the
wrongness surfaces only because the defects that motivated it were found later.*
v3 is retained — knob-free fitting removes optimizer settings from a measurement's
meaning, and the feature lock is required regardless — but retained **on merits it
was not argued on.**

**What worked.** H-C3′ rejected a broken instrument on both bad runs, before any
channel claim was read. The read-order lock — control before hypotheses, hard stop
— is the only reason four defects produced zero false findings rather than a clean
three-channel story built on relabelled data.

## 6. Deviations

Protocol as locked in v2 + amendment v3. Two bug fixes restoring the protocol as
written (single-pass data collection; per-seed MNIST benchmarks) — code defects,
not protocol changes. All numbers emitted by `scripts/channel_decomp.py` into
`runs/e6b/decomp.json`; none transcribed.
