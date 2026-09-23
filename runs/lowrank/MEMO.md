# Low-Rank Adapters — Result Memo

**Status:** Complete (seed 42, rank 32 + rank 64 + same-seed full-rank
reference). **Pre-registered branch (c) fired for both ranks.**

---

## 1. Verdict — branch (c)

| arm | AVG | RET | forgetting | storage | vs full-rank |
|---|---|---|---|---|---|
| full-rank (ref, seed 42) | 0.9470 | 0.9395 | 0.0383 | 12.29 MB | — |
| rank 64 (n=3) | **0.8719 ± 0.0250** | 0.8464 | 0.1279 | 2.01 MB | **−7.51pp** |
| rank 32 (n=3) | **0.6877 ± 0.0121** | 0.6161 | 0.3589 | 1.00 MB | **−25.93pp** |

Branch (c) as pre-registered: *"full-rank stands, the 10× cost stays owned in
print, low-rank returns to future work with a measured reason instead of a
hope."* The measured reason is §3.

## 2. NEW LIMITATION — the ER comparison is storage-contingent

This is the consequential finding, and it revises a headline claim.

| method | MB | AVG | n |
|---|---|---|---|
| adapters r32 | 1.00 | 0.6877 ± 0.0121 | 3 |
| ER 100/task | 1.57 | 0.7916 ± 0.0051 | 3 |
| **adapters r64** | **2.01** | **0.8719 ± 0.0250** | 3 |
| ER 200/task | 3.14 | 0.8592 | 1 |
| ER 500/task | 7.84 | 0.9002 | 1 |
| adapters full-rank | 12.29 | 0.9470 | 1 |

**The frontier crosses TWICE.** ER wins at the smallest budget (+10.4pp over
r32); **adapters win at ~2 MB** (r64 beats ER-200 by 1.3pp on 36% LESS storage);
full-rank wins at the top (+4.7pp over ER-500 on 57% more). The honest claim is
regime-dependent, not a blanket concession.

**Claim language, FINAL** (supersedes the "replay wins at matched storage"
draft, which was written before the r64 backfill and the ER sweep):

> *"Adapters are competitive-to-better above ~2 MB and dominated below it, while
> retaining no raw data at any point."*

*Superseded draft:* "…adapters outperform replay by 14.2pp … at matched storage
budgets, replay is the stronger method." Too pessimistic in one direction and
based on a single crossing; the measured frontier crosses twice.

The privacy/regime claim survives intact; the *efficiency* framing is dead, and
so is any unqualified "adapters beat replay."

## 3. Why low-rank fails: capacity, not initialization

Effective rank of the learned deviation `A_k − I` (full-rank adapters):

| task | ‖A−I‖_F | rank @90% energy | rank @99% | top-32 energy | top-64 energy |
|---|---|---|---|---|---|
| 0 | 47.68 | 40 | 144 | 0.864 | 0.951 |
| 1 | 36.82 | 56 | 197 | 0.802 | 0.919 |
| 2 | 29.59 | 65 | 220 | 0.771 | 0.898 |
| 3 | 26.29 | 70 | 229 | 0.761 | 0.889 |
| 4 | 24.36 | 74 | 234 | 0.748 | 0.880 |

The deviation is genuinely high-rank: 56–74 components for 90% of the energy,
~200+ for 99%. Rank 32 captures only 75–80%. Because initialization was verified
**exact** identity (`max|A(x)−x| = 0` at step 0), the drop is attributable to
capacity, not to a poor starting point — which is what makes branch (c)
interpretable rather than confounded.

## 4. The adapters are not approximating a target transform

Relative reconstruction error, learned low-rank vs same-seed full-rank adapter:

| task | rank 32 | rank 64 |
|---|---|---|
| 1 | 0.9331 | 1.2204 |
| 2 | 0.8782 | 1.1437 |
| 3 | 0.8231 | 1.1049 |
| 4 | 0.8158 | 1.1007 |

**Rank 64 is FURTHER from the full-rank solution than rank 32 (1.10 vs 0.82),
while scoring 13pp higher.** Fidelity to the full-rank solution and task
performance are not merely uncorrelated — they run in opposite directions here.
There is no single target transform being approximated at varying fidelity; each
rank converges to its own serviceable solution.

**This corroborates the Rotated MNIST invariance result from an independent
direction.** Rotated showed the mechanism works where exact inversion is
*impossible*; low-rank shows different-capacity variants find *unrelated*
solutions of differing quality. Both refute the motivating theory ("the adapter
represents `P_k⁻¹`"). Rank buys **expressive capacity**, not fidelity to any
particular map.

## 5. Scope

**Backfilled to n=3** (the frontier became a figure, which triggered the
pre-registered condition). Branch (c) is unaffected: r32 −25.9pp, r64 −7.5pp,
both far beyond the ±1-2.5pp seed spreads.

Seeding moved both points, and **r64 moved UP +3.3pp (0.8388 -> 0.8719) — the
first PESSIMISTIC n=1 in the project** (all four prior instances were
optimistic). r32 moved down −1.7pp. The n=1 rule protects in both directions.
