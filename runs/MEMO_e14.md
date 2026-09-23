# MEMO — E14: ResNet-50 on Split-CIFAR-100

**Contract:** `docs/E14_resnet_prereg.md` (signed before any training).
**Question:** is reader-dominance transformer-specific?
**Answer:** no. **Branch (A).**

**Status of every number below:** produced by a named script, run in the
execution environment, quoted from its recorded output. All preconditions
CLEARED. One of them — **P2(b)** — was discovered to have no launcher job at all
*after* the headline runs had landed; it was built, run, and passed before this
memo closed, and §10.1 records how it was missed (catch 33).

---

## 1. The result, in one table

| quantity | ResNet-50 | ViT-B/16 (E12) | scratch-trained (LSTM/MLP) |
|---|---|---|---|
| reader share | **98.81% [98.25, 99.38]** | 91.77% [90.88, 92.67] | 66–88% |
| `F_enc` | **0.0076 [0.0042, 0.0110]** | 0.0599 | 0.052–0.112 |
| `F_read` | **0.6428 [0.6080, 0.6777]** | 0.6723 | — |
| `F_total` | 0.6514 [0.6164, 0.6863] | 0.7154 | — |
| instrument `R` | **−0.0009 [−0.0023, +0.0004]** | +0.0168 | — |
| cells | 57 (3 seeds × 19 old tasks) | 95 (5 × 19) | — |

`runs/e14_hv1.json`, computed by `scripts/e12_hv1.py` over `runs/e14_decomp/`.

**The one claim E14 buys:** reader-dominance is **not transformer-specific among
pretrained backbones**. Nothing more — §9 states what it deliberately does not buy.

---

## 2. H-R1 (PRIMARY) — MET

Bar: reader share ≥ 0.50 pooled, CI lower bound above 0.50.
Measured: **98.81%, CI lower bound 98.25%.** A 48-point margin.

The refit is what carries it. Across all 57 old-task cells the deployed head
scores **0.2929** while a probe refit on the *same θ_T features* scores
**0.9358** — against a ceiling of 0.9434 and chance of 0.2000. Task 0 across
seeds: ceiling **0.9447** → deployed **0.2807** → refit **0.9313**.

Read plainly: **after 20 tasks of full fine-tuning, the ResNet's features still
support 93.6% accuracy on tasks it appears to have forgotten.** What is lost is
the mapping from those features to labels, not the features.

**Instrument gate:** pooled `R` = −0.0009 against the ±0.05 bar, **0 of 57 cells
over**, worst cell −0.0180 (seed 2024, T9). The identity
`F_enc + F_read − R = F_total` holds per cell with max residual **0.0e+00**.

---

## 3. H-R2 (descriptive, no gate) — the arc continues, and steepens

66–88% (scratch-trained small) → 91.8% (pretrained ViT) → **98.8% (pretrained
ResNet)**.

**Register — this is an ordering, not a measured gap.** The 7pp ResNet-vs-ViT
difference is between *pooled* quantities from two experiments that differ in
architecture, parameter count, and optimizer trajectory. E14's per-cell
reproduction floor is **10.8pp** (§6), so no per-cell version of this comparison
is licensed and none is offered. What survives the floor is the ordering itself,
because each arm's pooled share carries a CI under 1.2pp wide.

The prediction on record for share ≥ 0.85 was **~45%**. It fired.

---

## 4. H-R3 (descriptive) — F_enc is the surprise

**0.0076** — **7.9× smaller** than the ViT's 0.0599 and an order of magnitude
below the LSTM/MLP range of 0.052–0.112. The lowest encoder-side damage measured
anywhere in this program.

The contract's branch (C) anticipated the opposite ("F_enc materially *larger*
than ViT's 0.060 → the degree of reader-dominance is architecture-dependent even
where its sign is not"). It came in materially *smaller*, and the contract
provided no mirrored branch for that direction. The asymmetry is noted rather
than quietly resolved: branch (C)'s sentence is true as written with the
direction reversed — degree is architecture-dependent, and the ResNet is the
more extreme case, not the weaker one.

**The general lesson: a one-sided conditional is a bet that the surprise will
come from one direction.** Ours came from the other. The gap is a real drafting
error — conditionals should be two-sided whenever the quantity can move either
way — but it cost nothing here for a specific reason: *because the branch was
written explicitly, the asymmetry became visible instead of being absorbed.* An
unwritten expectation would have quietly accommodated either outcome.

---

## 5. H-R4 (descriptive) — span rotation, now on two backbone families

**Era control first**, as ruled. Projecting a task's θ_T features onto the span
of its own era prototypes costs, *at the era checkpoint*, a mean of **+0.0001**
(max +0.0100) against the 0.05 readability bar. Rank **5 of 2048** loses
essentially nothing, so the span statement below is readable rather than an
artifact of a lossy projection.

At θ_T that same projection recovers **+4.0%** of the reader gap (0.3185 against
a 0.2929 deployed and a 0.9358 refit). E15's ViT measured **+0.5%**.

**Feature stability does not confer span stability.** This now holds on a second
architecture, and the ResNet is the sharper instance: its features moved *least
of anything measured* (`F_enc` 0.0076) while its era-prototype span rotated just
as thoroughly out from under the deployed reader.

---

## 6. The reproduction floor — measured before the claims were written

`runs/e14floor_base_seed42_rep2` vs `runs/e14_base_seed42`, identical config
relaunched:

| | |
|---|---|
| bit-identical cells | **5 / 210** |
| mean \|Δ\| | 0.02531 |
| **max \|Δ\| — the floor** | **0.10800** |
| headline AVG delta | 0.00267 |

**This is the least reproducible arm in the program** — E5's seed 42 was
bit-identical in *both* arms; here 205 of 210 cells moved.

**What it forbids:** any per-cell ResNet comparison, and any cross-architecture
delta stated at cell granularity. **What survives:** everything in §§2–5, all of
which are pooled over 57 cells with CIs, and pooling is exactly what collapses a
10.8pp cell floor into a 0.27pp headline.

**fp16 clause — closed.** The era-checkpoint reload delta is **max 0.00400, mean
0.00067**, i.e. **27× below the floor**, so boundary-time values remain
authoritative and the delta is not a reportable quantity here. Worth recording
that E12's ViT measured exactly 0.00000: the contract called that "the
expectation, not the assumption," and E14 is why that wording was right.

**Ordering note.** E14 is the first arm whose floor was measured *before* its
claims were written rather than after. That ordering is why §§2–5 are
pooled-only. Had the floor landed after the memo, catch 19's asymmetry would have
applied and the honest response would have been a rewrite.

---

## 7. Preconditions and instrument

| # | clause | value | status |
|---|---|---|---|
| **P0** | build gate C1–C4 | 4/4 GREEN, `runs/e14_build_gate.json` | PASS |
| **P1** | frozen DIAG ≥ 0.85 | mean 0.9552, **min 0.8600**, n=60 | PASS |
| **P1** | trainable within 5pp of frozen | 0.9446 vs 0.9552, gap **+0.0106** | PASS |
| **P2(a)** | task-0 drop ≥ 0.15 | **0.6620** (0.788 / 0.684 / 0.514) | PASS |
| **P2(b)** | repeat-task control ≤ 0.05 | **0.0010** (0.0030 / 0.0000 / 0.0000) | PASS |
| **P3a** | `head(feature) == logits`, max \|Δ\| = 0 | **0.0e+00**, every cell | PASS |
| **P3b** | module identity = head *k* | `head[k]`, by `is` | PASS |
| **P3 controls** | both must FAIL | **both FIRED** (P3a Δ 2.435e+01) | PASS |
| **P4** | sequential loader, label alignment | assert printed, all cells | PASS |

P1/P2 computed by `scripts/e12_preconditions.py` → `runs/e14_preconditions.json`,
which prints `P1 PASS · P2(a) PASS · P2(b) PASS — PRECONDITIONS: CLEARED`.

**P2(b) is what licenses the word "forgetting."** Five tasks of *identical* data
with fresh heads forget **0.0010** — two seeds at exactly 0.0000. So the 0.6620
task-0 drop in P2(a) is caused by the sequence's *content*, not by the trunk
merely continuing to optimize. Without this clause, P2(a) alone could not
distinguish the two, and the entire decomposition would be decomposing an
artifact of continued training.

**Build gate detail** (`runs/e14_build_gate.json`, run on the execution GPU):
C1 `resnet50.a1_in1k`, feature (2, 2048), trunk grads flow, amendments
default-off. C2 timm parity max\|d\| **0.0e+00**, live P3a **0.0e+00**, and the
pre-pool positive control (spatial MAX — same layer, same shape, wrong
reduction) delta **2.457e-01 → FIRED**. C3 both controls fired. C4 timm 1.0.20,
25.6M params, **weight hash 4aa4ff84c7e0**.

**The controls are the point.** P3a and P3b were re-derived for avgpool→fc rather
than inherited from the ViT's CLS-post-norm, per the contract's preamble — a rule
learned on a ViT is a hypothesis about a ResNet. Both fired on their known-failure
cases before first live use.

**Deviation, C1.** The contract specifies ResNet weights *baked at image build*;
the image bakes only the ViT, so the ResNet weights are fetched at container
start. The **verification** C1 names — hash recorded and asserted — is satisfied
(C4, hash `4aa4ff84c7e0`), and a substituted upstream weight would change the
hash and be caught. The mechanism is weaker than contracted; the check is not.

**Ruled: record, do not rebuild.** A runtime fetch under a hash assert is
*provenance-equivalent* to a baked weight under a hash assert, because the assert
is what certifies identity in either case. What the contracted mechanism added
was defense-in-depth — no network at run time — so losing it costs robustness,
not correctness. Re-running for a strictly-stronger-but-equivalent guarantee
would spend a day for zero epistemic gain.

---

## 8. Catch 24 — the trivial use of the resource, priced

**Resource the setting assumes:** an ImageNet-pretrained ResNet-50.
**Dumbest possible use:** freeze it, train one linear head per task.

| | accuracy | forgetting |
|---|---|---|
| frozen trunk + per-task linear heads | **0.9552** | **0, by construction** |
| full fine-tuning, 20 tasks (what E14 diagnoses) | **0.3257** | 0.6514 |

**The trivial baseline beats the diagnosed setting by 62.95pp with zero
forgetting.** This does not touch H-R1 — E14 is a diagnosis of *why* fine-tuning
forgets, and that question is well-posed regardless of whether one should
fine-tune. But it fixes the register: **no deployment recommendation follows from
E14**, and any paper text implying one is wrong. Same structure as E9's ρ = 1.237
snapshot result, and the frozen arm was contracted partly for this purpose.

Stated positively: on a pretrained backbone at this scale, the interesting
question is not how to repair fine-tuning but why anyone fine-tunes — and the
reader-dominance result is a partial answer, since what fine-tuning destroys is
recoverable by refitting a linear map.

**For the discussion section, one sentence:** the ViT and ResNet sections now
carry the same shape — a diagnosis of a regime the benchmark itself does not
reward — and we priced the frozen baseline on *both* backbones before reporting
either. A reviewer who notices the frozen baseline will find we noticed first,
twice.

---

## 9. Limitations — carried forward from the contract, unchanged

**E14 does not separate pretraining from scale from architecture.** It varies
architecture family with pretraining held fixed, so it supports *"reader-dominance
is general among pretrained backbones"* and **not** *"pretraining causes
reader-dominance."* The monotone arc in §3 is *ordered along* the pretraining
axis but not *controlled* on it. The clean answer is a factorial (scratch-ViT,
scratch-at-scale, pretrained-small) and it belongs to **paper 2's opening**.

**Task-IL only.** Class-IL was not re-run; E12 established it as
corroborative-not-extending.

**No adapt arm.** H-V2 was untestable at E12's placement and E14 did not reopen it.

---

## 10. PROCESS APPENDIX

### 10.1 Catch 33 — a contracted precondition with no job defined

**P2(b) had no launcher entry.** The contract states P2 as two clauses — task-0
drop ≥ 0.15 **plus** a repeat-task control ≤ 0.05. Only clause (a) had a job.
E12 has `jobs_e12rptb` and reported 0.0040; E14 had no `jobs_e14rpt` at all, and
P2 would have entered this memo marked **PASS on half its definition**.

**Why the existing rules did not reach it.** Catch 20 audits *artifacts* ("do the
checkpoints this experiment needs exist?"). Catch 21 audits *premises* ("does the
contract's claim about arm X cite a field?"). Catch 30 audits *arm identity at
train time*. The build gate audits *the build*. **Nothing audited the mapping from
contract clauses to launcher jobs** — whether every precondition the contract
names has something that produces it.

**Why it was invisible:** P2(a) passed at **4.4× its bar**, so the P2 row read as
emphatically green. Catch 30's lesson in a new place — *the verdict's quality
tells you nothing about the premise's* — except here the missing half was not
misconfigured, it was absent, and absence produces no output to look wrong.

**RULE:** every contract ships a **clause → job → artifact** table, verified
before the headline runs launch, not after. A precondition with no job is not a
precondition; it is a sentence. The audit that catches this is mechanical — walk
§2 of the contract, name the launcher entry for each clause, and require a path.

**What it cost:** ~15 GPU-minutes, because it was caught before the memo landed
rather than after the ledger did. Had the memo shipped first, the correction
would have been a ledger retraction.

### 10.2 What went right

- The **positive controls fired** on a fresh architecture, exactly as the
  preamble demanded, and the pre-pool control (2.457e-01) is a real tensor from
  the real forward rather than noise — it fails only because the capture point
  genuinely matters.
- **P3a read 0.0e+00 on the first live attempt**, meaning the catch-28 correction
  transferred to a fourth backbone without rework.
- The **blast-radius regression passed at HEAD**, bit-identical to E11's matrices
  after four experiments of amendments — the E11 proof was not spent.
- The **floor was measured before the claims were written** (§6).

### 10.3 Calibration

| prediction on record | outcome |
|---|---|
| H-R1 MET, **~75%** | **fired** |
| share ≥ 0.85, **~45%** | **fired** — 98.81% |
| `F_enc` ≤ 0.15, **~70%** | **fired**, by 20× |

Three for three, and all three were *under*-confident. The contract's calibration
note said this program's magnitude predictions "run directionally right and too
narrow"; here they ran directionally right and too *timid*. The stated reason for
doubt — "convnet features are more spatially local and less linearly separable at
the penultimate layer" — was the right thing to worry about and empirically
backwards: global average pooling produces a *more* linearly separable 2048-d
vector than the ViT's CLS token, not less.

**This is the most useful kind of calibration entry, because the miss is in the
model rather than in the number.** It also generates a hypothesis: if linear
separability of the penultimate representation is what sets the ceiling on
`acc_refit`, then the arc in §3 may be ordered by **separability**, not by
recency of architecture — which would predict that a *less* separable pretrained
backbone lands lower regardless of how modern it is. **Speculative, flagged as
such, and a clean paper-2 hypothesis** — it is testable with the refit
instrument already built.

---

## 11. LEDGER EDIT — drafted here, applied second

**One new row:**

| claim | evidence | status |
|---|---|---|
| **98.8% [98.3, 99.4] on a pretrained ResNet-50** (task-IL, 20 tasks) — reader-dominance is **not transformer-specific** | E14 H-R1, `runs/e14_hv1.json`, 3 seeds × 19 old tasks = 57 cells; `F_enc` **0.0076**, `F_read` 0.6428; instrument R −0.0009, 0/57 cells over bar; identity residual 0.0e+00; P3a 0.0e+00, both positive controls fired; build gate 4/4 (`runs/e14_build_gate.json`) | **new** — a second pretrained family, and a convnet. The claim spans **four architectures** (LSTM, MLP, ViT, ResNet) and two pretraining regimes. |

**Two qualifying rows:**

| claim | evidence | status |
|---|---|---|
| feature stability does not confer span stability — **second architecture** | E14 H-R4: era control +0.0001 (max +0.0100, bar 0.05, rank 5/2048); θ_T span projection recovers **+4.0%** of the reader gap vs E15's ViT +0.5% | **extended** — measured on the arm with the *smallest* `F_enc` in the program, which is the sharper case |
| E14 carries **no deployment recommendation** | frozen trunk + per-task linear heads: **0.9552**, zero forgetting by construction, vs **0.3257** for the fine-tuned trunk it diagnoses — the trivial use of the resource wins by **62.95pp** | **scope lock** (catch 24) — the diagnosis is well-posed; the setting is not one to recommend |

**One-liner amendment** — replace *"and 91.8% [90.9, 92.7] on a pretrained
ViT-B/16 under full fine-tuning across 20 tasks — three architectures, one
instrument, verified exact on all of them"* with:

> **91.8% [90.9, 92.7] on a pretrained ViT-B/16 and 98.8% [98.3, 99.4] on a
> pretrained ResNet-50, both under full fine-tuning across 20 tasks — four
> architectures spanning recurrent, feedforward, transformer and convolutional
> families, one instrument, verified exact on all of them**

**Not claimed, and the one-liner must not imply it:** that pretraining *causes*
the higher shares. The arc is ordered along that axis and uncontrolled on it (§9).

**P2(b) landed and PASSED** (0.0010 vs the 0.05 bar). Preconditions CLEARED;
the ledger edit above is applied.
