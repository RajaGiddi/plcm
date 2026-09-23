# MAFC — Result Memo (Phase 0 + Phase 0′)

**Status:** Phase 0 complete; Phase 0′ complete. **Branch (b) of amendment v3
taken.** Phase 1 as contracted is NOT run — see §3.

---

## 1. Verdict

**Phase 0′ (5 tasks × 10 epochs, M-λ0 control alone) — now 3 seeds:**

| seed | AVG | RET (T0–3) | DIAG | forgetting | T0 final |
|---|---|---|---|---|---|
| 42 | 0.9470 | 0.9395 | 0.9776 | 0.0383 | 0.8724 |
| 1337 | 0.9279 | 0.9148 | 0.9775 | 0.0620 | 0.7970 |
| 2024 | 0.9243 | 0.9103 | 0.9786 | 0.0678 | 0.7825 |
| **mean ± std** | **0.9331 ± 0.0099** | **0.9215 ± 0.0129** | **0.9779 ± 0.0005** | **0.0560 ± 0.0128** | 0.8173 ± 0.0394 |

Branch (b) threshold was RET ≥ 85%; the **minimum** across seeds is 91.0%, so
the branch call holds on all three.

*Note on the original n=1 report:* seed 42 was the most favorable draw
(AVG 94.70 vs 93.31 mean). The three-seed figures supersede it throughout —
single-seed optimism of ~1.4pp, which is exactly why the row was seeded.

Per the pre-written branch: **"MAFC is solving a solved problem. The finding
becomes *per-task input adapters alone largely prevent encoder-mediated
forgetting*, which is a result — cheaper, simpler, and it goes straight into
paper one while the program re-derives what component one should be."**

## 2. The result that replaces the mechanism

| configuration | AVG | forgetting | DIAG |
|---|---|---|---|
| LSTM baseline | 0.4756 | 0.6219 | 0.9731 |
| unfrozen PLCM (no adapters, no hints) | 0.4640 | 0.6353 | 0.9722 |
| PLCM+EWC λ=200 (best weight-space) — **n=3: 0.4330 ± 0.0441** | 0.4330 | 0.6718 | 0.9704 |
| **M-λ0: unfrozen + adapters + hints, NO anchor loss** (n=3) | **0.9331 ± 0.0099** | **0.0560 ± 0.0128** | 0.9779 |
| frozen + adapters + hints (n=1) | 0.9767 | 0.0009 | 0.9766 |
| frozen + adapters + hints, memory OFF (n=1) | 0.9798 | 0.0002 | 0.9796 |

**Per-task input adapters, with no continual-learning mechanism of any kind,
take AVG from 44.0% → 93.3% and cut forgetting from 0.67 → 0.056** — while the
encoder stays fully plastic (DIAG 97.8% on every task including the last, and
notably stable across seeds: ±0.0005). Adapters buy **+49.3pp**.

### EWC is a NULL result, not a weak baseline (n=3, final)

    PLCM+EWC lambda=200 : 0.4330 +/- 0.0441
    vanilla LSTM        : 0.4399 +/- 0.0282
    -> EWC is 0.7pp WORSE than doing nothing, well inside noise.

**Weight-space regularization provided no measurable benefit on this benchmark,
at the best λ from a five-point sweep.** Every earlier sentence of the form "EWC
bought +1.9pp" or "PLCM+EWC beat the LSTM at λ=200" is **retracted** — those
compared against an n=1 EWC value of 0.4950.

This *strengthens* the thesis rather than weakening it. The argument is "the
field defends weights; we route the input instead." The weight-defense arm does
not merely underperform — it fails to move at all (+0pp) while input routing
moves **+49pp**. The central contrast is 50-vs-0, not 50-vs-2.

**EWC is also the least stable configuration measured** (±0.0441, the largest
seed spread in the project). Weight-space regularization on a recurrent net at 5
tasks is not just ineffective but *unstable* — consistent with Ehret et al. on
EWC's particularities in RNNs. Worth its own sentence.

### Methodological note for the process appendix (mandatory)

The λ=200 optimum was **selected at n=1** from {50, 100, 150, 200, 300}. With
±0.044 seed noise on this configuration, **that entire sweep was selecting on
noise** — the apparent 0.4950 peak at λ=200 is indistinguishable from the
0.4138–0.4521 readings at other λ once seeded. State this plainly: it is a
textbook illustration of why the n=1 rule exists, and a stronger appendix entry
than any of the other twelve catches, because the error was *invisible* until
the baseline was seeded.

**n=1 optimism is now 5/5**, worst instance −6.2pp (this one).

**The addressable gap MAFC was designed to close is therefore ~4.4pp, not the
imagined 47.6 → 97.7 span.** Phase 0 showed MAFC drives forgetting to exactly
0.0000 in the micro regime, so it would likely capture some of that residual —
a real but marginal improvement over a control that already does the work.

*Comparison caveat:* the M-λ0 row is n=3; the frozen rows are still n=1. The
"freezing buys ≈4pp" claim is therefore 3-seed vs 1-seed and should be seeded on
the frozen side before it goes in print.

## 3. Why Phase 1 is not run as contracted

H1's bar under the v2 Ruling-3 amendment (M-λ0 ≥ 55% ⇒ bar = control + 15pp) is
**94.7 + 15 = 109.7%** — unsatisfiable, exactly as H0a was in Phase 0. The
pre-registered hypotheses can no longer discriminate, and the mechanism they
were written to test addresses a 3pp residual. Running Phase 1 would spend two
days measuring a mechanism against an impossible bar.

## 2y. THE BUILT-IN CONTROL — DIAG flatness (for §5 opener)

**Sentence for the top of §5:**

> *Diagonal accuracy — each task measured immediately after training it — is flat at
> 0.970–0.978 across every configuration in this table (0.75pp total spread, and
> ±0.0005–0.0014 within each row), while final retention spans 0.296–0.974 (67.8pp).
> Retention variance is **91× the diagonal variance**. Every method learns every task
> to the same standard; the table's entire spread is a retention phenomenon measured
> with plasticity held constant.*

| row | DIAG | RET |
|---|---|---|
| vanilla LSTM | 0.9722 ± 0.0008 | 0.3081 ± 0.0352 |
| PLCM, no adapters | 0.9720 ± 0.0005 | 0.2961 ± 0.0137 |
| PLCM+EWC λ=200 | 0.9704 ± 0.0011 | 0.2998 ± 0.0545 |
| ER 100/task | 0.9705 ± 0.0014 | 0.7478 ± 0.0062 |
| adapters, unfrozen | 0.9779 ± 0.0005 | 0.9215 ± 0.0129 |
| adapters, frozen | 0.9764 ± 0.0007 | 0.9742 ± 0.0020 |
| **across rows** | **0.75pp spread** | **67.81pp spread** |

This is a control that was running the whole time without being named. It forecloses
the most common ambiguity in continual-learning tables — *"did the winner just trade
learning capacity for stability?"* — **by inspection**, not by argument. Reporting the
variance alongside the mean is what makes it airtight: DIAG is flat *and tight*, so
"flat" cannot be dismissed as merely "similar on average."

**Consequence for the results section:** **RET is the honest headline column**
(0.296 → 0.922), with AVG retained as the metric readers expect. AVG mixes plasticity
and retention; RET isolates the quantity that actually differs.

## 2z. CENTRAL FIGURE (consolidated, all rows n=3 except where noted)

**Permuted MNIST, 5 tasks × 10 epochs, task-incremental eval.**

| configuration | n | AVG | forgetting | DIAG |
|---|---|---|---|---|
| vanilla LSTM | 3 | 0.4399 ± 0.0282 | 0.6653 | 0.9722 |
| PLCM, no adapters (unfrozen) | 3 | 0.4304 ± 0.0109 | 0.6770 | 0.9720 |
| **Experience Replay** (100/task, stores raw data) | 3 | 0.7916 ± 0.0051 | 0.2236 | 0.9705 |
| **PLCM + adapters, unfrozen** | 3 | **0.9331 ± 0.0099** | 0.0560 | 0.9779 |
| PLCM + adapters, **frozen** encoder | 3 | 0.9739 ± 0.0020 | 0.0038 | 0.9765 |

**ER (prereg branch: ER-partial, as expected).** +36.1pp over the no-adapter
control single-variable — a genuinely strong baseline, not a strawman. Adapters
lead it by **+14.2pp** (frozen by +18.2pp) while retaining no raw examples.

**Storage honesty (report before a reviewer computes it):**

| method | retained at task 4 | size |
|---|---|---|
| ER buffer | 400 raw images | 1.25 MB |
| adapters | 5 × 784×784 | **12.29 MB** |

The adapters use **~10× more storage than the buffer they beat**. The defensible
claim is the *regime* — no raw examples retained (privacy) — **not** efficiency.

**FINAL — the frontier crosses TWICE; it is regime-dependent, not a single
concession.** All storage on one basis (retained after training):

| method | MB | AVG | n | retains |
|---|---|---|---|---|
| adapters r32 | 1.00 | 0.6877 ± 0.0121 | 3 | parameters |
| Experience Replay 100/task | 1.57 | 0.7916 ± 0.0051 | 3 | raw images |
| **adapters r64** | **2.01** | **0.8719 ± 0.0250** | 3 | parameters |
| ER 200/task | 3.14 | 0.8592 | 1 | raw images |
| ER 500/task | 7.84 | 0.9002 | 1 | raw images |
| adapters full-rank | 12.29 | 0.9470 | 1 | parameters |

**Head-to-head at comparable budgets:**

    ~1 MB : ER 100  0.7916  beats r32 0.6877   by +10.4pp  (ER, +57% storage)
    ~2 MB : r64     0.8719  beats ER 200 0.8592 by  +1.3pp (ADAPTERS, -36% storage)
    large : full    0.9470  beats ER 500 0.9002 by  +4.7pp (adapters, +57% storage)

Required claim language, final:

> *"Adapters are competitive-to-better above ~2 MB and dominated below it, while
> retaining no raw data at any point."*

That tells a practitioner which regime they are in — more useful than either
"replay wins at matched storage" (too pessimistic) or "we are efficient" (false).
The frontier figure should plot both curves, annotated with **what each
retains**, so the privacy distinction and the storage tradeoff are visible in one
graphic.

**The ER buffer sweep moved our own claim, in both directions.** It killed the
strawman objection *and* raised the baseline: ER-500 at 0.9002 is far stronger
than the 0.7916 we had been quoting. Adapters still lead (0.9470 full-rank) but
by **4.7pp, not 15pp**.

*All rows now n=3. The LSTM row shifted −3.57pp under seeding (n=1 was 0.4756) —
the largest single-seed optimism measured in this project, and the row carries
the table's highest variance (±0.0282).*

**Rotated MNIST (0/22.5/45/67.5/90°), same protocol.**

| configuration | n | AVG | forgetting |
|---|---|---|---|
| PLCM, no adapters (unfrozen) | 3 | 0.6446 ± 0.0100 | 0.4293 |
| **PLCM + adapters, unfrozen** | 3 | **0.9366 ± 0.0101** | 0.0538 |

**MLP backbone, Permuted MNIST (architecture generality).**

| configuration | n | AVG | forgetting |
|---|---|---|---|
| MLP, no adapters (unfrozen) | 3 | 0.6690 ± 0.0103 | 0.3858 |
| **MLP + adapters, unfrozen** | 3 | **0.9198 ± 0.0236** ᵃ | 0.0676 |

ᵃ largest seed spread in the project (~2× the LSTM arm's); effect is 25pp
against a ±1.9pp delta spread, so nothing is at risk — noted for honesty.

### DECOMPOSITION PANEL (the paper's strongest single result)

| condition | OFF | ON | delta |
|---|---|---|---|
| Permuted / LSTM (reference) | 0.4304 | 0.9331 | +50.3pp |
| Rotated / LSTM | 0.6446 | 0.9366 | +29.2pp |
| Permuted / MLP | 0.6690 | 0.9198 | +25.1pp |

**Change vs the reference condition:**

    Rotated : OFF +21.4pp    ON  +0.4pp
    MLP     : OFF +23.9pp    ON  -1.3pp

**The adapter arm spans 1.7pp across two architectures and two shift types
(0.9331 / 0.9366 / 0.9198), while baselines span 43–67%.** Every measured delta
reduction is **baseline relief, never mechanism degradation** — twice, along
independent axes. The effect cannot be attributed to permutation-specific
invertibility (refuted by Rotated) or to recurrence (refuted by the MLP arm),
because the endpoint does not move under either.

**Title scope updated accordingly** (was "…in Trainable Recurrent Encoders"):

> *Adapt the Input, Not the Weights: Input-Path Isolation Prevents Catastrophic
> Forgetting in Trainable Encoders*

**Three quantities the figure supports:**

1. **Adapters: +50.3pp** (Permuted, single-variable) — the entire system gain.
   The memory machinery contributes **−0.95pp, i.e. nothing measurable**
   (0.4304 vs 0.4399; well inside the ±0.0282 / ±0.0109 seed spreads).
   *Corrected from −4.5pp, which compared an n=3 arm against the n=1 LSTM.*
2. **Freezing: +4.08pp** (0.9331 → 0.9739, now seed-matched) — and this delta is
   **pure retention too** (§6.3 echo): DIAG is 0.9779 unfrozen vs 0.9764 frozen, i.e.
   *unchanged*, while RET moves 0.9215 → 0.9742. Freezing does not buy its 4 points by
   sacrificing plasticity; the decomposition stays clean all the way down. Freezing is a
   marginal refinement, not the active ingredient — correcting the project's
   original attribution.
3. **Invariance: +0.35pp** — the adapter arm scores 0.9331 (Permuted, exactly
   invertible shift) vs 0.9366 (Rotated, 24.9% signal loss under linear
   inversion). The mechanism does not depend on invertibility.

**Variance structure — RETRACTED as mechanism evidence, retained as observation.**
Seed spread by regime:

    vanilla LSTM (unprotected)      +/- 0.0282
    PLCM no adapters (unprotected)  +/- 0.0109
    PLCM + adapters (unfrozen)      +/- 0.0099
    ER (100/task)                   +/- 0.0051   <-- tighter, NO isolation
    PLCM + adapters (frozen)        +/- 0.0020

The unprotected regime is unstable (outcomes swing ~3 points on seed alone) and
every effective method is far tighter. **But ER — which has no gradient
isolation — is tighter than the unfrozen adapter arm.** So variance collapse
tracks *effectiveness in general*, not the isolation mechanism specifically.

*Superseded draft:* "variance collapse as a signature of the mechanism working /
independent evidence for gradient isolation." Not supportable: a rehearsal
method with no isolation achieves lower variance. Report as a descriptive
observation about protected vs unprotected regimes only.

**Methods sentence (n=1 optimism, 4/4):** *every configuration first measured at
n=1 proved optimistic on seeding — by 1.4pp (adapter row) and 3.57pp (vanilla
LSTM row) — which is why no row in this table is n=1.*

**Outstanding:** the vanilla LSTM row is still **n=1** and should be seeded
before the figure is final; it is the only asymmetry left.

## 2a. Attribution — single-variable control (both arms n=3)

Only `adapters.enabled` differs between the two PLCM rows; encoder unfrozen and
shared readout held constant on both sides (verified by config diff: exactly one
differing key).

| arm | n | AVG | RET | DIAG | forgetting |
|---|---|---|---|---|---|
| vanilla LSTM | 1 | 0.4756 | 0.3528 | 0.9731 | 0.6219 |
| PLCM, **no** adapters | 3 | 0.4304 ± 0.0109 | 0.2961 ± 0.0137 | 0.9720 ± 0.0005 | 0.6770 |
| PLCM, **with** adapters | 3 | 0.9331 ± 0.0099 | 0.9215 ± 0.0129 | 0.9779 ± 0.0005 | 0.0560 |

**Decomposition of the system delta:**

    memory machinery  (LSTM -> no-adapter PLCM):   -4.5pp AVG
    ADAPTERS          (no-adapter -> +adapters):  +50.3pp AVG
                                     per-seed  :  +50.2, +50.9, +49.7

The entire benefit — and then some — is attributable to one component. Seed
spread is 1.2pp against a 50pp effect, so the result is not a favorable draw.

**RET tells it more starkly than AVG: retention triples, 0.296 → 0.922**, while
DIAG barely moves (0.9720 → 0.9779) — nothing was traded away.

**Sentence for the paper (report plainly, do not bury) — REVISED after the LSTM
row was seeded:** *"The memory-augmentation machinery contributed no measurable
benefit over the vanilla baseline (−0.95pp, within seed noise); the entire
benefit is attributable to per-task input adapters."*

*Superseded draft:* "…contributed −4.5pp…". That figure compared the n=3
no-adapter arm against the **n=1** LSTM (0.4756). With the baseline seeded
(0.4399 ± 0.0282) the gap collapses to −0.95pp — no longer a penalty, simply
nothing. The corrected claim is weaker in sign and stronger in kind: the
machinery is not harmful, it is *inert*.

## 3a. Two items promoted from memo to PAPER text

**(i) Report the self-correction in the paper, not just here.** One line:
*"Our own single-seed estimate was optimistic by 1.4pp; we report three-seed
means throughout."* In a paper whose central figure compares small differences,
this costs nothing and pre-empts a whole class of reviewer doubt.

**(ii) DIAG stability is an ARGUMENT, not an observation — give it its own line
in Results.** Across seeds, DIAG = 0.9779 ± **0.0005** while RET carries all the
variance (± 0.0129). Plasticity is seed-invariant and essentially uncompromised;
retention is what moves. This is the cleanest available evidence that the result
is *not* the frozen-backbone result in disguise, and it is the sentence that
separates this method from every frozen-backbone approach in the literature:
**the backbone keeps learning at full strength (97.8% on every task including
the last) while retaining 92% of old tasks.**

## 4. Scope and caveats (stated so the finding is not oversold)

- **Task-incremental only.** Per-task adapters require task identity at eval.
  The task-free routing attempt (component two, prior work) failed at 20%
  routing accuracy; this result does not rescue it.
- **Single seed.** Branch (b) was pre-registered as a one-seed decision, so the
  *branch call* is contract-legitimate — but the *finding* needs 3 seeds before
  it appears in paper one.
- **Permutation-specific (R1).** Permuted MNIST's task shift is exactly a linear
  input map, and a linear adapter inverts it exactly. This is why adapters work
  so well here and why the result should not be claimed to transfer to semantic
  shifts (Split-CIFAR).
- **Not forgetting-mitigation.** As with the frozen result, this is parameter
  isolation: each task's gradient routes through its own input map, shielding
  the shared encoder. The encoder is *not* being protected by a mechanism that
  resists interference — it is being shielded by construction.

## 5. What was built and verified (retained regardless)

MAFC mechanism is implemented, tested, and correct — it simply has little left
to do:
- `functional_forward` (g∘f∘A, memory-bypassed) and `readout_from_state` (g_φ on
  stored bank triples).
- Two-term loss: `L_enc` (adapter-anchored, reaches θ and φ) + `L_read`
  (bank-anchored, reaches φ only). **Gradient structure verified empirically** —
  `L_read` gives θ exactly zero gradient, confirming prereg §1.1.
- Four arms (full / enc-only / raw-probe LwF / λ=0), snapshots, CLI, config.
- 10 structural tests; full suite 73/73 green.

## 6. Recommended next step

Three seeds of the M-λ0 control (~75 min) to make the §2 finding publication-
grade, then it goes into paper one as a headline row alongside the failure
catalog. The program's component one should be re-derived against the *measured*
3pp gap, not the imagined 50pp one — and the honest open question is no longer
"how do we stop encoder drift" but "what is the memory for, given adapters
already carry the load."
