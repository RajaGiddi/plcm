# MEMO — E15: ViT Tier-0, Prototypes-Only

**Status: COMPLETE.** Analysis-only, E12 base-arm era checkpoints, 5 seeds × 19
old tasks = **95 cells**. Contract: `docs/E15_vit_tier0_prereg.md` (signed
2026-08-10). Numbers from recorded output (`runs/e15_tier0/seed_*.json`).

**Verdict: H-P1 NOT MET — branch (B), with the rank control PASSED.**
And branch (B) with a passed rank control is **more informative than the
positive would have been**.

---

## 1. Read in contract order

### Preconditions

| | |
|---|---|
| P3a / P3b positive controls | **BOTH FIRED** |
| catch-32 alignment | order == dataset indexing **PASS**; shuffled control **FIRED** |
| reproduction floor quoted | E12's: aggregate ~0.4pp, **per-cell ~15pp** → pooled-with-CI primary |
| `acc_orig` consistency vs E12 v2 cells | exact on every cell (else the reused `acc_refit`/`R` would not belong to them) |

### Rank control (§2a) — PASS, decisively

| | |
|---|---|
| projection rank | **5 of 768** |
| mean (era ceiling − proj@era) | **−0.0015** |
| max | +0.0080 |
| bar | 0.05 |

**Rank 5 of 768 loses nothing at the era checkpoint.** The projection is
information-preserving, the θ_T verdict is readable, and **branch (C) is off the
table**. The control you sustained as a gate did exactly the work it was added
for: without it, everything below would have been ambiguous between *drift* and
*rank-5-of-768 is too lossy*.

### The cure — read only now

| | |
|---|---|
| **proj ρ, guarded** | **+0.005 [−0.001, +0.011]**, 95/95 cells |
| forced | **+0.005** — identical |
| bar | 0.30 → **NOT MET** |
| deployed → projected | 0.2161 → **0.2204** |
| refit / era ceiling | 0.8884 / 0.9315 |

**Projection recovers 0.4pp where a fresh probe recovers 67pp.** The CI includes
zero, no cells are excluded, and guarded equals forced — the verdict survives its
own bookkeeping completely.

**Era-head component:** printed as `acc_orig` = 0.2161, **degenerate by
construction** (head *k* is the deployed head for task *k*, P3b-verified) and
**never scored**, exactly as contracted.

---

## 2. What E15 actually delivered

**This is the program's first MECHANISM-LEVEL result about the reader channel
itself.** E12 established that forgetting is 92% reader-side on the ViT. E15
establishes *what kind* of reader damage that is:

- the era-prototype span is a **perfectly good** subspace for reading task *k* —
  at the era checkpoint it costs **nothing** (−0.0015);
- old-task information **persists** at θ_T — the refit recovers **0.888**;
- yet projecting θ_T features onto that same span recovers **0.4pp of a 67pp gap**.

The span did not degrade. The information did not vanish. **The discriminative
directions rotated out of the stored subspace** — at near-zero cost to
representational quality (F_enc = 0.06).

> **Feature stability and span stability are different quantities, and reader
> repair from stored prototypes tracks the second, not the first.**

Measured, not argued — and measured in **the strongest case the program could
construct for the opposite conclusion**, since the ViT is the regime where
features moved least.

### The counterintuitive number

| regime | tier-0 ρ | F_enc |
|---|---|---|
| HAR/OFF | 0.075 | 0.052–0.112 (family range) |
| MNIST/OFF | 0.120 | — |
| **ViT (task-IL)** | **0.005** | **0.06** |

**Tier-0 is worst where features moved least.** That is the single most
counterintuitive number in the program, and **the rank control is what makes it a
finding rather than an anomaly** — without it, the reading would have been "the
projection is too lossy at this rank" and the phenomenon would have been invisible.

### The cure section gains its explanatory paragraph

*Why does C3 work where projection fails?* Because **C3 refits the reader to
current geometry rather than projecting onto stored geometry.** Its regeneration
sidesteps span staleness entirely — it does not need the old subspace to still be
the right one. **The failed cure explains the successful one**, which is a better
cure section than two successes would have given.

---

## 3. Calibration — entry seven

| prediction | outcome |
|---|---|
| H-P1 ρ ≥ 0.30 (~45%, revised down from 65%) | **MISSED** — +0.005 |
| rank control passes (~70%) | **fired** — −0.0015, comfortably |
| era-head clause (withdrawn pre-run) | correct to withdraw — degenerate, 0.2161 = `acc_orig` exactly |

**The miss's content IS the finding.** The surviving clause of the prediction
rested on F_enc = 0.06 implying span persistence; the result is that those are
different quantities. *The forecast error and the scientific result are the same
fact* — which is the most useful shape a miss can have.

Withdrawing the era-head clause before the run was right: its null was already
sitting in E12's own table (the component was deployed all along, and forgetting
is 0.72 anyway).

---

## 4. Process note

**A gate pre-empted rather than discovered.** The rank control was added at
contract time because rank-5-of-768 could have made a null ambiguous — the
manufactured-comparison family's fourth appearance, and **the first caught before
the numbers rather than in them** (catch 31: control without a counterfactual;
catch 32: parallel loader; E12/H-V3: floor and probe sharing a masking; here:
drift and geometry sharing one null). It then passed, which is what licensed
reading the cure at all.

**Two wiring failures before the clean run, both mine.** `assert_path_identity`
is the LSTM-family gate — its catch-28 sizing block calls `model.lstm(...)`
expecting `(out,(h,c))` — and a `ViTEncoder` returns a single tensor; the ViT gate
is `scripts/e12_p3.assert_p3`. It now raises that message instead of an opaque
unpack error. Second: a CPU/CUDA mismatch between cached features and the head.
Neither touched a number; both were caught by the run.

**And a launcher hazard worth fixing before reuse:** two apps once ran
concurrently against the same output paths, because `spawn_analysis` does not
check for a live app of the same job set. Nothing was contaminated (no outputs
had been written), but that was timing, not design. The operational fix used
here: verify zero live apps **before** launching and exactly one **after**.

---

## 5. LEDGER CLAUSE — drafted here, applied second

> Read-time reader repair from stored class prototypes **fails universally across
> measured regimes** (HAR ρ 0.075, MNIST 0.120, ViT 0.005 [−0.001, +0.011]) — and
> **fails worst where features are most stable**. The mechanism is measured, not
> argued: at the ViT's θ_T the era-prototype span is information-preserving (rank
> control −0.0015 vs era ceiling), old-task information persists (refit 0.888),
> yet projection onto the stored span recovers **0.4pp of a 67pp gap**. Feature
> stability (F_enc 0.06) **does not confer span stability**; discriminative
> directions rotate out of stored subspaces even when representation quality is
> preserved. Reader repair requires **span stability**, which no measured regime
> provides. The cure family's load-bearing result is **E10's C3 alone**
> (ρ 0.974, forgetting −81.6%), whose known-map regeneration sidesteps span
> staleness by **refitting rather than projecting**.

---

## 6. Refinement carried to E14

E14's ResNet decomposition should print the **same rank-control-style diagnostic
per task** if the machinery is already there. "Span rotation under stable
features" is now a phenomenon with **one architecture's measurement**; a second
data point costs nothing once the projection is wired. **Descriptive only, no
bar**, per H-R2's register.
