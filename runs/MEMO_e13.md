# MEMO — E13: Tier-0 Stack — Re-Derivation + Cross-Regime MNIST Cell

**Status: COMPLETE.** Analysis-only, existing checkpoints, one pass. Contract:
signed, regenerated for execution 2026-08-10, with the catch-32 amendment (§2a)
added to the precondition block. Every number is from recorded script output
(`runs/e13_stack_mnist.json`, `runs/e13_stack_har.json`).

| hypothesis | verdict |
|---|---|
| **H-S1** re-derive the frozen 0.347 | **+0.075 [+0.052, +0.098]** — no bar; this value IS the ledger entry |
| **H-S2** cross-regime transfer (MNIST/OFF ≥ 0.30) | **NOT MET** — +0.120, robustly |
| **H-S3** composition with adapters (MNIST/ON) | **UNEVALUABLE at this residual scale** — see §4 for both readings |

**Branch (C)** — with a correction to what branch (C) means (§5).

---

## 1. Gates

| gate | HAR/OFF | HAR/v1-ON | MNIST/OFF | MNIST/ON |
|---|---|---|---|---|
| catch-32 alignment (+ shuffled control) | PASS / FIRED | PASS / FIRED | PASS / FIRED | PASS / FIRED |
| platform (`fwd_full` == recorded row) | 0.0000 (Modal) | 0.0000 (Modal) | 0.0000 (local) | 0.0000 (local) |
| pooled \|R\| ≤ 0.05 | −0.0045 PASS | −0.0061 PASS | +0.0016 PASS | +0.0027 PASS |
| catch-28 path identity | every cell | every cell | every cell | every cell |

**The platform gate did its job as a router, not just a check.** HAR printed
ABSENT locally — its checkpoints are volume-only and its executed data does not
reproduce on arm64 — and the HAR cells were run on Modal, where they do. No
number was read across an environment boundary.

Full test split throughout (the N_TEST cap removed per contract; the N_TRAIN cap
retained, because lifting it would change the refit *recipe*, not just the
sample).

---

## 2. Results — all four arms, three cures, each measured alone

| arm | era head alone | projection alone | **stack** | cells |
|---|---|---|---|---|
| HAR/OFF | +0.033 [+0.011, +0.056] | +0.063 [+0.040, +0.086] | **+0.075 [+0.052, +0.098]** | 12/12 |
| HAR/v1-ON | +0.029 [+0.005, +0.053] | +0.046 [+0.022, +0.071] | +0.067 [+0.043, +0.092] | 12/12 |
| MNIST/OFF | +0.062 [+0.055, +0.069] | +0.119 [+0.112, +0.125] | **+0.120 [+0.113, +0.127]** | 12/12 |
| MNIST/ON (guarded) | +0.144 | +0.217 | +0.270 | 6/12 |
| MNIST/ON (forced) | +0.491 | −0.088 | −0.084 | 12/12 |

Guarded and forced are **identical** on all three fully-populated arms.

---

## 3. H-S1 — the frozen literal, replaced

**Tier-0 stack on HAR/OFF = +0.075 [+0.052, +0.098].** Frozen literal: **0.347**.
**Delta: −0.272.**

No bar, by contract — a bar here would have been re-baring against a number the
program already distrusted. This value enters the ledger and supersedes the
literal, whatever it reads, and it reads far lower.

**What the literal was.** A hardcoded constant (`scripts/transport_estimate.py:51`)
and a registered table entry with **no computing artifact anywhere in the
repository**, frozen at the E11 audit and cited for two weeks on that basis. The
first time anyone computed it, it was wrong by a factor of ~4.6.

---

## 4. H-S3 — both readings, recorded verbatim

> By the evaluability threshold, H-S3 is evaluable and reads NOT MET at +0.270
> [+0.227, +0.313]; by the registered EV1 survival requirement, guarded and
> forced disagree in sign and no verdict survives; H-S3 is therefore recorded as
> unevaluable at this residual scale, both columns printed.

**Why the threshold did not fire:** exactly **6 of 12** cells are guard-excluded.
The pre-commitment says *more than half*. Six is half. It was not rounded up in
the spirit's name — the threshold stands exactly as written.

**Why no verdict survives anyway:** guarded **+0.270** against forced **−0.084**
is a **sign disagreement**. A number that flips sign under its own bookkeeping is
not a measurement with a caveat; it is the absence of a measurement. The EV1
requirement — forced-inclusion printed beside every guarded value *because
verdicts must survive their own bookkeeping* — is registered in §2 and inherited
without amendment.

Same practical consequence either way: **no composition claim.**

**Lesson for future contracts:** specify **≥ vs >** on evaluability thresholds
explicitly. Landing exactly on the boundary should not require a ruling.

---

## 5. H-S2 and what branch (C) actually means

**H-S2 NOT MET: +0.120 against the 0.30 bar**, with 12/12 cells kept and guarded
identical to forced. As clean as a negative gets. The cross-regime sentence is
**deleted, not hedged**.

**But the scoping sentence has to change too.** The plan was "the cure is
HAR-scoped." It is not:

> **HAR 0.075, MNIST 0.120 — MNIST is the better of the two.** The tier-0 family
> is weak in *both* regimes. It looked like a real partial repair only because
> the 0.347 that represented it was never computed.

### The composite rule's dividend

Measuring each cure alone — required by the composite rule, and the only reason
either fact is visible:

- **On MNIST, projection alone (+0.119) is the entire effect; the era head adds
  +0.001. The stack isn't a stack on MNIST.**
- On HAR the era head contributes a little more (+0.012 over projection alone),
  but projection still dominates.

### The anatomy finding: the ordering does NOT flip

Component ranking is **projection > era head in both regimes** — HAR
(+0.063 > +0.033) and MNIST (+0.119 > +0.062). The pre-flagged possibility was
that the ordering might invert across regimes; it does not. **A stable ordering
is a cleaner anatomy result than a flip would have been**, and it is only visible
because each component was measured alone before any stack.

### CORRECTED MECHANISM — the prior reading is retracted

**Retracted as written:** *"On MNIST the encoder absorbed the shifts internally so
the feature space rotated out from under the stored heads, while on HAR drift was
small enough that era readers stayed near-valid (hence 0.347-ish)."*

The measurement falsifies its premise. **HAR's era heads recover LESS than
MNIST's (+0.033 vs +0.062)**, so "HAR's readers stayed near-valid" is dead. It was
only ever plausible because the 0.347 literal made tier-0 look like a real
partial repair — i.e. the fiction was load-bearing for the mechanism story too.

**What survives is narrower and cleaner:** tier-0 is weak everywhere it has been
measured alone (0.075, 0.120); projection outranks the era head in both regimes;
and the composite rule is the only reason we know either fact.

---

## 6. Consequence for the paper

**The cure section's load-bearing result is now E10's C3 alone** — the certified
pseudo-refit, ρ **0.97**, end-to-end forgetting reduced **81.6%**. Tier-0 is
demoted from "cheap cure family" to **"weak baseline, measured honestly,
anatomized by component."**

That is survivable: C3 was always the certified result and tier-0's collapse does
not touch it. What it does is **raise the stakes on the ViT tier-0 check, which
becomes a discriminator rather than a confirmation**:

- stored era readers **work** where F_enc = 0.06 → the features-stayed-put theory
  lives, and the cure family gains a **scope law**: *reader repair works where
  features stayed put, in proportion to how put they stayed*;
- they **fail** there too → tier-0 is simply weak, the theory dies, and the cure
  section is **C3, full stop**.

**Prediction on record (before the protocol runs):** ViT tier-0 recovery lands
well above MNIST's 0.120, with the **era-head component dominating** — the
reverse of the ordering measured here. **~65%**, resting on F_enc = 0.06
directly, not on the HAR/MNIST contrast that E13 just falsified. Six misses
stand beside it.

---

## 7. Calibration — entry six, with its distinguishing feature

| prediction | outcome |
|---|---|
| H-S1 within ±0.10 of 0.347 (~55%) | **MISSED** — actual +0.075, off by 0.272 |
| H-S2 ≥ 0.30 on MNIST/OFF (~60%) | **MISSED** — +0.120 |
| H-S3 evaluable at all (~40%) | fired (unevaluable) |

**What distinguishes this miss:** it was a prediction about a number **nobody had
computed**, anchored to a literal that turned out to be fiction.

> **A frozen, unsourced number is not a prior. Anchoring a prediction to it
> launders the fiction into the forecast.**

Logged next to the streak-weighting lesson (a prediction made after seeing
partial results is not evidence of improved forecasting).

---

## 8. LEDGER EDIT — drafted here, applied second

Replace the tier-0 row:

| clause | evidence | status |
|---|---|---|
| tier-0 stack magnitude | **+0.075 [+0.052, +0.098]** on HAR/OFF — `runs/e13_stack_har.json`, 12/12 cells, guarded == forced, pooled R −0.0045, platform gate 0.0000 on Modal | **UNFROZEN.** Supersedes the literal 0.347, which was a **FROZEN LITERAL WITH NO COMPUTING ARTIFACT** — carried in the ledger on citation alone from the E11 audit until first computed here, and wrong by ~4.6×. |

Add:

| clause | evidence | status |
|---|---|---|
| tier-0 does **not** transfer cross-regime | MNIST/OFF stack **+0.120** vs the 0.30 bar (E8's own H-R4, reused); 12/12 cells, guarded == forced | **new** — branch (C). The cure family is weak in **both** regimes (HAR 0.075 < MNIST 0.120), not HAR-scoped. |
| tier-0 anatomy | projection alone > era head alone in **both** regimes (HAR +0.063 > +0.033; MNIST +0.119 > +0.062); on MNIST the era head adds **+0.001** | **new** — visible only because the composite rule measures each cure alone |
| tier-0 composition with adapters | MNIST/ON guarded +0.270 vs forced −0.084 — sign disagreement | **UNEVALUABLE at this residual scale**; both columns recorded |

**With this edit the ledger is fully sourced for the first time since the E11
audit** — no frozen numbers remain.
