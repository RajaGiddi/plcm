# MEMO — E16: Learning without Forgetting as a Matched Baseline

**Contract:** `docs/E16_lwf_prereg.md` (10 review issues) + the Deck-Close
Contract Item 1 (`docs/DECK_CLOSE_amendments.md`, 6 review issues).

**The result in one sentence:** *LwF protects exactly what Li & Hoiem say it
protects — and it does not help, because the channel it protects is not the one
carrying the forgetting.*

---

## 1. Preconditions

### 1a. The λ=0 positive control — PASS

`e16_mnist_lam0_seed42` vs `e4off_v2_seed42`: **15/15 bit-identical, max |Δ| =
0.000000**, AVG 0.444740 both.

**Non-tautological, and the witness proves it.** `lwf_distill_loss` read
**6.18 → 14.90** across task 1 — the teacher was snapshotted, forwarded, and
distilled against at full magnitude, then scaled to zero. So the control
demonstrates three things at once:

1. **the term is added, not multiplied** — a misplaced term perturbs the loss at λ=0;
2. **`fork_rng` held** — the teacher's forward consumed no draws from the global
   generator. Without it this fails for an entirely benign reason and gets waved
   through as "expected nondeterminism";
3. **silently, determinism** — bit-identity between two *different arms* across
   15 cells is only possible if both are deterministic on that path.

The rising trajectory is itself informative: an unconstrained student moving
*away* from a frozen teacher is what λ=0 must produce. A flat trajectory would
mean the teacher was tracking the student.

**The control runs on `mafc_off`**, which has a proven zero floor (four runs,
two waves). A bit-identity control is unachievable on an arm whose own floor is
nonzero — `plain_lstm` reproduces 0/15 at default threading, so the same control
there would have failed for reasons unrelated to LwF.

### 1b. The decomposition — 8/8 cells

| clause | status |
|---|---|
| era-checkpoint status uniform per arm | **PASS** — `[True, True, True]`, all 8 |
| identity `F_enc + F_read − R = F_total` | **0.0e+00**, all 8 |
| script version identical across arms | **PASS** — `38fbe28edb197c6b` / `e03f1f252d58cfec` |
| catch-32 label alignment | **structural** — `channel_decomp.load_task_data` imported, never reimplemented |
| cells | 12 per arm (3 seeds × 4 old tasks) |

---

## 2. The sweeps — and the two-trap exhibit

### MNIST (reference `mafc_off`: DIAG 0.9711 → floor 0.9211)

| λ | DIAG | gap | status | AVG | forgetting |
|---|---|---|---|---|---|
| **0.25** | 0.9697 | +0.0014 | **candidate** | 0.4095 | 0.7002 |
| 1.0 | 0.8462 | +0.1248 | EXCLUDED | 0.5268 | 0.3995 |
| 4.0 | 0.3462 | +0.6249 | EXCLUDED | 0.3282 | 0.0296 |
| 16.0 | 0.3007 | +0.6704 | EXCLUDED | 0.2937 | **0.0100** |

### HAR (reference `mafc_off`: DIAG 0.9100 → floor 0.8600)

| λ | DIAG | gap | status | AVG |
|---|---|---|---|---|
| **0.25** | 0.9101 | −0.0001 | **candidate** | 0.5826 |
| 1.0 | 0.8268 | +0.0832 | EXCLUDED | 0.7653 |
| 4.0 | 0.6692 | +0.2408 | EXCLUDED | 0.6668 |
| 16.0 | 0.6554 | +0.2546 | EXCLUDED | 0.6407 |

**3 of 4 λ values fail competence on both benchmarks.**

### The two traps — the contract's most valuable clause, materialised

**Trap 1, selection on forgetting.** λ=16 wins at **0.0100** — a *99% reduction
in forgetting*. Its DIAG is **0.3007** against the reference's 0.9711: the model
lost **67 points of per-task competence**. It forgets one percent because it
learns essentially nothing. Under §5's pre-committed LwF-wins branch this would
have been reported at **full prominence**, with our own honesty rules amplifying
it.

**Trap 2, selection on AVG without the floor.** λ=1.0 scores **0.5268**, beating
the reference's 0.4340 by **9pp** — and sits **12.5pp below competence**. This is
the more dangerous one: it *looks like a genuine win.*

**The metric and the floor catch different wrong answers. Only both together
leave λ=0.25.**

### Not noise

Across-λ AVG spread **0.2331** against a mean across-seed sd of **0.0198** —
**11×**. DIAG falls monotonically with λ on both benchmarks while forgetting
falls with it: the stability–plasticity trade visible in a single table.
*λ buys retention by purchasing it with competence.*

---

## 3. Verdicts — matched configuration, n=3

| | LwF λ=0.25 | reference | delta | 95% CI |
|---|---|---|---|---|
| **MNIST** | 0.4095 | 0.4174 | **−0.79pp** | ±4.42pp |
| **HAR** | 0.5826 | 0.5679 | **+1.46pp** | ±4.75pp |

> **At matched configuration, LwF's only competent setting is indistinguishable
> from doing nothing on both benchmarks.**

**Not "loses to."** The mixed-config numbers (−2.45pp / −2.86pp) were computed
against reference arms that differed on the era-checkpoint axis (§5).

**Resolution limit, stated rather than papered over:** with n=3 and per-seed
spreads of 3–6pp, ±4.4pp *is* the resolution. Distinguishing LwF from no-method
at this scale needs substantially more seeds.

**Floors: zero.** LwF λ=0.25 at 4 threads with era checkpoints — MNIST 15/15,
HAR 15/15, max |Δ| 0.000000 both. Measured on the arm, at the config, at the
threading the comparison uses.

---

## 4. The decomposition — where the protection lands

### 4a. H-D2's caveat, written before the numbers were read

λ=4 and λ=16 **collapsed on DIAG**. Their decompositions answer *"where did what
little it learned go"* and are **not comparable to a competent arm's**. F_enc and
F_read measure loss relative to what *was* learned; when little was learned, the
quantities are not commensurable across λ. Stated in the contract ahead of the
instrument firing, precisely because the temptation afterward is to read the λ
column as a trend.

### 4b. H-D1 — descriptive, no bar

| | arm | F_enc | F_read | reader share |
|---|---|---|---|---|
| **MNIST** | `mafc_off` (ref) | +0.2441 [+0.182, +0.306] | +0.4490 [+0.388, +0.510] | 64.8% |
| | λ=0.25 | **+0.1940** [+0.153, +0.235] | +0.4966 [+0.444, +0.549] | 71.9% |
| | λ=4.0 *(collapsed)* | +0.0448 | +0.4960 | 91.7% |
| | λ=16.0 *(collapsed)* | +0.0224 | +0.4717 | 95.5% |
| **HAR** | `mafc_off` (ref) | +0.0757 [+0.042, +0.110] | +0.3100 [+0.237, +0.383] | 80.4% |
| | λ=0.25 | **+0.0159** [+0.004, +0.027] | +0.3546 [+0.227, +0.482] | 95.7% |
| | λ=4.0 *(collapsed)* | −0.0018 | +0.1265 | 101.4% |
| | λ=16.0 *(collapsed)* | +0.0035 | +0.1339 | 97.4% |

### 4c. H-D3 — paired by (seed, task), 12 pairs per benchmark

| | ΔF_enc | ΔF_read | ΔF_total |
|---|---|---|---|
| **MNIST** | **−0.0501 [−0.0821, −0.0180]** | **+0.0476 [+0.0140, +0.0812]** | −0.0022 [−0.0348, +0.0304] |
| **HAR** | **−0.0599 [−0.0890, −0.0308]** | +0.0447 [−0.0226, +0.1119] | −0.0094 [−0.0524, +0.0337] |

**LwF's protection lands on the encoder, exactly where its authors say it does.**
F_enc falls on both benchmarks with CIs excluding zero — **−20% of the encoder
channel on MNIST, −79% on HAR**. Li & Hoiem's stated mechanism — preserving
outputs "retain[s] the important shared structures" — is **vindicated, not
refuted**.

**And F_total does not move.** Both CIs straddle zero. The encoder gain is spent:
F_read gets *worse* — **measured on MNIST** (CI excludes zero), **suggestive on
HAR** (CI includes zero). Two different evidential strengths, stated separately.

> **A method can work exactly as designed and still not help, when it operates
> on the channel that was not broken.**

This is the paper's thesis demonstrated through the baseline's *success* rather
than its failure, and it explains §3's null mechanistically instead of leaving it
as an absence.

**Mechanism for why F_read worsens: not tested.** Distillation on new-task data
constrains the readout toward the teacher's outputs *on the new distribution*,
which is not the same as preserving its behaviour on old ones — plausible, and
**labelled untested** rather than asserted. Two mechanisms died this week for
being plausible.

---

## 5. Process

### 5a. Era-checkpointing is configuration, not observability

`--era-checkpoints` does not merely write a file: `_save_era_checkpoint` reloads
through `PLCM.load_era`, evaluates, runs `_refit_ceiling`, and fires the P3 gate,
all inside training. Measured effect on final AVG: **MNIST +2.13pp, HAR
−1.67pp** — same flag, **opposite signs**, so no blanket correction exists. On
the reference arms at n=3 it was worth **−1.66pp (MNIST)** and **−4.32pp (HAR)**.

**Confirmed by prediction, not conjecture:** re-running the floor pairs *with*
the flag reproduced the sweep runs **bit-identically, 15/15, AVG gap 0.0000, on
both benchmarks**.

*General form for the paper: anyone adding checkpoint-time diagnostics to a
continual-learning pipeline and comparing against runs without them carries an
unlabeled ~2pp confound of benchmark-dependent sign.*

### 5b. The determinism map

| arm | 4 threads | 1 thread |
|---|---|---|
| MNIST `plain_lstm` | ✗ 0/15 | ✓ 15/15 |
| MNIST `mafc_off` | ✓ 15/15 (×2 pairs) | — |
| MNIST `lwf` λ=0.25 | ✓ 15/15 | ✗ 1/15 |
| HAR `mafc_off` | ✓ 15/15 | ✗ 0/15 |
| HAR `lwf` λ=0.25 | ✓ 15/15 | — |

**`plain_lstm` is the odd one out.** Every other arm tested is 4-thread
deterministic and 1-thread nondeterministic. No thread-count-monotone mechanism
produces both directions, and single-threaded BLAS has a fixed reduction order —
so the "BLAS reduction order" story is **retracted**.

> **Determinism belongs to (arm × config × platform × thread-count), established
> per-pair, mechanism unknown until demonstrated. Pinning is an axis, not a
> stabilizer.**

### 5c. Catches this experiment

| # | catch | cost |
|---|---|---|
| 1 | `--mafc-arm lwf` already meant an H3 ablation — a launcher written from memory would have produced a MAFC control wearing the field's baseline's name | caught at contract |
| 2 | selection metric unspecified → λ=16's fake 99% | caught at contract; **materialised in the data** |
| 3 | `jobs_e16floor_match` bypassed `_e16()` and silently dropped `--era-checkpoints` | 4 wasted runs; found by a 1.67pp number that did not fit |
| 4 | `scripts/e16_decompose.py` did not exist; `jobs_e16dec` had no λ | caught at contract review |
| 5 | "`mafc_off` already decomposed" — false premise | caught at contract review |
| 6 | checkpoint **directories** exist ≠ checkpoint **files** addressable | 8 failed jobs |

**Catch 6 deserves its line:** I wrote *"28 `ckpt_e16*` dirs exist, so the
expensive part is done"* — and the repo's own gotcha, quoted earlier the same
day, is *"a checkpoint exists" is not "a checkpoint loads."* The files were one
directory deeper. **The app still reported "✓ App completed"** over 8 failures,
because `_run()` catches non-zero exits and returns `{"ok": False}` — an
orchestrator that swallows child exit codes turns every job failure into a
silent success at the app level.

### 5d. Predictions

| prediction | outcome |
|---|---|
| λ-sweep noise-dominated, ~40% | **miss** — 11× structure |
| LwF > vanilla on MNIST, ~80% | **miss** — indistinguishable |
| bridging > LwF on HAR, ~55% | pending §7 |
| **protection concentrates in F_read, ~65%** | **MISS — it concentrates in F_enc** |
| collapsed arms uninterpretable under H-D2, ~80% | **fired** |

**Eleven entries, six misses.** The F_read miss is the most valuable: the
prediction was wrong and the truth is the better paper.

---

## 6. LEDGER EDITS — drafted, applied on sign-off

| claim | evidence | status |
|---|---|---|
| **LwF at its only competent λ is indistinguishable from doing nothing** | matched config n=3: MNIST **−0.79pp [±4.42]**, HAR **+1.46pp [±4.75]**; floors **zero** on both (15/15, max \|Δ\| 0.000000) at the arm/config/threading the comparison uses; λ=0 control 15/15 with a live witness | **new**. Reported at the instrument's resolution — ±4.4pp *is* the limit at n=3 |
| **LwF's protection lands on the ENCODER — its authors' stated mechanism is vindicated** | E16 H-D3, paired 12 cells/benchmark: ΔF_enc **−0.0501 [−0.0821, −0.0180]** (MNIST), **−0.0599 [−0.0890, −0.0308]** (HAR); −20% / −79% of the encoder channel | **new, and a counterexample to the prior** (~65% on F_read). Reported at equal prominence per contract |
| **…and it does not help, because that channel was not carrying the forgetting** | ΔF_total **−0.0022 [−0.0348, +0.0304]** (MNIST), **−0.0094 [−0.0524, +0.0337]** (HAR) — both straddle zero. ΔF_read **+0.0476 [+0.0140, +0.0812]** MNIST (**measured**), **+0.0447 [−0.0226, +0.1119]** HAR (**suggestive**) | **new** — the thesis via the baseline's success, not its failure. Mechanism for the F_read worsening **labelled untested** |
| **3 of 4 λ fail competence on both benchmarks** | MNIST DIAG gaps +0.125 / +0.625 / +0.670; HAR +0.083 / +0.241 / +0.255, against a 5pp bar | **new** — with both counterfactuals: forgetting-selection picks λ=16 (**0.0100**, a fake 99% reduction, DIAG **0.3007**); AVG-without-floor picks λ=1.0 (fake 9pp win at 12.5pp below competence) |
| **era-checkpointing is configuration, not observability** | MNIST **+2.13pp**, HAR **−1.67pp** on final AVG; on the reference arms **−1.66 / −4.32pp**; confirmed by 15/15 bit-identical reproduction when matched | **new, methods contribution.** Every comparison must be uniform on this axis, checked **across** arms |
| determinism map, 5 cells | §5b | **new** — `plain_lstm` is the outlier; "BLAS reduction order" **retracted** |

---

## 7. What closes, and what does not

**Closes:** LwF is a measured, matched, competently-tuned baseline with its
mechanism decomposed. The differentiation paragraph now rests on numbers.

**Does not close:** the HAR bridging comparison (§7.2), which needs the
bridging arm read against this reference under matched era-checkpoint status.

**Unrelated and flagged:** the repository has **zero commits**. The
script-version precondition was met with a content hash instead. That is a
reproducibility gap for the paper, priced at minutes, deliberately not folded
into a signed contract.
