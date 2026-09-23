# MEMO — Wave 1: Reference Closure

**Contract:** the Wave 1 execution contract (1a–1d), signed after a catch-33 pass.
**Purpose:** apply the wholesale-supersession standard to everything the §5
comparison table cites.
**Outcome:** the reference table closes verified. **Nothing qualitative changed.
One measurement method did.**

> **The catch-33 pass caught three unmechanized clauses in the contract that
> adopted catch 33.** That is the rule paying for itself on its first outing.

---

## 1. Preconditions

| # | clause | status |
|---|---|---|
| **P1** | catch-32 label alignment | N/A — no probe numbers in this wave |
| **P2** | arm identity from the run's own artifacts | **PASS**, 12/12 runs record `arm` |
| **P2b** | EWC witness: `ewc_penalty_mean` nonzero on tasks ≥ 1 | **PASS**, 3/3 seeds |
| **P3** | checkpoint audit by loading | N/A — era checkpoints struck, §6 |
| **P4** | floor pair before any delta is interpreted | **FAILED, then diagnosed** — §4 |

**Three clauses had no mechanism when the contract was signed**, found before any
container started:

1. **The EWC witness could not be satisfied.** `ewc_penalty` was computed and
   added to the loss, then never recorded. A precondition with no artifact.
2. **`arm_provenance()` did not record λ.** An artifact could prove *"this is an
   EWC model"* but not *"this is EWC at λ=200"* — one level coarser than the
   cited configuration.
3. **Era checkpoints on the LSTM arms would fail their own audit**, because
   `_save_era_checkpoint` audits through `PLCM.load_era`, which is PLCM-specific.

**Resolution: fix two, strike one.** Fixes 1 and 2 were built and gated on the E4
regression printing **bit-identical (2/2 arms, max |Δ| 0.0e+00)** — on a pipeline
whose determinism is proven for that arm, that check is decisive rather than
indicative. Era checkpoints were struck (§6).

**A near-miss inside the fix, recorded because it is worth as much as the catch.**
Fix 2 was first written `getattr(self.ewc, "lam", nan)` from memory. The real
attribute is `ewc_lambda`; the guess would have written **`nan` into every
artifact** — a field that exists, passes a presence check, and certifies nothing.
That is strictly worse than the missing field it replaced: the missing field
fails loudly at read time, the `nan` fails silently forever. Caught by reading
the source instead of shipping the guess. **General form: a witness field's
default is a forgery permit. Witness recordings assert the attribute exists;
they do not default.**

---

## 2. The verified reference table

All twelve runs record their own arm, thread config, and λ where applicable.

| arm | n | AVG | sd | forgetting | cited | Δ |
|---|---|---|---|---|---|---|
| `plain_lstm` | 3 | **0.4319** | 0.0151 | 0.6755 | 0.4399 | **−0.80pp** |
| `ewc_l200` | 3 | **0.4329** | 0.0471 | 0.6733 | 0.4330 | **−0.01pp** |
| `mafc_off` | 3 | **0.4340** | — | — | 0.4304 | +0.36pp |
| adapters (`mafc` ON) | 3 | **0.9183** | — | — | 0.9331 | **−1.48pp** |

**Attribution, verified-minus-verified:** `0.9183 − 0.4340 =` **+48.4pp [±1.9pp]**,
superseding **+50.3pp**. Per-seed 47.3 / 47.1 / 50.9.

**Forgetting reduction:** **89.0% [83.6, 94.3]**, mean-of-ratios, superseding 92%.
The interval is ±5.4pp against the attribution's ±1.9pp because forgetting is a
*difference of differences* and the variance compounds — computed, not adjusted.

**`plain_lstm` and `mafc_off` differ by 0.20pp** and are **reported separately by
name, never averaged** (§5). The attribution uses `mafc_off`.

**Predictions on record:** `plain_lstm` within ±1.5pp of 0.4399 — **~70%, fired**
(0.80pp). EWC null surviving verification — **~85%, fired** (§3).

---

## 3. The EWC null, with its floor and its alibi

**EWC λ=200: 0.4329. `plain_lstm`: 0.4319. Difference +0.10pp, against a measured
floor of 1.44pp.** The null survives verification.

**Reported at the resolution the instrument supports, and not re-run to polish
it.** A pinned re-run would convert *"indistinguishable, floor stated"* into
*"indistinguishable, smaller floor stated"* — precision the claim does not use.
The claim is existence, not magnitude, and the direction is already safe.

**Why the floor-stated version is the better one to publish.** This is the result
most likely to draw a hostile *"you didn't tune it / your implementation was
broken"* read. The answer is now three-layered, all from artifacts:

1. **Active** — `ewc_penalty_mean` nonzero on every epoch of tasks ≥ 1, 3/3 seeds.
2. **Correctly configured** — `ewc_lambda: 200.0`, `ewc_param_filter: None`
   (whole-model, distinguishing it from `plcm_ewc`'s `lstm.`-only filter).
3. **Honestly bounded** — a 1.44pp floor stating what the comparison can resolve.

**A null reported at exactly its instrument's resolution is more credible than a
null polished to a decimal, because the polish invites the question of what else
was polished.**

**The witness did more than it was designed to.** The penalty grows monotonically
within and across tasks — 3.10e−05 at the start of task 1 to 4.60e−03 by the end
of task 4, **~150×**. Fisher terms accumulate per task and drift compounds against
more anchors, so the mechanism's theory *predicts* that trajectory. A broken
accumulation (Fisher overwritten instead of summed; penalty against only the
latest anchors) would pass a nonzero-everywhere test and **fail this shape**.
*Where a mechanism's theory predicts a trajectory, record the trajectory: a
witness that confirms dynamics is strictly stronger than one that confirms
existence, at the same recording cost.*

**What the witness does not show.** The penalty is ~1e−3 against a loss of order
1. It proves EWC is **active**, not that it is **binding**. Whether λ=200
meaningfully constrains this model is answered by the three verified AVGs against
`plain_lstm`, not by any property of the penalty term.

**Seed 42 is the high seed in all three arms measured** — `plain_lstm` 0.4493 vs
0.4247/0.4218; `mafc_off` 0.4447 vs 0.4384/0.4187; `ewc_l200` 0.4836 vs
0.4249/0.3903. The old λ sweep was **seed-42-only**, so its λ=200 value (0.4950)
carries that tilt. One more reason the sweep's shape is indicative, not
load-bearing.

**EWC per-seed spread is 9.3pp on one configuration**, against the sweep's 8.1pp
*across all five λ values*. The EWC λ=200 lesson, re-confirmed on verified runs.

---

## 4. What the floor pair taught about floors

**The contract's one blocking condition fired.** `plain_lstm` seed 42, relaunched
at the same HEAD: **0/15 cells bit-identical, max |Δ| 0.1060**, AVG 0.4493 vs
0.4349. The wave stopped.

**The diagnosis wave, 2×2:**

| | 4 threads | 1 thread |
|---|---|---|
| **`mafc`** | **15/15 exact** — two independent pairs, four runs | (not run) |
| **`plain_lstm`** | **0/15**, max \|Δ\| 0.1060 | **15/15 exact**, both 0.4349 |

**The cause is thread count.** Multithreaded BLAS reduces partial products in
completion order; floating-point addition is not associative; the same seed lands
on different bits. Pinning to one thread removes it entirely.

**The generalization error, stated plainly.** Earlier in this program I wrote that
*"the permuted-MNIST CPU path is bit-deterministic"* and used it to argue that
historical deltas were **attributable to code state with certainty**. That was
established on **one arm** and generalized to **the path**. It does not hold for
`plain_lstm`. The certainty claim is **retracted, not softened** — certainty
claims do not get hedged.

> **A measured property belongs to the exact configuration that measured it;
> every generalization is a new claim needing its own evidence.**

Third instance this week of the same shape: a property of the *benchmark* read as
a property of the *run*; a property of the *arm* read as a property of the *path*.

**The floor rule gains a term, measured rather than assumed:** determinism belongs
to **(arm × config × platform × thread-count)**. "Platform" always included
threading in principle; this is the first time it was tested.

**Forward rule adopted.** Future `plain_lstm`/`lstm_ewc` **floor pairs run
pinned** — 2.2× cost, and floors are cheap runs — while **comparison triples may
run at default threading carrying the measured 1.44pp floor**. Pin where you are
measuring the instrument; stamp-and-state where you are measuring the effect.
E16 inherits this: its MNIST floor pair runs pinned; its comparison rows carry
whichever floor their threading earns.

**One asymmetry observed and left open:** `mafc` is bit-exact at the *same* four
threads where `plain_lstm` is not, so the LSTM path exposes a thread-sensitive
reduction the MAFC path does not trigger. **Stated as an observed platform
property, mechanism not investigated.** Inventing a dispatch story from memory is
exactly the failure mode this week has been spent removing.

### CORRECTION appended 2026-08-16, after E16's HAR OFF floor pair

**The 2×2 above stands as measured. The mechanism paragraph attached to it is
RETRACTED, and the forward rule it produced is replaced.**

E16's HAR OFF arm was floor-paired at **one thread** and came back **0/15
bit-identical, max |Δ| 0.1069** — the two pinned runs disagreeing with *each
other* (AVG 0.5238 vs 0.5344) and sitting up to **22pp** from the same arm's
4-thread runs. Meanwhile that arm at **4 threads is bit-identical 15/15**, and
reproduces two-week-old runs of the same config exactly.

So HAR is **4-threads-deterministic, 1-thread-nondeterministic** — the exact
inverse of MNIST's `plain_lstm`. **No thread-count-monotone mechanism can
produce both**, and single-threaded BLAS has a fixed reduction order, so
"multithreaded reduction order" cannot explain a pinned run diverging from
itself. That sentence was ours and it was wrong.

**Second dead mechanism in one week, same shape as the first.** "Certainly code
state" (retracted §7) and "BLAS reduction order" (retracted here) were both *a
plausible mechanism attached to a true observation, generalized one
configuration beyond its evidence.* The observations survive both times; the
stories did not.

**Final form of the rule, which never contained a mechanism and so survives
intact:**

> **Determinism belongs to (arm × config × platform × thread-count), established
> per-pair, mechanism unknown until demonstrated.**

**The forward rule is replaced.** Not *"floor pairs run pinned"* — **pinning is
an axis, not a stabilizer.** A floor is measured **at the threading the reported
runs use**, or it bounds a configuration nobody reported. Consequences applied:
E16's verdict floors re-run at 4 threads (`jobs_e16floor_match`), the pinned
pairs retained as data on the pinning question, and the HAR comparison confirmed
sound — sweep at 4 threads, reference deterministic at 4 threads.

**Fourth control-scoping instance this week**, after benchmark→run, arm→path,
and one-arm→pipeline.

**Consequence for the drift question: there is no drift to explain.**
`plain_lstm`'s verified 0.4319 sits **0.80pp** from the cited 0.4399, **inside its
own 1.44pp floor**. The old number was never in conflict with the new one.

---

## 5. Naming (1b)

The bare-word audit returned ~20 hits, of which **two** were real — and that ratio
is the finding. The corpus uses "baseline" in **two senses**:

- **arm baseline** — `plain_lstm`, `mafc_off`. The hazard: two arms 0.20pp apart,
  interchangeable-looking, ~1pp of attribution riding on which a reader picks,
  and **no output inspection would catch the swap** (catch 30's defining
  property).
- **method baseline** — tier-0, the snapshot at ρ 1.237, era-head/era-prototypes.
  *The thing a method must beat.* Unambiguous and correct.

**Narrowed rule:** ***"vanilla" is never an arm name; "baseline" is fine for a
method and never for an arm.*** A blanket ban would have trained everyone to skip
the warning — *a rule that fires mostly on false positives will be ignored where
it matters.*

| hit | disposition |
|---|---|
| `docs/E16_lwf_prereg.md:135` "vanilla 0.4399" | **renamed** → `plain_lstm`, plus the not-`mafc_off` warning inline; row updated post-supersession |
| `docs/E8_prereg.md:116,174` "the 0.3736 baseline" | **annotated** — HAR/OFF `mafc_off`, forgetting |
| E9:23-24, E15:8, E16:32, ledger:72 | **no change** — method baselines, correctly used |

---

## 6. Deviation record

**Era checkpoints struck from 1a.** The contract asked for them on both arms;
`_save_era_checkpoint` audits by loading through `PLCM.load_era`, so on
`lstm`/`lstm_ewc` the clause cannot succeed and would have written checkpoints
that fail their own audit. Making the audit model-aware is a shared-trainer change
with blast radius across every experiment.

**Priced follow-up, named and unscheduled:** model-aware era-checkpoint audit,
~1–2h plus a regression pass. It becomes real only if a decomposition question
lands on an LSTM-family arm; nothing on the roadmap currently does.

**The honest consequence:** if these arms ever need decomposition, they **re-run**
under a model-aware audit. We do not pretend today's checkpoints would have been
audit-clean.

---

## 7. `fullrank_ref` and the storage table (1d)

**Restated exclusion.** `runs/fullrank_ref` is **not citable as an adapter
operand**. The earlier reasoning ("2.91pp from a bit-deterministic pipeline, no
third option") is **retracted** — `plain_lstm` now demonstrates 1.44pp of same-seed
spread at default threading, so a delta of that magnitude is not by itself
disqualifying.

**The exclusion never rested on the delta.** Two independent grounds stand:
**no `arm` field** (configuration unverifiable), and **a name placing it in the
low-rank study**. Unverifiable configuration is sufficient.

**Storage-frontier provenance pass:**

| cell | source | n | arm recorded |
|---|---|---|---|
| rank-32 0.69 | `runs/lowrank_r32` | 1 | no — **but `_seed1337`/`_seed2024` exist** |
| rank-64 0.87 | `runs/lowrank_r64` | 1 | no — **but siblings exist** |
| **full-rank 0.95** | **`runs/fullrank_ref`** | 1 | **no, and no siblings** |
| ER 0.79 | `runs/er_seed*` | 3 | clean |

**Price drops from ~9 runs to 3.** Both low-rank cells recompute from existing
sibling dirs at n=3 with an arm-provenance caveat; only the full-rank cell needs
new compute (`jobs_w3c_fullrank`, 3 seeds, recorded config).

**The orphanhood resolves the identity question as a side effect.** A lone
reference run with no seed set, beside two arms that have them, is the shape of a
*study reference* — consistent with its name. It was never an `e4_on` seed; it
was the storage study's own full-rank arm, cited into the adapter table by an
arithmetic coincidence. **Same run, two tables, one legitimate home** — and wave
3c gives the legitimate one a verified replacement so the illegitimate citation
retires without taking a real result down with it.

---

## 8. The era presumption

Catch 34 has **two independent instances**, both found in Wave 1, both
structurally identical — *one value from an exploratory file + two
unrecorded-config run dirs, averaged into a citation*:

- adapters **0.9331** = mean(`fullrank_ref`, `e4_on_seed1337`, `e4_on_seed2024`)
- EWC **0.4330** = mean(**0.4950 read from inside `plcm_ewc_sweep.json`**,
  `ewc_seed1337`, `ewc_seed2024`)

The second was found by **audit**, not by accident — the graduation the process
exists to produce.

**Catch 34 is not an incident; it is a pattern from a specific era of this repo**
— pre-arm-recording, pre-run-directory discipline. **Every number whose lineage
passes through that era is presumptively unverified until re-run or traced to an
artifact recording its own configuration.**

Notably, **the EWC number was right all along** (0.4329 vs 0.4330). Being right
is not the same as being verifiable, and the wave's job was the second.

---

## 9. LEDGER EDITS — drafted here, applied on sign-off

| claim | evidence | status |
|---|---|---|
| `plain_lstm` **0.4319** (n=3, permuted MNIST) | `runs/w1_vanilla_seed{42,1337,2024}`, arm recorded in all three | **supersedes 0.4399**; Δ −0.80pp, **inside** the arm's own 1.44pp 4-thread floor — the old number was never in conflict |
| `ewc_l200` **0.4329** (n=3) | `runs/w1_ewc_l200_seed*`; `ewc_lambda` 200.0 and `ewc_param_filter` None recorded; witness nonzero on tasks ≥1, 3/3 | **supersedes 0.4330** (Δ −0.01pp). Replaces the mixed-provenance triple, one member of which lived inside `plcm_ewc_sweep.json` |
| **the EWC null survives verification** | EWC 0.4329 vs `plain_lstm` 0.4319 = **+0.10pp**, against a measured **1.44pp** floor | **new, with alibi**: active (witness, growing 150×), correctly configured (λ + filter in-artifact), honestly bounded (floor stated). Reported at its instrument's resolution; **not re-run to polish** |
| determinism is **(arm × config × platform × thread-count)** | 2×2: `mafc` 15/15 at 4 threads (2 pairs); `plain_lstm` 0/15 at 4 threads, **15/15 at 1 thread** | **new, and a retraction**: "the path is bit-deterministic" was generalized from one arm and is **withdrawn**. `mafc`'s zero floor stands on four runs |
| `fullrank_ref` exclusion **restated** | grounds are **no `arm` field** + **name lineage**, not the delta | **narrowed**: the "no third option" certainty claim is retracted; unverifiable configuration remains sufficient |
| storage-frontier table | 3 of 4 cells are n=1 unrecorded-config; low-rank siblings exist | **flagged**, price **3 runs** (`jobs_w3c_fullrank`), not 9 |

**Unchanged and re-affirmed:** adapters **0.9183**, attribution **+48.4pp
[±1.9pp]**, forgetting reduction **89.0% [83.6, 94.3]**. None depended on the
retracted determinism claim; the attribution's interval was always cross-seed.

---

## 10. What the wave cost and bought

**Cost:** one writing day, ~20 CPU runs.

**Bought:** a reference table verified on both sides; two superseded numbers and
one confirmed-but-now-verifiable; the EWC null with a three-layer alibi; a new
term in the floor rule, measured rather than assumed; a witness that turned out
to certify dynamics rather than existence; four rules in `CLAIM_LEDGER`/`CLAUDE.md`;
and one certainty claim retracted before it reached a paper.

**Three errors of mine are in this record** — the arithmetic-match identification,
the too-quick recantation to "noise," and the determinism over-generalization.
All three were caught by instruments this program built, none by luck. That is
the argument Appendix E makes, and it is stronger for being made on our own
reference table than on anyone else's.
