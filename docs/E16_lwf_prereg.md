# Pre-Registration: E16 — Learning without Forgetting as a Matched Baseline

**Status: SIGNED**, 2026-08-16, after contract review. Nine issues raised, nine
accepted; issues 1–3 were blocking and are resolved in §§1, 3, 4b below. A tenth
was found while building the launcher and is resolved in §0. **One day of CPU —
not GPU** (issue 10). Runs in parallel with manuscript drafting; blocks nothing.

**Why this exists.** LwF (Li & Hoiem, 2017) is the baseline every reviewer will
name when they read synaptic bridging, because both use a stored model to supply
labels. Without a measured comparison, our differentiation is a conceptual
argument; with one, it is a table row.

**What it is not.** Not a hypothesis test. No bar, no kill criterion, no branch
structure on the *result* — the output is a set of numbers that enter the
comparison tables. A contract is still required because **a baseline run under a
weaker protocol than our own method is worse than no baseline**.

**Margin note, carried from the review and load-bearing throughout:**
> *Low forgetting is what "LwF working" looks like, which is precisely why it
> cannot be the selection metric.*

---

## 0. Review amendments — what changed and why

| # | issue | resolution |
|---|---|---|
| 1 | selection metric unspecified → λ→∞ degenerate arm wins on forgetting | **select on AVG**; DIAG printed per λ; competence floor as a P1-shaped gate (§1) |
| 2 | §3's decomposition had no input — needs era, not final, checkpoints | `--era-checkpoints` on **every** LwF run incl. sweep; audit table (§3) |
| 3 | `--mafc-arm lwf` already means an H3 ablation control | new arm **`lwf_lh17`**; arm-identity witness asserted from artifacts (§4b) |
| 4 | no clause → job → artifact table | §7, built before launch |
| 5 | catch 24 not applied | snapshot baseline (ρ = 1.237) enters the comparison table (§2b) |
| 6 | storage is the stated differentiator but not measured | storage row, in bytes, in §1 |
| 7 | correctness certificate had no positive control | λ_LwF = 0 ≡ vanilla, **separate RNG generator** for the teacher pass (§4c) |
| 8 | checkpoint audit in the weak "it loads" form | matrix-reproduction form via `load_era` (§4) |
| 9 | HAR arm pairing unspecified | pinned to the **OFF** arm (§2) |
| 10 | draft said "one day of **GPU**" | **CPU** — every arm E16 is compared against is an existing CPU run (below) |

**Issue 10, found while writing the launcher entries.** The draft budgeted GPU.
Every arm E16 is compared against — the MNIST input-path table and the HAR OFF
arm — is an existing **CPU** run, and `modal_runner.py` already encodes the rule:
*every arm of a comparison runs on the same device as the arms it is compared
against.* Catch 19 makes it load-bearing, because the reproduction floor is a
property of (seed × config × platform × BLAS); a device swap on one side is the
infrastructure form of an asymmetric relaunch, which biases the delta in an
unknown direction. E16 is therefore **absent from `GPU_EXPERIMENTS` and
`GPU_JOB_PREFIXES` by design**, and the launcher comment says so, so that a
future "why is this slow?" edit cannot quietly undo it.

Issue 1 is the one that would have produced a **wrong number rather than a
missing one**, and wrong in our favour. Its sibling is catch 30: a defect that
yields an emphatic reading sails, because emphatic is what passing looks like.

---

## 1. The fairness constraint (the contract's entire purpose)

LwF receives **exactly what our method received**, on every axis:

| Axis | Our arms | LwF arm |
|---|---|---|
| Seeds | 5 headline / 3 control | **5 headline / 3 sweep** |
| Epochs per task | matched to the arm compared against | **matched** |
| Optimizer, LR schedule, batch size | as recorded | **identical** |
| Task sequence, class order, shifts | fingerprinted | **identical fingerprints** |
| Test split | full, no subsampling | **full** |
| Reproduction floor | measured per batch | **measured for this arm** |
| **Storage at deployment (bytes)** | **measured** | **measured** |

**Storage row (issue 6).** If storage is the differentiator §5 claims it is, it
is a measurement, not prose. Recorded in bytes for: LwF (one stored model copy),
synaptic bridging (generator + era metadata), ER-100 (buffer), the snapshot
baseline (one model per task), adapters (per-task params). Computed by
`scripts/e16_storage.py` from the artifacts themselves, never from parameter
counts done by hand.

### Hyperparameter budget

λ_LwF sweep: **{0.5, 1.0, 2.0, 5.0} at 3 seeds each**, temperature T fixed at
the `configs/default.yaml` value (2.0) and reported. Best configuration then run
at **5 seeds** for the headline row. **The full sweep is reported in the
appendix, not just the winner.** Our own method's comparable sweep is cited
beside it; where we never swept a comparable axis, the caption says so.

### SELECTION METRIC — pinned (issue 1, blocking)

**Selection is on AVG (final average accuracy). Never on forgetting.**

LwF's canonical degenerate mode is that large λ_LwF prevents the model from
learning new tasks at all: **forgetting → 0 because nothing was learned to
forget.** A forgetting-selected sweep picks that corner, and §2 calls the
forgetting comparison "the comparison that matters" — so the artifact would have
entered under §5's LwF-wins branch at pre-committed full prominence.

Three mechanical consequences:

1. **AVG is the selection quantity.** It penalises both failure modes jointly.
2. **DIAG is printed for every λ in the sweep table**, so the degenerate corner
   is visible rather than inferred.
3. **Competence floor, P1-shaped:** a λ whose mean DIAG falls more than **5pp
   below the vanilla arm's DIAG** is a **failed run, not a candidate** — it is
   excluded from selection and reported as excluded, with its DIAG, in the
   appendix table. An exclusion rate is a reportable fact about the sweep.

### The EWC lesson, applied prospectively

Report across-λ spread against across-seed spread. If λ-variation sits inside
seed noise, that is a finding about the benchmark, not a claim about LwF's
ceiling — exactly as EWC's λ=200 "optimum" dissolved.

---

## 2. Where it runs

| Benchmark | Arm compared against | Why |
|---|---|---|
| Permuted MNIST (LSTM) | adapters, ER-100, EWC, vanilla — **all four re-cited from artifacts, §2a** | the headline input-path table |
| UCI HAR (subject-disjoint, E10 construction) | synaptic bridging on the **OFF arm** (forgetting 0.4812 → 0.0886) | **the comparison that matters** |

**HAR pairing pinned (issue 9):** LwF runs the **OFF-arm configuration**
(`--no-adapters`), because bridging's 0.4812 → 0.0886 is the OFF arm. Any other
pairing moves two variables.

Split-CIFAR-100 is **excluded**: the ViT/ResNet arms are diagnostic, not method
arms, and a method baseline there invites a method-comparison reading of a
section that makes no method claim. Stated in the paper, not left to inference.

### 2a. Reference-number provenance audit (catch 21) — RUN, and it found something

The draft cited four MNIST numbers with no artifact. Audited:

| cited | status | resolution |
|---|---|---|
| ER-100 **0.7916** | **CONFIRMED** | mean of `runs/er_seed{42,1337,2024}/mafc_results.json` = 0.7916 (0.7918 / 0.7852 / 0.7977) |
| `plain_lstm` **0.4399** | traced, **memo-only**; **superseded by Wave 1a** | `runs/drift/MEMO.md` L81. Reconstructs as mean(`plcm_ewc_sweep.baseline_lstm` 0.47558, `lstm_seed1337` 0.4375, `lstm_seed2024` 0.4067) — catch 34's shape again. **Renamed from "vanilla"**: this is `plain_lstm`, NOT `mafc_off` (0.4304), and the attribution uses the latter. |
| EWC **0.4330** | traced, **memo-only** | `runs/drift/MEMO.md` L82 (λ=200, seeded). Local `runs/ewc_seed*` hold **2** seeds averaging **0.4020** — a *different* configuration. |
| adapters **0.9331** | **RESOLVED AND SUPERSEDED -> 0.9183** | This row's original finding ("does not reproduce locally") was correct but incomplete. Re-run under a recorded config: seed 42 reads **0.9179**, verified 3-seed mean **0.9183** (`runs/e4on_v2_seed*`). The seed-42 member of the old average was `runs/fullrank_ref`, now **positively excluded** — 2.91pp from a pipeline measured bit-deterministic (15/15, max \|Δ\| 0.000000). See `runs/fullrank_ref/PROVENANCE.md` and the ledger's Appendix E chain. |

**Lock:** none of these four appears in a caption until it resolves to run
artifacts audited by the completion definition. This is the E11 lesson at the
level of *cited numbers* rather than checkpoints — "a number appears in a memo"
is not "a number has an artifact," the same cheap-signal substitution.
`jobs_e16ref` (§7) re-runs whatever the audit cannot resolve.

### 2b. Catch 24 — the trivial use of LwF's assumed resource

**Resource LwF assumes:** a stored copy of the previous model.
**Dumbest use of exactly that:** keep the snapshot and run old tasks through it —
already measured in this program at **ρ = 1.237, above every ceiling in it.**

The snapshot baseline **enters E16's comparison table as a row**, at its measured
storage cost. This also corrects §5's frame, which the review improved:

> **Both LwF and bridging compete against "just keep the model." The axis that
> separates them is storage and deployment constraints, not accuracy.**

---

## 3. What is measured

Standard reporting for both benchmarks: per-task accuracy matrix, AVG, RET,
DIAG, forgetting — same quantities, same scripts, as every other arm.

**Additionally, the scientifically interesting part:** the **three-channel
decomposition run on LwF's checkpoints**. LwF constrains outputs on new data;
our finding says the damage is readout-side. Does LwF reduce `F_read`, `F_enc`,
or both? No bar, descriptive only — but if LwF's protection concentrates in one
channel, that is a mechanistic result about a method the field already uses, and
it belongs in §6 ("what readout aphasia explains"), not in a baseline table.

### ERA CHECKPOINTS — required, and the reason the draft was unrunnable (issue 2)

The draft said the decomposition runs on "LwF's **final** checkpoints." It
cannot. `F_enc` and `F_read` are defined against `acc_ceiling`, which is measured
**at the era checkpoint** — the model as it stood at the end of task *k*. Final
checkpoints alone produce no ceiling and therefore no decomposition.

**Every LwF run carries `--era-checkpoints`, including the sweep arms** — §1's
own logic is that any configuration may become the compared one, and catch 20's
corollary is to save checkpoints for both arms whenever an arm might later be a
baseline. Storage is trivial at LSTM scale.

**Checkpoint audit table — verified before the decomposition is read:**

| arm | seeds | era ckpts expected | audited by |
|---|---|---|---|
| `lwf_lh17` MNIST headline | 5 | 5 × (T−1) | `scripts/audit_checkpoints.py`, criterion `PLCM.load_era` succeeds |
| `lwf_lh17` HAR/OFF headline | 5 | 5 × (T−1) | same |
| `lwf_lh17` sweep (both benchmarks) | 4λ × 3 | 12 × (T−1) each | same |

---

## 4. Instrument and protocol (inherited, no amendments except where noted)

- P3a capture-point identity with its positive control, re-fired in this script.
- Catch-32 label-alignment assert printing PASS before any probe number; feature
  extraction routes through `channel_decomp.load_task_data`, not a fresh path.
- **Checkpoint audit in MATRIX-REPRODUCTION form (issue 8):** the criterion is
  that a reloaded era checkpoint **reproduces the recorded accuracy matrix**,
  not that it loads. Loading goes through `PLCM.load_era` so `task_stats` is
  restored — `state_dict()` omits it and `forward()` gates on it (catch 29).
- Per-cell guards applied per cell, forced-inclusion printed beside.
- Reproduction floor measured for this arm before any delta is read; the
  comparison partner's floor must exist on the same platform, or **both** are
  relaunched (never one side — an asymmetric relaunch biases the delta).
- Verdict strings and table values computed from printed arrays.
- Provenance: every number in the memo from recorded output.

### 4b. ARM IDENTITY (issue 3, highest severity)

**`--mafc-arm lwf` already exists in this repo and is NOT Li & Hoiem.** It is
*"both terms but RAW probe inputs (H3 control)"* — a MAFC ablation
([scripts/train.py:280](../scripts/train.py#L280)). A launcher entry written from
memory produces a MAFC control arm wearing the field's baseline's name, and every
downstream table inherits it. Catch 30, pre-loaded before any code exists.

- **New arm name: `lwf_lh17`.** The citation lives in the name so a future
  collision of the same kind cannot form silently.
- **Arm-identity witness, asserted from the run's own artifacts:** every epoch
  dict must contain a **nonzero `lwf_distill_loss`**. Verified at read time by
  `scripts/e16_verify_arm.py`, never from the config — `train.py` overwrites the
  config from CLI flags, which is how catch 30 happened.
- The witness is **non-tautological**: it fails on a run where the distillation
  term was never engaged, which is exactly the misconfiguration to fear.

### 4c. POSITIVE CONTROL for the correctness certificate (issue 7)

§6's "parity against a reference implementation" is a gate that will only ever
have passed. It ships with a case where it **must fail**:

> **λ_LwF = 0 must reproduce the vanilla arm exactly.**

Non-tautological: it fails if the distillation term is wired into the wrong
branch, or scaled where it should be added.

**The RNG caveat is part of the control, not a footnote.** The teacher's forward
pass can consume draws from the global generator and change the student's
trajectory for an entirely benign reason — the control would then fire for the
wrong cause and be waved through as "expected nondeterminism." **The teacher pass
uses a separate `torch.Generator`, and the λ=0 run must be bit-identical to the
vanilla arm's recorded matrix.** Anything less is not this control.

---

## 5. How the result is used, decided in advance

**If LwF underperforms bridging on HAR/OFF:** the differentiation paragraph
points at the numbers, and the mechanism sentence is stated — LwF distils on
*new* data during training to preserve old behaviour; bridging refits *only the
readout*, at read time, on regenerated old-distribution inputs, with no gradient
to the trunk.

**If LwF matches or beats it:** reported at full prominence in the comparison
table and in the abstract's framing. The diagnosis is untouched either way —
where forgetting lives does not depend on which repair wins — and the cure
section is rescoped to "a data-free readout repair competitive with LwF while
retaining no raw data," with storage and privacy as the differentiator rather
than accuracy. **Written now so the writing cannot be tempted later.**

**If the λ sweep is noise-dominated:** reported as such, exactly as EWC's was,
with the seed-spread comparison printed.

**Frame correction (§2b):** in all three branches, both methods are also priced
against the snapshot baseline, which beats both on accuracy and loses to both on
storage. The paper's sentence is about the frontier, not about a winner.

### 5b. Optional fourth arm — LwF + bridging (contingent on the timebox)

They are **composable, not alternatives**: LwF changes training, bridging repairs
the readout at read time on an unchanged trunk. A repair that still helps *on top
of* the field's standard method is a stronger sentence than one that merely beats
it. Nearly free once both exist. **First thing cut if the day runs long** — it
extends the result rather than securing it.

---

## 6. Locks

- Matched budget on every axis in §1; any deviation stated in the caption.
- Full sweep reported, not just the winner; exclusions counted.
- Timebox one day. If the implementation is not verifiably correct within it —
  §4c's control plus reference parity — **report the state and stop.** A baseline
  we are not confident in is not admissible.
- No new claims about LwF beyond the measured numbers.
- Selection on AVG. Never on forgetting. (§1)
- Era checkpoints on every run. (§3)

---

## 7. CLAUSE → JOB → ARTIFACT (catch 33 — built before launch)

| clause | launcher entry | artifact |
|---|---|---|
| §1 λ sweep, MNIST (4λ × 3 seeds) | `jobs_e16sweep_mnist` | `runs/e16_mnist_lam{λ}_seed{s}/mafc_results.json` |
| §1 λ sweep, HAR/OFF (4λ × 3 seeds) | `jobs_e16sweep_har` | `runs/e16_har_lam{λ}_seed{s}/mafc_results.json` |
| §1 headline, MNIST (winner, +2 seeds → 5) | `jobs_e16head_mnist` | `runs/e16_mnist_head_seed{s}/` |
| §1 headline, HAR/OFF (winner, +2 seeds → 5) | `jobs_e16head_har` | `runs/e16_har_head_seed{s}/` |
| §1 competence floor / DIAG per λ | *(analysis)* `scripts/e16_sweep.py` | `runs/e16_sweep.json` |
| §1 storage row, in bytes | *(analysis)* `scripts/e16_storage.py` | `runs/e16_storage.json` |
| §1 reproduction floor, this arm | `jobs_e16floor` | `runs/e16_{mnist,har}_floor_rep2/` |
| §2a unresolved reference numbers | `jobs_e16ref` | `runs/e4_on_seed42/`, EWC/vanilla per-seed dirs |
| §2b snapshot baseline row | *(analysis)* `scripts/e16_snapshot_row.py` | `runs/e16_snapshot.json` |
| §3 decomposition on LwF checkpoints | `jobs_e16dec` | `runs/e16_decomp/{bench}_{s}.json` |
| §3 checkpoint audit | *(analysis)* `scripts/audit_checkpoints.py` | `runs/e16_ckpt_audit.json` |
| §4b arm-identity witness | *(analysis)* `scripts/e16_verify_arm.py` | `runs/e16_arm_witness.json` |
| §4c λ=0 ≡ vanilla positive control | `jobs_e16ctrl` | `runs/e16_{mnist,har}_lam0_seed42/` |
| §5b LwF + bridging (optional) | `jobs_e16combo` | `runs/e16_combo_har_seed{s}/` |

**Every clause has a path. No clause is a sentence.**

> **AMENDMENT 2026-09-15 — catch 33 in the prereg itself.** Four launcher /
> script entries in this table were never built: `scripts/e16_sweep.py`,
> `scripts/e16_storage.py`, `scripts/e16_verify_arm.py`, and
> `scripts/e16_snapshot_row.py`, with their artifacts `runs/e16_sweep.json`,
> `runs/e16_storage.json`, `runs/e16_arm_witness.json`, `runs/e16_snapshot.json`.
> The sweep, the selection on AVG with the competence floor, and the
> determinism map were computed and are recorded as prose and tables in
> `runs/MEMO_e16.md` §§2, 5b; the arm witness and the selection are now
> implemented from artifacts in `scripts/s72_row.py` (which runs them for the
> `har_subject` construction). The storage row and the snapshot row were not
> computed. **A table naming a script is not a script**; this note is the
> record so the table cannot be read as evidence that they ran.
>
> **Second amendment, same date — the HAR arm's construction.** §2 pins "UCI
> HAR (subject-disjoint, E10 construction)"; `modal_runner.E16_HAR` wrote
> `--benchmark har`, the E5 shared-window construction, and every HAR run in
> this contract records `benchmark: 'har'` in its `arm` field. Every
> within-E16 comparison was uniform on `har` and stands. The §7.2 comparison
> against bridging (a `har_subject` number) was never comparable as planned;
> it was closed on 2026-09-15 by relaunching both arms on `har_subject` with
> uniform flags — the full λ sweep again, λ\* = 1.0 there — see
> `runs/MEMO_c0deg.md` §6 and `runs/s72_row.json`. Bridging > LwF **fired**.

---

## 8. Predictions on record (restoring the series — 8 prior entries, 5 misses)

| prediction | odds |
|---|---|
| λ-sweep noise-dominated (across-λ spread < across-seed spread) | **~40%** — LwF's λ genuinely matters more than EWC's in most reports, but our benchmarks are small |
| LwF's protection concentrates in `F_read` rather than `F_enc` | **~65%** — distillation constrains outputs, and outputs are the reader's territory; if it fires, §6 gains its fifth practice |
| Bridging ahead of LwF on HAR/OFF | **~55%** — unchanged from the draft |

*Calibration note:* E14 ran three-for-three and all three **under**-confident,
with the stated reason for doubt empirically backwards. Treat the intervals as
wide and expect the miss to be in the model, not the number.
