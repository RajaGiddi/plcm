# MEMO — E12: ViT-B/16 on Split-CIFAR-100

**Status: COMPLETE. Hypothesis set fully adjudicated.** Contract:
`docs/E12_prereg.md` (signed 2026-08-03). Every number is from recorded script
output, computed in the execution environment (x86_64/Linux).

| hypothesis | verdict |
|---|---|
| **H-V1** decomposition generality (PRIMARY) | **MET** — reader share **91.77%** [90.88, 92.67], base arm, 95 cells |
| **H-V2** engagement prediction | **UNTESTABLE at this placement** — the adapter's best demonstrated effect (1.34pp) is below H-V2's own 5pp bar |
| **H-V3** class-IL (SECONDARY) | **MET** — masked reader share **97.86%** [97.27, 98.45]; corroborates, does not extend |
| **H-V4** depth/scale (descriptive) | F_enc **0.060** vs the LSTM/HAR range 0.052–0.112 — cross-family caveat attached |

**Branch (A).** The decomposition generalizes to a pretrained ViT at 20 tasks.

---

## 1. The result, in one table

Reader share — the fraction of forgetting carried by reader–encoder mismatch
rather than representational damage:

| regime | reader share |
|---|---|
| small, trained-from-scratch encoders (LSTM, MLP; 3 modalities) | **66–88%** |
| pretrained ViT-B/16, per-task heads (task-IL) | **91.8%** [90.9, 92.7] |
| pretrained ViT-B/16, shared head (class-IL) | **97.9%** [97.3, 98.5] |

**The more modern the setup, the more completely the reader carries the damage.**
The monotonicity across three regimes is ours; the class-IL row **corroborates
rather than extends**, because the recency-bias literature (BiC, WA, LUCIR;
Masana et al. TPAMI 2022) independently predicts readout-dominated failure there.
The claim rests on the task-IL row, measured where recency bias **cannot** be the
mechanism — per-task frozen heads, selection verified by P3b with both positive
controls fired.

**Two independent instruments arrive at the same sentence.** A probe sweep and an
exact decomposition, from opposite directions:

> **0.974** (frozen pretrained features) → **0.888** (fine-tuned features, refit)
> → **0.224** (deployed). *The features survived; the reader did not.*

**Consequence for the field's dominant recipe.** The freeze-the-backbone
consensus treats the trunk as the fragile component. F_enc = 0.06 says the trunk
was nearly fine all along — freezing works largely because it **pins the reader's
target**, not because it protects endangered features.

---

## 2. Method scope — what E12's model is

The ViT path reads **post-norm CLS → head** and bypasses the memory bank, GGC
composition, and the learned output gate. P3a registers that tensor as the
deployed feature; wrapping it in `o_t·tanh(c′)` would measure a hybrid whose
readout no reader of the ViT literature would recognise.

So **E12 measures forgetting in a plain fine-tuned ViT**, with PLCM contributing
task bookkeeping and head routing only. What is shown to transfer is **the
instrument, not the architecture**. H-V4's comparison is therefore cross-family
and the caveat travels with the number.

---

## 3. H-V1 — decomposition generality (PRIMARY): MET

`runs/e12_decomp_v2/`, via `scripts/e12_hv1.py` → `runs/e12_hv1_v2.json`.
5 seeds × 19 old tasks = 95 cells per arm.

| | base | adapt |
|---|---|---|
| F_total | 0.7154 [0.6939, 0.7369] | 0.7230 [0.7018, 0.7442] |
| **F_enc** | **0.0599** [0.0533, 0.0665] | **0.0560** [0.0504, 0.0615] |
| **F_read** | **+0.6723** [+0.6511, +0.6935] | **+0.6821** [+0.6606, +0.7037] |
| **reader share (raw)** | **91.77%** [90.88, 92.67] | **92.29%** [91.49, 93.09] |
| reader share (adjusted) | 94.01% | 94.26% |
| instrument gate R | +0.0168 [+0.0140, +0.0196] PASS | +0.0151 [+0.0124, +0.0178] PASS |
| cells kept | 95/95 | 95/95 |

**H-V1 MET**: CI lower bound **+90.88%** against the 50% bar.

Task 0: ceiling **0.9236** → deployed **0.2240** → **refit 0.8536**. Across all 95
cells the refit averages **0.8884** (range 0.780–0.962) against chance 0.2000.

Base vs adapt, F_total: Welch **t = −0.494, df = 188** — statistically
indistinguishable.

---

## 4. H-V2 — the engagement prediction: UNTESTABLE at this placement

| benchmark | adapters off | adapters on | delta | Welch |
|---|---|---|---|---|
| permuted-pixel (framework predicts **engagement**) | 0.2790 | 0.2924 | **+0.0134** | t = +1.14, df 3.7 |
| semantic Split-CIFAR-100 (predicts **inertness**) | 0.2584 | 0.2505 | −0.0079 | t = −0.63, df 3.4 |

The efficacy control does not establish the adapter can help. On the terrain
where the framework says it *should* engage, it buys **+1.34pp, not significant**.

**The decisive problem is dynamic range: the adapter's best demonstrated effect
(1.34pp) sits below H-V2's own 5pp bar.** "Contributes ≤ 5pp" cannot be read as
confirmed inertness when the adapter has never been shown to produce 5pp of
anything. The semantic arm's −0.79pp is *consistent with* the engagement
condition and is stated as consistency, **never as confirmation**.

**Cost, stated plainly:** the paper loses its cleanest prospective sentence —
"the theory predicted its own boundary on foreign terrain". The patch-embedding
placement (a weaker adapter class than paper 1's pixel-space maps, declared in
the contract §1) goes in limitations as the honest suspect.

---

## 5. H-V3 — class-IL (SECONDARY): MET, corroborative

Read under `docs/` protocol with semantics fixed **before** any number was
visible. `runs/e12_decomp_cil_v3/`, 3 seeds × 19 tasks = 57 cells.

**Preconditions (all six, printed before the hypothesis):**

| | |
|---|---|
| P3a | max \|Δ\| **0.0e+00**; control fired (Δ = 1.112e+02) |
| catch-32 label alignment | loader order == dataset indexing, **bitwise, 64 samples — PASS** |
| identity residual | **0.0e+00** |
| instrument bias | pooled R **+0.0167** [+0.0130, +0.0204] — PASS |
| per-cell guard | **57/57 kept**, 0 excluded |
| reproduction floor | aggregate ~0.4pp, per-cell ~15pp — quoted with every per-cell claim |
| P3b | **N/A by construction** — single shared readout, so "which head" has no referent |

| quantity | value |
|---|---|
| acc_ceiling | 0.9448 [0.9380, 0.9517] |
| acc_orig (deployed, unmasked all-seen) | **0.0017** |
| **masked refit** (within-task, chance 0.200) | **0.9411** [0.9331, 0.9491] |
| unmasked refit (all-seen, chance 0.010) | 0.6393 [0.6251, 0.6534] |
| F_enc | **0.0205** [0.0149, 0.0261] |
| F_read | **0.9394** [0.9311, 0.9477] |
| **masked reader share (VERDICT COLUMN)** | **97.86%** [97.27, 98.45] |

**Verdict, in the pre-committed words:** *Class-IL agrees with task-IL, as the
recency-bias literature would predict; because that literature already attributes
class-IL failure to readout bias, this arm corroborates rather than extends the
task-IL result, which is where our claim rests.*

**Within-task information is undamaged.** Masked refit **0.9411** against a
ceiling of **0.9448** — F_enc 0.0205, a third of task-IL's already-small 0.06 —
while the deployed system sits at **0.0017**. Total functional collapse over a
nearly perfect substrate: the cleanest instance of the thesis in the program.

### The unmasked column: a joint-access reader CEILING, not a repair target

The 100-way probe is fit **once over all 20 train splits jointly**, which grants
it a resource the deployed class-IL system never has: simultaneous access to
every task's data. So it **upper-bounds what any reader could recover** — which
is what makes it the clean measure of recoverability-plus-discriminability, and
exactly why it **must never be quoted as what a cure would achieve**. A
storage-honest repair has no such access. *(The caveat lives in
`scripts/e12_decompose.py` beside the computation, not only here — code outlives
prose.)*

**The 0.639 → 0.941 gap decomposes the class-IL burden itself.** Even a reader
with impossible joint access recovers only 0.64 across 100 classes, so roughly a
third of class-IL's difficulty is **genuine cross-task discriminability in
feature space**, and the remainder is readout misalignment that a per-task oracle
erases. Descriptive, clearly bounded — and a measurement the recency-bias
literature discusses without having.

---

## 6. H-V4 — depth/scale (descriptive, no gate)

F_enc **0.060** (task-IL) / **0.021** (class-IL) against the post-E11 LSTM/HAR
range **0.052–0.112**. Encoder damage does **not** grow with capacity; on the
largest model in the program it is the smallest. Cross-family comparison — the
LSTM values come from arms where the full PLCM machinery participates and E12's
do not.

---

## 7. Preconditions and instrument

**Build gate P0: 6/6.** Fingerprints, all computed where executed — class order
`4da9de3736a6`, shift spec `24b6131095fd`, ViT weight hash `3416d11351bb`.
B1/B6 fingerprints are **identical on arm64/Darwin and x86_64/Linux**, closing
the E10 cross-environment lesson by measurement rather than hope. Seeds 7 and
1234 registered later to meet the five-seed lock (`ecaa61aa1c21`,
`5e00e4c7a8cf`), also identical across environments.

**Frozen-probe arm** (the trivial use of the assumed resource, catch 24):
per-task DIAG **0.9741** (min 0.8940), forgetting **0.0000 by construction**.

**P1** (scale-free): trainable mean DIAG **0.9334** vs frozen 0.9741 → gap
**+0.0407**, inside the 5pp margin **by 0.93pp — a marginal pass, stated as one.**

**P2(a)**: mean task-0 drop **0.7233** (0.538 / 0.778 / 0.854) vs the 0.15 bar.
**P2(b)** repeat-task control: **0.0040** (0.0065 / 0.0000 / 0.0055) vs 0.05 —
so **drift accounts for 0.55% of the collapse; 99.45% is task-sequence
interference.**

**Instrument proofs, every arm:** P3a max |Δ| **0.0e+00**; identity residual
**0.0e+00**; fp16 reload delta **0.00000**; R inside bar everywhere.

**Reproduction floor** (catch 19, identical relaunch):

| arm | AVG \|Δ\| | forgetting \|Δ\| | **max cell \|Δ\|** |
|---|---|---|---|
| base | 0.0042 | 0.0049 | **0.1500** |
| adapt | 0.0036 | 0.0071 | **0.1680** |

Aggregates reproduce to ~0.5pp; **single cells move up to 16.8pp**. Pooled-with-CI
is primary throughout the ViT section; per-seed values are shown and never leaned
on. H-V1's effect is ~50× the aggregate floor over 95 cells.

---

## 8. PROCESS APPENDIX — what it took to trust these numbers

### 8.1 The retraction chain (catch 32)

**E12 was first reported as the opposite of its result.** The decomposition
script extracted era features in one pass and θ_T features in a second pass over
a `shuffle=True` loader, then paired the second pass's features with the first
pass's labels. The θ_T probe trained on randomly relabelled data and read chance.

| | reported (defective) | corrected |
|---|---|---|
| refit at θ_T | 0.198 (≈ chance 0.200) | **0.888** |
| F_enc | 0.7505 | **0.0599** |
| F_read | −0.0183 | **+0.6723** |
| reader share | −4.89% | **+91.77%** |
| conclusion | branch (C): encoder-side architecture boundary | **branch (A): reader-side, strongest in the family** |

`channel_decomp.load_task_data` had **already solved this**, with the fix in its
docstring — *"produced R ~ −0.63 and was misread as optimizer underfitting"*. A
fresh extraction path re-implemented the job without inheriting it.
**Rule: a new data path routes through the audited loader, or the contract states
why it cannot; any parallel implementation ships a bitwise equivalence test.**

**What caught it: one number that did not fit.** The class-IL arm's deployed head,
restricted to a task's own classes, scored 0.7075 on θ_T features while a probe
*fit to convergence* on those same features scored 0.1982. A freshly fit linear
probe cannot lose to a fixed linear head by 50pp on identical inputs.

**The corrupted result was more dramatic and more publishable-sounding than the
truth** — an architecture boundary, a clean sandwich figure, a tidy narrative. It
survived two rounds of scrutiny, a full memo draft, and an odds recalculation.
What killed it was refusing to write around the anomaly.

**And the control cited against it could not have caught it.** "The same recipe
on era checkpoints gets 0.95" was offered as ruling out a probe bug; era features
and era labels came from the *same* pass, so an upstream misalignment cancels
there and surfaces only at θ_T. It was a control against *encoder-quality*
explanations, never against *label-alignment* explanations — and it was allowed
to stand for both. **State the failure mode a control would catch before citing
it.**

### 8.2 Three generations, all retained

| generation | status |
|---|---|
| `e12_decomp/` (v1) | **defective** — label misalignment; kept as the record |
| `e12_decomp_v2/`, `e12_decomp_cil_v2/` | corrected loader; class-IL **incomplete** (no unmasked column) |
| `e12_decomp_cil_v3/` | **to protocol** — both label spaces, alignment assert printed |

### 8.3 The protocol's first save

H-V3's read protocol forced the masked/unmasked semantics **before any number was
visible**, and caught that the v2 class-IL run had masked *deployed* accuracy
standing where an unmasked *probe* belonged. The defective object was not a wrong
value but a **wrong kind** — floor and probe drawn from the same label space,
manufacturing reader share by construction. **Rebuilt rather than read.**

Third variant of the manufactured-comparison family: catch 31 (control without a
counterfactual), catch 32 (parallel loader), and this (floor and probe sharing a
masking).

### 8.4 Catches logged this experiment

- **Catch 30** — the strength of a gate's pass is not evidence it ran on the right
  configuration. `E12_CFG` omitted `--task-heads`/`--no-adapters`; the "base arm"
  ran adapters-ON with one shared head, and P2(a) passed at **5× its bar**. The
  defect produced a *comfortable* number, not a suspicious one.
- **Catch 31** — a gate that cannot READ. The efficacy control ran only the
  treatment arm, so "the adapter helps" had no counterfactual.
- **Catch 32** — a hardened helper's value is lost the moment a new script
  re-implements its job (§8.1).
- **P3b's positive control** fired on the amendment it was written for: the ViT
  path routed heads by hint *regardless of the flag*, so every live assert passed
  vacuously. **A control that fires on the known-broken case is the only evidence
  a flag does anything at all.**
- **Dynamic range** (new standing rule) — *a hypothesis bar must sit inside the
  instrument's demonstrated range, or the hypothesis is unreadable regardless of
  the data*, and the range must be demonstrated **before** the bar is set.

### 8.5 Infrastructure

The precondition re-run took four launches; three produced **no artifact at all**
(attached client killed; `--detach` with a blocking entrypoint cancelled;
~3h queued for an L4 that never came). Fixed structurally: `.spawn()` so no live
client is required, and `memory=16384 → 8192`, which Modal itself named as the
constraint. One preemption on `e12pb`, restarted from scratch — not a correctness
event (catch 19's preemption diagnosis was retracted; the pattern is
seed-correlated).

---

## 9. Calibration

| prediction | outcome |
|---|---|
| frozen-probe DIAG clears 0.85 comfortably (~90%) | **fired** — 0.974 mean |
| P2 passes, drop ≥ 0.15 (~60%) | **fired**, badly conservative — cleared at ~5× |
| task-0 drop lands 0.25–0.50 | **missed** — 0.7233; reasoning (frozen-head protection) mechanistically wrong |
| H-V1 share ≥ 0.50 (~60%, blind) | **fired** |
| H-V1 share 80–90%, F_enc 0.05–0.15 | **fired** — 91.8% / 0.060; *made after seeing 11 corrected cells, so not evidence of improved forecasting* |
| H-V3 masked ≥ 0.50 (~80%, blind) | **fired** |
| H-V3 masked ≥ 0.85 (~55%, blind) | **fired** — 97.9% |

---

## 10. LEDGER EDIT — drafted here, applied second

> Catastrophic forgetting can be eliminated at the input path when the shift is
> expensive to absorb and the encoder has something to lose. Where the input path
> fails to engage, forgetting is carried predominantly by reader–encoder
> mismatch, not representational damage: 66–88% on small trained-from-scratch
> encoders (LSTM, MLP; three modalities), and **91.8% [90.9, 92.7] on a
> pretrained ViT-B/16 under full fine-tuning across 20 tasks** — three
> architectures, one instrument, verified exact on all of them. Old-task features
> survive fine-tuning nearly intact (F_enc 0.06; refit probes 0.888 vs 0.974
> pretrained); the readout is what breaks. Certified-honest generative refit
> recovers the channel on sensor data (0.97, forgetting −81.6%); exact input
> correction is harmful; the reader-channel claim's prior "regime boundary" is
> **RETRACTED** — the boundary was an instrument artifact (label misalignment),
> and its retraction is part of the record.

**Carried forward unchanged:** the H-X1/H-X2 clauses and all standing caveats.
**Added on the strength of §5:** *…and the same split holds under
class-incremental evaluation, where prior work independently attributes failure
to readout bias.*
