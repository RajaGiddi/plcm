# MEMO — catch 28 supersession (E11)

**Status: IN PROGRESS.** Steps 1–3 complete on both corpora. Step 4 (recompute
chain) is ready to run and **must run on Modal** (see "Where the recompute runs").
One ruling is needed before E6b can be recomputed: the **E7 task-head arms fail
P-B** (see step 1).

HAR data restored 2026-08-02 20:09 and verified against the frozen fingerprints
before any read: `partition_fingerprint` **5d047e4213d1** and shift spec
**3de66e205eb7**, both matching `docs/E10_prereg.md:17-18`.

Contract: `docs/E11_instrument_correction.md`. Every number below is from
recorded script output, cited to the script that printed it.

---

## What E11 was called for, and what it actually found

The contract named **one** defect: `channel_decomp.py::features_and_logits`
hand-rebuilt the readout from the raw cell state `c_t`, while the deployed
classifier reads the composed state's `h' = o_t ⊙ tanh(c_t′)`.

Executing it found a **second, independent defect of the same family**, which the
contract did not anticipate and which is **larger in blast radius**:

> **A reloaded checkpoint is not the model that was trained.** `PLCM.task_stats`
> is a plain dict, not a buffer, so it is absent from `state_dict()`.
> `forward()` gates coordinate alignment on `self.task_stats` being non-empty.
> Every checkpoint-loading analysis in this program has therefore evaluated a
> model with the coordinate-alignment path **silently disabled**.

Both are the same error shape: *an instrument's premise about identity, asserted
in a docstring and never checked.* Catch 28 was about reading the wrong tensor
**from** the model. This one is about reading the right tensor from the **wrong
model**. The first was caught by a consistency check nobody required; the second
by the three-column check the contract ordered as a scope bound.

---

## Step 1 — fix with proof, not assertion  ✅ GREEN (MNIST arms)

`scripts/verify_path_identity.py`. Feature and logits are now taken from
`PLCM.forward` itself (`output["readout_feature"]`), and two properties are
checked per arm-task cell, permanently:

- **P-A (path identity)** — the tensor the instrument reports is *bitwise* the
  tensor every readout module consumed in that same call, verified from forward
  hooks. **8/8 cells PASS.**
- **P-B (single-head readout)** — deployed logits == `head_k(feature)`.
  **8/8 cells PASS**, `max|Δlogit| = 0.000e+00`. Note this is structural for
  these arms: MAFC λ0 arms set `use_task_heads=False`, so `forward` falls through
  to the shared classifier with no routing blend. **P-B is untested on task-head
  arms** (E7), which are HAR-only — see blocker.

**Positive control (catch 25) — the gate fires.** P-A was run on v3's own feature
(`o_t ⊙ tanh(c_t)`), injected at the module boundary so the gate itself is
unmodified, and it **failed as required**. A gate that has only ever passed is
indistinguishable from one that cannot fail; this one has now been shown to fail
on the exact defect it exists to catch.

**Size of the defect in feature space:** max |v3 feature − deployed feature| =
**1.756** (E4/OFF) and **1.762** (E4/ON) — on a feature bounded to [−1, 1].

## Step 2 — scope confirmation  ✅ MNIST (E4) / ⛔ HAR (E5) blocked

`scripts/three_column.py --benchmark mnist --seeds 42`
→ `runs/e11_three_column_mnist_s42_fixed.json`

| arm | task | matrix | fwd_full | instr **corrected** | instr **v3 (old)** | \|corrected−mat\| | \|v3−mat\| |
|---|---|---|---|---|---|---|---|
| E4/OFF | 0 | 0.0978 | 0.0978 | 0.0978 | 0.0940 | 0.0000 | 0.0038 |
| E4/OFF | 1 | 0.1870 | 0.1870 | 0.1870 | 0.2665 | 0.0000 | 0.0795 |
| E4/OFF | 2 | 0.2730 | 0.2730 | 0.2730 | 0.3235 | 0.0000 | 0.0505 |
| E4/OFF | 3 | 0.7010 | 0.7010 | 0.7010 | 0.6590 | 0.0000 | 0.0420 |
| E4/ON | 0 | 0.8724 | 0.8724 | 0.8724 | 0.8160 | 0.0000 | 0.0564 |
| E4/ON | 1 | 0.9393 | 0.9393 | 0.9393 | 0.8980 | 0.0000 | 0.0413 |
| E4/ON | 2 | 0.9693 | 0.9693 | 0.9693 | 0.9540 | 0.0000 | 0.0153 |
| E4/ON | 3 | 0.9772 | 0.9772 | 0.9772 | 0.9625 | 0.0000 | 0.0147 |

*(`instr corrected` and `fwd_full` are both on the full test split; the matrix
column is the recorded final row.)*

**Verdict, computed from the columns above:**
- **E4/OFF — floor AFFECTED.** Old instrument off by **0.0439** mean absolute.
- **E4/ON — floor AFFECTED.** Old instrument off by **0.0319** mean absolute.
- The corrected instrument reproduces the deployed path **exactly** — 0.0000 in
  all 8 cells, not "closely".

**Third finding, unpriced until now — the instrument's own subsample.**
`N_TEST = 2000` of a 10,000-example MNIST test split contributes **0.0178**
(OFF) / **0.0165** (ON) mean absolute error *by itself* — the same order as the
catch-28 defect it sat underneath. On E10's HAR-subject splits (~340 windows per
task) the cap is inactive and this term vanishes, so it is a **MNIST-only** term.
It must be either removed (evaluate on the full split) or reported as instrument
noise wherever a MNIST floor is quoted.

## Step 3 — the residual  ✅ DIAGNOSED AND CLOSED

The contract recorded `PLCM.forward` sitting **+0.056 systematically** above the
matrix on E10/OFF/s42 and listed two candidates (test-loader construction; the
checkpoint holding a different state). **Both are wrong.** The cause is the
`task_stats` serialization gap.

`scripts/checkpoint_fidelity.py --benchmark mnist --seeds 42`
→ `runs/e11_ckpt_fidelity_mnist_s42.json`

```
state_dict keys matching 'task_stat': NONE
running_mean present: True   running_var present: True   stats_initialized: True
```

| arm | task | matrix | reloaded (bare) | restored | rel−mat | res−mat |
|---|---|---|---|---|---|---|
| E4/OFF | 0 | 0.0978 | 0.1018 | 0.0978 | **+0.0040** | +0.0000 |
| E4/OFF | 1 | 0.1870 | 0.1967 | 0.1870 | **+0.0097** | +0.0000 |
| E4/OFF | 2 | 0.2730 | 0.3066 | 0.2730 | **+0.0336** | +0.0000 |
| E4/OFF | 3 | 0.7010 | 0.7280 | 0.7010 | **+0.0270** | +0.0000 |
| E4/ON | 0 | 0.8724 | 0.8692 | 0.8724 | −0.0032 | +0.0000 |
| E4/ON | 1 | 0.9393 | 0.9407 | 0.9393 | +0.0014 | +0.0000 |
| E4/ON | 2 | 0.9693 | 0.9691 | 0.9693 | −0.0002 | +0.0000 |
| E4/ON | 3 | 0.9772 | 0.9779 | 0.9772 | +0.0007 | +0.0000 |

**The OFF arm's error is one-directional (+0.0186 mean, all four cells positive)
and the ON arm's is not (0.0014, mixed sign)** — the same signature the contract
reported for E10/OFF at +0.056. Restoring the stats reproduces every cell
**exactly**. The diagnosis is a repair, not an argument: `task_stats[k]` is by
construction the running (mean, var) at the k→k+1 boundary, which is exactly what
`task{k}_epoch9.pt`'s buffers hold, so it is reconstructed from recorded state,
never estimated.

**Why the arms differ:** the OFF arm's old-task accuracies sit near chance
(0.098–0.273), where small logit changes flip many predictions; the ON arm's sit
at 0.87–0.98, where they flip few. The defect's *magnitude* tracks how close the
arm is to the decision boundary — so it is largest exactly where forgetting is
being measured.

### Fix, in three places
1. `PLCM.load_era(dir, task)` — loads the deployed model **era-correctly**:
   a task-k model gets exactly k stats entries, asserted, never task 4's
   (restoring later stats onto a ceiling checkpoint would leak future state into
   a ceiling measurement — a different error in the same family).
2. `ContinualTrainer` now writes `task_stats` into every checkpoint, so nothing
   after E11 needs reconstruction.
3. `channel_decomp.load()` delegates to `load_era`, so all four consuming scripts
   (`cure_screen`, `transport_estimate`, `rho_percell`, `generative_certificate`)
   inherit the fix without edits.

`pytest tests/ -q` → **124 passed**.

---

## Revised scope table (supersedes the contract's "inherited scope")

| script (experiment) | calls `forward()`? | catch 28 | `task_stats` |
|---|---|---|---|
| `channel_decomp` (E6b) → `cure_screen` (E8, E10), `transport_estimate` (E9), `rho_percell`, `generative_certificate` | now yes | **YES** (known) | **YES** |
| `drift_probe` (E4b) | no — `model.lstm` direct | **YES** — docstring: "the LSTM cell state, **which is what PLCM reads out**". It is not. | no |
| `subspace_drift` (E4b) | no | **YES**, compounded — projects raw `c_t` drift onto the row space of `classifier.weight`, a matrix that acts on `h′`. Two different spaces. | no |
| `drift_anatomy` (E6) | no | **YES** — same raw-`c_t` premise | no |
| `rbst_feasibility` (E4b) | no | **no** — declares "Transported object: raw cell state `c_t` (**NOT** `c_prime`)". A declared scope, not a false premise. | no |

**The defect predates E6b.** The contract scoped it to "every number computed
through this path since E6b"; E4b and E6 carry the same false premise in their
own instruments. Their arithmetic is unaffected (they measure real properties of
`c_t`); what is affected is every inference from those numbers **to the deployed
readout's behaviour** — which for `subspace_drift` is the whole point of the
measurement. Consequence **unmeasured**: E4b/E6 are not in the contract's Step 4
recompute chain. Ruling needed — recompute, or explicitly rescope those claims to
raw `c_t`.

**`rbst_feasibility` is the control that makes the point.** Same era, same raw
tensor, no defect — because it *declared* what it was reading instead of
asserting what it was equivalent to.

---

## Step 1/2 on HAR — the numbers that matter

`scripts/verify_path_identity.py --skip-mnist`: **P-A passes on 24/24 HAR cells**,
positive control fires. **P-B FAILS on 8/8 task-head cells** — see the ruling
below. The four λ0 arms (E10/OFF, E10/ON, E5/OFF, E5/v1-ON) pass P-B cleanly.

`modal run modal_runner.py::analysis --argv "scripts/three_column.py --benchmark
har_subject --seeds 42 --batch-size 128"` — **on Modal, the platform the numbers
were produced on**:

| arm | task | matrix | fwd_full | instr **corrected** | instr **v3 (old)** | \|corrected−mat\| | \|v3−mat\| |
|---|---|---|---|---|---|---|---|
| E10/OFF | 0 | 0.4010 | 0.4010 | 0.4010 | 0.2641 | 0.0000 | 0.1369 |
| E10/OFF | 1 | 0.1570 | 0.1570 | 0.1570 | 0.3140 | 0.0000 | 0.1570 |
| E10/OFF | 2 | 0.1556 | 0.1556 | 0.1556 | 0.1582 | 0.0000 | 0.0026 |
| E10/OFF | 3 | 0.8042 | 0.8042 | 0.8042 | 0.5535 | 0.0000 | 0.2507 |
| E10/ON | 0 | 0.5648 | 0.5648 | 0.5648 | 0.4939 | 0.0000 | 0.0709 |
| E10/ON | 1 | 0.4767 | 0.4767 | 0.4767 | 0.6395 | 0.0000 | 0.1628 |
| E10/ON | 2 | 0.4821 | 0.4821 | 0.4821 | 0.5026 | 0.0000 | 0.0204 |
| E10/ON | 3 | 0.7128 | 0.7128 | 0.7128 | 0.5483 | 0.0000 | 0.1645 |

**The corrected instrument is EXACT on all eight cells. The old one was wrong by
0.1368 (OFF) / 0.1046 (ON) mean absolute — 10–14 percentage points, bidirectional.**
On MNIST the same defect was worth 3–4pp; on E10 it is an order of magnitude
worse, because HAR-subject accuracies sit near chance where the composition's
contribution decides the argmax.

`acc_orig` is the floor in **every** ρ — numerator and denominator both. E10's
headline ρ = 0.615 was computed from floors carrying this error.

## Step 4 — recompute chain  ⏸ READY, one ruling outstanding

Everything is in place: the checkpoint audit is green (33/33) and the data is
fingerprint-verified. Two things must be settled first.

### RULING NEEDED — the E7 task-head arms fail P-B

| arm | tasks | max \|Δlogit\| | prediction disagreement |
|---|---|---|---|
| E7/heads | 0,1,2,3 | up to 5.439 | 0.0% |
| E7/heads+ad | 0,1,2,3 | up to 4.114 | **up to 75.0%** (T2), 46.9% (T1) |

On these arms `forward` routes a **convex blend of all five task heads** by
retrieval attention mass, then optionally blends stored memory logits through the
context-head gate. So the deployed prediction is not `head_k(feature)` for any k
— on E7/heads+ad task 2, three quarters of the probe's predictions differ.

The corrected instrument's **floors are still right** for these arms (it returns
the deployed logits). What breaks is the *comparison*: `F_read = acc_refit_t4 −
acc_orig` sets a refit **single head** against a deployed **routed blend**, so it
absorbs a readout-class difference on top of the reader-walk it is meant to
measure. That is the same confound-stacking v3's feature lock was written to
remove, one level up.

The contract says stop and report, do not special-case. Reported. Options:
**(a)** recompute E6b on the four P-B-clean arms and report the E7 rows
separately as "deployed readout is a routed blend; F_read not interpretable as
reader-walk"; **(b)** redefine the refit target for head arms to match the
routing; **(c)** something else.

**Note H-C1 and H-C2 are unaffected either way** — they read HAR/OFF and
v2-control, both λ0 arms that pass P-B. **H-C3′ IS affected**: it pools R across
*all* HAR arms into one mean before testing |R| ≤ 0.05, so the P-B-failing arms
enter the gate that licenses every channel claim. This is catch 26's shape
exactly — a guard applied after the average, where a failed cell is already
averaged away. The recompute will print R per arm and per cell beside the pooled
value. The bar stays at 0.05; only the granularity changes.

### Where the recompute runs — Modal, not locally

Re-evaluating E10 checkpoints **locally** misses the recorded matrix by 0.0065
(OFF) / 0.0126 (ON) mean absolute; the **same script on Modal reproduces it
exactly, 0.0000 in all eight cells**. Ruled out by measurement, not argument:
same weights (audit green), same data (fingerprints match), `shuffle=False` test
loaders, and batch size irrelevant — 128 and 256 agree cell-for-cell. What
remains is the BLAS: macOS ARM vs Modal Linux, last bits apart, flipping
argmaxes wherever predictions sit near a tie. On E10/OFF/s42 that was **exactly 1
sample** on task 2 and **9** on task 3.

E4/MNIST reproduced exactly *locally* because those runs were trained on this
laptop — the same platform, so no gap. That contrast is the control.

**Consequence:** a recompute that supersedes a Modal-computed number runs on
Modal, or the old→new delta carries a ~0.6–1.3pp platform term nobody can
separate from the correction. `modal_runner.py::analyze` was added for this and
is verified working.

**The chain must be re-run with BOTH fixes**, not just catch 28's: the recompute
changes the composed state itself, so every floor, ceiling, refit and ρ moves for
two independent reasons.

## Step 4 RESULTS — the recompute chain (all on Modal)

### 4.1 E6b decomposition — the 75–90% claim

`/runs/e11_e6b/decomp.json`. Identity residual **0.00e+00**. Read share = F_read /
(F_enc + F_read), old beside new:

| arm | OLD share | NEW share | OLD R | NEW R |
|---|---|---|---|---|
| v1-ON | 82.9% | **75.6%** | +0.0262 | −0.0096 |
| OFF | 79.2% | **66.2%** | +0.0589 | −0.0050 |
| v2-ON | 84.0% | **77.2%** | +0.0571 | −0.0068 |
| v2-control | 88.4% | **87.9%** | +0.0514 | −0.0095 |
| *E7-heads* | *79.6%* | *78.9%* | *+0.0773* | *−0.0002* |
| *E7-heads+ad* | *85.2%* | *78.9%* | *+0.0495* | *−0.0022* |

*(E7 rows italic — routed blend, F_read not interpretable as reader-walk.)*

**H-C1 PASS** (OFF F_read 0.2180 ≥ 0.10). **H-C2 PASS** (v2-control 0.3785 >
0.0523). **Branch (A) holds.**

**THE LEDGER'S RANGE BREAKS AT THE BOTTOM.** "75–90%" becomes **66.2%–87.9%**
across the four P-B-clean arms. The qualitative claim — reader channel dominant
in every arm — survives; the OFF arm at 66.2% no longer supports a 75% floor.

**The instrument's own bias collapsed, which is the correction validating
itself.** Old pooled R over all HAR arms was **+0.0534 — above the 0.05 bar —
with 28/72 individual cells exceeding it**. Corrected: pooled **−0.0077** with
**0/48 cells** over the bar. R measures the deployed-vs-optimal *fitting* gap;
reading the wrong tensor inflated it sevenfold. H-C3′ was not a marginal pass
before — it was failing, and the per-cell view the E11 ruling introduced is what
makes that visible.

### 4.2 E8 and E10 screens — the headline ρ

`scripts/rho_percell.py` (mean-of-ratios, per-cell denominators — like-for-like
with the ledger's published figures):

| quantity | OLD | NEW |
|---|---|---|
| **E10 C3 / OFF** | **0.615** [0.548, 0.682] | **0.974** [0.898, 1.050] |
| E10 C3 / ON | 0.449 [0.335, 0.563] | **0.998** [0.803, 1.193] |
| E10 C2 / OFF (oracle) | 0.664 | 0.979 |
| E8 C3 / OFF (pooled screen) | 1.037 → 0.798 re-grade | **1.004** |

**H-X1 (bar 0.80): NOT MET → MET.** The CI's lower bound, 0.898, clears the bar.

**Mechanism, and why it is not a surprise.** The reference table moved coherently:

| | OLD | NEW | Δ |
|---|---|---|---|
| OFF floor `acc_orig` | 0.4838 | 0.4212 | −0.0627 |
| OFF matched-late ceiling | 0.8316 | 0.7987 | −0.0329 |
| OFF C3 accuracy | 0.7212 | 0.7714 | +0.0502 |
| **OFF R** | **+0.0945** | **+0.0058** | **−0.0887** |
| **ON R** | **+0.1375** | **−0.0049** | **−0.1424** |

The ledger records *"H-C3′ instrument gate fails both arms (R = +0.0945 /
+0.1375)"* as an established finding about the era-head cure family. **That gate
failure was a symptom of catch 28, not a property of the benchmark.** Corrected,
both arms pass.

**THE "OVERSTATEMENT" CLAUSE DOES NOT SURVIVE.** The ledger says a
reconstruction-permissive benchmark read this cure at ~1.0 while the honest one
read 0.615 — an overstatement "measured at ~0.2". Recomputed, E8 reads 1.004 and
E10 reads 0.974: **the gap is ~0.03, not ~0.2.** E10's benchmark repair remains
correct and worth having; what it does not do is cut this cure's score, and the
program's published claim that it did was an instrument artifact.

**SCRUTINY OWED BEFORE THIS ENTERS THE LEDGER.** This moves every number in the
direction that flatters the method — the direction this program treats as most
suspect. ρ ≈ 1.0 says a storage-honest cure reaches the labelled matched-late
reader. Three things to attack first: (i) the C0/C0C1 cells remain degenerate
(cell sd 8.6 on OFF), so the per-cell guard is doing real work and its exclusions
must be read, not skimmed; (ii) 1/12 cells now excluded on each arm where 0 and 2
were before — the exclusion set moved and should be reconciled; (iii) catch 24
still applies to C3's assumed resource and is untouched by this correction.

### 4.3 Step 4b — E4b/E6 geometry in the read space

`/runs/e11_subspace/`. The harness reproduces both published values in the
published space before changing anything: MNIST v3 **0.2139** vs published 0.214;
HAR v3 **0.9704** vs published 0.970.

| | v3 (published space, raw c_t) | v4 (read space, h′) |
|---|---|---|
| MNIST ON/OFF drift_S ratio | 0.2139 | **0.2041** |
| HAR v1-ON/OFF drift_S ratio | 0.9704 | **1.0556** |
| MNIST in-S energy, ON | 13.36% | **18.79%** |
| MNIST in-S energy, OFF | 26.19% | **45.93%** |
| HAR in-S energy, v1-ON | 5.38% | **40.26%** |
| HAR in-S energy, OFF | 4.46% | **41.96%** |
| HAR in-S energy, v2-ON | 5.64% | **38.64%** |
| HAR in-S energy, v2-control | 4.39% | **31.89%** |
| *(random rank-10 baseline)* | *3.91%* | *3.91%* |

**The ratios hold; the aim table inverts.** MNIST moves 0.0099 (prediction on
record was >0.05 at ~50% — it did not), HAR moves 0.0852, and the qualitative
split (MNIST steered / HAR unsteered) **survives in both**.

But the in-S energy fractions are a different story. In the published space HAR's
drift sat at **4–6%, indistinguishable from the 3.91% random baseline** — the
basis for reading HAR drift as *unaimed*. In the space the classifier reads it is
**32–42%, an order of magnitude above chance**. The conclusion inverts: drift is
strongly concentrated in the readout subspace. Anywhere the outline says HAR
drift is not aimed at the readout, that sentence is superseded.

**Blast radius, recorded not acted on:** `drift_anatomy.py:45` sets
`H_B1_BAR = 0.43  # 2 x MNIST's measured ON/OFF drift_S ratio of 0.214`. A
pre-registered bar in E6 was derived from a measured quantity in E4b that this
recompute moves to 0.2041. The bar is **not** adjusted — re-baring after seeing
data is the error this program exists to avoid — but its derivation input has
changed and that is now a fact about its provenance.

## The three pre-ledger evaluations  (`scripts/e11_evaluations.py`, output in `runs/e11_evaluations_e10.txt`)

### EV1 — exclusion-set reconciliation: THE POPULATION MOVED, and it moved in our favour

The guard (`RHO_MIN_DENOM = 0.02`) does not act on the same cells before and
after. For C3 the old computation excluded **nothing**; the new one excludes one
cell per arm, because those denominators collapsed:

| cell | old raw D | new raw D | |
|---|---|---|---|
| OFF s42/t3 | +0.2977 | **+0.0104** | crossed the guard |
| ON s2024/t3 | +0.3185 | **+0.0131** | crossed the guard |

**Both excluded cells would have scored BELOW the kept mean** — ρ = +0.000 (OFF)
and −0.800 (ON) — so excluding them **raises** the headline. Disclosed, with the
number the reader is owed:

| arm | guarded (11 cells) | forced-inclusion (12 cells) | Δ |
|---|---|---|---|
| OFF | **+0.974** | **+0.893** | −0.081 |
| ON | **+0.998** | **+0.848** | −0.150 |

Exclusion remains **correct** — the guard is pre-existing and principled (catch
26), and a ρ computed on D = 0.01 is noise amplified 100×. But **H-X1's verdict
must be shown to survive the bookkeeping, and it does: 0.893 and 0.848 both clear
the 0.80 bar.** The era-head cures moved the other way (old excluded 2 cells per
arm, new excludes 0–1), so the populations are not comparable cure-by-cure across
the correction and should not be differenced without this table beside them.

### EV2 — degenerate-cell audit: C3 is robust, the era-head family is not

| arm/cure | mean | sd | worst cell | mean without it |
|---|---|---|---|---|
| OFF/**C3** | +0.974 | 0.197 | s1337/t3 ρ +1.278 | **+0.944** |
| ON/**C3** | +0.998 | 0.204 | s1337/t3 ρ +1.410 | **+0.957** |
| OFF/C0 | −2.949 | 8.637 | s42/t3 **ρ −30.000** (D=+0.0209) | −0.490 |
| OFF/C0C1 | −2.881 | 8.647 | s42/t3 **ρ −30.000** | −0.416 |
| OFF/C2 | +0.979 | 1.105 | s42/t3 ρ +4.375 | +0.670 |

**The headline cure does not rest on any single cell** (shifts 0.030 / 0.041).
**The era-head family does**: C0's mean of −2.949 is one cell at ρ = −30.000,
whose denominator is **+0.0209 — above the 0.02 guard by 0.0009**. The "era-head
cures are unevaluable at these test-set sizes" clause therefore **stands, and now
has its mechanism**: the guard admits cells that are degenerate in every sense
but the threshold's.

### EV3 — catch 24: C3 IS DOMINATED BY ITS OWN INGREDIENT

Scored on the same instrument, same cells, same denominators as C3 — the
snapshot baseline is `acc_ceiling`, already recorded in every screen row:

| arm | C3 | snapshot (deploy the era model) | gap |
|---|---|---|---|
| OFF | +0.974 [+0.898, +1.050] | **+1.136 [+1.051, +1.221]** | **+0.162** |
| ON | +0.998 [+0.803, +1.193] | **+1.794 [+0.708, +2.879]** | +0.796 |

On the OFF arm the intervals **separate** (snapshot's lower bound 1.051 sits above
C3's upper bound 1.050). On ON the point estimate dominates but the interval is
wide (sd 1.725) and the comparison is not resolved there.

C3 requires the era snapshot **to label its pseudo-data**. Deploying that same
snapshot directly scores higher at strictly less storage. **The
deployment-shaped claim ("use C3") is not supportable.** The mechanism-shaped
claim is, and is what the evidence actually carries.

*(The registered 1.237 figure recomputes to 1.136 here — close, but the number
that matters is that this is now measured on the same instrument and cells as the
thing it is compared against, which the registered figure was not.)*

---

## LEDGER EDIT — DRAFTED HERE, APPLIED SECOND

### Proposed one-liner

> Catastrophic forgetting in LSTMs from input-space task shifts can be eliminated
> at the input path when the shift is expensive to absorb and the encoder has
> something to lose. Where the input path fails to engage, **66–88%** of
> forgetting is reader–encoder mismatch, and the drift carrying it is **aimed at
> the readout subspace** (32–42% of drift energy in S against a 3.9% chance
> baseline), not scattered. That channel is **repairable at read time without
> stored data** — certified-honest generative refit recovers **0.97 [0.90, 1.05]**
> of the recoverable gap on subject-disjoint sensor data — but the cure is
> **dominated by the era snapshot it already requires** (ρ = 1.14 at strictly
> less storage), so the finding is **mechanistic**: it demonstrates the damage is
> reader-side rather than representational, and does not recommend a deployment.
> Exact input correction at read time is harmful; the era-head cure family remains
> unevaluable at this benchmark's test-set sizes.

### Clause-by-clause

| clause | disposition |
|---|---|
| input-path elimination | **held**, untouched by E11 |
| ~~75–90% reader–encoder mismatch~~ | **revised → 66–88%** (E6b recompute, 4 P-B-clean arms) |
| ~~generative refit recovers 0.62 [0.55–0.68]~~ | **revised → 0.97 [0.90, 1.05]**; provisional marker cleared |
| ~~reconstruction benchmark read the cure at ~1.0 / overstatement ~0.2~~ | **RETRACTED.** E8 1.004 vs E10 0.974 — the gap is ~0.03. The overstatement was an instrument artifact, not a benchmark property. |
| exact input correction harmful | **held** (C0 negative both arms, both benchmarks) |
| era-head family unevaluable | **held**, mechanism added (guard admits D=0.0209 → ρ=−30) |
| ~~H-C3′ instrument gate fails both arms (R = +0.0945/+0.1375)~~ | **RETRACTED** — symptom of catch 28; corrected R = +0.0058 / −0.0049 |
| tier-0 stack magnitude | **still frozen**, untouched by this pass |
| **NEW** — drift is aimed at S | 32–42% in-S energy vs 3.9% chance; supersedes every "unaimed" sentence |
| **NEW** — C3 dominated by its ingredient | ρ_snapshot 1.14 > ρ_C3 0.97 at less storage (EV3) |
| **NEW** — E7 supersession | branch (D) becomes "an attention-blended multi-head readout dominated by the latest head worsens forgetting"; the frozen-early design **was never executed** (α_own 0.165, own head top-weighted in 0/24 cells) |
| **NEW** — E10 partition as-executed | registered `5d047e4213d1` is the laptop's; runs used `1104af185c87`; documented as-executed, x86 required |

### Calibration appendix (three predictions, as ruled)

| prediction | outcome |
|---|---|
| C3's recomputed ρ within ±0.15 of 0.615 (~50%) | **MISSED by 0.36** — the worst quantitative miss of the program. Priced as noise-like; it was bias-like, both floors moving the same direction. |
| MNIST 0.214 moves by >0.05 (~50%) | **did not fire** — moved 0.0099 |
| qualitative structure survives (~65%) | **fired** — split intact on both datasets |

## The E10 partition is environment-dependent — and the frozen fingerprint does not certify the executed data

Found while recomputing the E10 screen, which printed partition fingerprint
**1104af185c87** where `docs/E10_prereg.md:18` registers **5d047e4213d1** — the
value that verified MATCH on the laptop at the start of this pass.

`scripts/partition_provenance.py`, run in both environments (same numpy 2.4.6,
same torch 2.13.0):

| | laptop (arm64) | Modal (x86_64) |
|---|---|---|
| task 0 train | [1, **5**, 11, 13, 16] | [1, **2**, 11, 13, 16] |
| task 3 train | [**2**, 4, 6, 18, 27] | [**5**, 4, 6, 18, 27] |
| test subjects | [25],[29],[26],[30],[28] | identical |
| per-task window counts | 1658/1705/1669/1684/1673 | identical |
| calibration μ sha1 | 16cd4d8b45f5 | **4727dff6b818** |
| task-0 test X sha1 | cb4a9286ef1c | **932a1fab10a6** |

Subjects 2 and 5 are an **exact tie** in the greedy balancer — the window counts
are identical either way — and the tie breaks differently across architectures.

**This is not cosmetic, because calibration is computed from task 0's TRAIN
windows only.** Swapping which subject sits in task 0 changes μ and σ, which
normalize **all five tasks**, so **every task's test tensor hashes differently**
between the two environments. Labels and shapes are identical; the values are not.

**MY EARLIER DIAGNOSIS IN THIS MEMO WAS WRONG AND IS CORRECTED HERE.** I
attributed the local-vs-Modal accuracy gap to BLAS — float non-associativity
flipping near-ties. That story survived the batch-size control (128 and 256 agree
cell-for-cell) and was written into this memo and into `modal_runner.analyze`'s
docstring as established. The partition control killed it. **The data differed;
the arithmetic never did.** Catch 19's lesson, reproduced exactly: a plausible
causal story survived one round of evidence before the control arrived.

The operational conclusion is unchanged but its justification is stronger — the
recompute must run on Modal not to match its arithmetic but because **it is the
only environment that reproduces the data the checkpoints were trained on**.
Evidence that Modal's partition is the executed one: on Modal `fwd_full`
reproduces the recorded accuracy matrix exactly in all 8 cells; locally it does
not, and locally the test tensors differ.

**What this costs the pre-registration.** E10's prereg claims the partition was
frozen with a fingerprint before any run. The registered value describes the
laptop's partition; every E10 run used the other one. The benchmark is still a
valid subject-disjoint benchmark — test subjects, disjointness and window counts
are exactly as registered — but the registered fingerprint does not certify the
executed data, and reproducing E10 requires x86. `partition_fingerprint()` should
gate, not merely print (`cure_screen` exits on a shift-spec mismatch and only
prints the partition), and the registered value needs re-derivation in the
execution environment.

## Checkpoint audit — catch 20, strengthened to "loads", not "exists"

`scripts/audit_checkpoints.py` → `runs/e11_ckpt_audit.json`. The first run found
**14 of 33 checkpoint dirs unusable**: the entire E5 family missing or truncated
(`TRUNCATED(0B)`, `RuntimeError: failed finding central directory`) plus two E7
dirs. Separately, **68 of 160 result dirs were empty** — directories present,
contents absent.

This is the CLAUDE.md gotcha one level down. "A run finished" is not directory
existence; **"a checkpoint exists" is not "a checkpoint loads."** A partial
`modal volume get` leaves plausible-looking files — one was 4.9MB and still
unreadable. The audit criterion is therefore `PLCM.load_era()` succeeding, which
is the same call the analysis makes, so anything it passes the chain can read.

Re-synced the epoch-9 files for 14 dirs from the volume; audit now **33/33, 0 bad
files**. The volume is the authoritative copy — the local `runs/` mirror was
never complete.

## Step 5 — ledger  ⛔ NOT STARTED

`docs/CLAIM_LEDGER.md` still carries the provisional marker on the 0.615 clause.
It stays until Step 4 produces the replacement number. **The tier-0 freeze and
every other clause are untouched by this pass so far.**

---

## Calibration

| prediction on record | outcome |
|---|---|
| the assert passes after the fix | held (definitional) |
| E5/E4 floors AFFECTED (~90%) | **CONFIRMED, and understated.** E4 0.0319–0.0439 mean abs; **E10 0.1046–0.1368** — an order of magnitude larger than the MNIST case the prediction was anchored on. |
| F_read-dominant structure survives (~65%) | unmeasured |
| C3's recomputed ρ within ±0.15 of 0.615 (~50%) | unmeasured |

**Not on the record at all: the `task_stats` defect.** The contract's step 3
listed two candidates for the residual and the true cause was neither. Both
listed candidates were about *which artifact* was read (loader, file); the actual
cause was about *what a loaded artifact omits*. The contract's own instinct —
"diagnose by content, not filename" — was right, and still under-reached: the
content check it implied was of the file, and the missing state was never in the
file to be checked.
