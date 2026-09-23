# MEMO — C0deg: the controls, the row, and the number that did not fit

**Status (2026-09-15, end of day):** every step of the sequence ruling is
done — *controls → ledger row → E16 §7.2 closes → bridging's regime → E18 arm
table.* Controls run and ruled; CC′ passed on both checkpoint sets; ledger
edits applied including **Ruling A** (headline → S72) and **Ruling B**
(bridging's regime); four standing rules in `CLAUDE.md`; E16 §7.2 closed with
every floor measured; **`docs/E18_prereg.md` drafted around C0deg, for
sign-off.**

**Origin.** The E18–E20 contract review (B1). `cure_screen.py` labels C0deg —
old-task inputs re-laid-out into the **current** frame via `M_4·M_k⁻¹`, scored by
the **current** model — a "CEILING ARTIFACT, never a cure", on a justification
written for E8's shared-window HAR. E10 replaced that construction with
subject-disjoint content; the label was carried across unexamined. Catch 21
("validated is context-bound") applied to an exclusion. On E10 the column was
computed in `runs/e11_e10/cures_e10.json` and never read.

---

## 1. Controls — `scripts/c0deg_controls.py` → `runs/e10/c0deg_controls.json`

| control | what it rules out | result |
|---|---|---|
| **G1** subject disjointness, from `HARSubjectBenchmark.test_subjects` | that the current model has seen the windows it is scored on | **PASS** — every test subject ∉ any training group; 5 test subjects pairwise distinct |
| **G2** map algebra | that the base layout is not the no-shift layout; generator composition | **PASS** — `channel_affine(0) == (I, 0)`; round-trip worst 4.8e-08 |
| **CA** no-shift positive control (ruling a) | that C0deg's code path differs from the deployed path; bounds what the map can recover | **PASS, bit-identical** — 12/12 cells max\|Δlogit\| = 0.0e+00, C0deg == deployed exactly |
| **CB** artifact consistency, OFF and ON | that the stored column is not what these checkpoints produce | **PASS** — live C0deg and acc_orig equal the stored values to **0.0e+00**; acc_orig equals the matrix final row to 2.3e-08 / 2.6e-08 |
| **CC** wrong-map must-fail (my addition, not in the ruling) | that the map's *identity* is not load-bearing | **FAIL as pre-committed** — see §2c |

The two ruled controls (a) and (b) pass. Path identity (P-B) held on every live
cell, both arms. The artifact reads `all_pass: false` because of CC, and per its
own rule the row is not citable until that is ruled on.

### 1a. CA's second reading — the scope statement's number

On the no-shift arm the map is the identity and recovers nothing *by
construction*; what the control prints beside that is the arm's own forgetting:

| no-shift arm (`e10_noshift_seed*`) | value |
|---|---|
| forgetting, diag form `mean_j(R[j,j] − R[T−1,j])` | **−0.0497** |
| forgetting, recorded field (peak-clipped, `metrics.py`) | 0.02508 (the ledger's P2 number) |

Under the diag form, subject change alone produces **negative** forgetting: the
final model, having trained on 25 subjects, is better on each held-out subject
than the era model that had seen 5. **E10's forgetting is presentation drift
entirely**; there is no content-drift component for any map to recover. That is
the scope statement: *C0deg recovers what the calibration shift took, and on E10
that is everything.*

---

## 2. The row, both formulas — `runs/e11_hx2_c0deg.json`, `runs/e11_hx2_c0deg_peak.json`

Same instrument as the ledger's H-X2 row (`scripts/hx2_forgetting.py`), same
11-cell intersection set, `--include-c0deg`. The flag-off path was verified
**byte-identical** to `runs/e11_hx2.json` before and after the amendment.

| HAR/OFF, intersection (11 cells) | diag form | peak-clipped form (`metrics.py`) |
|---|---|---|
| none | 0.4812 ± 0.013 | 0.5028 ± 0.013 |
| C3 (bridging) | 0.0886 [0.0759, 0.1013] — MET | **0.1198 [0.1071, 0.1325] — NOT MET** |
| **C0deg** | **−0.0247 [−0.0346, −0.0148]** | **0.0129 [0.0030, 0.0228]** |
| SNAP | 0 by construction | 0.0216 (positive backward transfer, folded in) |
| ranking C0deg vs C3 | RESOLVED — intervals separate | RESOLVED — intervals separate |

| HAR/ON, intersection (11 cells) | diag | peak-clipped |
|---|---|---|
| none | 0.4598 | 0.5047 |
| C3 | 0.1113 — NOT MET | 0.1636 [0.1513, 0.1759] — NOT MET |
| **C0deg** | **−0.0314 [−0.0403, −0.0225]** | **0.0356 [0.0266, 0.0445]** |

Per cell (all 12): C0deg > C3 in **11/12** (OFF, mean +0.106) and **10/12** (ON,
mean +0.132). On the ON arm C0deg suppresses the learned adapter through the
deployed path and substitutes the analytic map — a different arm, reported as
such; the OFF arm is the bridging benchmark's.

**Resources.** C0deg needs `M_k` and `M_4` — the same known-transform premise
C3's generator needs — and nothing else: no era snapshot, no refit, no
pseudo-labels, one encoder, one head. A strict subset of bridging's resources.
The storage row is computed by `scripts/e16_storage.py`'s method from
artifacts, not by hand; it is not in this memo.

### 2c. CC — reported as pre-committed, then read

Pre-commitment: *if the identity of the map is load-bearing, the wrong map
(task k re-laid-out as if it were task j = k+1 mod 4) scores below the right
map in every cell.* Result: **9/12 (OFF), 11/12 (ON)**; mean gap +0.263 / +0.346.

| failing cell | right map | wrong map | wrong ≡ treating k as |
|---|---|---|---|
| OFF s42/t1 | 0.8924 | 0.9709 | task 2 (z-rotation) |
| OFF s1337/t0 | 0.8753 | 0.8998 | task 1 (gain + offset) |
| OFF s1337/t1 | 0.9099 | 0.9360 | task 2 (z-rotation) |
| ON s1337/t0 | 0.8802 | 0.8851 | task 1 (gain + offset) |

Every failure is at k ∈ {0, 1}, where right and wrong maps differ by a gain/
offset or a 30° z-rotation only. Where they differ by a channel permutation
(k = 2 → 3, k = 3 → 0) the wrong map collapses to 0.19–0.51 in every cell. So
the *observation* — not a mechanism — is that the θ_T encoder is partially
invariant to the mild shift components and not to re-wiring, consistent with
E5d's `A_k ≈ I`. The mean gap says the map's identity is load-bearing; the
per-cell bar said "in every cell", and that is what failed.

**This is the review's own rule turned on its author:** *a control rules out
only the defects that would BREAK it.* CC as written cannot distinguish "the map
is load-bearing" from "the encoder is invariant to the difference between this
particular right and wrong map", and for adjacent shift types that difference is
small. The bar was too strong for the benchmark's shift set, and I chose it
without pricing that. It stands as FAIL. **Any reformulation now is post hoc
and must be labelled as such**; the options are (i) strike CC — the ruling did
not ask for it, and G1/CA/CB carry the row's claim; (ii) keep CC in mean-gap
form, labelled post hoc, with the four cells reported as the invariance
observation; (iii) a fresh must-fail stated before reading, e.g. the
permutation-only wrong map, which the existing cells already say must collapse.
**Ruling requested.** The row is BLOCKED until then.

**RULED 2026-09-15: CC stands as FAIL; option (iii).** The bar was set before
the run and does not move after it. CC's cause is recorded as a scope finding
and enters the row (§4). CC′ below is the control that tests what CC was meant
to test.

### 2d. CC′ — stated BEFORE launch (`scripts/c0deg_cc2.py` → `runs/e10/c0deg_cc2.json`)

Same checkpoints the row cites (`ckpt_e10_{off,on}_seed*`, Modal x86). Two
**permutation-only** wrong maps, each applied on top of the right relayout
`x_cur = M_4·M_k⁻¹(x − b_k) + b_4`, so the only thing wrong is a wiring:

| variant | Q | what it models |
|---|---|---|
| CC′-within | `PERM = [2,0,1,5,3,4,7,8,6]`, the benchmark's own wiring revision, applied once more | axes relabelled within each sensor triple |
| CC′-across | channel `i → (i+3) mod 9` | sensor triples swapped (acc → gyro → total_acc) |

**Bar, per cell, both variants, both arms:** `acc(Q·x_cur) < C0deg` in **12/12
cells per arm**. Verdict derived from the printed values. If any cell fails,
CC′ fails as CC did, and is reported so.

**CD — component ablation, DESCRIPTIVE, no bar.** The ruling's scope finding
("recovery comes from the permutation components; the encoder partially absorbs
gain/offset and small rotations") rests on CC's four failing cells, and one
rotation cell contradicts it (OFF s2024/t1: right 0.9012, wrong-as-rotation
0.6715). A claim that enters the row needs a direct measurement. For each old
task k and each subset S ⊆ {gain, offset, rot, perm}, the input is corrected
in the components of S only (task-4's values for S, task-k's for the rest,
composed exactly as `channel_affine` composes them, then `/sd` as the loader
does). Self-checks, both asserted: S = ∅ reproduces the deployed input; S = all
reproduces `x_cur` (max err < 1e-5). Reported: accuracy per (cell × S), with
`{perm}` alone and `all − {perm}` printed beside `C0deg` and `none`. No
prediction is registered for CD; its reading is written after, and marked so.

### 2e. CC′ and CD — results (`runs/e10/c0deg_cc2.json`, Modal x86, partition as-executed)

| control | OFF | ON |
|---|---|---|
| CC′-within (`PERM` once more on top of the right map) | **12/12 below C0deg — PASS** | **12/12 — PASS** |
| CC′-across (triples swapped) | **12/12 — PASS** | **12/12 — PASS** |

Every permutation-only wrong map lands at 0.19–0.51 against right-map values
of 0.76–0.99. The map's permutation component is load-bearing; the row's
controls are G1, CA, CB, CC′ (CC recorded FAIL, cause below).

**CD reading — written after the run, marked as such.** Mean over 12 cells:

| S corrected | OFF | ON |
|---|---|---|
| none | 0.4212 | 0.4376 |
| {perm} only | **0.8057** | **0.7646** |
| all − {perm} = {gain, offset, rot} | 0.4414 | 0.4480 |
| all (= C0deg) | 0.8773 | 0.8950 |

Correcting the permutation alone recovers **84%** (OFF) / **71%** (ON) of
C0deg's gain; correcting everything *but* the permutation recovers **4% / 2%**.
The ruled scope finding stands with its qualifier: the recovery is carried by
the permutation component, and the encoder *partially* absorbs
gain/offset/rotation — partially, because with the permutation already right
(task 3, whose `P_3 = P_4`) task-4's gain/rotation/offset still add **+8.6pp**
OFF / **+13.4pp** ON. Task 2's `{perm} ≈ all` is trivial (its rotation equals
task 4's). Consistent with E5d's `A_k ≈ I` and §5's engagement condition; one
sentence there.

---

## 3. The number that did not fit — two forgetting formulas, one word

CA printed **−0.0497** for the no-shift arm; the ledger's P2 row lists the same
three matrices at **0.02507**. Chased before writing around it:

- `src/training/metrics.py:85-106` — the trainer's `forgetting` — is
  `mean_j max(0, max_l R[l,j] − R[T−1,j])` (peak-minus-final, clipped at 0).
  Every recorded `forgetting` field, AVG table, and the P2 row use it.
- `scripts/hx2_forgetting.py`'s docstring stated that "the standard formula
  (src/training/metrics.py)" is `mean_j(R[j,j] − R[T−1,j])` — **it cited
  metrics.py for a formula metrics.py does not implement**, and the H-X2 row
  (0.4812 → 0.0886) was computed in that diag form.
- The forms differ by the backward transfer the peak form folds in plus the
  clipping. On E10/OFF the difference moves C3 across the H-X2 bar: **diag
  0.0886 MET; peak 0.1198 [0.1071, 0.1325] NOT MET.** On ON both forms are
  NOT MET.

The prereg's words ("the standard forgetting formula") point at metrics.py. So
either H-X2 is read under the registered formula and its verdict flips, or the
diag form is ruled the registered one and the row says so explicitly. Either
way the row currently carries a verdict under a formula it does not name.
**RULED 2026-09-15: diag is the paper's form** — §3.1 defines
`F_total(k) = A_k(h_k,θ_k) − A_k(h_k,θ_T)`, diag by construction, and every
decomposition quantity inherits it; peak-clipped references the maximum ever
reached, not the era checkpoint, and would break the identity. **The
registration was defective**: H-X2 is *ambiguously registered* — MET under the
form the instrument computed and the paper defines, NOT MET under the form the
prereg's citation pointed to. The ledger row carries both; the paper does not
describe bridging as "pre-registered and met" without the qualifier. Standing
rule (now in `CLAUDE.md`): every registered hypothesis names its formula by
displayed equation; file citations are witnesses, not definitions. The ranking
C0deg ≪ C3 is unchanged under both.

Instrument amendment: `--formula {diag,peak}` (default `diag`; flag-off path
byte-identical, verified); flag-on artifacts record `formula` and `honest`;
`runs/e11_hx2_peak.json` is the H-X2 row alone under the registered formula.

**Catch, unnumbered — for the process record.** A docstring cited a file for a
formula the file does not contain, and a registered clause ("the standard
formula") was executed against the docstring rather than the file. Catch 28's
shape (a premise carried in a docstring), at the level of a *definition* rather
than a tensor. Found the way catch 32 was: one number from a control that did
not match the same arm's number in the ledger.

---

## 4. LEDGER EDITS — APPLIED 2026-09-15 (after the two rulings and CC′)

Applied to `docs/CLAIM_LEDGER.md`: the one-liner's cure sentences; the
C3-dominated row (strengthened); the new C0deg row (with CC's cause as its
scope finding and CD's shares); the H-X2 row (ambiguously registered, both
formulas); the "exact input correction" row (narrowed to the base layout);
annotations on the E11 H-X2 and E10 P2 history lines; a History entry above
E14. The table below is the draft as it stood; the applied text adds CC′/CD
and the rulings. **Not yet in the ledger: the S72 three-way row (§6) — it
waits on the λ\* floor pair.**

| claim | evidence | status |
|---|---|---|
| **NEW — re-layout into the current frame via the known map recovers more than the era checkpoint had.** HAR/OFF forgetting **−0.0247 [−0.0346, −0.0148]** (diag) / **0.0129 [0.0030, 0.0228]** (peak-clipped); ON −0.0314 / 0.0356. Uses the maps only — no snapshot, no refit, one encoder | `runs/e11_hx2_c0deg.json`, `runs/e11_hx2_c0deg_peak.json`, 11-cell intersection; controls `runs/e10/c0deg_controls.json` G1/G2/CA/CB PASS, CC FAIL as pre-committed; `runs/e10/c0deg_cc2.json` CC′ PASS 12/12 × 2 × 2, CD; per-cell C0deg > C3 11/12 OFF, 10/12 ON | **new — APPLIED.** Scope: recovers presentation drift; E10 has no content-drift forgetting to recover (no-shift arm −0.0497 diag); the recovery is carried by the permutation component (84% / 71%). Bridging benchmark = the OFF arm |
| end-to-end forgetting after the best storage-honest cure: ~~C3 OFF 0.4812 → 0.0886, H-X2 MET at the estimate~~ | C0deg is the best storage-honest method on both arms, both formulas, ranking resolved. Under the **registered** formula (`metrics.py`, peak-clipped) C3 reads **0.1198 [0.1071, 0.1325] — NOT MET**; under the diag form the row was computed in, 0.0886 MET | **amended** — the row names its formula; the C3 verdict is formula-dependent and the diag form's citation of metrics.py was false (§3) |
| C3 is dominated by its own required snapshot (EV3) | + dominated by the trivial use of its *other* required resource, the known map, at strictly less storage | **strengthened** — bridging is dominated on both resources it assumes (catch 24, twice). Whatever bridging's regime is, it is not "the map is known" |
| exact input correction at read time is harmful (C0 negative, both benchmarks, both arms) | true of correction into the **BASE** layout (C0) only; correction into the **CURRENT** layout (C0deg) is the best method in the program | **narrowed** — the sentence was scoped by an inherited exclusion |
| P2 94.4% shift-attributable (no-shift 0.02507 vs shifted-OFF 0.44937) | both are the peak-clipped form; in the diag form no-shift reads **−0.0497** — content change alone produces no forgetting | **annotated** — formula named; the diag reading strengthens P2 (all of E10's forgetting is presentation) |

---

## 5. Process record — five errors, two authors

From the contract review, the contract author's (recorded at their request):

1. **A1 defined in the non-deployable direction.** Task-1 test data needs no
   inversion for the task-1 snapshot; that arm is SNAP. The deployable
   inversion goes the other way and needs no old encoder.
2. **"The sweep found no prior input-inversion oracle on Permuted MNIST."**
   `runs/e8/MEMO.md` §3a had ruled the benchmark family out for exactly this
   method. The sweep did not include the repo's own memos.
3. **"Nineteen prior entries, six misses"** — produced from memory; matches no
   artifact (appendix: 14 scored / 3 misses; MEMO_e16: 11 / 6). The witness
   rule failing inside the contract that invokes it.

From this pass, mine:

4. **CC's bar was chosen without pricing the benchmark's shift set** (§2c). A
   must-fail whose failure can be produced by a property of the encoder rather
   than a defect in the instrument is not a must-fail; it is a measurement.
5. **The forgetting-formula catch was inherited, not caught earlier** — the
   H-X2 docstring's false citation sat under a row I had already re-run twice
   in this pass before a control produced the number that exposed it.

Two more from the S72 launch, both caught by exercising the path before it ran:

6. **The E16 §7.2 pair differed on TWO axes.** The LwF HAR arm and its floor
   pair ran `benchmark: 'har'` (E5 shared-window) per their own `arm` field;
   bridging is `har_subject`. The E16 contract pinned the E10 construction
   (`docs/E16_lwf_prereg.md` §2); `modal_runner.E16_HAR` wrote `har`. Catch
   30's shape on the benchmark axis — nothing checked the artifact's
   `benchmark` field against the contract's sentence until this closure tried
   to put both arms in one row. Every within-E16 comparison was uniform on
   `har`, so no E16 row moves; only the cross-benchmark comparison was never
   valid as planned. Consequence: both arms relaunched on `har_subject`, LwF's
   λ sweep with them.
7. **A latent trainer bug on the era-audit failure path.**
   `_save_era_checkpoint` sets `rec["loads"]` before the audit's
   evaluate/refit; an exception there sets `rec["error"]` but leaves the
   summary print to format `None`, so an audit failure at any task boundary
   raised `TypeError` and killed the training run. Found because the S72 smoke
   ran on the laptop's `mps` and hit a device mismatch at task 1 (`load_era`
   leaves `task_stats` on CPU — a device-agnosticism defect, unfixed here,
   harmless on Modal's CPU). Guarded (`src/training/trainer.py`); the success
   path's output is unchanged.

Also found, not acted on: `docs/E16_lwf_prereg.md` §7 names
`scripts/e16_verify_arm.py`, `scripts/e16_sweep.py`, `scripts/e16_storage.py`
and `runs/e16_sweep.json`, `runs/e16_arm_witness.json`, `runs/e16_storage.json`,
`runs/e16_snapshot.json` — **none exist**. The sweep table and the λ selection
live in `runs/MEMO_e16.md` §2 as prose. Catch 33's own table naming artifacts
that were never built. `scripts/s72_row.py` implements the witness, the
selection and the competence floor from artifacts, in one place.

Standing-rule text — **placed in `CLAUDE.md` 2026-09-15** (three entries: the
inherited exclusion, the displayed-equation rule, the must-fail bar). The
original draft:

> **A registered clause that names a formula by description ("the standard
> forgetting formula") is executed against the FILE the description points to,
> and the artifact records which formula it used.** Two formulas that share a
> word are two quantities; a verdict is not comparable across rows until each
> row names its own. (H-X2: diag 0.0886 MET vs peak-clipped 0.1198 NOT MET on
> the same cells.)

---

## 6. Step 3 — E16 §7.2 CLOSED (`scripts/s72_row.py` → `runs/s72_row.json`)

**The pair as E16 planned it differed on two axes.** The ruling named era
status. The artifacts' own `arm` field says the LwF HAR arm and its floor pair
ran `benchmark: 'har'` — the E5 shared-window construction — while bridging's
0.4812 → 0.0886 is `har_subject` (E10). `docs/E16_lwf_prereg.md` §2 pinned the
E10 construction; `modal_runner.E16_HAR` wrote `har`. So both arms were
relaunched on `har_subject` with one flag set — `--era-checkpoints
--fp32-shadow`, 4 threads — and, because LwF's λ had been selected on the
other construction, its sweep moved with it (`jobs_s72_off`, `jobs_s72_lwf`,
`jobs_s72_lwf_floor --lam`; 21 CPU jobs, all finished by `verify_runs.check`).
The fp32 shadow is required: fp16 era reloads miss the matrix by one window
(measured +0.00034 on E16; +0.0058 in the smoke), the H-X2 consistency gate is
1e-6, and the shadow reloads at exactly 0.0 (every boundary, every run).

### 6a. The row — `har_subject` OFF, uniform, arm-recorded, 12/12 cells valid

| method | diag (paper's §3.1 form) | peak-clipped (`metrics.py`) |
|---|---|---|
| none | 0.3864 ± 0.013 | 0.4414 |
| LwF λ\* = 1.0 | **0.2259 [0.212, 0.240]** (per seed 0.159 / 0.333 / 0.186) | 0.2831 |
| C3 (bridging) | **0.0954 [0.084, 0.107]** — MET at the estimate, CI upper exceeds | 0.1506 — NOT MET |
| **C0deg** | **−0.0379 [−0.047, −0.029]** | 0.0372 — MET |
| SNAP | 0 by construction | 0.0550 |

Both comparisons RESOLVED (intervals separate), both formulas. **E16's pending
prediction "bridging > LwF on HAR/OFF, ~55%" — FIRED**, under both formulas;
LwF's worst seed (0.333) sits above C3's mean. **C0deg is ahead of both.** Per
cell on these checkpoints C0deg > C3 in 11/12 again. The row's `none`
recomputed from the matrices agrees with the screen's uncured baseline to
4e-9 — two scripts, one run.

### 6b. LwF on this construction

| λ | AVG (sd) | DIAG gap vs reference | forgetting diag | competent |
|---|---|---|---|---|
| 0.25 | 0.4988 (0.011) | −0.2pp | 0.4210 | yes |
| **1.0** | **0.6145 (0.075)** | −4.2pp | 0.2259 | yes — margin **+0.8pp** |
| 4.0 | 0.6244 (0.059) | −14.8pp | 0.0815 | **EXCLUDED** |
| 16.0 | 0.6097 (0.021) | −19.3pp | 0.0433 | **EXCLUDED** |

λ\* = **1.0**, selected on AVG (never forgetting); the shared-window
construction had selected 0.25. Across-λ spread 3.0× the across-seed sd —
structure, not noise. The two excluded λ show E16's trap again: forgetting
0.04–0.08 bought with 15–19pp of competence.

### 6c. Floors, per quantity — every delta in 6a clears its floor

| pair (seed 42 twice, same flags, 4 threads) | AVG \|a−b\| | cells identical | forgetting \|a−b\| |
|---|---|---|---|
| OFF (`e10off_ec`) | **0.0000** | **25/25** | 0.0000 |
| LwF λ\* = 1.0 | **0.0000** | **25/25** | 0.0000 |
| LwF λ = 0.25 (prior; floor observation) | 0.0543 | **1/25** | 0.0453 diag / 0.0009 peak |
| C3, C0deg per cell (screen on the OFF pair) | — | — | **0.0000** |

**Determinism is per-λ**: the same LwF arm is bit-exact at λ = 1.0 and 1/25 at
λ = 0.25, same platform, same threading. The floor-tuple rule gains an instance:
(arm × config × platform × threading × quantity × seed), established per pair,
mechanism unclaimed — and "config" includes the regularizer's weight.

### 6d. Controls carried to the new checkpoints

CC′ re-run on `e10ec`: **12/12 PASS**, both wirings (`runs/e10ec/c0deg_cc2.json`).
CD on these checkpoints: {perm} alone 0.7688 of none 0.4665 → all 0.8908 —
**71%** of the recovery; all-but-permutation **15%** (old checkpoints: 84% /
4%). Same direction, different runs.

### 6e. What the relaunch says about the old runs — for ruling

`e10_off` (Aug 2, E11 checkpoints, no `arm` field, era OFF) vs `e10off_ec`
(today): AVG **+14.3 / +6.9 / −9.7pp** by seed, **0–5 of 25 cells identical**.
That is era + fp32 shadow + two weeks of code state, unseparated — the old
runs record nothing that would separate them. Under the unrecorded-config
rule, these are now the bridging benchmark's citable OFF arm. The E11-era
C3/C0deg rows stand as measurements of *those* checkpoints — same sign, same
ranking (C3 0.0886 → 0.0954; C0deg −0.0247 → −0.0379; none 0.4812 → 0.3864
on 11 vs 12 cells). **RULED (A): the headline moves to S72.** The
unrecorded-config rule applied to the paper's most-quoted number. Bridging's
reduction reads **0.386 → 0.095, −75%** (was 89%); ρ **1.03 [0.83, 1.23]**
(`rho_percell --bench e10` on `runs/e10ec/cures_e10.json`, 12/12 cells, no
exclusions; was 0.974); C0deg **−0.038**. E11-era rows retained, superseded
by provenance, not withdrawn. The shipped 89% was true under the ledger when
sent; future communication uses S72 and explains the supersession if asked.
Applied to the one-liner, the H-X1, H-X2 and C0deg rows, and History.

### 6f. Step 4 — bridging's regime, from the artifacts

Now measured on one benchmark under uniform arms: **LwF 0.226 > bridging
0.095 > C0deg −0.038**, all against a known map. Bridging beats the field's
baseline and loses to the trivial use of its own resource. Its surviving
candidates are unchanged from the earlier ruling: the map-free-at-inference
variant (a self-contained head after one-time repair, versus a map applied
per query) — a deployment preference, reported as dominated on accuracy — and
Split-CIFAR's stranded head, which has no map and no generator. Two phenomena
with one signature under the decomposition; §4 says so.

**RULED (B): bridging has no regime where the map is known, and the paper
says so.** Position: *a measured intermediate — better than regularization,
worse than applying the map — with a possible regime the paper names and
does not claim* (a self-contained head where the validated pipeline cannot
acquire a per-query preprocessing step; untested). Not a method contribution.
One row, one scope sentence (ledger, clause sourcing).

**Consequence — E18 redrafts around C0deg**, as the known-map arm in the
known-vs-estimated question: `docs/E18_prereg.md` (DRAFT v2 for sign-off).
Six arms — nothing / SDC-port / LDC-port / C0deg / bridging / reset-with-data
— with anchors (SNAP, NCM-era, NCM-oracle, C2-oracle, tier-0) and the LwF row
from S72. **Premise check before drafting, catch 21:** the estimated-drift
family already has in-repo numbers — E9's chained Procrustes → E13 tier-0,
ρ 0.075 on HAR/OFF (E11-era, to be re-run on S72), and C2 oracle-prototype
Procrustes ρ **0.687** on S72 — so the ports are priced against the program's
own estimator before they run, not discovered after (B1's error in reverse).
Disjoint-content MNIST constructions per B2; P3 fired live and on the
shared-content must-fail; every formula displayed; fp32 shadow everywhere;
CLAUSE → JOB → ARTIFACT and the checkpoint audit table in the draft.

## 7. Artifacts written this pass

| path | what |
|---|---|
| `scripts/c0deg_controls.py` | the controls; `--algebra-only` runs G1/G2 anywhere, scoring gated on the as-executed partition |
| `runs/e10/c0deg_controls.json` | control output, Modal x86, all per-cell rows |
| `scripts/hx2_forgetting.py` | `--include-c0deg`, `--formula`; docstring correction; flag-off path byte-identical to `runs/e11_hx2.json` (verified twice) |
| `runs/e11_hx2_c0deg.json` | H-X2 with C0deg, diag form |
| `runs/e11_hx2_c0deg_peak.json` | H-X2 with C0deg, peak-clipped form |
| `runs/e11_hx2_peak.json` | H-X2 row alone under the registered formula |
| `scripts/c0deg_cc2.py` → `runs/e10/c0deg_cc2.json`, `runs/e10ec/c0deg_cc2.json` | CC′ (PASS on both checkpoint sets) and CD |
| `docs/CLAIM_LEDGER.md` | ledger edits applied: C0deg row, H-X2/H-X1/C0deg headlines moved to S72 (Ruling A), Ruling B row, S72 row, History |
| `CLAUDE.md` | four standing rules: the inherited exclusion, the displayed-equation rule, the must-fail bar, fp32 shadow + construction as an arm-identity field |
| `docs/E16_lwf_prereg.md` | amended: the never-built §7 entries, and the `har`/`har_subject` construction |
| `docs/appendix.tex` | E16 prediction row: pending → fired (S72); tally 15 scored / 3 misses |
| `docs/E18_prereg.md` | **DRAFT v2 for sign-off** — E18 around C0deg |
| `scripts/cure_screen.py` | `--arms e10ec / e10ecfloor` (fp32 shadow dirs, a/b replicates); C0deg label per arm set; docstring correction; default path additive-only |
| `scripts/hx2_forgetting.py` | `--arms`, `--formula`, `--include-c0deg`; intersection cell list on flag-on outputs; flag-off byte-identical (verified 3×) |
| `modal_runner.py` | `S72_OFF`, `_s72`, `jobs_s72_off`, `jobs_s72_lwf`, `jobs_s72_lwf_floor` (λ required) |
| `src/training/trainer.py` | era-audit summary print guarded on the failure path |
| `scripts/s72_row.py` → `runs/s72_row.json` | the three-way row: finish, arm identity across arms, sweep + selection, both formulas on the screen's population, floors, E16's prediction |
| `runs/e10off_ec_seed*`, `runs/e10off_ec_floor4tec_{a,b}` | the bridging arm relaunched, era + shadow, arm-recorded (checkpoints `ckpt_e10off_ec_*` on the volume, fp32 shadow beside each) |
| `runs/e16s_har_lam{0.25,1.0,4.0,16.0}_seed*`, `runs/e16s_har_floor4tec_lam{0.25,1.0}_{a,b}` | LwF on `har_subject`, full sweep + two floor pairs |
| `runs/e10ec/{cures_e10,hx2_c0deg,hx2_c0deg_peak}.json`, `runs/e10ecfloor/…` | the screen and H-X2, both formulas, on the new checkpoints and the floor pair |
