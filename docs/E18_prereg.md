# Pre-Registration: E18 — Known map versus estimated drift

**Status: SIGNED 2026-09-16, with four conditions (§0a).** v2 supersedes the
E18a section of the E18–E20 draft reviewed 2026-09-15 (blocking findings B1,
B2; Rulings A and B in `runs/MEMO_c0deg.md`). E19 and E20 are unchanged from
the accepted review amendments and are contracted separately; nothing here
depends on them. **Order of work, ruled:** E20 first (a day; feeds §4
regardless) → the MNIST constructions and the `w2d_mafc` retrain (the long
pole) → E19 in parallel once the trainer flag has its regression → the ports
last, after the human read (condition 1).

**Why this exists.** Two literature sweeps found no prior work that *applies* a
known input transform for readout repair, and no input-inversion oracle on
Permuted MNIST. The program then found that it had already run the arm — `C0deg`
in `runs/e11_e10/cures_e10.json`, excluded under a label earned on a different
construction — and that it is the best storage-honest method measured on the
bridging benchmark (S72, uniform arms: **C0deg −0.038 < bridging 0.095 < LwF
0.226**, diag form). Ruling B: bridging has no regime where the map is known.
So the question E18 answers is the one the sweep called interesting, with the
arm the sweep said nobody has: **on benchmarks with a characterizable
transform, does applying the map beat estimating the drift?** Known map → apply
it; unknown map → estimate it; here is what each costs. Bridging is one row.

**What it is not.** Not a method paper's headline experiment. C0deg is the
trivial use of a resource (catch 24) — the point is that trivial use *wins*,
and the paper's §7 is a costed decision table, not a contribution claim.

**Standing rules applied, by name:** displayed equations for every registered
quantity (§3); reference numbers cited to artifacts (§2a); trivial-use pricing
per resource (§2b); checkpoint audit table (§5); every gate ships with a
positive control (§6); CLAUSE → JOB → ARTIFACT before launch (§8); predictions
two-sided (§9); fp32 shadow on every run entering a 1e-6-gated comparison;
constructions recorded in `arm` and verified from artifacts, not from this
document. No timebox.

---

## 0. Review amendments carried in from the 2026-09-15 review

| # | finding | resolution here |
|---|---|---|
| B1 | A1 was defined in the non-deployable direction (= SNAP); the deployable inversion is C0deg, already measured, beats bridging | C0deg is the known-map arm; SNAP is its own anchor row; bridging is one row (Ruling B) |
| B2 | Permuted-MNIST-family benchmarks cannot distinguish generative repair from replay (`runs/e8/MEMO.md` §3a); 90° rotation is a pixel permutation (measured: 90→0 generator max\|err\| 0.00e+00) | **disjoint-content constructions** for both MNIST benchmarks (§1); P3 fired on the live constructions AND on the shared-content must-fail (§6) |
| catch 24 (twice) | bridging dominated on both assumed resources | resource table with trivial-use rows (§2b); SNAP and C0deg are anchors, not cures |
| catch 20 | "runs on existing checkpoints" needs an audit table | §5 |
| catch 21 | "per-task linear heads" premise was false for scratch arms | scratch arms are one shared head (`use_task_heads: False`, cited in §2a); repairs are per-task by construction and say so |
| catch 33 | no CLAUSE → JOB → ARTIFACT | §8, before launch |
| Ruling 2 (formula) | H-X2 was ambiguously registered | §3 displays every formula; artifacts record `formula` |
| S72 | fp16 era reloads miss the 1e-6 gate; constructions drift under names | `--fp32-shadow` on every run; `benchmark`, `disjoint_content`, angles recorded in `arm` and asserted from artifacts (§6, P8) |
| prior work already in-repo | the estimated-drift family exists: E9's chained Procrustes → E13's tier-0 stack, ρ **+0.075 [+0.052, +0.098]** on HAR/OFF (E11-era); C2 oracle-prototype Procrustes ρ **0.687** on S72 | SDC/LDC ports are the field's named members of a family already measured weak here; tier-0 and C2 are anchors on the estimated side (§2c) — so the ports are priced against the program's own estimator before they run |

### 0a. Conditions of sign-off (2026-09-16) — each binds before the job it names

| # | condition | where it binds |
|---|---|---|
| **1** | **The port equations get a human read against the papers.** P10's transcription into `docs/E18_ports.md` is written "at the level I'm confident of"; the gate is met only when both papers have been read in full and the transcription verified line by line. An unfaithful port of LDC is the one outcome worse than no port. | `jobs_e18_ports` does not launch before this. The MNIST constructions and the A0 / C0deg / bridging / reset arms may run now. |
| **2** | **Which rows are the paper's table.** Main text (§7, 1.25 pages): **nothing, LwF, SDC-port, LDC-port, C0deg, reset-with-data** — six rows — with **bridging seventh if space allows** (a measured intermediate). Appendix: **SNAP, NCM-era, NCM-oracle, C2-oracle, tier-0.** | `scripts/e18_row.py` emits both tables directly (`runs/e18_row.json` → `main`, `appendix`), so no transcription sits between the artifact and §7. |
| **3** | **The crux prediction is named** (§9): *does LDC's port close the gap to C0deg, or does the known map beat the best estimate by a margin?* The internal anchors (tier-0 ρ 0.075, C2-oracle 0.687, C0deg above the refit ceiling) set the prior at "known ≫ estimated"; **LDC ≥ 0.9 on HAR changes §7's shape**, and its probability is visible in the table. | §9, first row. |
| **4** | **Compute is stated** (§11), from measured per-run times, so no Sunday-style surprise recurs. | §11. |

*Wording ruled with the sign-off:* bridging's ρ **1.03 [0.83, 1.23]** is
"approximately one", not "above one" — the interval spans 1. Bridging recovers
roughly the era-checkpoint accuracy; nothing says it exceeds it. Applied to the
ledger's one-liner and H-X1 row.

---

## 1. Benchmarks and constructions

| benchmark | construction | transform family | map k → current | arms exist? |
|---|---|---|---|---|
| **HAR** | `har_subject` (E10; subject-disjoint content, held-out test subject per task; partition as-executed `1104af185c87`, x86 only) | 9×9 affine calibration per task: gain, offset, z-rotation, wiring permutation (`src/data/har_shift.py` `SHIFT_SPEC`) | `M_4·M_k⁻¹(x − b_k) + b_4`, exact (round-trip 4.8e-08) | **yes** — S72 OFF arm, `ckpt_e10off_ec_seed{42,1337,2024}` + fp32 shadow (§5) |
| **Permuted MNIST, disjoint content** (new) | `PermutedMNISTBenchmark(disjoint_content=True)`: the 60k train images split into 5 disjoint 12k subsets by a seeded index permutation, the 10k test images into 5 disjoint 2k subsets, one subset per task; task 0 identity permutation as before | pixel permutation | undo `P_k`, apply `P_4`; exact | no — new runs, MLP and LSTM-OFF backbones |
| **Rotated MNIST, disjoint content** (new) | `RotatedMNISTBenchmark(disjoint_content=True)`, same split rule; angles 0 / 22.5 / 45 / 67.5 / 90 (`src/data/rotated_mnist.py:120-122`) | image rotation, bilinear | rotate by `θ_4 − θ_k`; **exact for k = 0 (90° is a permutation), lossy for k = 1, 2, 3** (round-trip rel. MAE 0.216) | no — new runs, MLP backbone |
| Permuted MNIST, shared content (existing) | `PermutedMNISTBenchmark` as shipped — every task the full MNIST | — | — | **used only as P3's must-fail control** (`ckpt_e17_mlp_seed*`, fp16 suffices there) |

**Why disjoint content, stated once.** With shared content, bridging's pseudo-old
data *is* the old training data (the E8 ruling), and C0deg on task k *is* task
4's own test evaluation — both tautologies. With disjoint subsets, pseudo-old
data is novel content in old coordinates, C0deg scores the current model on
images it never saw, and reset-with-data (task k's own subset) is a different
input from bridging's. The construction is E10's repair, transposed. The split
applies to **both** train and test, for the reason given in the second
sentence.

**`disjoint_content` ships opt-in, default off**, so the flag-off path is the
construction every existing MNIST artifact was produced on. Regression, before
launch (§6, P0): the flag-off loaders' first 50 batches hash identically to the
current loaders' at seeds 42/1337/2024, and the deterministic anchor pair
`runs/w2d_mafc_seed42_{a,b}` (25/25 bit-identical, `benchmark: 'permuted'`,
4 threads) is reproduced 25/25 by a retrain on the amended code. A syntax check
cannot see this; the retrain can.

**Backbones.** MNIST-MLP is the E17 arm (`E17_MLP`: `--backbone mlp
--no-adapters`, 266,752 params, one shared head, `use_task_heads: False`);
MNIST-LSTM is the `mafc_off` arm (`W1_BASE + --no-adapters`). HAR is the S72
OFF arm. **Every arm is scratch-trained with one shared head**; the repair arms
below refit or re-read *per task* by construction, and the table says so in its
caption. No task-incremental per-task-head arm exists in this program (catch
21's own example), and none is claimed.

---

## 2. Arms

### 2a. The six arms, plus anchors — per benchmark, per old task k at θ_T

| arm | mechanism | resource assumed | on HAR (S72) | on MNIST |
|---|---|---|---|---|
| **A0 nothing** | deployed head on the drifted encoder | — | `acc_orig` | new |
| **SDC-port** | era prototypes translated by drift estimated from *current* data through the previous and current model (Yu et al., CVPR 2020: Gaussian-weighted mean of feature displacements around each old prototype); NCM readout over compensated prototypes; applied sequentially at every boundary | previous model available during the next task's training; stored era prototypes | new (from era checkpoints, §2d) | new |
| **LDC-port** | era prototypes mapped by a linear projector fit on current data to send old-model features to new-model features (Gomez-Villa et al., ECCV 2024); NCM readout; sequential | same as SDC | new | new |
| **C0deg** | re-layout task-k input into the current frame with the known map; current encoder, current head; no refit | the map | **−0.0379 [−0.047, −0.029]** | new |
| **bridging (C3)** | pseudo-old inputs (current-task inputs re-laid into task k's frame), labelled by the era snapshot, convex head refit on θ_T features | the map **and** the era snapshot | **0.0954 [0.084, 0.107]** | new |
| **reset-with-data** | convex head refit on task k's own training subset at θ_T (`acc_refit`) | old data stored | screen has it | new |
| *anchor* SNAP | era snapshot deployed as-is | era snapshot | 0 by construction | 0 |
| *anchor* NCM-era | uncompensated era prototypes, NCM at θ_T | era prototypes | new | new |
| *anchor* NCM-oracle | era prototypes recomputed with θ_T on stored old data, NCM | old data (oracle) | new | new |
| *anchor* C2-oracle | Procrustes with oracle θ_T prototypes, era head (`cure_screen.py`) | old data (oracle) | ρ **0.687** | new |
| *anchor* tier-0 | E9/E13 chained per-boundary Procrustes + era head | previous model at boundaries; stored 256×256 maps | E11-era ρ 0.075 → **re-run on S72** | new |
| *row* LwF λ\* | training-time regularization, AVG-selected per construction | previous model during training | **0.2259** (S72, λ\* = 1.0) | not run here (a different construction needs its own sweep; scoped out, stated in the caption) |

Cited fields for the HAR values: `runs/s72_row.json` (`row.diag`), `runs/e10ec/cures_e10.json`
via `scripts/rho_percell.py --bench e10` (C2). Every "new" cell is produced by
§8's jobs; none is written down before its artifact exists.

**The paper's table (condition 2).** Main text: nothing · LwF λ\* · SDC-port ·
LDC-port · C0deg · reset-with-data, in that order, per benchmark, with the
storage column beside accuracy (§7); bridging as a seventh row if space
allows. Appendix: SNAP · NCM-era · NCM-oracle · C2-oracle · tier-0, same cells,
same population. `scripts/e18_row.py` writes both from one pass over the same
artifacts (`runs/e18_row.json` keys `main` and `appendix`); the caption states
the arm names as the artifacts record them and that every scratch arm has one
shared head.

**The comparisons that matter, in order.**
1. **C0deg vs SDC-port / LDC-port** — applying the known map vs estimating the
   drift, same benchmark, same population of cells. The question.
2. **SDC/LDC vs their own anchors** — NCM-era below, NCM-oracle and C2-oracle
   above. A port that does not beat NCM-era is not repairing; a port near its
   oracle has nothing left to estimate. Composite rule: each measured alone.
3. **bridging vs everything** — one row, wherever it lands (Ruling B).
4. **Rotated MNIST, lossy pairs (k = 1, 2, 3)** — the only place a lossy map can
   lose to an estimator. k = 0 is exact and is reported as the permutation
   case it is.

### 2b. Catch 24 — trivial use of every assumed resource, priced first

| resource | trivial use | measured (HAR, S72) | in the table as |
|---|---|---|---|
| the map | apply it (C0deg) | −0.0379 | the arm under test |
| the era snapshot | deploy it (SNAP) | 0 by construction | anchor, never ranked |
| the previous model during training | LwF (matched, swept, competence-floored) | 0.2259 | training-time row |
| era prototypes | NCM over them at θ_T (NCM-era) | new | anchor |
| old data | refit the head on it (reset-with-data) | screen | ceiling row |

If a method's row does not beat the trivial use of what it assumes, the row
reports that, at full prominence. That sentence is why this contract exists.

### 2c. What the ports add to what is already measured

The program's estimated-drift family is tier-0: a per-boundary orthogonal map
estimated while the previous encoder is in memory, chained, read with the era
head (E9 design, E13 measurement: ρ 0.075 on HAR/OFF, 0.120 on MNIST/OFF,
"weak everywhere measured"). SDC differs in *what* is estimated (a per-class
translation, not a rotation) and LDC in *the map's class* (general linear,
least-squares on current data, not orthogonal Procrustes). They are the
field's named baselines, and a reviewer will name them; they are ported
faithfully and priced against tier-0 and C2. **A port that reads far above its
anchors is checked before it is believed** (catch 30: emphatic is what passing
looks like).

### 2d. Port design decisions — `docs/E18_ports.md`, written before launch (P10)

Recorded here so the decisions are visible; transcribed equations live in the
appendix table once the papers have been read in full, which is a precondition
and not a formality.

- **Readout.** Both papers compensate *prototypes* and classify by nearest class
  mean. The port keeps that: stored era prototypes (class means of era training
  features under θ_k, computed at the boundary — post hoc from the fp32 era
  checkpoint is identical, and the storage cost is `n_classes × d` per task),
  NCM readout at θ_T. Converting the compensation to a linear-head update would
  be *our* method, not theirs; the NCM-era anchor is the port's own baseline
  and the linear-head arms are compared to it on the same cells.
- **Sequential application.** Compensation at every boundary t−1 → t using the
  previous model on the current task's training data, composed across
  boundaries, as the papers deploy it. The era checkpoints θ_0..θ_4 with fp32
  shadow supply the previous model at every boundary; the current task's
  training data supplies the estimation set. Nothing oracle enters.
- **SDC's σ** (kernel width) is a hyperparameter selected on AVG at n ≥ 3 on a
  held-in construction (HAR), never by the launcher; the sweep is reported in
  full. **LDC's projector** is linear with bias, least squares to convergence
  (closed form, knob-free), regularization as in the paper if the paper
  regularizes — to be transcribed.
- **Controls (§6, P5):** zero-drift (θ_t = θ_{t−1} ⇒ SDC's Δ = 0 and LDC's
  P = I to solver tolerance ⇒ port == NCM-era, exactly); oracle-drift (estimate
  replaced by the true prototype displacement from stored old data ⇒ port ==
  NCM-oracle, exactly, by construction for SDC); E9's estimated-vs-oracle map
  agreement `‖Q_est − Q_oracle‖_F / ‖Q_oracle − I‖_F` printed per boundary as
  a diagnostic, no bar.

---

## 3. Quantities — every formula displayed, recorded in the artifact

Accuracy matrix `R[i, j]` = accuracy on task j after training task i; `T − 1`
the final row; old tasks `j < T − 1`. For a method m with accuracy `a_m(j)` on
task j at θ_T:

```
forgetting_m   = mean_j ( R[j,j] − a_m(j) )                      # diag form — the paper's §3.1 F_total; REPORTED
forgetting_m^p = mean_j max(0, max_l R[l,j] − a_m(j))            # peak-clipped, metrics.py — printed beside, never the verdict
ρ_m            = mean_j ( a_m(j) − a_0(j) ) / ( a_refit(j) − R(j) − a_0(j) )   # bias-adjusted, per cell, guard D > 0.02 per cell
recovery_m     = ( a_m − a_0 ) / ( a_reset − a_0 )                # share of the reset-with-data gain, pooled means, reported beside ρ
```

`R(j)` is the instrument bias `acc_refit_ceiling − acc_ceiling`. Intersection
population: cells valid under the guard for **every** ranked method; full
population printed beside; exclusions counted. `formula`, `honest`, and the
cell list are written into every artifact (`hx2_forgetting.py` convention).
Means of ratios, never ratios of means. Intervals: binomial SE of the method
term propagated through the per-seed mean, 1.96×; floors added per §6 P6.

---

## 4. Readings, pre-committed, two-sided

| outcome | what §7 says |
|---|---|
| C0deg ≥ both ports by more than the floor on HAR and on Permuted-disjoint | *known map → apply it.* Estimation is what you do when you cannot characterize the change. The hardware-revision message in one sentence. |
| a port ≥ C0deg on any exact-map benchmark | the estimator recovers something the map misses — encoder drift beyond the layout. Report at full prominence; C0deg's scope sentence narrows to "layout drift". |
| a port ≥ C0deg on Rotated lossy pairs only | *the map's loss is the boundary*: applying a lossy map costs interpolation, estimation does not. §7 states the crossover as a function of map fidelity, measured. |
| ports ≤ NCM-era | the ports do not repair here; reported as such with the anchors; the estimated-drift family's weakness in this program (tier-0) generalizes to the field's versions. Not a claim about the papers' own benchmarks. |
| bridging above C0deg anywhere | reported, with the resource table beside it; Ruling B's scope sentence is revisited *only* in that benchmark's caption |
| P3 live fails on a disjoint construction | that construction cannot license generative repair either; bridging's row on it is struck; C0deg and the ports are unaffected (they generate nothing) |

---

## 5. Checkpoint audit table (catch 20) — verified before sign-off

| arm | seed | dir on volume | fp32 shadow | loads (`PLCM.load_era`) | reproduces matrix (1e-6) | status |
|---|---|---|---|---|---|---|
| HAR S72 OFF | 42 | `ckpt_e10off_ec_seed42/mafc_seed42` | `…_fp32` | ✓ (screen ran) | ✓ 4.0e-09 (`runs/s72_row.json`) | **audited** |
| HAR S72 OFF | 1337 | `ckpt_e10off_ec_seed1337/…` | ✓ | ✓ | ✓ | audited |
| HAR S72 OFF | 2024 | `ckpt_e10off_ec_seed2024/…` | ✓ | ✓ | ✓ | audited |
| HAR S72 floor | a, b | `ckpt_e10off_ec_floor4tec_{a,b}/…` | ✓ | ✓ | ✓ | audited |
| MNIST shared (P3 must-fail only) | 42/1337/2024 | `ckpt_e17_mlp_seed*` | fp16 only | ✓ (E17) | n/a — not a 1e-6 row | usable for P3 only |
| Permuted-disjoint MLP / LSTM | — | **none** | — | — | — | new runs |
| Rotated-disjoint MLP | — | **none** | — | — | — | new runs |

Every task boundary 0..4 present for the S72 arm (the screen loaded all five).
`scripts/audit_checkpoints.py` is re-run on the new directories before any
analysis reads them (criterion: `load_era` on the fp32 shadow, then matrix
reproduction).

---

## 6. Preconditions — each a gate, each with the case where it must fail

| P | gate | positive control / failure mode it must detect |
|---|---|---|
| **P0** build gate | `disjoint_content` opt-in; flag-off loader batches hash-identical (3 seeds × 50 batches); `w2d_mafc_seed42_{a,b}` reproduced 25/25 by retrain on the amended code; `arm` records `benchmark`, `disjoint_content`, `angles` | flip the flag on: hashes must differ and the per-task index sets must be pairwise disjoint (asserted) |
| **P1** checkpoint audit | §5, re-run on new dirs | a deliberately truncated copy must fail `load_era` |
| **P2** capture-point identity | `assert_path_identity` per backbone (MLP, LSTM), on every live cell, both eras | its existing positive control (wrong module hooked ⇒ inequality) fires per script |
| **P3** generative honesty | `generative_certificate.py` v3 on each disjoint construction (live) | **must FAIL on shared-content Permuted MNIST** (`ckpt_e17_mlp_seed*`) — the E8 construction transposed; if it passes there it cannot fail and licenses nothing |
| **P4** C0deg controls, per MNIST construction | G1′ content disjointness from the benchmark's index sets; CA′ identity permutation ⇒ C0deg == deployed, bit-identical logits; CC′ wrong permutation on top of the right relayout ⇒ below C0deg in **every** cell (12/12 per arm) — bar stated here | CC′ is the must-fail; it passed 12/12 × 2 × 2 on HAR twice |
| **P5** port controls | zero-drift ⇒ port == NCM-era exactly; oracle-drift ⇒ port == NCM-oracle exactly | a port with a sign error in the displacement fails zero-drift by construction; a port that ignores its input fails oracle-drift |
| **P6** floors, per quantity, 4 threads | floor pair (seed 42 twice) for every new arm; per-method forgetting and ρ floors from the screen on the pair; the MLP-MNIST arm is **0/15 deterministic** (MEMO_e17 §5) — its floors are nonzero and are read, not assumed; LwF-per-λ lesson noted: floors are per config | a pair with a flag mismatch (era on/off) must be caught by P8 before it is read as a floor |
| **P7** formulas | §3 displayed; `formula` in every artifact; verdict strings computed from the printed arrays | a hand-written verdict string is refused by review |
| **P8** arm identity from artifacts | benchmark, construction flags, `era_checkpoints`, fp32-shadow delta 0.0 every boundary, threads, backbone, `use_task_heads: False`; **uniformity checked ACROSS arms per benchmark** | the E16 `har`/`har_subject` case: a construction mismatch must fail this before any row is read |
| **P9** predictions filed | §9, in this document, before the first job | — |
| **P10** ports read in full | `docs/E18_ports.md` with transcribed equations, checked against the papers, before `jobs_e18_ports` exists | a port whose equations are not in the table is not built |
| **P11** era-status and shadow uniform | every run in a comparison carries `--era-checkpoints --fp32-shadow`; checked across arms | S72's rule; mixed status fails P8 |

---

## 7. Storage row — computed from artifacts, never by hand

`scripts/e18_storage.py` writes bytes per task for: the map (HAR: 9×9 + 9
floats; Permuted: a 784-entry permutation; Rotated: one angle), the era
snapshot (fp32 shadow file size), era prototypes (`n_classes × d` floats),
tier-0's stored map (256×256), LwF's previous model (one model), the reset
buffer (task k's training subset as stored tensors). The §7 decision table
prints accuracy and bytes side by side. This is the row E16's contract named
and never built; here it has a job (§8).

---

## 8. CLAUSE → JOB → ARTIFACT (catch 33) — every clause has a path

| clause | launcher / script | artifact |
|---|---|---|
| §1 Permuted-disjoint MLP, 3 seeds + floor pair | `jobs_e18_pmd_mlp` | `runs/e18_pmd_mlp_seed{s}/`, `runs/e18_pmd_mlp_floor4tec_{a,b}/`, `ckpt_…` + `_fp32` |
| §1 Permuted-disjoint LSTM-OFF, 3 seeds + floor pair | `jobs_e18_pmd_lstm` | `runs/e18_pmd_lstm_seed{s}/`, floor pair, ckpts |
| §1 Rotated-disjoint MLP, 3 seeds + floor pair | `jobs_e18_rmd_mlp` | `runs/e18_rmd_mlp_seed{s}/`, floor pair, ckpts |
| P0 build gate | `scripts/e18_build_gate.py` | `runs/e18_build_gate.json` (loader hashes, disjointness assert, `w2d_mafc` reproduction 25/25) |
| P1 checkpoint audit | `scripts/audit_checkpoints.py` | `runs/e18_ckpt_audit.json` |
| P3 generative honesty, live + must-fail | `scripts/generative_certificate.py --construction {pmd,rmd,shared}` | `runs/e18_p3_{pmd,rmd}.json` (live), `runs/e18_p3_shared_mustfail.json` |
| §2a screen: A0, C0deg, C3, reset, SNAP, C2-oracle, per benchmark | `scripts/cure_screen.py --arms {e10ec,e18_pmd_mlp,e18_pmd_lstm,e18_rmd_mlp}` (MNIST relayouts: permutation existing; rotation added) | `runs/e18_{bench}/cures.json` |
| §2a tier-0 on S72 and on the MNIST constructions | `scripts/e13_stack.py --arms …` | `runs/e18_{bench}/tier0.json` |
| §2a SDC-port, LDC-port, NCM-era, NCM-oracle + P5 controls | `scripts/e18_ports.py` (after P10) | `runs/e18_{bench}/ports.json`, `runs/e18_ports_controls.json` |
| §2d SDC σ sweep (HAR, n = 3, AVG) | `scripts/e18_ports.py --sweep-sigma` | `runs/e18_sdc_sigma.json` |
| P4 C0deg controls on MNIST constructions | `scripts/c0deg_controls.py --bench {pmd,rmd}` (G1′/CA′), `scripts/c0deg_cc2.py --arms {…}` (CC′) | `runs/e18_{bench}/c0deg_controls.json`, `c0deg_cc2.json` |
| §3 H-X2-form table, both formulas, cell lists | `scripts/hx2_forgetting.py --arms {…} --include-c0deg [--formula peak]` | `runs/e18_{bench}/hx2{,_peak}.json` |
| §7 storage row | `scripts/e18_storage.py` | `runs/e18_storage.json` |
| P6 floors | floor pairs above + screen on `…floor` sets | `runs/e18_{bench}floor/…` |
| the §7 decision table | `scripts/e18_row.py` (finish, P8 across arms, P6, both formulas, storage) | `runs/e18_row.json` |

**Every clause has a path. No clause is a sentence.** `jobs_e18_ports` does not
exist until `docs/E18_ports.md` does (P10).

---

## 9. Predictions on record — filed before the first job (calibration series: 15 scored, 3 misses)

| prediction | odds |
|---|---|
| **CRUX (condition 3) — HAR (S72), LDC-port ρ.** `≥ 0.9` (the estimate closes the gap; §7 changes shape) / `[0.5, 0.9)` (estimation gets most of the way; §7 is subtler) / `[0.0, 0.5)` (known ≫ estimated; §7 as drafted) / `< 0` (harmful) | **10% / 15% / 60% / 15%** — the prior is set by the program's own estimator (tier-0 0.075) and its oracle ceiling (C2 0.687), both below C0deg by a wide margin. *This row decides the paper.* |
| HAR (S72): C0deg beats both ports by more than the floor (ρ gap ≥ 0.3) | **~85%** — the estimated family is weak here (tier-0 0.075) and C0deg's accuracy (0.891) sits above the refit ceiling (0.772) |
| HAR: SDC-port ρ ∈ [0.0, 0.5] / > 0.5 / < 0 | **55% / 20% / 25%** — a translation is a weaker model of this drift than a rotation, and tier-0's rotation already reads 0.075 |
| HAR: tier-0 re-run on S72 within ±0.10 of E11-era 0.075 | **~60%** |
| Permuted-disjoint (MLP): C0deg forgetting ≤ 0.02 (≈ DIAG) | **~85%** |
| Permuted-disjoint (MLP): bridging recovers > 80% of the reset-with-data gain — *now a real test* | **~60%** |
| Permuted-disjoint: C0deg > bridging | **~80%** |
| Permuted-disjoint: LSTM-OFF and MLP rankings agree | **~65%** |
| Rotated-disjoint: C0deg on the lossy pairs (k = 1–3) reads ≥ 5pp below the exact pair (k = 0) | **~55%** |
| Rotated-disjoint: a port beats C0deg on at least one lossy pair | **~25%** — the crossover, if it exists, is the result worth most |
| P3 live passes on both disjoint constructions | **~75%** (must-fail on shared is a requirement, not a prediction) |
| MLP-MNIST floors: per-method forgetting floor > 2pp for at least one method | **~50%** — the arm is 0/15 deterministic; the screen floor on HAR was 0.0000 because the checkpoints were |

*Calibration note:* the E14 set was three-for-three under-confident; E16 missed
three of four in the direction of more structure than predicted; E17 fired
twice. The misses have all been *more structure than expected*. Odds above are
not shaded for that.

---

## 11. Compute (condition 4) — from measured per-run times, Modal CPU at 4 threads unless stated

| item | count | per-run (measured) | CPU-hours | wall |
|---|---|---|---|---|
| Permuted-disjoint MLP: 3 seeds + floor pair | 5 | 19.2 min (`e17_mlp_seed42`) | 1.6 | fanned out |
| Permuted-disjoint LSTM-OFF: 3 seeds + floor pair | 5 | 29.1 min (`e4off_ec_seed42`) | 2.4 | fanned out |
| Rotated-disjoint MLP: 3 seeds + floor pair | 5 | ≈ 19 min (same arm, rotation costs nothing at train time) | 1.6 | fanned out |
| `w2d_mafc` retrain for P0 (one replicate, compared against both existing) | 1 | ≈ 30 min (adapters-ON LSTM, MNIST) | 0.5 | **gates the 15 above** |
| screens (`cure_screen` on 4 sets + 3 floor sets), tier-0, P3 live ×2 + must-fail, C0deg controls ×2, storage | ~14 analysis jobs | 3–8 min each (S72 screens: ~5 min) | ~1.2 | serial or fanned |
| SDC σ sweep (HAR, 4–5 values × 3 seeds, analysis-only on stored era checkpoints) + ports on 4 sets + P5 controls | ~10 analysis jobs | ≈ 5 min each | ~0.8 | after condition 1 |
| **E18 total** | 16 training + ~24 analysis | | **≈ 8 CPU-hours** | **two Modal batches: ~40 min (build gate) then ~45 min (runs), analysis under an hour; the long pole is the human read** |
| E20 (for reference, own contract): MLP-MNIST probes on `ckpt_e17_mlp_seed*` | 1 analysis job | minutes | < 0.2 | — |
| E19 (own contract): HAR OFF incl + excl relaunch (6) + floors (4), MNIST-MLP excl (3) + floor pair (2) | 15 training | 5 / 19 min | ≈ 2.5 | one batch |

At Modal's CPU rate (4 vCPU, ~$0.6 / hour) the whole E18–E20 program is
**under $15** and about **two half-days of wall clock**, dominated by waiting.
E14's ResNet arm is not in this program (E20 relocated to the MLP arm as
primary; a ResNet secondary would add ~1–2 L4 GPU-hours and is not budgeted
here until asked for).

---

## 10. Decision — what §7 becomes, with no date attached

With HAR, Permuted-disjoint and Rotated-disjoint in hand:

- **Applying beats estimating everywhere the map is exact** → §7 is a decision
  table — known map → apply it; unknown → estimate it, and here is what
  estimation costs — with the hardware-revision story told as *characterize
  the change and apply it*. Bridging one row. SDC/LDC two rows. E18b
  (Split-CIFAR-100 with a characterized simulated revision, ResNet-50, with
  CoTTA/EATA supervised ports) proceeds as the pretrained instance of the same
  table.
- **A crossover on the lossy pairs** → §7 gains the map-fidelity axis, measured;
  E18b's simulated revision is designed to sit on both sides of it.
  **Correction, 2026-09-16, from the first read:** bilinear rotation on 28×28
  is *not* lossy in any way an MLP classifier sees — the rotated lossy pairs
  read within 0.3pp of the exact one. So E18b's perturbation sweep cannot use
  geometric interpolation as its error source; the map-fidelity axis needs
  **actual calibration error — gain and offset on the sensor channels**, which
  is what a real hardware revision carries anyway. A correction to the
  amendment, not a finding about the crossover.
- **A port beats C0deg on an exact-map benchmark** → the estimator sees drift the
  layout does not carry; C0deg's scope narrows to layout drift, the paper says
  so, and the decomposition (F_enc on that benchmark) is re-read beside it.

No Sunday. The runs are cheap; the reading is not, and it is written after the
artifacts exist.
