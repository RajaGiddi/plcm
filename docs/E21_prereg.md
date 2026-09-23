# Pre-Registration: E21 — Repair Under an Approximate Map

**Status: SIGNED 2026-09-17** (v2; v1 reviewed the same day; four blocking
findings accepted, §0). Three notes recorded with the sign-off, §0a. **Smoke
run; budget written from it (§9). Sweep launched 2026-09-17 00:20 (36 jobs), STOPPED at 00:27 on instruction, RELAUNCHED 11:38 — swap family complete (10/10); **the gain family died 11/13 on an unbounded line search (§3 amendment); gain + offset re-run bounded (26 jobs).** Runs on S72 checkpoints only — no new
construction, no training. **Default landing: October, paper 2.** Nothing here
enters the September 25 submission; the cost clause (§9) makes the 22nd
unreachable and the contract does not pretend otherwise.

**The question.** Every repair result so far assumes the map $M_k$ is exact.
Real hardware revisions ship with calibration error. Under
$\hat M_k = M_k + \epsilon$: re-layout (A1) applies a wrong map to every query
forever, and its head — the current model's, fit on unshifted data — reads an
ε-off frame at test time. Bridging (A2) generates its pseudo-data through the
wrong map at *refit* time, its era teacher labels an ε-off frame, and the
head it fits is trained ε-off and tested on the true frame. **The mechanism
question is therefore narrower than v1 wrote it:** *is a head fit on shifted
data more robust to the unshifted test than a fixed head fit on unshifted
data is to the shifted test?* That is asymmetric and empirical, and a weaker
basis for a crossing than "absorption". It is the one regime in which
bridging could be worth a row of its own, and the one the hardware-revision
application lives in.

**Deployment constraint, stated verbatim from E18 §0a and load-bearing here:**
*one deployed encoder; the era model may be consulted at repair time and is
then discarded.* Without that sentence the regime belongs to SNAP (deploy the
era model: forgetting 0 by construction, no map needed), which is printed as
an unranked anchor in every table.

**Prior work this arm is positioned against** (named before the run):
Mummadi et al. 2021 (arXiv:2106.14999) learn an input transformation at test
time from the frozen model's confidence with a diversity term — the closest ML
prior; CoTTA (Wang et al., arXiv:2203.13591) uses a past weight-average as a
pseudo-labeling teacher but corrects the model, not an input map; **Kanai et
al. 2023 (arXiv:2308.02153) — UNVERIFIED**: cited from sweep 3 as refining
few-parameter extrinsics against a frozen network's outputs; the citation is
checked against the paper before the positioning sentence is written, and
until then it is not relied on. The combination here — a stored era
checkpoint supervising a few-scalar input map — was not located in the sweep
and does not exist in this repository (`grep` of `scripts/`, `src/` for
test-time map refinement: none; E9's transport estimate is feature-space;
the learned adapters are training-time).

---

## 0. Review amendments (v1 → v2)

| # | finding | resolution |
|---|---|---|
| B1 | v1 wrote bridging's head as fit "through $\hat M_k^{-1}$ on the corrupted distribution". The generator maps **forward** ([cure_screen.py:280](../scripts/cure_screen.py#L280) `har_relayout(xcur_tr, CUR, k, maps)`); the teacher labels an ε-off frame; the head is trained ε-off, tested true | §2's **where-ε-enters table**, one line per arm, each citing the implementing line; $M_4$ exact; the mechanism question restated above |
| B2 | the ε = 0 identity check conflated harness correctness with optimizer behaviour for A3–A5 | §5: exact identity for A0/A1/A2/A6 against the **seeded** S72 artifacts; for A3–A5 the ε = 0 cell is a two-sided *measurement* of drift from a correct map |
| B3 | "most recent head" as the wrong-direction teacher is CoTTA's teacher, not a known-wrong one | §5: **deranged-teacher must-fail** — the era model with outputs permuted by a fixed derangement, per-cell bar 12/12, stated here |
| B4 | A2 and A5 consume the era checkpoint, whose trivial use (SNAP) dominates both | the deployment constraint above; SNAP an unranked anchor in every table |
| predictions | "A1 degrades linearly" contradicted CD (gain 0%, offset 1%, permutation 71% of C0deg's recovery, `runs/e10ec/c0deg_cc2.json`); "A2 never beats A6" cannot fail; "twenty entries" matched no artifact | §7: per-family predictions; A6 an anchor; the count **read from `docs/appendix.tex` at the moment of writing** (brought current first) |
| sections | no ε-realization seeds, optimizer unpinned, no CLAUSE → JOB → ARTIFACT, no audit table, adaptation data unstated, no A2-on-refined-map | §3 (realizations), §4 (optimizer), §8 (tables), §2 (data), arm A7 |
| cost | "4–6 CPU-hours" did not price the optimized arms through the LSTM | §9: **smoke one cell first**; budget written from the measured per-fit time; October |
| citations | rotation error cited as 4e-7 from a console check | `runs/e18/rotation_roundtrip.json`: 90° round trip max\|err\| **5.4e-6** on 2000 real test images through `rotated_relayout`; the lossy pairs 0.218 rel. MAE |

### 0a. Notes recorded with the sign-off

1. **A5's teacher reads the true frame.** Different from A2's teacher, which
   reads the ε-off pseudo-frame. A5 uses the era model on what it was trained
   on — the query stream in task k's own coordinates — and asks the current
   model, through the refined map, to agree: the cleanest use of the stored
   self in any arm so far, and the one where the teacher's accuracy (0.85
   ceiling on this arm; **0.78 on the smoke cell**) is the actual ceiling on
   the supervision. One sentence in the memo when it reads.
2. **The transductive protocol matters for the crossing.** A3–A5 see
   $\mathcal{X}_k$ during adaptation; A1 and A2 do not. The crossing at matched
   map quality is A5-then-A1 versus A7; the raw A1/A2 crossing is the
   deployment comparison without adaptation. Both belong in the row; they
   answer different questions.
3. **31 scored, 7 misses, every miss toward more structure and less
   determinism.** The per-family predictions in §7 were written against that
   bias deliberately; if they miss in the same direction again, Appendix F
   gets a sentence about what the series has taught about the author's
   priors.

---

## 1. Perturbation families — displayed, seeded, realized

The deployment's map is $\hat M_k$; $M_4$ (the current device) is **exact**
throughout. Maps are in calibrated units as `har_maps` gives them
([cure_screen.py:121-133](../scripts/cure_screen.py#L121-L133)): $(M_k, c_k)$ with
$x_k = M_k Z + c_k$. Per family, per level $\ell$, per realization $r$
(seed $= 20260917 \cdot 100 + 10\ell + r$, recorded):

```
P-gain    :  M̂_k = diag(1 + g_ℓ · s) · M_k,   ĉ_k = c_k,          s_i ∈ {−1,+1} drawn per realization,  g_ℓ ∈ {0, 0.05, 0.10, 0.20, 0.30}
P-offset  :  M̂_k = M_k,   ĉ_k = c_k + o_ℓ · σ_k ⊙ s,             σ_k,i = std of task-k train windows, channel i (calibrated units),  o_ℓ ∈ {0, 0.05, 0.10, 0.20, 0.30}
P-swap    :  M̂_k = Q_r^{(m_ℓ)} · M_k,   ĉ_k = Q_r^{(m_ℓ)} · c_k,   Q = product of m_ℓ disjoint channel transpositions drawn per realization,  m_ℓ ∈ {0, 1, 2, 3}
```

Rotation/interpolation is **excluded**: `runs/e18/rotation_roundtrip.json`
(90° exact to 5.4e-6; the lossy pairs read within 0.3pp of the exact one in
E18), so geometric interpolation is not a usable error source here.

**Realistic calibration range, pre-registered:** ≤ 5% gain ($g \le 0.05$),
≤ 0.1σ offset, **0 swaps** — from the shift spec's own datasheet comment
(consumer MEMS sensitivity tolerance a few percent;
[har_shift.py:100-131](../src/data/har_shift.py#L100-L131)). The shift itself is
±15% gain, so the sweep covers errors larger than the shift. **A P-swap
crossing, if found, is outside the realistic range and the sentence says so.**

**Three realizations per level**, seeded and recorded — one drawn sign
pattern or one chosen transposition is n = 1 on the perturbation axis, and
CC′ showed within-triple and across-triple wirings differ. Level 0 has one
realization by construction (ε = 0).

## 2. Arms, and where ε enters each one

All on HAR `har_subject` OFF, S72 checkpoints (`ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32`),
12 cells (3 seeds × 4 old tasks), probe subset seed 20260916 recorded.
$\mathcal{X}_k$ = task-k **test** inputs in the true frame (344–409 windows) —
the query stream. A3–A5 adapt on it and are then scored on it: TENT's
transductive protocol, stated as such.

| arm | mechanism, by implementing line | where ε enters | resources | role |
|---|---|---|---|---|
| **A0** none | deployed θ_T on $\mathcal{X}_k$ ([cure_screen.py:234](../scripts/cure_screen.py#L234)) | nowhere | — | floor |
| **A1** re-layout with $\hat M_k$ | $x \mapsto M_4 \hat M_k^{-1}(x - \hat c_k) + c_4$, current model, no refit ([cure_screen.py:141-150, 249, 261](../scripts/cure_screen.py#L141-L150)) | **at query time**, every query, forever | map | the arm to beat |
| **A2** bridging with $\hat M_k$ | pseudo $= \hat M_k M_4^{-1}(x_4 - c_4) + \hat c_k$ on task-4 train inputs ([cure_screen.py:280](../scripts/cure_screen.py#L280)); labels $\arg\max p_{\theta_k}(\cdot \mid \text{pseudo})$ ([:285-286](../scripts/cure_screen.py#L285-L286)); head refit on $\theta_T$ features of pseudo, **tested on $\mathcal{X}_k$ in the true frame** ([:287-288](../scripts/cure_screen.py#L287-L288)) | **at refit time**: the generator and the teacher's input; not at test time | map, era model (repair time), refit | the method |
| **A3** TENT-on-map | $\phi^* = \arg\min_\phi \mathbb{E}_{x \in \mathcal{X}_k} H\!\left(p_{\theta_T}(\cdot \mid \mathrm{relayout}_{\hat M_k(\phi)}(x))\right)$; then A1 with $\hat M_k(\phi^*)$ | at query time, through the refined map | map params, current model's entropy | **trivial use of self-supervision** |
| **A4** Mummadi-objective on map params | confidence maximization with a diversity regularizer on the batch marginal, over the same $\phi$ and $\mathcal{X}_k$ — **the exact form is transcribed from Mummadi et al. §3 before launch and written here; this line is the intent, not the transcription** | as A3 | map params, confidence + diversity | second self-supervised baseline |
| **A5** era-teacher map refinement | $\hat y(x) = \arg\max p_{\theta_k}(\cdot \mid x)$ for $x \in \mathcal{X}_k$ (the teacher reads the **true** frame; no map); $\phi^* = \arg\min_\phi \mathbb{E}_x\, \mathrm{CE}\!\left(p_{\theta_T}(\cdot \mid \mathrm{relayout}_{\hat M_k(\phi)}(x)),\, \hat y(x)\right)$; then A1 with $\hat M_k(\phi^*)$; θ_k discarded | as A3 | map params, era model at repair time | **the arm sweep 3 cleared** |
| **A6** re-layout with exact $M_k$ | A1 with $M_k$ | nowhere | exact map | ceiling; **anchor**, constant across ε, never ranked |
| **A7** bridging on the refined map | A2 with $\hat M_k(\phi^*)$ from A5 | at refit time, through the refined map | as A2 + A5 | the matched-map-quality comparison for the crossing question |
| *anchor* SNAP | θ_k on $\mathcal{X}_k$ | nowhere | era model at query time | 0 by construction; printed, never ranked |

**A5's distinction from A3/A4 is the supervision signal:** entropy has minima
wherever the current model is confident, including where it is confidently
wrong; the era teacher's minimum is near the true map if the teacher is
mostly right on the true frame (it is: ceiling 0.85 on this arm).

## 3. Parameterization of the refinement, per family

```
P-gain, P-offset :  φ ∈ ℝ^9  (log-gain corrections / offset corrections, init φ_0 = 0), continuous
P-swap           :  φ ∈ {identity} ∪ {36 transpositions}, discrete; for m_ℓ ≥ 2 a greedy sequence of up to 3 best single transpositions
```

The refined map is $\hat M_k(\phi) = \mathrm{diag}(e^{\phi}) \hat M_k$ (gain),
$\hat c_k + \phi$ (offset), or $Q_\phi \hat M_k$ (swap). A3–A5 never touch the
full 9×9 matrix and never touch the model.

**AMENDMENT 2026-09-17, after the first sweep — the continuous search space is
BOUNDED.** Under the unbounded form, LBFGS's strong-Wolfe line search took
trial steps that sent some $\phi_i$ to huge magnitudes: $e^{\phi_i}$
underflowed to 0 and the relayout's inverse met a zero diagonal — **11 of the
13 gain-family jobs died** (`torch._C._LinAlgError: linalg.inv: the diagonal
element … is zero`; one more by float overflow), while the offset family (no
inverse in $\phi$) and the swap family (discrete) completed. The contract's
own realistic range licenses the bound: a gain correction outside $(\tfrac12,
2)$ or an offset correction beyond $1\sigma$ is outside every calibration
error the sweep considers (a 30% gain error needs a correction of 0.77 or
1.43). Amended parameterization, stated before the re-run:

```
gain    :  M̂_k(φ) = diag( exp( ln 2 · tanh φ ) ) · M̂_k          per-channel correction in (1/2, 2)
offset  :  ĉ_k(φ) = ĉ_k + σ_k ⊙ tanh φ                            per-channel correction within ±1σ_k
swap    :  unchanged
```

$\phi = 0$ is still the identity, so C-ID, C-DRIFT's meaning and the jitter
inits are unchanged. **Both continuous families are re-run bounded** so the
optimizer is identical across A3/A4/A5 *and* across families (the offset
family's unbounded artifacts are kept beside, superseded, never read). The
swap family's artifacts stand. The harness records the parameterization in
every artifact (`optimizer.parameterization`).

## 4. Optimizer — pinned, identical across A3/A4/A5

```
continuous families :  torch.optim.LBFGS on φ, max_iter 50, tolerance_grad 1e-6, line_search strong_wolfe,
                       full batch over 𝒳_k;  3 init jitters φ_0 + N(0, 0.01²) with seeds {0,1,2}; n_iter recorded per fit
discrete family     :  exhaustive over the 37 candidates per greedy step; no seed axis
```

The three optimized arms share solver, budget, jitters and data. Convergence
witness per fit (`n_iter < max_iter`, gradient norm); an unconverged fit is
excluded and counted. The per-cell **spread over the three jitters** is that
arm's optimizer floor.

## 5. Controls — each with the case where it must fail

| control | statement | bar |
|---|---|---|
| **C-ID** identity at ε = 0 | A0/A1/A2/A6 at level 0 reproduce `runs/e10ec_seeded/cures_e10.json` (`acc_orig`, `C0deg`, `C3`, `C0deg`) per cell | max \|Δ\| < 1e-6 (float32 representation), 12/12; otherwise the harness is wrong and nothing is read |
| **C-DRIFT** A3–A5 at ε = 0 | a *measurement*, two-sided: does the objective move a correct map? Report $\|\phi^*\|$ and A3/A4/A5 − A1 per cell | no bar; reading pre-committed in §6 |
| **C-DT** deranged teacher | A5 with $\hat y \to \pi(\hat y)$, $\pi(y) = (y + 1) \bmod 6$, a fixed derangement | A5-deranged **< A5 in 12/12 cells at every level ≥ 1**; if not, A5's gain is not coming from the teacher |
| **C-FLOOR** per level | spread over 3 realizations × 3 jitters, per cell, per arm, per quantity; the model-seed axis is the population, not a floor | no delta between arms below its floor is read |
| **C-WIT** arm identity | S72 artifacts' `arm` fields (benchmark, era, shadow, adapters off, threads) re-asserted at the top of the harness; path identity (`assert_path_identity`) on every cell | as in every consuming script |
| **C-EXACT** | A6 constant across all levels and realizations to 1e-6 | it does not depend on ε; a change is a harness error |

## 6. The measurement and the pre-committed readings

For each family, level, realization, arm: forgetting in the **diag form**
(the paper's §3.1 definition; peak-clipped printed beside, never the verdict),
pooled over 12 cells, intervals over the three model seeds; the curve of
forgetting against $\|\epsilon\|$ per arm.

```
forgetting_arm(ℓ, r) = mean_j ( R[j,j] − acc_arm(j; ℓ, r) )
```

**Headline 1 — the A1/A2 curves, per family.** Three shapes:

| shape | reading |
|---|---|
| A1 below A2 at every level | bridging has no regime under map error either; the dominance result is unconditional on this benchmark and §7 says so |
| **A1 and A2 cross at some $\epsilon^*$** | bridging wins above $\epsilon^*$; report $\epsilon^*$ in calibration units against the pre-registered realistic range — inside it, a regime; outside it, a curiosity, stated as such |
| both degrade together, gap within floor | the error passes through both; the *place* of repair does not matter for this error class |

Also read: **A7 vs A1-on-refined-map** — the crossing question at matched map
quality.

**Headline 2 — A5 vs A3/A4.** If A5 does not beat both by more than the floor
in any family, the era teacher is replaceable by self-supervision — the fifth
time a stored-self component reads as redundant — reported as such.

**C-DRIFT reading (two-sided):** if A3 or A4 at ε = 0 reads below A1 by more
than its floor, self-supervision moves a correct map — a finding about the
objective, not the harness, and a caveat on any A3/A4 gain elsewhere; if A5
does not, the teacher's minimum is where it should be.

## 7. Predictions, registered before the run — per family, against CD

CD on these checkpoints (`runs/e10ec/c0deg_cc2.json`): gain carries 0% of
C0deg's recovery, offset 1%, the permutation 71%; the current encoder absorbs
gain/offset and not re-wiring.

| family | A1 shape | crossing | odds |
|---|---|---|---|
| P-gain | near flat within its floor across the realistic range | | **~15%** |
| P-offset | near flat within its floor | | **~15%** |
| P-swap | cliff at the first wrong transposition (≥ 20pp at $m = 1$) | both arms collapse; A2's collapse is at refit and may be partial | **~40%** |

| prediction | odds |
|---|---|
| A1 within its floor of its ε = 0 value across the realistic range, P-gain and P-offset | **~80%** |
| A1 loses ≥ 20pp at one transposition | **~85%** |
| a crossing $\epsilon^*$ exists in at least one family | **~40%** |
| if it exists, it is in P-swap | **~60%** |
| A5 beats both A3 and A4 by more than the floor in at least one family | **~45%** |
| A5's margin over A3 concentrates where the perturbation induces *confident* errors (the mechanism claim) | **~40%** |
| C-DRIFT: A3 or A4 reads below A1 by more than its floor at ε = 0 in at least one family | **~35%** |
| A6 constant; A2 never above A6 | anchor, not a prediction |

*Calibration series, read from `docs/appendix.tex` (`app:predictions`) at the
moment of writing, after the E18/E20 rows were added: **thirty-one scored
entries, seven misses**; every miss in the direction of more structure and
less determinism than predicted.*

## 7a. Rulings after the sweep (2026-09-18) — recorded before the row is read

**C-DT — recorded FAIL with cause, not re-barred.** 391/396 cells; the five
failures are seed 42, gain family, tasks 1–2. Cause, printed from the fit
records: the deranged objective is flat at φ ≈ 0 (LBFGS stops after 2–3
iterations, |φ|max ≤ 0.075, so A5d = A1 exactly), while the *true* teacher walks
the map to the bound (|φ|max 3.6–13.9) and loses 8–10pp — already at ε = 0
(s42/t1: A5 0.811 vs A1 0.901). The bar assumed A5's refinement helps; where it
hurts, a refinement that fails to move beats one that moves wrong. CC's shape.

**C-ID for A2 — RULED 2026-09-18: two test windows per cell** (`--a2-windows 2`;
per-cell bar 2/n_test,k; max 2/409 = 0.0049). The ruling was on the *measured*
floor — "max 0.0049 observed" — and that number is the ruling. It was first
glossed as "one window" on my report that 0.0049 was one window of a
~204-window set; task 0 has **409** windows, so 0.00489 is **two** windows
(s1337/t0 −2, s2024/t0 +2; tasks 2–3 one each, 1/392 and 1/383). A
description does not override the measurement it describes, and a bar at one
window would have sat below the floor — R3's defect. Carried as a floor on
every A2 and A7 delta (pooled 0.0053, below every such delta in the row: the
readings stand). The row prints A2's deltas in windows (−1, −2, +2, −1): an
integer is a witness that the row reads the right test set. A0/A1/A6 stay at
1e-6, exact in 36/36. The floor is cross-platform (arm64 screen vs x86 jobs)
and A2-only — the refit's per-pair determinism, one level down.

**Recorded for the memo:** the row's inputs were within a factor of two of a
bar that would have withheld the headline (A2/A7 unread → no wrong-locus row).
The result is unchanged under either bar; the gate was not. Floors on refit
quantities are stated in windows from here.

## 8. CLAUSE → JOB → ARTIFACT (catch 33) and the audit table (catch 20)

| clause | script / job | artifact |
|---|---|---|
| §9 smoke: one cell, one family, one level, all arms, timed | `scripts/e21_smoke.py` (Modal `analysis`) | `runs/e21/smoke.json` (per-fit seconds, per-arm) |
| §1 families, realizations, seeds | `scripts/e21_perturb.py` (the harness; families as displayed) | `runs/e21/{family}/level{ℓ}_real{r}.json`, 12 cells × 8 arms + SNAP |
| §5 C-ID, C-EXACT | first block of the harness at level 0 | `runs/e21/identity.json` |
| §5 C-DRIFT | level-0 cells of A3–A5 | in the level-0 file |
| §5 C-DT | A5-deranged in every level file | in each file, `A5_deranged` |
| §2 A4 transcription | `docs/E21_mummadi.md` (equations from the paper, checked) — precondition | — |
| §6 curves, crossings, headlines, floors, predictions | `scripts/e21_row.py` (written before the sweep landed; reads only when all 36 artifacts exist) | `runs/e21_row.json` |
| launch | `modal_runner.jobs_e21` → `spawn_analysis --experiment e21` (CPU fan-out `analyze_one`, 36 jobs; gated on `runs/e21/smoke.json`) | `runs/e21/*.log`, one per job |
| audit | `scripts/audit_checkpoints.py` on the fp32 shadow dirs | `runs/e21/ckpt_audit.json` |

| arm | seeds | checkpoints | precision | loads | matrix reproduced | status |
|---|---|---|---|---|---|---|
| HAR S72 OFF | 42, 1337, 2024 | `ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32`, tasks 0–4 | fp32 shadow | ✓ (S72 screen; E20 decompositions) | ✓ 4.0e-9 (`runs/s72_row.json`) | **audited** |

Every clause has a path; `e21_perturb.py` does not launch a family sweep
before `runs/e21/smoke.json` exists (§9).

## 9. Cost — measured, then budgeted

**Smoke (`runs/e21/smoke.json`, Modal CPU, one cell s42/t1, P-gain 0.10,
realization 0, every arm):** forwards 0.6 s; A2 and A7 (bridging refits)
1.5 s each; **A3 / A4 / A5 / A5d 59–68 s each** (LBFGS through the LSTM, three
jitters, 39–50 iterations); **cell total 255 s**. v1's 4–6 hours priced
the forwards; the optimized arms are 99% of the cell.

**Budget, from the smoke:** continuous families — 12 cells × (3 realizations ×
4 non-zero levels + level 0) × 2 families = 312 cell-runs × 255 s ≈
**22 CPU-hours**; swap family — 120 cell-runs of an exhaustive
no-grad search (≤ 4 arms × 3 steps × 37 candidates, bounded at ~116 s) ≈
**4 CPU-hours**; total **≈ 26 CPU-hours**, ~2 hours wall
fanned out at 16 containers, under $20. The 22nd is off the table; default
landing **October, paper 2**, one paragraph and one row if it lands earlier.
The A4 transcription (`docs/E21_mummadi.md`) is a precondition for reading A4,
not for launching the sweep — A4's intent form runs alongside and is not read
until the transcription lands.

**Smoke sanity, NOT a reading** (one cell, one realization): C-DT's direction
holds (A5d 0.46 < A5 0.81); A1 lost 0.6pp to a 10% gain error while A2 lost
~8pp — the first cell says bridging is the *more* sensitive arm to gain error,
which is the B1 mechanism (era teacher on an ε-off frame; head trained ε-off,
tested true) showing its sign; A3 read 0.91 against A6's 0.90, so TENT
moved the map past exact on this cell. None of this enters any table.

**Out of scope:** the no-map / shared-label regime (its own construction and
contract), the SDC/LDC ports (gated on the human read), and any claim about
approximate maps on constructions other than HAR.
