# MEMO — E20: the probe control that would not pass, the band under uniform arms, and an unseeded draw

**Status (2026-09-16, evening):** R1 and R2 **ruled and executed** (§6).
E20-A read under the amended gate on seeded artifacts — the gate **fails on
both arms**, for different reasons; a tolerance ruling is requested (§6). E20-B
band **seeded and in the ledger**: 74.8–89.1 by three-seed means. S72 seeded:
forgetting unchanged to four decimals, ρ 1.03 → 1.02. Every cited number now
records its probe draw.

---

## 1. E20-A — the positive control, seven designs, one mechanism

Contract §4: before the MLP probe reads a live cell, a target the linear probe
**fails** and the MLP **passes**, on the same features, same recipe. Every
design was stated before its run; every failure is on disk under
`runs/e20/positive_control_*`.

| v | target | synthetic | live (MLP-MNIST θ_T, task 0) | what it taught |
|---|---|---|---|---|
| 1 | `(y mod 2) XOR [PC1 > 0]` | **FAIL** — linear 0.9635 | — | PC1 is a function of the class under separated clusters; XOR of two class functions is a dichotomy of 10 means, linearly separable in 256-d |
| 2 | parity XOR class-mean split of a random residual direction | PASS 0.51 / 0.84 | **FAIL** — linear **0.679**, MLP 0.77 | real within-class distributions are skewed; a mean split is not 50/50; parity + a per-class constant decodes it |
| 3 | same, per-class **median** split | PASS | **FAIL** — linear **0.6725** | still linearly decodable on real features |
| 4 | XOR of two random half-spaces in **whitened** space | FAIL — linear 0.503 as designed, **MLP 0.54** | — | a needle: an XOR in two random directions of 256-d is not findable from 4000 samples. A target both instruments fail is not a control |
| 5 | band-vs-tails on PC1 ("linear ≤ 0.75 by geometry") | **FAIL** — linear 0.89 | — | the bound assumed the other 255 coordinates carry nothing about the band; the tails of PC1 *are* particular classes |
| 6 | XOR of median-split PC1, PC2 | **FAIL** — linear 0.758 | — | dominant PCs assign whole clusters to quadrants — v1's mechanism again |
| 7 | XOR of per-class-median splits of the top-2 **within-class residual** PCs, anisotropic synthetic | PASS 0.51 / 0.97 | **FAIL** — linear **0.7055**, MLP 0.80 | even a within-class, per-class-balanced XOR is linearly decoded on these features |

**The pattern.** Every live failure is "the weaker instrument passes" (linear
0.67–0.71), never "the stronger fails" (the MLP reads 0.77–0.80, +0.10 over
linear, every time). Every synthetic pass that failed live differed from the
live features in one property: the live features are not unimodal within
class. **Hypothesis, stated as such, mechanism unclaimed:** in 256-d, any
labeling of up to ~257 separated modes is linearly separable, and an XOR only
defeats a hyperplane where the density crosses the checkerboard boundary
continuously; on these features it does not. A target that is a function of
mode identity — which, on multimodal features, nearly every smooth target is —
is a dichotomy of modes.

**What is validated and what is not.** The MLP *recipe* (capacity, solver,
convergence, init spread) is validated on the anisotropic synthetic (v7: margin
0.46, three seeds, converged). What is **not** validated is the live half the
contract required — a target on *these* features that linear provably fails.
Under §4 as signed, the MLP-probe decompositions (`runs/e20/*_mlpprobe.json`,
computed after the failed control) are **not read**; their numbers are not
carried into this memo.

---

## 2. Instrument finding — the probe subset was an unseeded draw

E20 §5 required the linear re-run on the E17 checkpoints to reproduce
`runs/e16_decomp/mnist_mlp.json` to 1e-9. It **did not**, and the per-field
comparison localizes the cause:

| field | max \|Δ\| over 12 cells |
|---|---|
| `acc_orig`, `acc_ceiling`, `d_logit` | **0.00e+00** — reload and deployed extraction reproduce exactly |
| `acc_refit_ceiling` | 4.0e-03 |
| `acc_refit_t4`, `F_enc`, `F_read` | **1.6e-02** |

`channel_decomp.load_task_data` takes the first `N_TRAIN = 4000` samples of a
`shuffle=True` loader and nothing seeded torch's global generator, so the
probe's training subset depended on process history and library version, not
on anything recorded. The same cause put the S72 OFF decomposition 2.9e-3 off
the screen's own `acc_refit` on the same checkpoints (two scripts, two draws).
Pooled effect on the E17 arm: F_enc 0.0707 → 0.0719, share 82.58 → 82.26%.

**Fixed, opt-in:** `load_task_data(..., probe_subset_seed=)` seeds the draw per
task and the caller records the seed (`probe_subset_seed` in every E18 screen
row; `PROBE_SUBSET_SEED = 20260916`). Default `None` keeps the pre-E20 path,
so no existing artifact is silently re-drawn. Verified: the seeded draw is
identical across RNG states; the unseeded one is not.

**Blast radius, stated:** every decomposition and every screen in the ledger
was produced unseeded. Their reload/extraction fields are exact; their refit
fields carry a draw of unrecorded provenance with the measured spread above
(≤ 1.6pp per cell, ~0.1pp pooled, ~0.3pp on a share). Nothing in the ledger
sits within that of a bar except where already noted (C3's H-X2 CI upper).

---

## 3. E20-B — the band on `har_subject`, uniform arms

15 runs (`jobs_s72_band`), all finished by `verify_runs.check`; every arm's
identity from its own artifact — `benchmark: har_subject`, adapters on, era +
fp32 shadow (delta 0.0 at every boundary), 4 threads — and the v1/v2
distinction from the artifact's own witnesses (`epochs[0].warmup` on tasks ≥ 1;
`adapter_dist_from_identity` = 0.0 in the warmup epoch for v2-control, > 0 for
v2-ON). `warmup_epochs`/`warmup_mode` are now also recorded in `arm` for
future runs. Linear decompositions, unseeded draw (pre-fix; one draw per job).

| arm | share (3 seeds) | F_enc | F_read | R | per-seed shares | E11-era on `har` |
|---|---|---|---|---|---|---|
| OFF | **77.9%** | 0.087 | 0.306 | +0.007 | 64.0 · 84.1 · 82.1 | 66.2 |
| v1-ON | **86.9%** | 0.050 | 0.333 | +0.005 | 81.4 · 92.8 · 84.7 | 75.6 |
| v2-ON | **74.8%** | 0.104 | 0.309 | +0.042 | 57.7 · 77.8 · 85.8 | 77.2 |
| v2-control | **89.1%** | 0.045 | 0.368 | +0.046 | 88.9 · 80.6 · 100.8 | 87.9 |

Identity residual 0.0e+00 on every cell. **The band on the paper's construction
reads 75–89% by three-seed means; per-seed shares run 58–101%.** Two cells read
F_enc < 0 (share > 100%): v2-control seed 2024, v1-ON seed 42 on its floor
replicates. R on the v2 arms (+0.042, +0.046) sits just under the 0.05 gate.

**Determinism, three launches of seed 42 per arm (main + floor a + floor b):**

| arm | main vs a | a vs b | share across the three |
|---|---|---|---|
| OFF | 25/25 | 25/25 | 64.0 (one value) |
| v1-ON | **0/25** | 25/25 | 81.4 vs **103.8** |
| v2-ON | 25/25 | **1/25** | 57.7 vs 66.4 |
| v2-control | 1/25 | 1/25 | 88.9 · 91.5 · 89.1 |

A floor **pair** that reads 25/25 is one observation of determinism, not a
proof — v1-ON's pair agreed with itself and disagreed with the main run
launched in the same batch. The share floor is the spread over all three
launches: v1-ON **22pp**, v2-ON 8.8pp, v2-control 2.6pp, OFF 0. **The ON arms'
shares at seed 42 are not readable at better than ±10pp.**

**Predictions (§8 of the contract), scored:** all three ON arms inside 66–88 —
**miss** (v2-control 89.1); band narrower than 22pp — fired (14.3); E11 ordering
preserved — **miss** (v2-ON < OFF); v2-control ≥ 85 — fired; every ON floor
pair bit-identical — **miss** (2 of 3 pairs, and the pair test itself is
insufficient). Three of five missed; the misses are all "more structure and
less determinism than expected".

**Ledger consequence, pre-committed in the contract:** the one-liner's
"66–88% across four arms of a scratch-trained LSTM on UCI HAR" is replaced by
the `har_subject` band with its construction and floors named, and the E11-era
band cited beside it as superseded by provenance and construction. **Not yet
applied** — the numbers above were produced with the unseeded draw, and the
seeding ruling (§5) decides whether they are re-run first (minutes).

---

## 4. E18 — status

Build gate PASS on all three checks (loader hashes 60/60 identical; disjoint
constructions correct; `w2d_mafc` anchor reproduced 25/25 by retrain on the
amended code). 15 runs launched 11:36, finished by 12:02, all verified,
construction recorded in every artifact. Screens running on Modal (seeded
draw, recorded). Determinism: Rotated-MLP three launches identical;
Permuted-MLP floor pair identical but the main run differs (v1-ON's pattern);
Permuted-LSTM 0/25.

---

## 5. Rulings requested

**R1 — the MLP-probe control.** Seven designs could not produce, on these
features, a target the linear probe fails; the recipe is validated on a
synthetic shaped like the features. Options: (a) accept the synthetic
certification for the *recipe* and replace the live half with a weaker,
non-vacuous live check — on every live cell the MLP refit must not fall below
the linear refit by more than the probe's init spread (an underfit MLP fails
it; a working one cannot); read E20-A under that gate, with the seven-design
record in the paper's appendix as a finding about the feature geometry;
(b) keep searching for a live-failing target (not recommended — the mechanism
is structural); (c) withdraw E20-A: the lower-bound sentence in §3.1 stays a
logical argument. I recommend (a), stated in the contract as an amendment
with the record beside it.

**R2 — the unseeded draw.** Options: (a) seed going forward only (done,
opt-in); existing artifacts annotated with the measured floor (≤ 1.6pp per
cell, ~0.1pp pooled); (b) additionally re-run the S72 screen, H-X2, ρ, and the
E20-B band seeded (minutes each) and cite those as final — same instrument,
now with a recorded config; the E11-era artifacts stay as they are, already
superseded by provenance. I recommend (b): every S72/E20-B number then has a
recorded probe draw, and the deltas from today's values will be inside the
floor above.

Order after the rulings: E18 screens → hx2/ρ → P3 → C0deg controls on the
MNIST constructions → `e18_row.py`; E19 once the exclusion flag has its
regression; ports after the human read.

---

## 6. Rulings executed (2026-09-16) and one more requested

**R1 (a)** — the recipe is certified on the anisotropic synthetic; the live half
is replaced by the gate *MLP refit ≥ linear refit − init spread, every cell,
both eras*; seven designs to the appendix. **R2 (b)** — S72 screen, H-X2, ρ,
the band, and both E20-A arms re-run with `probe_subset_seed = 20260916`;
those are the citable values. Two parallel `modal run` streams conflicted
(`APP_STATE_STOPPED`) and cost the afternoon; the missing two jobs were re-run
serially. Every seeded artifact records its seed; identity 0.0e+00 throughout.

**S72 seeded vs unseeded:** none 0.3864, C3 0.0954, C0deg −0.0379 — identical to
four decimals (none of these depends on the draw); ρ(C3) 1.029 → **1.020
[0.824, 1.215]**, C2 0.687 → 0.691. No ranking moves. Applied to the ledger.

**E20-B seeded:** OFF 77.8 · v1-ON 86.7 · v2-ON 74.8 · v2-control 89.1 (was
77.9 / 86.9 / 74.8 / 89.1). Share floors over three seed-42 launches: 0.0 /
**22.4** / 9.5 / 2.3pp. v1-ON's swing is between the main run and two replicates
that are identical to each other and share 0/25 cells with it — different
models; survives the seeded draw; appendix. In the ledger.

**E20-A under the amended gate, seeded artifacts (`runs/e20/mnist_mlp_row.json`,
`runs/e20/har_s72off_row.json`):**

| arm | regression (reload / refit) | R gate | R1 gate | printed row (NOT READ) |
|---|---|---|---|---|
| primary MLP-MNIST | reload fields 0.0e+00; refit fields 1.4–1.9pp (the draw, measured) | linear +0.0006, MLP −0.0002 OK | **FAIL 5/24** — shortfalls 0.1–0.6pp on cells at 0.93–0.98, every one inside the test set's 1.96·SE (0.6–1.1pp, n = 2000) | pooled ΔF_enc −0.0073, 0/12 overfit flags, share 82.2 → 84.0 |
| secondary HAR S72 OFF | — | OK | **FAIL 6/24** — two substantive (6.9pp, 6.6pp below linear, outside resolution), one unconverged cell | pooled ΔF_enc +0.0011, below the probe floor (0.030) |

Same gate, two kinds of failure. On the primary the gate's tolerance (the
init spread, as small as 0.1pp) is finer than the measurement's resolution; on
the secondary the MLP probe genuinely underperforms linear on unbounded LSTM
cell states, which is the failure mode the gate exists to catch. **Ruling
requested (R3):** tolerance `max(init spread, 1.96·SE_binomial(n_test))` —
stated here before it is applied. Under it the primary passes 24/24 and reads
**TIGHT** (|ΔF_enc| ≤ 0.01: the linear bound is tight against MLP-recoverable
information; share moves 82.2 → 84.0); the secondary still fails 2/24 and its
MLP reading stays withheld. Under the gate as ruled, both are withheld and
§3.1's sentence stays a logical argument.

**R3 — approved and applied (2026-09-16).** Tolerance `max(init spread,
1.96·SE_binomial)`; stated before it was applied; the test that it corrects
rather than relaxes is that the secondary's substantive cells still fail.

| arm | R1/R3 gate | reading |
|---|---|---|
| primary MLP-MNIST | **24/24 PASS** (the five prior failures sit inside resolution, above the spread — listed in the artifact) | **TIGHT** — pooled ΔF_enc −0.0073, 0/12 overfit flags, share **82.2 → 84.0%**. §3.1's lower-bound argument confirmed empirically on the arm where F_enc was large enough to move, in the favourable direction. One sentence in §4; seven designs to the appendix. |
| secondary HAR S72 OFF | **22/24 FAIL** — (42,2) ceiling 0.9847 → 0.9158; (1337,2) t4 0.852 → 0.787; one unconverged cell | **WITHHELD.** Standardization is on train statistics in both probes (`channel_decomp.py:273, 316`). On both failing cells the MLP reaches **train accuracy 1.000** (loss ≈ 5e-4): it *overfits* 4000 unbounded LSTM cell states even standardized and reads below the convex fit. An optimization finding about nonlinear probes on recurrent states — the paper's stated reason for linear probes with standardization. The LSTM row's lower-bound claim rests on the logical argument, stated as such. |

E20 predictions (§7): primary `|ΔF_enc| ≤ 0.01` (~45%) **fired**; `ΔF_enc <
−0.01` (~40%) did not fire (−0.0073); overfit flag on ≥ 1 primary cell (~35%)
did not fire (0/12); MLP floor > 0.01 on ≥ 1 cell (~50%) fired (max 0.0145);
`|R^mlp| > 0.05` (~15%) did not fire; secondary sign agreement (~60%) —
unreadable (withheld).

**Abstract range.** The LSTM band 75–89% and the ResNet 98.8% make the
abstract's "66–99%" into **"75–99%"**; every endpoint traces to a seeded
artifact. (The abstract text is not in this repository; the ledger's
one-liner is the source.)
