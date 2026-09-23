# MEMO — E25: Three Boundary Experiments

**Status (2026-09-21):** contract `docs/E25_prereg.md` signed v4.2. The launch
order's free steps are done: the count line was re-read at signing, **R1 and R2
are read** (`scripts/e25_reads.py`, `runs/e25/r1_c2.json`,
`runs/e25/r2_a4_lam1.json`), and **L0 is in flight** on Modal
(`spawn_analysis --experiment e25audit`). Neither read carries odds; both were
recorded as reads under §0c's integrity ruling.

**Count line at signing**, re-read from `docs/appendix.tex:609`: *"Sixty-seven
scored entries, fourteen misses (2026-09-20)."* Unchanged since E23-B. E25
registers fifteen predictions; none is scored yet.

## 0. L0 — the checkpoint audit: **GREEN, 22/22**

`runs/e25/ckpt_audit.json`. Criterion **LOADS** through `PLCM.load_era`, run on
the volume. **22 of 22 directories fully loadable, 0 bad files.**

| step | directories | verdict |
|---|---|---|
| A (ViT, ResNet, S72 HAR, E18 MLP) | 12/12 | **CLEAR** |
| B (four D1 scratch arms + E23-B A3) | 15/15 | **CLEAR** |
| ctrl6 (`ckpt_e17_mlp_seed42` regression target) | 1/1 | **CLEAR** |

**L4–L8 are unblocked.** E11's audit found 14 of 33 directories unusable when
opened, so a clean table is a result rather than a formality.

Extension notes:

- The E11 chain hardcoded `TASKS = [0..4]` and `EPOCH = 9`. E25's arms differ on
  **both** axes: ViT and ResNet are 20 tasks at epoch 4, E23-B's A3 is 20 tasks
  at epoch 9, the rest are 5 at 9. `audit_dir` now takes `tasks`/`epoch` with
  defaults equal to the old globals, and `--chain` defaults to `e11`, so **the
  path E11's table was produced on is unchanged**.
- The chain audits the directory **each analysis actually opens**: the `_fp32`
  shadows for the scratch arms, the plain fp16 directories for ViT and ResNet,
  which have no shadow. Auditing a directory the analysis does not open would
  prove nothing about the analysis.
- **Routing, stated rather than inherited.** `e25audit` is deliberately left off
  the CPU tuple in `spawn_analysis` so it runs on `analyze_one_gpu`, the only
  container proven to construct both pretrained backbones. Whether a file loads
  is not device-dependent; device uniformity is a property a *number* needs.
  Falling to GPU by accident is the hazard the contract's routing rule names.
  Falling to GPU on purpose, said out loud, is not that.

Run locally first against the partial mirror: **9 of 22 green**, and every
failure is a directory this laptop never pulled. That is why the audit is
authoritative only on the volume.

## 1. R1 — C2, the existing labeled repair

`cure_screen.cure_c2`: per-class prototypes from **full** training labels on both
encoders, orthogonal Procrustes between them, read through the era head. All
four arms at the seeded probe draw `20260916`; `runs/e10ec/cures_e10.json` is
the same HAR arm at an unrecorded draw and is **not** used.

| arm | acc_orig | C2 | acc_refit | ceiling | C2 recovery | valid cells | forced |
|---|---|---|---|---|---|---|---|
| LSTM · HAR | 0.4665 | 0.6862 | 0.7723 | 0.8529 | **+0.7158** | 11/12 | +0.6978 |
| LSTM · Permuted | 0.1953 | 0.5422 | 0.7675 | 0.9348 | **+0.6108** | 12/12 | +0.6108 |
| MLP · Permuted | 0.7676 | 0.9014 | 0.9430 | 0.9617 | **+0.7439** | 10/12 | +0.6800 |
| MLP · Rotated | 0.5117 | 0.8459 | 0.9291 | 0.9674 | **+0.8053** | 9/12 | +0.7673 |

Recovery is `(C2 − acc_orig) / (acc_refit − acc_orig)`, mean of ratios over cells
clearing a **per-cell** denominator guard of 0.05, with the forced-inclusion
pooling printed beside it (catch 26: the guard applies per cell, never to a
pooled denominator).

**What it says.** A repair that is fully labeled *and* constrained to be
orthogonal already recovers **61–81%** of the refit gap on every scratch arm.
That is the number A has to be read against. A's registered bar, recovery ≥ 0.8
at n = 10, sits at or above what C2 reaches with *every* label — so the question
A answers is not "can a linear repair work" but "how much of C2's recovery
survives at ten labels per class, and does dropping orthogonality buy the rest."
D1 measured this drift at spread 8–12 with 54–66% of singular values below 0.5,
far from orthogonal, which is the room an unconstrained repair can occupy.

**Every excluded cell is task 3**, the last old task, where the deployed head has
lost little and the denominator collapses: HAR (2024, 0.026); MLP·Permuted (42,
0.018) and (1337, 0.024); MLP·Rotated (42, 0.041), (1337, 0.040), (2024, 0.047).
The exclusion rate is a reportable fact about the benchmark, not a footnote: six
of 48 cells have almost no gap to recover.

## 2. R2 — A4 at λ = 1, already run

Transcription supplied by the contract's §0: A4 as executed is
`slr − 1.0·H(p̄)`, and `slr` is Mummadi §3.2.2's SLR exactly since
`Σ_{i≠c} p_i = 1 − p_c`. Their objective `L_div + 0.025·L_slr` rescales to
`slr − 40·H(p̄)`, so **A4 as run is Mummadi's form at 1/40th the diversity
weight**. The artifacts still carry `"mummadi_form": "INTENT -- transcription
pending (docs/E21_mummadi.md); A4 not read"`, and that document does not exist.

Pooled accuracy over 12 cells, mean over realizations. Floor replicated from
`scripts/e21_row.py:139-141`, not reinvented: `max(realization spread, mean
per-cell jitter spread)`.

| m | A1 | A3 | A4 | A4 − A3 | floor | verdict |
|---|---|---|---|---|---|---|
| 0 | 0.8908 | 0.8816 | 0.8924 | +0.0108 | — | **not read** |
| 1 | 0.7454 | 0.8445 | 0.8769 | +0.0324 | 0.0492 | within floor |
| 2 | 0.5435 | 0.7521 | 0.8304 | +0.0783 | 0.1752 | within floor |
| 3 | 0.4465 | 0.6342 | 0.7236 | +0.0894 | 0.1666 | within floor |

**Reading, stated in the contract before this read:** A4 at λ = 1 does **not**
beat A3 by more than floor at m = 1, so **C at λ = 40 is the only remaining
test** of whether the diversity term gives entropy direction on the discrete
family.

**Sign pattern, a measurement and not a cleared claim.** A4 sits above A3 in
point estimate on 3/3 readable levels (+0.0324, +0.0783, +0.0894) while none
clears its floor. Same form as E23's (E1) row. The floors are realization spread
alone (0.049–0.175); the jitter column is 0.000 at every level because the swap
search is `torch.no_grad()` and deterministic.

**A defect in this script's own floor, caught and recorded.** The first run
printed m = 0 as "A4 BEATS A3, floor 0.0000". Level 0 has **one** realization —
`e21_row.py:135` runs eps = 0 once because it is identical for every r — so
`max(vals) − min(vals)` is zero *by construction*, and the jitter spread is zero
because the search is deterministic. A floor that cannot be anything but zero is
not a floor: any non-zero delta clears it. Catch 25's shape inside a floor rather
than inside a gate. Level 0 now carries no floor and is not read, and the m = 0
row above is printed without a verdict.

## 2a. The paired estimator, computed before C is amended — it does not rescue m = 1

E21's memo (§5) names the floor's defect: "a realization is *which channels are
transposed* — a treatment axis, not a noise axis", and registers a paired read
blocked on the wiring as the follow-up. Before ruling on whether C should adopt
it, it was applied to the λ = 1 data already in hand. A3 and A4 see the **same**
perturbed map in every cell, so the pairing is exact.

| m | mean A4 − A3 | per-realization pooled | blocked t95 (n = 3) | verdict | cells positive |
|---|---|---|---|---|---|
| 1 | +0.0324 | +0.0234, +0.0628, +0.0111 | ±0.0671 | **includes zero** | 8/36 |
| 2 | +0.0783 | +0.0628, +0.1234, +0.0488 | ±0.0986 | **includes zero** | 12/36 |
| 3 | +0.0894 | +0.0959, +0.0904, +0.0818 | ±0.0176 | clears zero | 17/36 |

**The paired read does not discriminate at m = 1**, which is exactly where C's
headline prediction lives. So adopting it cannot be moving a bar to meet an
outcome: it demonstrably does not produce the outcome at the level that matters.
That is what makes it safe to register after R2's values were seen.

**And a number that does not fit, which reaches back into E21.** The
realization-level sign pattern is unanimous while the cell-level one is not,
for **both** arms:

| pair | m | realizations positive | cells positive |
|---|---|---|---|
| A4 − A3 | 1 | 3/3 | **8/36** |
| A5 − A3 | 1 | 3/3 | **11/36** |
| A5 − A3 | 2 | 3/3 | 18/36 |
| A5 − A3 | 3 | 3/3 | 28/36 |

E21's memo reports A5 − A3 "positive in **9 of 9** realizations". True, and at
m = 1 A5 is above A3 in **11 of 36 cells** — the three realization *means* are
unanimous because a few cells with large gains carry each one. The registered
paired follow-up, blocked on the wiring, would reproduce the 9/9 statistic and
inherit the over-reading. **Catch 26's rule applied to signs: pooling is a way
of hiding a failed cell, and a sign count taken after the average is taken where
the failures have already been averaged away.** Recorded as catch 35(b).

**Consequence for E21's row:** its paired follow-up prints the per-cell count
beside the per-realization one, or it is a claim about means wearing a claim
about consistency. R2's own registered reading is unaffected — it fires under
both estimators.

## 3. Launch record — L3 through L8, 2026-09-21

| step | job set | jobs | artifact |
|---|---|---|---|
| L3 | `e25c` | 10 | `runs/e25/c/lam40/level{l}_real{r}.json` |
| L3 | `e25cregress` | 1 | `runs/e25/c/lam1_regression_raw.json` |
| L3 | `e25csignflip` | 3 | `runs/e25/c/signflip/level1_real{r}.json` |
| L4 | `e25b` | 4 | `runs/e25/b/{arm}/steps_seed{s}.json` |
| L8 | `e25ba3` | 3 | `runs/e25/b/e23b_t20/steps_seed{s}.json` |
| L5+L7 | `e25ascratch` | 2 | `runs/e25/a/{arm}/curve_seed{s}.json` |
| L6 smoke | `e25apresmoke` | 2 | `runs/e25/a/smoke_{arm}/curve_seed42.json` |
| L6+L7 | `e25apre` | 6 | held until the smoke is green |

C runs **A4 alone** at λ = 40: A1, A3, A6, A0 and SNAP do not depend on λ, so A3
comes from E21's artifacts and the cheap deterministic arms are recomputed as the
C-ID witness. `run_cell` now gates A2's bridging refit on `"A2" in arms`, and the
default tuple still contains it, so E21's own path is unchanged.

## 4. What the build found

**(a) `refit_probe` does not reach its own optimum.** Its docstring says "fit to
OPTIMALITY. No tuning knobs", and it runs sklearn at the **default `tol=1e-4`**.
Measured on `e18_pmd_mlp` seed 42 task 0, 256-d features:

| fit | iterations | objective | test accuracy |
|---|---|---|---|
| sklearn, default tol 1e-4 (what `refit_probe` does) | 45 | 120.579123 | 0.9200 |
| sklearn, tol 1e-10 | 490 | 119.444907 | 0.9190 |
| E25's solver, γ = 0.5 | 541 | **119.444907** | 0.9190 |

The new solver's convention is confirmed exactly: γ = 0.5 at C = 1 reproduces
sklearn's objective to six decimals. The 0.001 accuracy gap is **sklearn stopping
1.13 objective units early**, not a disagreement about the objective.
**The recorded convergence witness cannot see this.** `probe_n_iter` is checked
against `max_iter = 5000`, which detects exhausting the iteration budget and is
blind to stopping on *tolerance* — 45 iterations never trips it. Same family as
catch 35: the witness cannot vary in the direction that would falsify the claim.
Effect here is two test samples in two thousand. **`refit_probe` is NOT changed**:
every refit number in the program comes from it, and a change is the user's
ruling, not a build decision. E25's equivalence control therefore bars on the
**objective** against sklearn-at-convergence and prints the default-tol value
beside it.

**(b) Control 3 caught a half-pinned target on its first use.** A-ridge pulls
toward `h_k`; with the intercept unpenalised, as sklearn leaves it, γ → 10⁶ pinned
`W` while `b` kept fitting the data, so the limit read **0.6895 against h_k's
0.6185** — a 7.1pp gap that looks like a broken must-fail and is really a target
pinned in one of its two parts. The target is a *readout*, so both parts are
pulled; the limit now reproduces `acc(h_k Z_T)` to **0.000000**. The sklearn
equivalence keeps the intercept free, which is why the two are separate knobs.

**(c) A launcher verification that could not fail, caught by the rule that names
it.** The first check fed every generated argv to the script with `--help`
appended and reported "all argv accepted". Argparse fires `--help` **before** it
validates unrecognised arguments, so the check was blind to exactly the defect it
was written for: eight pretrained-A jobs carried `--root` where the script
defines `--data-root`. The sound check compares each argv's flags against the
flags the parser defines, and it found all eight. Third instance of *a
verification must exercise the failure mode it claims to rule out*, and the first
where the unsound check was mine and written in the same turn as the rule.

**(d) The C flag regression passes on its subject, and the two arms that move
land exactly on E21's ruled floor.** Re-running λ = 1 with the default arms on
swap level 1 realization 0, against `runs/e21/swap/level1_real0.json`:

| arm | max abs delta over 12 cells |
|---|---|
| A0, A1, A3, **A4**, A5, A5d, A6, SNAP | **0.000e+00** |
| A2 | 2.907e-03 |
| A7 | 4.890e-03 |

Eight of ten arms are **bit-identical**, including A4, which is the arm the flag
touches and the arm C is about. The two that move are the only two that go
through `bridging()` and therefore through `refit_probe` — and their deltas are
**whole test windows**: 2.907e-03 = **1/344**, one window on task 1; 4.890e-03 =
**2/409**, two windows on task 0. That is E21's ruled A2/A7 window floor
(`--a2-windows 2`, max 2/409 = 0.004890) reproduced to six decimals, arriving
from an independent direction. So the 1e-6 bar "fails" on exactly the two arms
the program already ruled cannot be held to it, by exactly the ruled amount.
It is the same phenomenon as finding (a): the refit's stopping point is
platform-sensitive because it stops on a loose tolerance, and its output is
discrete in test windows. **C's numbers are usable**: the flag is inert on
everything deterministic.

**(e) An early signal from B's local smoke, not read.** On `e18_pmd_mlp` seed 42
the per-step residual is 0.42–0.46 against the registered 15% bar, and close to
the single-shot residual. One arm, one seed, no floor yet. Recorded so it is not
discovered later as a surprise, and not read.

## 5. Rows read — C, B, and the probe floor (2026-09-21)

### 5a. C — the addendum earns its place

`runs/e25c_row.json`. **C-ID passes exactly**: A0, A1, A6 and SNAP, none of which
depends on λ, reproduce E21 to **0.000e+00**, so this is E21's harness.

| m | A4 (λ=40) | A3 | delta | contracted floor | verdict |
|---|---|---|---|---|---|
| 1 | 0.8914 | 0.8445 | +0.0469 | 0.0492 | within floor |
| 2 | 0.7952 | 0.7521 | +0.0431 | 0.2995 | within floor |
| 3 | 0.6780 | 0.6342 | +0.0438 | 0.1965 | within floor |

**The contracted prediction did not fire** (registered 40%). The addendum's
paired estimator reads differently, and this is the case it was registered for:

| m | λ | mean d | paired t95 | paired | realizations + | cells + |
|---|---|---|---|---|---|---|
| 1 | 40 | +0.0469 | ±0.0191 | **clears zero** | 3/3 | **11/36** |
| 2 | 40 | +0.0431 | ±0.1794 | includes zero | 2/3 | 15/36 |
| 3 | 40 | +0.0438 | ±0.1012 | includes zero | 3/3 | 19/36 |
| 1 | 1 | +0.0324 | ±0.0671 | includes zero | 3/3 | 8/36 |
| 2 | 1 | +0.0783 | ±0.0986 | includes zero | 3/3 | 12/36 |
| 3 | 1 | +0.0894 | ±0.0176 | clears zero | 3/3 | 17/36 |

**At Mummadi's own weight the paired read clears zero at one transposition;
at 1/40th of it, it does not.** That is the distinction the addendum was
registered to make, and the λ = 1 table published before launch is what shows it
was not built to produce this. **Catch 35(b) applies immediately:** the same cell
clears the paired bar with **11 of 36 cells positive** — unanimity across three
realization means, a minority of cells. Both numbers are reported; the reading
rests on the mean and says so.

**A wording defect in the contract, recorded not repaired.** Prediction 2,
"λ = 40 beats λ = 1", does not name a level. It is true at m = 1 (+0.0469 vs
+0.0324) and **false at m = 2 and m = 3** (+0.0431 vs +0.0783, +0.0438 vs
+0.0894). Scored at m = 1, the headline level, with the reversal stated. A
one-sided prediction over an unstated index is the same family as E14's
one-sided branch.

**Controls.** Sign-flip must-fail **PASS 36/36**: rewarding collapse instead of
diversity is worse than A3 in every cell. Flag regression bit-identical on 8/10
arms (§4d).

Predictions: 4 scored, 3 fired, 0 missed, 1 did not fire.

### 5b. B — the pre-committed third branch fired

`runs/e25b_row.json`. Pooled over cells, three seeds per arm.

| arm | family | cells | per-step residual | single-shot residual | deployed | composed | single-shot | ceiling |
|---|---|---|---|---|---|---|---|---|
| s72_off | LSTM | 12 | 0.457 | 0.447 | 0.8514 | 0.8794 | 0.8663 | 0.8529 |
| e18_pmd_mlp | MLP | 12 | 0.418 | 0.410 | 0.9582 | 0.9607 | 0.9605 | 0.9617 |
| e18_pmd_lstm | LSTM | 12 | 0.466 | 0.452 | 0.9266 | 0.9365 | 0.9365 | 0.9348 |
| e18_rmd_mlp | MLP | 12 | 0.420 | 0.415 | 0.9679 | 0.9655 | 0.9667 | 0.9674 |
| **e23b_t20** | LSTM | 57 | **0.502** | 0.496 | 0.8678 | **0.8303** | 0.8824 | 0.8582 |

**READING (pre-committed): per-step drift is as non-linear as endpoint drift.**
No arm clears the 15% bar, and on every arm the per-step residual sits within
0.01 of its own single-shot residual. Splitting the boundary buys nothing. §B's
third branch: **no rolling method has a foundation here.**

**And over twenty steps composition actively hurts.** A3's composed repair
(0.8303) is **below its own deployed accuracy** (0.8678) and 5.2pp below the
single-shot fit (0.8824), while the four five-step arms all stay within 5pp.
Errors accumulate with the number of boundaries, which is the second branch's
failure mode showing up only where there are enough steps to see it.

**Controls.** C-SHUF must-fail passes on every cell of every arm (12/12 ×4,
57/57). The composed affine equals the chained `.apply` to ≤6.2e-14 everywhere,
so the exactness claim is witnessed, not asserted. C-PLUMB is 0.12–0.29 on the
MNIST arms and **4.15 on HAR**, matching D1's note that ~30 of HAR's 256
directions are near-dead. Sharing is arm-dependent: rotated MLP 36/36 pairs
within resolution, permuted 25–27/36, HAR 10/36, A3 13/60.

Predictions: 5 scored, 1 fired, 2 missed, 2 did not fire. Prediction 5 is
**scored against the ceiling, not the refit** — the contract names the refit and
`e25b_steps.py` records `acc(h_k, Z_k)`. Stated rather than absorbed; the ruling
is the user's.

### 5c. The probe floor — measured, and the direction is NOT safe

`runs/e25/probe_floor/*.json`. Per cell, the same features fit twice: sklearn at
its default tolerance, which is what `refit_probe` does, and at `tol=1e-10`.

| arm | cells | max abs F_enc delta | max abs acc delta | objective excess | iters default vs tight | F_enc inflated / deflated / same |
|---|---|---|---|---|---|---|
| e18_pmd_mlp | 12 | 0.0020 | 0.0020 | 1.60 | 18 vs 291 | 5 / 5 / 2 |
| e18_rmd_mlp | 12 | 0.0030 | 0.0020 | 1.86 | 20 vs 317 | 5 / 5 / 2 |
| e18_pmd_lstm | 12 | 0.0020 | 0.0020 | 1.82 | 91 vs 666 | 2 / 9 / 1 |
| **s72_off** | 12 | **0.0116** | 0.0116 | 0.54 | 132 vs 1502 | 3 / 2 / 7 |
| **e14_base** (2048-d) | 57 | **0.0140** | 0.0140 | 1.50 | 21 vs 434 | 25 / 24 / 8 |

**The POOLED floor is what belongs beside a pooled $F_{enc}$, and it is 20–50×
smaller than the per-cell maximum**, because the per-cell errors are unsigned and
cancel:

| arm | cells | pooled F_enc | pooled floor | per-cell max |
|---|---|---|---|---|
| e18_pmd_mlp | 12 | 0.0205 | **0.0000** | 0.0020 |
| e18_rmd_mlp | 12 | 0.0387 | 0.0002 | 0.0030 |
| e18_pmd_lstm | 12 | 0.1717 | 0.0006 | 0.0020 |
| s72_off | 12 | 0.0866 | 0.0007 | 0.0116 |
| e12_base (ViT) | 38 | 0.0567 | 0.0005 | 0.0080 |
| **e14_base (ResNet)** | 57 | **0.0079** | **0.0003** | 0.0140 |

**Table 1's ResNet row stands.** Its pooled $F_{enc}$ of 0.0079 sits **26×** above
its pooled probe floor of 0.0003. The per-cell maximum of 0.0140 is the wrong
comparator for a pooled quantity precisely because the direction is unsigned —
57 cells inflate and 59 deflate, so they average out. Print the pooled floor
beside the pooled term, not the per-cell one.

**The paper's corrected sentence can carry a number: the default tolerance
terminates within 0.3pp of the tight-tolerance optimum on the MNIST and rotated
arms and within 1.2pp on HAR.** HAR's is larger because its test sets are 344–409
windows, so one window is 0.24–0.29pp and 0.0116 is exactly **four windows on
task 1** — the same discreteness that produced §4d's regression deltas.

**The direction argument does not hold and should not be made.** The expectation
was that a probe stopping short recovers less, inflating `F_enc` and deflating
the readout share, running the safe way. Measured, `F_enc` moves **both ways**:
inflated in 15 cells, deflated in 21, unchanged in 12. Both refits stop short,
and which of the two loses more decides the sign per cell. The default tolerance
runs 10–15× fewer iterations than convergence, and the resulting error is
unsigned. Pretrained arms (2048-d ResNet) are in flight.

### 5d. A — scratch curves in, one control failing on the ViT smoke

Recovery fraction, mean of ratios over cells clearing the 0.05 guard:

| arm | cells | n=1 | n=5 | n=10 | n=20 | n=50 |
|---|---|---|---|---|---|---|
| s72_off (HAR) | 11/12 | +0.327 | +0.699 | +0.740 | +0.797 | +0.940 |
| e18_pmd_mlp | 10/12 | −0.006 | +0.646 | +0.743 | +0.823 | +0.898 |
| smoke ViT (2 cells) | 2/2 | +0.384 | +0.532 | +0.621 | +0.750 | +0.851 |
| smoke ResNet (2 cells) | 2/2 | +0.406 | +0.668 | +0.766 | +0.834 | +0.910 |

**Ten labeled examples per class already match C2's fully-labeled orthogonal
repair** (R1: 0.61–0.81). Controls 1, 3 and 4 pass on both scratch arms and on
the ResNet smoke.

**Control 4 restated before the headline ran (2026-09-21 ruling).** The old bar,
"shuffled-label A_n does not exceed A0 beyond binomial resolution", discriminates
**nowhere**. On permuted CIFAR the deployed accuracy is itself at chance, 0.22
against 0.20, so it compares noise to noise and failed the ViT smoke for that
reason. On the scratch arms it could not fail either: shuffled labels read ~0.17
on HAR against a deployed 0.47, so it passed trivially. Catch 25 from two sides
at once — a gate that has only ever passed and a gate that cannot discriminate
are the same defect.

Restated against **chance**, which is what "the label path is not leaking"
actually means and does not depend on how good A0 happens to be:

$$|\,\mathrm{acc}(\text{shuffled}) - 1/C\,| \;\le\; 1.96\cdot \mathrm{SE}_{\text{binom}}(1/C,\, n_{\text{test}})$$

Chance is 0.20 on the CIFAR heads, 1/6 on HAR, 0.10 on MNIST. Verified
non-trivial on e18_pmd_mlp: shuffled reads **0.0930 against chance 0.1000, a
0.0070 gap under a 0.0131 bar** — it now has room to fail. Applied to every arm
including the scratch ones, which were relaunched for uniformity, and the
superseded bar is printed in every artifact so the change is visible rather than
silent. Written before the pretrained headline was read; the headline
(`e25apre`, 6 jobs) released on that ruling.

### 5e. The restated control 4 FAILS on HAR, and the bar does not move again

| arm | shuffled | chance | bar | verdict |
|---|---|---|---|---|
| e18_pmd_mlp | 0.0940 | 0.1000 | 0.0131 | **PASS** |
| **s72_off (HAR)** | **0.2860** | 0.1667 | 0.0360 | **FAIL by 12pp** |

**Class imbalance is ruled out, measured not assumed.** HAR's subject-disjoint
test sets are near-balanced: per-task class frequencies 0.128–0.201, and the
**best constant predictor scores 0.181–0.201, mean 0.191**. The shuffled probe
reads 0.286, which is 9.5pp above the majority-class rate and 12pp above uniform
chance — roughly 49 of 409 windows. Whatever it is, it is not the target being
wrong.

**The bar does not move.** It was restated once today, before the headline, on
an argument about what the control means. Restating it a second time *because it
now fails* is precisely the move the program forbids: *a must-fail's bar is
priced against the benchmark before launch, and it does not move after.* Recorded
as FAIL with the diagnosis so far.

**Consequence: HAR's A reading is HELD.** The curve is computed and stored but
not read, because a label-path control failing on that arm is exactly the defect
that would make its recovery numbers meaningless. The MNIST arm passes
non-trivially and its reading stands. The pretrained arms' control-4 values are
not in yet.

**DIAGNOSED (2026-09-21): neither hypothesis. The failure is n = 1.**
`scripts/e25a_c4_diag.py` ran four variants over 20 draws per cell:

| variant | pooled over 4 tasks × 20 draws | per-draw range |
|---|---|---|
| **A label-vector shuffle** (what the control runs) | **0.1664** | [0.038, 0.367] |
| B feature-row shuffle | 0.1527 | [0.006, 0.372] |
| C class relabel map (the hypothesis) | 0.1951 | [0.000, 0.628] |
| D no shuffle | 0.7327 | [0.535, 0.958] |

Chance is 0.1667. **The control's own statistic pools to 0.1664 — three ten-
thousandths from chance.** The 0.286 that failed was ONE draw from a
distribution spanning 0.038 to 0.367. The bar was built from binomial SE on the
test set, which captures test sampling and omits the draw-to-draw variance of
which 120 training points and which shuffled labeling were used; here that
omitted term is 4–8× larger. The program's own rule, applied to a control's
measurement: *nothing at n = 1*, and R3's form, tolerance =
max(the instrument's own spread, binomial SE).

**The class-relabel hypothesis is ruled out twice.** By implementation —
`rng.permutation(y_tr[sel])` permutes the label VECTOR and is not a consistent
renaming, verified directly — and by measurement: variant C pools to 0.1951 with
1.0–1.4 fixed points, not to 0.286. Its arithmetic fit two observations to
within 0.02 and was still the wrong mechanism, which is why it was measured
rather than argued away.

**MNIST's pass was equally n = 1**, as suspected, though for this reason rather
than fixed points.

**The target does not move; the ESTIMATOR does.** Chance was and remains right.
Control 4 now averages 20 draws and bars at max(binomial SE, draw t95), with the
contracted single-draw value retained beside it so the change is visible. Which
is the control is the user's ruling.

**A defect in the diagnostic itself, recorded.** Its first run was local, and
`har_subject`'s partition is x86-only: the laptop builds `5d047e4213d1` where the
executed runs are `1104af185c87`. `e25a_curve.py` asserts the fingerprint and
refused to run; the diagnostic had no such assert and silently measured a
partition no checkpoint was trained on. The assert is now in it, and the
numbers above are **pending x86 re-run** (`e25c4diag`, launched). *A diagnostic
that does not verify its own construction is the defect it was written to find,
one level up.*

**Superseded next step, kept for the record:** the shuffle is
`rng.permutation(y_tr[sel])` over a class-balanced `sel` whose indices are
concatenated per class, so `Xtr_u[sel]` is ordered by true class. The
permutation breaks the pairing, but it does not break any structure carried by
the ROW ORDER. Test whether the failure survives shuffling the feature rows
instead of the labels, and whether it survives drawing `sel` unordered. If it
does not, the control has been measuring a draw artifact on the one arm whose
feature ordering is class-blocked and 256-dimensional with only 120 training
points.

### 5f. Superseded reading — why the old control 4 "failed" C4 asks that a
shuffled-label probe not exceed the deployed accuracy beyond binomial
resolution. On B6-permuted ViT the deployed accuracy is itself near chance
(0.22 against 0.20 for five classes), so the control compares two near-chance
numbers and its bar sits below the measurement's resolution — R3's category. It
is a control that cannot discriminate on this arm, not a leak. **The pretrained
headline (`e25apre`) stays held** until the bar is re-stated against chance, with
the new bar written before it is applied.

## 6. Artifacts

`runs/e25/r1_c2.json`, `runs/e25/r2_a4_lam1.json`, `runs/e25/ckpt_audit.json`
(pending). Scripts: `scripts/e25_reads.py`; `scripts/audit_checkpoints.py`
(`--chain e25`). Launcher: `jobs_e25_audit`, `ANALYSIS_JOBS["e25audit"]`.
