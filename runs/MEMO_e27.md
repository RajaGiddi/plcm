# MEMO — E27: the equivariance defect as a predictor of readout-repair loss

**Status (2026-09-22):** contract `docs/E27_prereg.md` signed v2. Audit green
18/18 (arm B audited for the first time). All 18 jobs landed in ~7 minutes.
Row: `runs/e27_row.json`. **The primary reading did not fire, by 0.004.**

## 0. Controls — all pass, after four pricings of one of them

| control | result | bar |
|---|---|---|
| positive, square invertible, $\lambda = 0$, 10 draws | max $\delta$ **1.0e-9** (784), **1.7e-9** (1152) | 0.01 |
| floor witness, random 784→256 | 0.8491 | ~0.85 |
| shuffled-pairing shape, synthetic | 1.034 vs 0.849 unshuffled | must exceed |
| plumbing, $k = T$ | **0.0** on every arm | exact |

**The positive control was priced four times before launch and each of the
first three fixed a real defect.** v1's version was algebraically impossible: a
dimension-reducing $W$ cannot have full column rank, and at 784→256 it read
0.849 against a 0.01 bar, so it would have fired against a working instrument.
v2 set $\lambda = 10^{-6}$ citing 0.0027 — measured at width 256 when the
contract names 784 and 1152, where it reads 0.037. $10^{-8}$ cleared 784 and
failed 1152 on a bad draw; $10^{-10}$ still failed one. The residual of a ridge
fit to an *exact* linear map scales with $\lambda$ **and** with $\mathrm{cond}(W)$,
which ranges 1,200–69,307 across draws, so no positive $\lambda$ is safe.
An exact map needs no ridge: **$\lambda = 0$ reads $\le 1.7 \times 10^{-9}$ on
every draw.** The fourth pricing removed the knob instead of tuning it, and gave
the control its own seeded stream, since reading it off a shared sequential rng
made its value depend on what had been drawn before it.

## 1. Every arm is LESS equivariant than its own random initialisation

Per-cell guard: $\delta > 1$ means the affine fit is worse than predicting
zero, so the fit failed and the number is not a measurement of equivariance.
**Two of 117 cells excluded, both named.**

| arm | family | cells | excluded | $\delta$ | $\delta^{\text{rand}}$ | $\tilde\delta$ |
|---|---|---|---|---|---|---|
| LSTM · HAR | LSTM | 12 | 1 | 0.616 | 0.417 | **1.51** |
| LSTM · Permuted | LSTM | 12 | 0 | 0.626 | 0.388 | **1.85** |
| A3, 20 tasks | LSTM | 57 | 0 | 0.600 | 0.281 | **2.19** |
| MLP · Permuted | MLP | 12 | 0 | 0.501 | 0.426 | **1.18** |
| arm B, MLP · HAR | MLP | 12 | 1 | 0.597 | 0.482 | **1.24** |
| MLP · Rotated | MLP | 12 | 0 | 0.579 | 0.433 | **1.34** |

**$\tilde\delta > 1$ on all six arms.** Training on twenty (or five) formats
left every encoder *further* from equivariant to the format change than its own
untrained initialisation. The random reference is what makes this readable: raw
$\delta$ is 0.50–0.63 everywhere and says little, because a random encoder of
the same shape already reads 0.28–0.48 from dimension reduction alone.

**LSTM arms are further from equivariant than MLP arms**, 1.85 against 1.25,
and the gap holds arm by arm with no overlap (LSTM 1.51/1.85/2.19, MLP
1.18/1.24/1.34). Prediction 2 fires at 70%.

## 2. One cell of 117 would have flipped a registered comparison

Arm B, seed 2024, task 0 reads $\delta = 15.104$ — twenty-five times any other
cell — against a training-split $\delta$ of 0.386 and a validation $\delta$ of
0.525. The fit succeeds on train and validation and explodes on test only.

**Pooled, that one cell moves arm B from $\delta$ 0.597 to 1.806 and
$\tilde\delta$ from 1.24 to 3.58**, which would have put the MLP family mean
(2.03) *above* the LSTM family mean (1.88) and reversed prediction 2. It is
catch 26 in its exact form: pooling is a way of hiding a failed cell, and the
mean over cells is the wrong summary when one cell is an order of magnitude out.
The per-cell guard is what surfaced it, the row names both exclusions, and the
forced-inclusion figures are printed beside the guarded ones.

**The same cell is the one must-fail violation** (116/117). That is coherent
rather than a second defect: where the honest fit already fails, a shuffled fit
cannot be made to look worse than it.

## 3. The primary reading — did not fire, by 0.004

$\rho(\tilde\delta, G)$ over the 46 surviving primary cells, $G = \text{C0deg} - \text{C3}$,
bootstrap resampling **seeds** rather than cells, because cells within a seed
share a checkpoint:

| quantity | value |
|---|---|
| $\rho(\tilde\delta, G)$ | **+0.596** |
| bootstrap 95% over seeds | [+0.571, +0.706] |
| registered bar | $\rho \ge 0.6$ **and** interval excluding zero |
| verdict | **did not fire** (registered 40%) |

The interval half is met; the point estimate falls **0.004** short of the
threshold. The bar is not moved. It is worth recording that a 0.6 threshold sits
well inside this measurement's own interval, so the pass/fail boundary is finer
than the resolution — R3's category — but that is an observation about the
contract's design, made after the fact, and it does not license a different
verdict.

**Two findings inside the primary that the contract anticipated.**

*Raw $\delta$ predicts better than $\tilde\delta$:* $\rho = +0.715$ against
$+0.596$. The normalisation by the random reference was introduced to remove
architecture from the quantity, and it removed predictive power instead. Catch
25's normaliser corollary: a normaliser is a degree of freedom, and this one
varies with something that was part of the signal.

*The correlation is heterogeneous across arms*, which is the contract's second
branch:

| arm | within-arm $\rho$ | cells |
|---|---|---|
| LSTM · HAR | **+0.918** | 11 |
| MLP · Permuted | +0.629 | 12 |
| arm B, MLP · HAR | +0.191 | 11 |
| LSTM · Permuted | +0.063 | 12 |

The within-arm mean is +0.45 against a pooled +0.596, so part of the pooled
figure is the between-arm architecture split rather than cell-level prediction.
On one arm the defect predicts the loss almost perfectly; on another it does not
predict it at all.

**Reading, per §4's table: positive but not at the bar, and partly carried by
the architecture split — a diagnostic, not a predictor.** Paper 1 does not get
the §6 paragraph as contracted. What it can carry is §1's result, which is
cleaner than the predictor was: training moves every encoder away from
equivariance to the format it will later be asked to repair, and it does so more
for recurrent encoders than for feed-forward ones.

## 4. Predictions — 6 scored, 3 fired, 2 missed, 1 did not fire

| prediction | odds | outcome |
|---|---|---|
| $\rho(\tilde\delta, G) \ge 0.6$, interval excluding zero | 40% | did not fire (+0.596) |
| mean $\tilde\delta$ on LSTM exceeds MLP | 70% | **fired** (1.85 vs 1.25) |
| $\tilde\delta < 1$ on MLP arms | 55% | **MISS** (1.18, 1.24, 1.34) |
| $\tilde\delta > 1$ on LSTM arms | 50% | **fired** (1.51, 1.85, 2.19) |
| positive control below 0.01 | 95% | **fired** (1.7e-9) |
| must-fail on every cell | 90% | **MISS** (116/117) |

## 5. Artifacts

`runs/e27/ckpt_audit.json`, `runs/e27/{arm}/defect_seed{s}.json` (6 arms × 3
seeds), `runs/e27_row.json`. Scripts `scripts/e27_defect.py`,
`scripts/e27_row.py`; `audit_checkpoints.py --chain e27`; launchers
`jobs_e27_audit`, `jobs_e27_defect`.
