# MEMO — Class-IL probe convergence audit

**Scope:** an audit of existing artifacts. No model retrained, no checkpoint
regenerated. Three changes inside the probe-fitting path, one re-run of the 57
class-IL cells against the stored ViT era checkpoints, one memo.

**Outcome:** the 100-way probe converged on every cell. The class-IL caption may
use the same convergence language as the rest of the paper.

---

## 1. Guard hits

**0 of 57 cells.** Across all **171 fits** (57 × three probes: era-ceiling
refit, θ_T refit, 100-way all-seen refit), no fit reached `n_iter_ ≥ 5000`.

The 100-way probe now carries the guard `refit_probe` always had — same
`LBFGS_MAX_ITER` constant, same message string, so one grep audits both paths.

## 2. Iteration distribution (57 cells, cap 5000)

| fit | n | min | median | max |
|---|---|---|---|---|
| era-ceiling refit (5-way) | 57 | 6 | 11 | 15 |
| θ_T refit (5-way) | 57 | 16 | 32 | 149 |
| all-seen refit (100-way, one per seed) | 3 | 463 | 552 | 564 |
| **all fits** | **171** | **6** | **32** | **564** |

The most expensive fit in the paper — 100 classes over all 20 train splits —
stops at **11% of the cap**. The 26 task-IL decomposition cells were **not**
re-fit (out of scope); their convergence evidence remains the retained stdout
logs, 0 guard hits of 26.

## 3. Reproduction against v3 (the recorded values)

| quantity | v3 (recorded) | v4 (re-run) | per-cell max \|Δ\| | cells identical |
|---|---|---|---|---|
| masked reader share | **97.86%** [97.26, 98.46] | **97.86%** [97.26, 98.47] | — | Δ = −0.0002 pp |
| 100-way refit | **0.6393** [0.6248, 0.6537] | **0.6392** [0.6247, 0.6536] | 0.0100 | 40/57 |
| within-task refit (`acc_refit`) | 0.9411 | 0.9411 | 0.0020 | 55/57 |
| `acc_refit_ceiling` | — | — | **0.0000** | **57/57** |
| `acc_orig_masked` | — | — | **0.0000** | **57/57** |
| `F_enc` / `F_read_masked` | — | — | 0.0020 | 55/57 |

Ledger values reproduce: 97.86% [97.27, 98.45] and 0.6393 [0.6251, 0.6534].
**Correction to this memo's first draft**, which called the one-unit interval
difference "rounding": it was an interval-method difference. The table above
used a t-interval (df = 56); the ledger's script (`e12_hv1.py`) uses
1.96 × SE. **Under the ledger's own method v4 gives exactly [97.27, 98.45]** —
identical to v3 and to the ledger. There was no discrepancy; the memo
manufactured one by switching methods mid-comparison, which is its own small
instance of the rule that a comparison is made with one instrument.

**Reading the nonzero deltas — the rule, applied.** No reproduction floor exists
for the class-IL arm at this configuration: the only E12 floor pair is the
task-IL base arm's *training* floor, and no same-checkpoint re-decomposition had
been performed on any arm before this one. **This run is therefore the first
measurement of that floor, not a comparison against one.** The only prior
reference point is the v3 fp16 reload delta, **2.8 × 10⁻⁸**, which says the
checkpoints reload bit-faithfully.

What the deltas are: the deployed-path quantities (`acc_orig_masked`,
`acc_refit_ceiling`) reproduce **57/57 exactly**. The refit quantities differ on
a handful of cells by one or two test samples (5-way: 0.002 = 1/500; 100-way:
0.010 = 5/500). The 100-way fit is one fit per seed shared across 19 cells, so
17 cells shifting together on one seed is one fit shifting once. **Observation,
not mechanism**: the deployed path is exact and the refit path moves at the
single-sample level — consistent with a sub-ULP difference in GPU feature
extraction reaching a decision boundary through the solver, and not explained by
it. Recorded as the arm's same-checkpoint re-decomposition floor:
**0.002 (5-way) / 0.010 (100-way) per cell, 0.0002 pp on the share.**

## 4. Platform

**x86/Linux, NVIDIA L4, Modal** — matching v3. This matters: two local
re-fits of a different cell (MLP floor, MNIST) on macOS/ARM read **81.45%** and
**82.21%** on identical checkpoints, against **82.03%** on Modal — the local
platform carries its own ~0.8 pp share floor where Modal carried 0.01 pp. A
local re-run of these 57 cells would have been uninterpretable against v3 for
that reason alone.

## 5. scikit-learn version

**1.9.0**, recorded in every row of every v4 artifact — the first time the
version behind a Modal-produced decomposition is known rather than inferred
(the image installs it unpinned). At 1.9.0 the `multi_class` parameter no longer
exists; L-BFGS is multinomial by construction. Since v4 reproduces v3 to the
single-sample level, whatever version produced v3 fit the same probes.

## 6. Changes made (all inside the probe-fitting path)

| change | files |
|---|---|
| `refit_probe(…, info=None)` — optional dict receives `probe_n_iter`; return type unchanged, existing callers unaffected | `channel_decomp.py` |
| 100-way probe: guard mirrored from `refit_probe`, literal `5000` → shared `LBFGS_MAX_ITER` (same value) | `e12_decompose.py` |
| `probe_n_iter` in every row, one key sub-keyed per fit (`ceiling`, `t4`, `unmasked_100way`) | `channel_decomp.decompose`, `e12_`, `e14_`, `e16_decompose` |
| `sklearn_version` per row; top-level plus `probe_n_iter_max` in the E16 driver | all four paths |

Untouched: `max_iter`, `C`, solver, standardization, training code, checkpoint
loading, every other experiment's cells. v3 artifacts preserved; v4 written to
its own directory.

## 7. Consequence for the appendix

`app:hyperparams` §Probe fitting: the `\todo` on the 100-way probe resolves —
*"the iteration guard fired on 0 of 26 retained decomposition cells and 0 of 57
class-incremental cells (171 fits; max n_iter 564 against a cap of 5000)."*
The version clause resolves to **scikit-learn 1.9.0, recorded per artifact from
this audit forward; the version behind pre-audit artifacts is not recorded, and
those artifacts reproduce under 1.9.0 to the single-sample level.**
