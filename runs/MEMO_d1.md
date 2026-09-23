# MEMO — D1: The Structure of Feature Drift (scratch arms)

**Status (2026-09-19):** four shared-head scratch arms run and read
(`jobs_d1_scratch`, Modal x86, 4 jobs; `scripts/d1_fit.py` → `d1_diagnose.py`
→ `d1_row.py`; `runs/d1/`, `runs/d1_row.json`). All twelve registered predictions scored (§3a added 2026-09-19 for the two
pretrained-arm D3 rows). Contract: `docs/D1_prereg.md`, signed v2.

## 0. Controls

| control | result |
|---|---|
| C-ID | `acc_orig`, `acc_ceiling` reproduce the seeded decompositions to **0.0e+00** in 48/48 cells |
| C-WIT | shared head, adapters off, era + shadow, construction fingerprints — asserted from the artifacts; path identity (P-A, P-B) on every cell of both eras; HAR on the as-executed partition `1104af185c87` |
| C-INPUT | test-input sha1 recorded per cell: **4 distinct on HAR** (the test population is seed-independent), 12 on each MNIST arm (content split by seed) — a witness the row reads the right sets |
| C-PLUMB (printed, not scored) | ‖S − I‖_F on (Z_T, Z_T): MNIST 0.12–0.78; **HAR 2.96–5.47**. Algebra: M − I = −λn(AᵀA + λnI)⁻¹, so ‖M − I‖²_F ≈ the number of Z_T directions with variance ≪ λn — on HAR about 30 of 256 directions of the final encoder's re-laid features are near-dead. Consistent with D2's collapse count; not a plumbing defect (the identity returns on MNIST). To be confirmed by re-extraction, which this pass did not keep |
| **C-SHUF** (must-fail) | own − own_shuf > 1.96·SE in **12/12 cells on every arm, both frames**: the fit uses the pairing. PASS |
| C-TARGET | own(k) vs the era accuracy: HAR pooled +1.3 pp *above* the ceiling (3/12 cells over resolution), MNIST within 0.2 pp (1–3/12 over). The repair recovers the era accuracy |
| C-λ | verdicts identical across λ ∈ {10⁻⁴, 10⁻³, 10⁻²}. **Record:** the first run flagged HAR on an own() shift of 0.8 pp under a 0.5 pp threshold I had coded — finer than HAR's 3.3 pp resolution, R3's error inside my own check; the bar is the resolution, and the flag was corrected before anything was read from it |

## 1. D1 — is the drift shared? Yes, once presentation is removed; no, before.

Transfer loss own(k) − cross(j→k), pooled [range over the 36 ordered pairs per arm], resolution 1.96·SE (HAR 2.6–4.4 pp, MNIST 1.0 pp):

| arm | adjacent, re-laid | distant, re-laid | pairs > res / > 5 pp | adjacent, **old frame** | distant, old frame | deploy(k) − own(k) |
|---|---|---|---|---|---|---|
| LSTM · HAR | +0.2 pp [−9.1, +11.0] | +2.1 [−16.3, +20.3] | 15/36 / 9/36 | **+19.3** | **+31.0** | −1.8 pp (3/12 cells below own by > res) |
| LSTM · Permuted | +0.6 [−0.7, +3.3] | +0.6 [−0.8, +2.9] | 10/36 / 0/36 | **+60.9** | **+62.1** | −0.3 (0/12) |
| MLP · Permuted | +0.2 [−0.4, +1.5] | +0.8 [−0.1, +2.1] | 8/36 / 0/36 | +14.6 | +24.1 | 0.0 (0/12) |
| MLP · Rotated | +0.0 [−0.5, +0.8] | +0.2 [−0.5, +0.8] | 3/36 / 0/36 | +9.5 | +47.1 | 0.0 (0/12) |

**Reading.** After re-lay, task j's repair applied to task k costs ≤ 0.6 pp
adjacent and ≤ 2.1 pp distant on every arm: **encoder drift is shared** (one
S per checkpoint), with HAR noisy pair-by-pair (negative losses too — another
task's map is sometimes better than one's own). Constraint 1 holds in the
re-laid frame. In the old frame it fails by 10–62 pp: the presentation
component is per-task by construction (a different M_k per task), so **a
repair fit without the map does not transfer**. The deployable form
(`deploy`: W_T reading the repaired features) inherits the repair on the
three MNIST arms exactly and loses 1.8 pp pooled on HAR.

## 2. D2 — is the drift near-orthogonal? No, on every arm.

Spectrum of the standardized drift map (T re-laid | T' old frame), residual of the repair on test:

| arm | spread log(σ₁/σ_d) | mass in [0.9, 1.1] | σ_i < 0.5 | residual S (re-laid) | residual S' (old) |
|---|---|---|---|---|---|
| LSTM · HAR | 11.7 \| 11.6 | 0.04 | 138/256 | 0.45 | 0.57 |
| LSTM · Permuted | 9.5 \| 11.1 | 0.04 | 166/256 | 0.45 | 0.61 |
| MLP · Permuted | 8.5 \| 8.5 | 0.04 | 168/256 | 0.41 | 0.40 |
| MLP · Rotated | 8.3 \| 8.9 | 0.06 | 144/256 | 0.42 | 0.49 |

The registered bar was spread < 0.5 and mass > 0.8. Measured: spread 8–12,
mass 4–6%, and 54–66% of the singular values below 0.5. Era features are all
live (sd ≥ 0.05 on every dimension), so this is not standardization; and
C-PLUMB returns ≈ I on the MNIST arms, so it is not the fitter. **Constraint 3
fails on every arm.** Two cautions on what the spectrum *is*: the singular
values of a least-squares map measure linear predictability, not geometry
alone — a direction of Z_T that Z_k cannot predict linearly is shrunk toward
zero — and the residual says 40–45% of the feature norm is exactly that. The
honest sentence is "the drift is neither orthogonal nor linear", not
"the features collapse in 60% of directions".

**F_enc against the fit.** Across the 48 cells: r(F_enc, residual) = **+0.81**
(partial, given log σ_d: +0.75); r(F_enc, log σ_d) = **−0.55** (partial,
given the residual: −0.33). The encoder term tracks what a linear map cannot
capture more than it tracks collapse. Both registered predictions fired (r <
−0.5 at 55%; residual the better explainer at 45%).

**Re-laid F_enc — E23's (E1) on the scratch arms, all four within resolution
of zero:** HAR −0.017 (res 0.033), P-LSTM +0.011 (0.011), P-MLP +0.007
(0.008), R-MLP +0.004 (0.008), against old-frame F_enc of 0.087 / 0.172 /
0.021 / 0.039. **With the map applied, the encoder term vanishes on every
known-map arm** — the two-origins claim measured on F_enc itself, not only on
the deployed accuracy. Fired at 55%.

## 3. D3 — the subspace lemma, on the arms where the lemma's head is the stored one

`D_out·Wᵀ` ≤ 4×10⁻¹⁴ on every cell: the identity as corrected holds. F_frozen
= acc_ceiling − acc(h_k Z_T') regressed on ‖D_in‖, ‖D_out‖ (h_k projector,
old-frame T'):

| arm | F_frozen | β_in [95%] | β_out [95%] | R² | ΔH |
|---|---|---|---|---|---|
| LSTM · HAR | 0.390 | +0.002 [−0.040, +0.043] | +0.002 [−0.031, +0.034] | 0.02 | −0.3 pp |
| LSTM · Permuted | 0.728 | −0.054 [−0.086, −0.022] | +0.050 [+0.014, +0.087] | 0.62 | +1.1 |
| MLP · Permuted | 0.179 | +0.035 [−0.079, +0.150] | +0.043 [−0.022, +0.108] | 0.82 | +1.5 |
| MLP · Rotated | 0.435 | −0.044 [−0.190, +0.103] | +0.117 [+0.022, +0.213] | 0.97 | +2.1 |
| pooled (48) | 0.433 | +0.005 [−0.037, +0.046] | −0.002 [−0.032, +0.029] | 0.01 | +1.1 |

Secondary on these arms (the deployed head is W_T; h_k is a snapshot), and
the registered D3 rows are the pretrained arms', still pending. What the
scratch arms show is reading (b): where anything tracks F_frozen it is
‖D_out‖, the component the lemma says the head cannot see — which means the
fitted linear T' is not the drift the head sees. The old-frame residuals
(0.40–0.61) say the same thing from the other side. Not read further until
the pretrained arms, where F_frozen ≡ F_total, are in.

### 3a. D3 on the pretrained arms — the direct test (2026-09-19, `jobs_d1_pretrained`)

`runs/d1/{e12_base,e14_base}/cells_seed{s}.json`, `runs/d1/pretrained_diag.json`.
Here the deployed head is the frozen per-task W_k, so F_frozen ≡ F_total.
C-ID 0.0e+00 on 57/57 cells per arm; C-SHUF 57/57 both; `D_out·Wᵀ` ≤ 4×10⁻¹³.

| arm | d | n_train | F_frozen | β_in [95%] | β_out [95%] | R² | T′ spread | σ_i < 0.5 | residual S′ | linear repair: deployed → own (ceiling) |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | 768 | 2,500 | 0.713 | −0.003 [−0.010, +0.005] | +0.003 [−0.006, +0.013] | 0.01 | 27.6 | 500/768 | 0.41 | 0.22 → **0.89** (0.94) |
| ResNet-50 | 2048 | 2,500 | 0.651 | −0.0002 [−0.0004, +0.0000] | +0.0001 [−0.0000, +0.0002] | 0.06 | 18.4 | 834/2048 | 0.70 | 0.29 → **0.89** (0.94) |

**Reading.** Neither projection norm tracks the frozen head's loss (R² 0.01 /
0.06): the registered β_in > 0 **misses** (60%) and "β_out includes 0" fires
(55%) vacuously, because nothing tracks. The linear drift model does not
explain F_total on the pretrained arms any more than it explained F_frozen on
the scratch arms. **Caveat that must travel with these two rows:** n_train =
2,500 against d = 768 / 2,048, so the fits sit at n ≈ d (ResNet) — the spectra
and residuals are not comparable to the scratch arms' (n = 4,000 against d =
256), and the residual of 0.70 on ResNet is partly the fit's own
under-determination. The lemma's algebra is intact (`D_out·Wᵀ = 0` witnessed
to 4×10⁻¹³); its *empirical* clause — that the fitted drift's in-subspace part
predicts the loss — does not hold on any arm. What does hold: a linear repair
fit on paired era/final features and read by the **frozen deployed head**
recovers 0.89 of a 0.94 ceiling on both backbones (from 0.22 / 0.29 deployed) —
the reader-dominance result seen from the repair side, with old data.

## 4. D4 — class balance

Train label fractions: HAR 0.127–0.204 (uniform 0.167); MNIST 0.086–0.115
(uniform 0.100). Stated; one line for Appendix C.

## 5. Predictions — 12 scored

| prediction | odds | outcome |
|---|---|---|
| D1 transfer loss ≤ 5 pp adjacent, all arms | 60% | fired |
| D1 ≤ 5 pp all pairs, all arms | 35% | fired |
| D1 deploy within resolution of own on ≥ 3 arms | 40% | fired (4/4) |
| D2 spread < 0.5, MLP arms | 65% | **miss** (8.3–8.5) |
| D2 spread < 0.5, LSTM arms | 45% | did not fire (9.5–11.7) |
| D2 r(F_enc, log σ_d) < −0.5 | 55% | fired (−0.55) |
| D2 residual explains F_enc better than σ_d | 45% | fired (+0.75 vs −0.33 partial) |
| D2 re-laid F_enc within resolution of 0 on ≥ 3 arms | 55% | fired (4/4) |
| D3 β_in > 0 excluding 0, pretrained | 60% | **miss** (R² 0.01 / 0.06) |
| D3 β_out interval includes 0, pretrained | 55% | fired (vacuously: nothing tracks) |
| D3 fit residual < 10% | 70% | **miss** (41–45%) |
| C-SHUF 12/12 every arm | 90% | fired |

Both misses point the same way as E18's and E21's: the encoder is *less*
structured than predicted — the drift is neither near-orthogonal nor
linear. The series' two miss directions are now 4 (more structure, less
determinism) against 4 (less encoder structure than predicted).

## 6. What this decides (§9 of the contract)

The method sketched in §9 needed three structural constraints. Measured:
**shared drift — yes, but only with the map applied** (without it the
per-task presentation dominates and nothing transfers); **readout-subspace
drift — not on the scratch arms** (pretrained pending); **near-orthogonal —
no, on every arm**. And with the map applied, re-laid F_enc is zero within
resolution on all four arms, so there is no encoder term left for a
readout re-alignment to recover beyond what C0deg already does. On the
known-map benchmarks the method has no foundation; on the no-map regime it
has no foundation either, because the constraint it would rest on fails
there. The diagnostics stand as §4 findings: drift is shared after
re-lay and per-task before it; the encoder term is what a linear map cannot
capture; the map removes it.

## 7. Artifacts

`runs/d1/{s72_off,e18_pmd_lstm,e18_pmd_mlp,e18_rmd_mlp}/{cells,fits}_seed{42,1337,2024}.{json,npz}`,
`runs/d1/<arm>/diag.json`, `runs/d1/controls.json`, `runs/d1_row.json`,
`runs/d1/<arm>/job.log`; inputs `runs/head_drift/*.json`,
`runs/e20/seeded/har_s72_off_linear.json`, `runs/e18_*/decomp.json` (also on
the volume). Scripts: `scripts/d1_fit.py`, `d1_diagnose.py`, `d1_row.py`;
launcher `jobs_d1_scratch`.
