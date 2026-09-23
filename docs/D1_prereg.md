# D1 — The Structure of Feature Drift (v2)

**Status:** **SIGNED 2026-09-19** (v2, on the description; the four † odds in
§6 set by the analyst as written). Smoke: `e18_pmd_mlp` seed 42, 11 s, C-ID
0.0e+00, `D_out·Wᵀ` ≤ 8×10⁻¹⁶; era feature sd ≥ 0.05 on every dim, so the
sub-0.5 singular values the smoke shows (137–166 of 256) are drift, not
standardization. Launched 2026-09-19 as `jobs_d1_scratch` (4 CPU jobs, x86); **read the same day** — `runs/MEMO_d1.md`, `runs/d1_row.json`: 10 predictions scored (7 fired, 2 misses, 1 did not fire), D3 on the pretrained arms pending `jobs_d1_pretrained`.
After the v1 review (six blocking findings, all accepted; §0a). Analysis only, on existing era
checkpoints; no training, no new construction. **Not a method contract**: it
measures whether four structural properties of the drift hold on the paper's
own data. Each measurement is a finding for §4 whether or not a method is
ever built on it; the method sketched in §9 is not contracted here and gets a
literature sweep before it is.

## 0. Verified before signing — pasted from the repo, 2026-09-19

**Era checkpoints and fp32 shadow, four arms × three seeds.** All twelve
`plcm-runs:/ckpt_{e10off_ec,e18_pmd_mlp,e18_pmd_lstm,e18_rmd_mlp}_seed{42,1337,2024}/mafc_seed{s}_fp32/`
hold `task{0..4}_epoch9.pt` (5/5). Every run artifact records
`era_checkpoints: True`, `use_task_heads: False`, `use_input_adapters: False`,
five `fp32_shadow` records with max |delta| **0.0**. Locally: the E18 MNIST
checkpoints (45 files) are pulled; S72's are on the volume, and `har_subject`'s
partition is x86-only (the laptop builds `5d047e4213d1`, the executed
partition is `1104af185c87`, gated in every consuming script), so **every HAR
job runs on Modal**.

**Era loader.** `PLCM.load_era` ([plcm.py:868](../src/models/plcm.py#L868)) →
`load_from_checkpoint` ([plcm.py:835](../src/models/plcm.py#L835)), which
restores `task_stats` (catch 29); analysis entry `channel_decomp.load`
([channel_decomp.py:104](../scripts/channel_decomp.py#L104)).

**Paired-feature path: exists.** [cure_screen.py:228-231](../scripts/cure_screen.py#L228-L231)
extracts era and final features from the same `xtr, ytr` tensors; E9's
[transport_estimate.py:96-97](../scripts/transport_estimate.py#L96-L97) does
the same and fits a Procrustes map (`procrustes` :58); and
[head_drift.py](../scripts/head_drift.py) (2026-09-19) applies the era head to
final features per cell with C-ID against the seeded decompositions. D1 reuses
that pairing; only the least-squares fit and the diagnostics are new.

**Audited loader.** `load_task_data`
([channel_decomp.py:422](../scripts/channel_decomp.py#L422)): single pass,
in-memory tensors, `probe_subset_seed` recorded; `N_TRAIN = 4000`,
`N_TEST = 2000` caps (:73–74). Both encoders consume the *same tensor*, so
`torch.equal(y_era, y_final)` holds by construction and would be a tautology;
the non-tautological witness is the input's hash, recorded per cell (§4).

**Count line.** [appendix.tex:581](appendix.tex#L581): *"Thirty-nine scored
entries, six misses (2026-09-18)."*

**Measured input, already on disk (head drift).** `runs/head_drift/*.json`:
on every D1 arm the deployed head is the shared W_T, and
`F_read = F_read^frozen + ΔH` with ΔH pooled **−0.3 / +1.1 / +1.5 / +2.1 pp**
(HAR-LSTM / P-LSTM / P-MLP / R-MLP), C-ID 48/48 cells at 1e-6. On the
pretrained arms the per-task heads are bit-identical across era checkpoints
(ResNet-50 heads 0 and 10, ViT head 0, seed 42), so there `F_frozen ≡ F_total`.

### 0a. Review amendments (v1 → v2)

| # | v1 | v2 |
|---|---|---|
| B1 | Z_k = f_{θ_k}(re-laid x): the era encoder read frame T, which it never saw | Z_k = f_{θ_k}(x_k) in frame k; Z_T = f_{θ_T}(relay_T(x_k)) — both encoders at home on the same samples; old-frame Z_T' reported beside |
| B2 | D3 regressed F_read (a T-free quantity) on properties of T, with W_k as "the readout" | D3's target is the frozen-head loss F_frozen; W_k is the deployed head only on the pretrained arms; ΔH reported as the part no feature-drift model explains |
| B3 | displayed identity `W(T−I)(I−P_W) = 0` (numerically 16.2) | `(T−I)(I−P_W)Wᵀ = 0` in the row convention the fit uses; cites the appendix's subspace lemma |
| B4 | control 1 (T at k = T ≈ I) presented as a control | labelled plumbing; the must-fail is the shuffled pairing with a numeric bar |
| B5 | "≤ floor" with zero reproduction floors | resolution = 1.96·SE_binomial per cell (R3) |
| B6 | scripts named, no launcher entries; laptop cost | `jobs_d1_*` on Modal (x86 for HAR); one L4 job for the pretrained D3 arm |
| — | T_j⁻¹ inverted | repair direction S fit directly; T fit for the spectrum |
| — | h_k only | transfer read with h_k (diagnostic) and W_T (deployable) |
| — | — | re-laid F_enc on the scratch arms (E23's (E1) for free); the σ_d–F_enc test as the sharp prediction |

## 1. Arms

| Arm (artifact) | Encoder | d | c | head | n_train / n_test per task |
|---|---|---|---|---|---|
| `s72_off` — HAR subject-disjoint | LSTM | 256 | 6 | shared W_T (era h_k stored) | ≤ 1,700 (all) / 344–409 |
| `e18_pmd_lstm` — Permuted-disjoint | LSTM | 256 | 10 | shared | 4,000 cap / 2,000 |
| `e18_pmd_mlp` — Permuted-disjoint | MLP | 256 | 10 | shared | 4,000 / 2,000 |
| `e18_rmd_mlp` — Rotated-disjoint | MLP | 256 | 10 | shared | 4,000 / 2,000 |
| `e12_base`, `e14_base` — Split-CIFAR-100 | ViT-B/16, ResNet-50 | 768, 2048 | 5 | per-task, **frozen (verified)** | full task set / 500 |

Twelve cells per scratch arm (3 seeds × 4 old tasks). The pretrained arms
have no known map, so they enter D2 and D3 (old-frame drift only) and not D1's
re-laid pairing; on them D3 is the *direct* test, because the deployed head is
the frozen W_k.

## 2. The measurement — displayed, with the frame of every term

Per arm, seed, old task k < T, on the audited loader's task-k **train** draw
(`N_TRAIN = 4000`, seed 20260916; HAR uses the whole task), row convention
throughout (samples are rows):

```
Z_k   = [ f_{θ_k}(x_i) ]_i            x_i in FRAME k  — the era encoder at home; the features h_k was fit on
Z_T   = [ f_{θ_T}(relay_T(x_i)) ]_i   the same samples carried into FRAME T with the known map — the final encoder at home
Z_T'  = [ f_{θ_T}(x_i) ]_i            x_i in frame k read by θ_T — the deployed input, presentation + encoder drift together

S_k   = argmin_S ‖ Z_k − Z_T S ‖_F² + λ‖S‖_F²      REPAIR direction (final → era), the map a method would apply; no inversion
T_k   = argmin_T ‖ Z_T − Z_k T ‖_F² + λ‖T‖_F²      DRIFT direction (era → final), for the spectrum in D2 and the projector algebra in D3
S'_k, T'_k                                          the same two fits on Z_T' (old frame)
```

`relay_T` is `har_relayout` / `mnist_relayout` / `rotated_relayout`
([cure_screen.py:141-150, 280-286](../scripts/cure_screen.py#L141-L150)) —
C0deg's input path. Features standardized on the era pass's train statistics;
λ = 10⁻³ recorded, swept over {10⁻⁴, 10⁻³, 10⁻²} and the reading must not
change across the sweep. The fit residual ‖Z_k − Z_T S_k‖_F / ‖Z_k‖_F is
recorded per cell: it is the linear model's own adequacy, read in D2.

The difference between the re-laid and old-frame fits is the presentation
term; on the scratch arms it is the quantity Table 1's re-layout column
removes.

## 3. Four diagnostics, readings pre-committed, all two-sided

### D1 — Is the drift shared across tasks?

Per seed, compare S_1, …, S_{T−1}. The measure that matters is transfer:

```
own(k)       = acc( h_k · Z_T^{test}(k) S_k )        task k's own repair, era head
cross(j→k)   = acc( h_k · Z_T^{test}(k) S_j )        task j's repair applied to task k
deploy(k)    = acc( W_T · Z_T^{test}(k) S_k )         the same repair read by the DEPLOYED head — the only deployable form
```

Transfer loss = own(k) − cross(j→k). Resolution per cell: 1.96·SE_binomial on
n_test (HAR 2.6–4.4 pp, MNIST 1.0 pp). `own` is also what control C-TARGET
reads.

| reading | meaning |
|---|---|
| transfer loss ≤ resolution on every pair | drift is shared: one S per checkpoint. Constraint 1 holds |
| ≤ resolution for adjacent tasks, > 5 pp for distant | shared locally; S composes across steps. Constraint 1 holds in composed form |
| > 5 pp on most pairs | task-specific drift. Constraint 1 fails |
| `deploy(k)` below `own(k)` by > resolution | the repair works only with the stored era head — SNAP's resource; the deployable form does not inherit it, reported as such |

### D2 — Is the drift near-orthogonal, and does its spectrum explain F_enc?

Singular values σ₁ ≥ … ≥ σ_d of T_k (and T'_k):

```
spread = log(σ₁ / σ_d)        mass = fraction of σ_i in [0.9, 1.1]        residual = ‖Z_k − Z_T S_k‖_F / ‖Z_k‖_F
```

| reading | meaning |
|---|---|
| spread < 0.5, mass > 0.8 | near-orthogonal: information-preserving rotation (Masip et al., **unverified until read**) |
| spread > 1 with small σ_d | collapse in some directions |
| spread large with σ₁ ≫ 1 | stretch; report |

**The sharp prediction.** Under a linear, invertible drift a linear refit
absorbs T entirely, so F_enc = 0. Table 1 reads F_enc = 0.02–0.17 on these
arms; therefore the drift is non-linear (large residual) or singular on
class-relevant directions (small σ_d), or both. Across the 48 scratch cells,
regress F_enc (old-frame, the Table 1 quantity, on T'_k) on `residual` and on
`log σ_d`; report both coefficients with intervals. The re-laid variant —
F_enc^relaid = acc_refit_ceiling − refit(Z_T) — is computed beside it: it is
E23's (E1) on the scratch arms, and its two-sided reading is E23 §6's.

### D3 — Does the subspace lemma hold, on the arms where it applies?

The lemma ([appendix.tex, "Subspace lemma"](appendix.tex#L112)): for a
**fixed** head W with row-space projector P_W = Wᵀ(WWᵀ)⁻¹W, drift outside the
row space does not move that head's logits. In the row convention of §2:

```
D_in  = (T − I) P_W          D_out = (T − I)(I − P_W)          D_out · Wᵀ = 0   identically
```

(v1 displayed `W(T−I)(I−P_W) = 0`, which is false — 16.2 on a random instance
where `W(I−P_W)` is 3×10⁻¹⁵.) The quantity the lemma governs is the fixed
head's own loss:

```
F_frozen(k) = acc( W_k Z_k ) − acc( W_k Z_T' )
```

On the pretrained arms W_k is the deployed head (bit-identical across eras)
and **F_frozen ≡ F_total**; on the scratch arms W_k = h_k is the stored era
head and F_frozen is `acc_ceiling − acc_frozen` from `runs/head_drift/`,
with the remainder ΔH = acc(h_k Z_T') − acc(W_T Z_T') already measured. F_read
is **not** the target: it is refit − deployed on Z_T' alone and contains no T.

Regress F_frozen on ‖D_in‖_F and ‖D_out‖_F across cells (T'_k on every arm;
T_k re-laid beside it on the scratch arms).

| reading | meaning |
|---|---|
| coefficient on ‖D_out‖ within its interval of 0, ‖D_in‖ positive | the linear drift model reproduces the lemma; the readout-relevant drift is c-dimensional. Constraint 2 holds |
| ‖D_out‖ coefficient nonzero | the fitted T is not the drift the head sees (non-linear residual): read with D2's residual, which must be large in those cells or the fit is wrong |
| neither tracks | the linear model does not explain the frozen-head loss; the method's premise fails |

### D4 — Class balance

Label marginals per task per arm from the loader; one line in Appendix C.

## 4. Controls — each with the case where it must fail

| control | statement | bar / must-fail |
|---|---|---|
| **C-ID** | `acc_orig`, `acc_ceiling` per cell reproduce the seeded decomposition (`runs/e20/seeded/har_s72_off_linear.json`, `runs/e18_*/decomp.json`) | 1e-6, 48/48, else nothing read (as `head_drift.py` does) |
| **C-INPUT** | the era pass and the final pass consume the same tensor: its sha1 recorded per cell, asserted equal for both passes | the non-tautological form of the label assert |
| **C-WIT** | arm identity from the artifacts (shared head, adapters off, era + shadow, construction fingerprint); path identity on every cell (P-A raises, P-B must hold) | as every consuming script |
| **C-PLUMB** | fit S on (Z_T, Z_T): returns ≈ I up to ridge shrinkage **by algebra** — cannot fail; a plumbing check, not a control | printed, not scored |
| **C-SHUF** must-fail | fit S on shuffled pairs (era features of sample i against final features of a different sample); `own_shuf(k)` | `own(k) − own_shuf(k)` > 1.96·SE_binomial in **12/12 cells per arm**, stated here; the pairing must be what the fit uses |
| **C-TARGET** | `own(k)` targets the **era accuracy** `acc_ceiling` (h_k reading recovered era features), not C0deg's re-laid number, which is h_T's and exceeds the era's | gap `acc_ceiling − own(k)` read against 1.96·SE per cell; a gap above it is fit error, reported |
| **C-λ** | readings identical across λ ∈ {10⁻⁴, 10⁻³, 10⁻²} | any reading that flips with λ is not read |

## 5. What the findings do for the paper, independent of any method

- **D1** → §4.2: whether drift is a property of the encoder's trajectory or of each task; and whether a repair fit with the era head survives the deployed head.
- **D2** → the σ_d / residual account of F_enc; Masip's taxonomy cited only after the paper is read.
- **D3** → the subspace lemma stays a lemma (algebra) and gains an empirical clause on the arms where the head is fixed; on the shared-head arms ΔH is stated as the part outside it.
- **D4** → Appendix C.

## 6. Predictions — registered before the fits run

*Read from `docs/appendix.tex` (`app:predictions`, line 581) at the moment of
writing:* **"Thirty-nine scored entries, six misses (2026-09-18)."**

| prediction | odds |
|---|---|
| D1: transfer loss ≤ 5 pp on adjacent tasks, all four arms | ~60% |
| D1: transfer loss ≤ 5 pp on all pairs, all four arms | ~35% |
| D1: `deploy(k)` within resolution of `own(k)` on ≥ 3 of 4 arms | ~40% † |
| D2: spread < 0.5 on the MLP arms | ~65% |
| D2: spread < 0.5 on the LSTM arms | ~45% |
| D2: F_enc correlates with log σ_d across the 48 cells (r > 0.5) | ~55% |
| D2: the residual explains F_enc better than σ_d does (larger partial r) | ~45% † |
| D2: re-laid F_enc within resolution of 0 on ≥ 3 of 4 scratch arms | ~55% † |
| D3: ‖D_in‖ coefficient positive with interval excluding 0, pretrained arms | ~60% |
| D3: ‖D_out‖ coefficient's interval includes 0, pretrained arms | ~55% † |
| D3: fit residual under 10% of ‖Z_k‖ | ~70% |
| C-SHUF passes 12/12 on every arm | ~90% |

† proposed here for sign-off; the others carry over from v1 where the clause
survived.

## 7. CLAUSE → JOB → ARTIFACT (catch 33) and the audit table (catch 20)

| clause | script / launcher | artifact |
|---|---|---|
| §2 paired extraction + fits, scratch arms | `scripts/d1_fit.py --arm {s72_off,e18_*}`; `jobs_d1_scratch` → `analyze_one` (x86; 4 jobs) | `runs/d1/{arm}/fits_seed{s}.npz` (S, T, S', T', residuals, input hashes) |
| §2 fits, pretrained arms (old frame, D2/D3 only) | `scripts/d1_fit.py --arm {e12_base,e14_base}`; `jobs_d1_pretrained` → `analyze_one_gpu` (2 jobs) | `runs/d1/{arm}/fits_seed{s}.npz` |
| §3 D1–D4 + §4 controls | `scripts/d1_diagnose.py` (first block: C-ID, C-INPUT, C-WIT, C-PLUMB, C-SHUF, C-TARGET) | `runs/d1/{arm}/diag.json`, `runs/d1/controls.json` |
| §3 D3's F_frozen and ΔH | **exists**: `scripts/head_drift.py` | `runs/head_drift/{arm}.json` |
| §6 row | `scripts/d1_row.py` | `runs/d1_row.json` |
| audit | `scripts/audit_checkpoints.py` by `PLCM.load_era` | `runs/d1/ckpt_audit.json` |

**Checkpoint audit table (2026-09-19):** scratch arms — 60/60 era files on
the volume, loaded today by `channel_decomp.load` in `head_drift.py` (48
cells, path identity on all); `ckpt_e12_base_seed{42,1337,2024,1234,7}`,
`ckpt_e14_base_seed{42,1337,2024}` — on the volume, loaded by the E12/E14
decompositions; task 0/10/19 of seed 42 loaded today for the head comparison.

## 8. Cost

Scratch arms: extraction is two forward passes per cell on ≤ 4,000 samples
(the head-drift pass took minutes per arm on CPU); 256×256 ridge fits are
seconds. Four Modal CPU jobs, well under one CPU-hour. Pretrained arms: one
L4 job each (95 and 57 cells; d = 768 / 2048 fits). Nothing trains.

## 9. What comes after

If D1–D3 hold, the method — one shared S restricted to the readout subspace,
constrained near-orthogonal, fit by entropy with a class-balance term — has a
foundation and gets its sweep before it is contracted; if `deploy(k)` fails
where `own(k)` passes, the method needs the era head and is priced against
SNAP first (catch 24). If they do not hold, the diagnostics are findings and
the method is not pursued.
