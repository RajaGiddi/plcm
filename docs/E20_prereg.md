# Pre-Registration: E20 — Nonlinear probe control for the readout-share lower bound

**Status: SIGNED 2026-09-16**, with one addition (§1, the S72 OFF linear
decomposition is a first-class output) and one ruling folded in (§8, E20-B: the
66–88% band relaunched under uniform arms). **Amended the same day by two
rulings after the first live pass: R1 (§4 — the live positive control could
not be built on these features; replaced by a non-vacuous gate, the recipe
certified on a synthetic shaped like the features, seven designs on record)
and R2 (§5 — the probe's training subset was an unrecorded draw; every cited
E20 number is re-run with a recorded seed). Record: `runs/MEMO_e20.md`.** Analysis-only on existing
checkpoints, plus E20-B's 15 training runs. First in the ruled order (E18 §0a) because it is a day and feeds
§4 regardless of E18's outcome. Review amendments of 2026-09-15 are carried in
whole: relocated to the arm with the largest linear `F_enc`, solver pinned,
three init seeds as the probe's floor, a positive control on a known-nonlinear
target, the identity re-printed under the same gate, and a train/test-gap bar
instead of a story for the third branch.

**Why.** §3.1 argues the reported readout shares are lower bounds because a
linear probe may miss nonlinearly recoverable information. That is a logical
argument. One MLP probe converts it to an empirical one — and it must be placed
where a nonlinear probe *could* move the number. The ResNet's linear `F_enc` is
0.0076, so "MLP within 0.01" there can only miss upward (catch 25's shape: a
test placed where it cannot fail in the interesting direction). The MLP-MNIST
arm's linear `F_enc` is **0.0707** pooled (0.009–0.184 per cell); the S72
HAR-OFF arm's is **0.087**. Those are the primary and secondary.

---

## 1. Arms and checkpoints — audit table (catch 20), verified before sign-off

| arm | seeds | checkpoints (volume) | era | precision | loads | used before | status |
|---|---|---|---|---|---|---|---|
| **primary** MLP / Permuted MNIST (E17 arm: `--backbone mlp --no-adapters`, one shared head) | 42, 1337, 2024 | `ckpt_e17_mlp_seed{s}/mafc_seed{s}`, task0..4 | ON | fp16 | ✓ (E17 decomposition, `runs/e16_decomp/mnist_mlp.json`, era `[True,True,True]`) | yes | **audited** |
| **secondary** LSTM-OFF / `har_subject` (S72 arm) | 42, 1337, 2024 | `ckpt_e10off_ec_seed{s}/mafc_seed{s}_fp32`, task0..4 | ON | **fp32 shadow** | ✓ (S72 screen; matrix reproduced 4e-9) | yes | audited |
| MLP floor pair (probe floor is measured differently, §4; listed for completeness) | a, b | `ckpt_e17_mlp_floor_{t}/mafc_seed42` | ON | fp16 | ✓ | yes | not needed |
| **first-class output — S72 OFF, LINEAR decomposition** (the LSTM band's OFF share on citable provenance; replaces `runs/e11_e6b` HAR/OFF 66.2% under Ruling A) | 42, 1337, 2024 | same as the secondary row | ON | fp32 shadow | ✓ | S72 screen only | **audited; a paper input, not a by-product** — cross-checked per cell against `runs/e10ec/cures_e10.json` (`acc_refit`, `acc_ceiling + R`, `acc_orig`): same instrument, two scripts, one run, must agree to 1e-6 |
| **E20-B (§8) — band ON arms relaunched on `har_subject`**: v1-ON, v2-ON, v2-control | 42, 1337, 2024 + floor a/b each | `ckpt_e10{v1on,v2on,v2ctl}_ec_seed{s}/mafc_seed{s}_fp32` | ON | fp32 shadow | — | — | **new runs** (`jobs_s72_band`, 15 jobs); audited by `audit_checkpoints.py` + matrix reproduction before decomposition |

**fp16 on the primary is acceptable here and the reason is stated:** E20
compares two probes fit on the *same* reloaded features per cell (paired), so
reload imprecision enters both sides identically and cancels in `ΔF_enc`. No
cell of this experiment sits behind the 1e-6 matrix gate; the identity is
re-printed per cell under each probe (§3). The secondary has shadows anyway.

**Reference numbers, cited:** linear `F_enc` per cell from
`runs/e16_decomp/mnist_mlp.json` (`rows[*].F_enc`; pooled 0.0707, `F_read`
0.3351, share 82.6%); HAR-OFF S72 linear `F_enc` from
`runs/e10ec/cures_e10.json` (`acc_ceiling + R − acc_refit`, pooled 0.087) —
the latter is re-derived by the same script here rather than copied.

---

## 2. The probe — pinned, knob-free where possible, floored where not

```
linear (existing, unchanged):  StandardScaler(train) → LogisticRegression(lbfgs, C=1, max_iter=5000), n_iter_ recorded
MLP   (new):                   StandardScaler(train) → MLPClassifier(hidden=(512,), relu, solver=lbfgs,
                                                       alpha=1e-4 [sklearn default, stated], max_iter=2000,
                                                       random_state ∈ {0, 1, 2}), n_iter_, loss_ recorded per fit
```

- **Same standardization, same train/test split, same N_TRAIN/N_TEST caps** as
  the linear probe: the extraction routes through
  `channel_decomp.load_task_data` and `features_and_logits` (catch 32 — no
  parallel path), and the MLP probe is a second `refit_probe` in the same
  `decompose()` loop, selected by a `probe` argument whose default reproduces
  the existing artifact (§5, regression).
- **lbfgs, full batch**: no learning rate, no batch size, no epochs. What is
  left — initialization — is floored: three `random_state` values per fit,
  the per-cell **mean** is the number and the per-cell **spread** is the
  probe's floor. Convergence guard: `n_iter_ < max_iter` on every fit, or the
  cell is marked unconverged and excluded, with the exclusion counted.
- **Not tuned on the outcome.** `alpha` stays at the default unless the
  overfit flag (§3, branch 3) fires; then an `alpha` sweep {1e-4, 1e-3, 1e-2,
  1e-1} is selected on a **20% validation split of the train set**, n = 3
  seeds, reported in full, never on test.

---

## 3. Quantities, formulas displayed, and the three readings

Per cell (seed × old task k), under probe `p ∈ {linear, mlp}`:

```
R^p       = acc_refit_ceiling^p − acc_ceiling            # instrument bias under p
F_enc^p   = acc_refit_ceiling^p − acc_refit_t4^p
F_read^p  = acc_refit_t4^p      − acc_orig
F_total   = acc_ceiling          − acc_orig               # probe-free
identity: F_enc^p + F_read^p − R^p = F_total               # printed per cell, per probe (holds algebraically; printed anyway — catch 22)
ΔF_enc    = F_enc^mlp − F_enc^linear                       # PAIRED within cell; the verdict quantity
share^p   = F_read^p / (F_enc^p + F_read^p)                # pooled means, both probes
gap^p     = acc_train^p − acc_test^p  (t4 fit)             # overfit witness
```

**Gates.** `|R^mlp|` pooled per arm ≤ 0.05 (the E14 bar, reused unchanged) —
a probe that cannot reproduce the era model's own accuracy on the era
features is not reading the features, and the reading is withheld. Identity
residual < 1e-6 per cell, both probes.

**Readings, pre-committed, two-sided (pooled `ΔF_enc` over 12 cells, both arms):**

| outcome | reading |
|---|---|
| `|ΔF_enc| ≤ 0.01` | the linear bound is tight against MLP-recoverable information; §4 gains one sentence and the shares stand as reported |
| `ΔF_enc < −0.01` (MLP recovers more) | the linear probe understated survival; the readout share is **higher** than reported — favourable, reported as such, with the MLP share beside the linear one |
| `ΔF_enc > +0.01` | **no story is pre-assigned.** The overfit flag decides what is read: if `gap^mlp − gap^linear > 0.05` on the cells driving it, the MLP reading is withheld pending the alpha sweep (§2); if the flag does *not* fire, the era features are more nonlinearly separable than the final ones — an encoder-side finding, reported |

Floors: the probe's own (init spread) for `ΔF_enc`, per cell, printed beside
it; no pooled `ΔF_enc` below the mean per-cell spread is read as a sign. The
checkpoint-level floor (`mnist_mlp_floor.json`, task-0 `F_enc` 0.153 vs 0.145)
does not apply to a paired within-checkpoint difference and is not cited as
if it did.

---

## 4. Positive control — before the first live read (catch 25)

A known-nonlinear target the linear probe **must** fail and the MLP probe
**must** pass, on the same features, same recipe:

```
y' = (y mod 2)  XOR  [ w · ( z − μ_y(train) ) > 0 ],   z = standardized features, w a fixed random unit vector
```

The second bit is computed on the **within-class residual** (train class means
removed), so it carries no class information by construction; the first is
linearly decodable; their XOR is not linearly separable. **Bar:**
`acc^mlp(y') − acc^linear(y') ≥ 0.15` on the held-out split, every seed. Run
twice: (a) on **synthetic** features locally, before any job launches, so the
recipe is shown to work at all; (b) on the **live** features of task 0 at θ_T,
on Modal, as the first thing the live script does — if (b) fails, nothing
after it is read. Both recorded (`runs/e20/positive_control_{synthetic,live}.json`).

**v1 of this control FAILED on the synthetic run, and the record is kept**
(`runs/e20/positive_control_synthetic_v1_FAIL.json`). v1's second bit was
`[PC1 > 0]`; the *linear* probe scored **0.9635** on it. With 10 well-separated
class clusters PC1 is a function of the class, so the XOR collapsed to a
dichotomy of 10 cluster means — always linearly separable in 256-d. A control
the linear probe passes cannot detect a broken MLP probe; the synthetic run
exists so that this is found before a live cell is read. v2 (above):
linear **0.5095**, MLP **0.842 / 0.846 / 0.838**, margins ≥ 0.33, PASS.

**AMENDMENT — R1, ruled 2026-09-16 (option (a)).** v2 passed on synthetic
features and **failed live** (linear 0.679); so did every design after it —
seven in all, each stated before its run, each on disk
(`runs/e20/positive_control_{synthetic,live}_v*`): on the live MLP-MNIST
features every target built to be non-linear was linearly decoded at
0.67–0.71, while the MLP read +0.10 above linear every time. That is a fact
about the features, recorded, not a defect in the design: the live half of
this control **cannot be built on these features**. Ruling:

- the **recipe** is certified on the anisotropic synthetic shaped like the
  features (v7: linear 0.51, MLP 0.97, three seeds, converged), on disk;
- the live half is replaced by a **non-vacuous gate**, applied on every cell
  and both eras before the row is read: `acc_refit^mlp ≥ acc_refit^linear −
  spread^mlp`. A broken (underfit) MLP probe fails it; a working one cannot,
  since the MLP contains the linear solution. (`scripts/e20_row.py`, "R1 gate".)
- the seven designs go to the paper's appendix as the record of what was
  tried, with the linear numbers, and the observation that on these features
  "nonlinear" targets are linearly separable enters §4 as a property of the
  feature geometry — mechanism unclaimed.

The MLP-probe decompositions computed after the failed live control are
**not the cited ones**: under R2 they are re-run seeded, and the row is read
from the seeded artifacts under the amended gate.

**R3, ruled 2026-09-16 — the gate's tolerance.** The R1 gate as first written
used the MLP's init spread alone as its tolerance; on the primary it failed
5/24 with shortfalls of 0.1–0.6pp on cells at 0.93–0.98 — all inside the test
set's own 1.96·SE (0.6–1.1pp at n = 2000). The init spread measures the
probe's variance and omits the test set's, a real component the tolerance
should have carried. **Tolerance = `max(init spread, 1.96·SE_binomial(p_linear,
n_test))`**, stated before it was applied. A correction rather than a
relaxation: it rescues only the noise-level failures — the primary passes
24/24, the secondary still fails 2/24 on shortfalls of 6.9pp and 6.6pp, and
stays withheld. Outcomes: **primary TIGHT** (ΔF_enc −0.0073; share 82.2 →
84.0, the favourable direction); **secondary withheld** — the MLP probe
overfits unbounded LSTM cell states (train 1.000, test 7pp below linear;
standardization confirmed on train statistics in both probes), which becomes
the paper's stated reason for linear probes.

---

## 5. Regression — the amendment must not spend E17's proof

`decompose(probe="linear")` re-run on the primary checkpoints must reproduce
`runs/e16_decomp/mnist_mlp.json` **row for row to 1e-9 on every key of the
cited artifact** (the extraction is unchanged and lbfgs is deterministic).
One key is **new beside them and stated**: `probe_gap` (the train/test gap of
each fit, the overfit witness of §3) — additive, no existing key's value
changes. `runs/e20/regression.json` records the max deviation; a nonzero one
stops the experiment.

**IT STOPPED, and the cause is a finding (R2, ruled 2026-09-16).** The
re-run reproduced `acc_orig`, `acc_ceiling` and `d_logit` to **0.00e+00** —
the reload and the deployed-path extraction are exact — and moved the refit
fields by up to **1.6e-2** per cell (`acc_refit_t4`; 4e-3 on the ceiling
refit; 0.12pp pooled `F_enc`). `channel_decomp.load_task_data` takes the
first `N_TRAIN` samples of a `shuffle=True` loader and nothing seeded the
draw: **the probe's training subset was an unrecorded configuration axis**,
in every decomposition and screen artifact in the ledger. Fixed as
`probe_subset_seed` (opt-in, recorded per artifact; default `None` keeps the
pre-fix path so no old artifact is silently re-drawn). Ruling (b): the S72
screen, H-X2, ρ, the E20-B band and both E20-A arms are **re-run seeded and
those are the citable values**; the unseeded artifacts stay on disk marked
"measured under an unrecorded draw", with the effect above as their floor.
The regression's role is rewritten accordingly (`e20_row.py --regress`): the
reload/extraction fields must be identical to 1e-9 (E17's proof, intact);
the refit fields' delta is *reported* as the draw effect, never passed.

---

## 6. CLAUSE → JOB → ARTIFACT (catch 33)

| clause | script / job | artifact |
|---|---|---|
| §2 MLP probe in the shared loop | `scripts/channel_decomp.py` (`refit_probe_mlp`, `decompose(..., probe, probe_seeds)`) | — |
| §5 regression, primary | `modal run …::analysis --argv "scripts/e16_decompose.py --arm mlp --bench mnist --probe linear --out /runs/e20/mnist_mlp_linear.json"` + `scripts/e20_row.py --regress` | `runs/e20/mnist_mlp_linear.json`, `runs/e20/regression.json` |
| §4 positive control, synthetic | `scripts/e20_positive_control.py --synthetic` (local) | `runs/e20/positive_control_synthetic.json` |
| §4 positive control, live | `scripts/e20_positive_control.py --live --arm mlp` (Modal, before the live read) | `runs/e20/positive_control_live.json` |
| §3 primary, MLP probe | `… e16_decompose.py --arm mlp --bench mnist --probe mlp --probe-seeds 0,1,2 --out /runs/e20/mnist_mlp_mlpprobe.json` | `runs/e20/mnist_mlp_mlpprobe.json` |
| §3 secondary, both probes, on the S72 arm | `… e16_decompose.py --arm s72_off --bench har_subject --probe {linear,mlp} …` (new ARMS entry, new bench choice: `HARSubjectBenchmark`, x86 partition gate) | `runs/e20/har_s72off_{linear,mlpprobe}.json` |
| §3 row: paired `ΔF_enc`, floors, gates, overfit flags, verdicts from printed arrays | `scripts/e20_row.py` | `runs/e20_row.json` |
| §2 alpha sweep (only if branch 3's flag fires) | `… --probe mlp --alpha-sweep` | `runs/e20/alpha_sweep.json` |

Every clause has a path. `--probe` defaults to `linear`; the flag-off path is
the regression of §5.

---

## 7. Predictions on record (series: 15 scored, 3 misses)

| prediction | odds |
|---|---|
| primary: `|ΔF_enc| ≤ 0.01` pooled | **~45%** |
| primary: `ΔF_enc < −0.01` (MLP recovers more; share rises) | **~40%** — the MLP-MNIST arm has the most encoder-side room in the program (0.0707) |
| primary: `ΔF_enc > +0.01` | **~15%** |
| secondary (HAR S72) agrees with the primary in the sign of `ΔF_enc` | **~60%** |
| overfit flag fires on ≥ 1 cell of the primary (`gap^mlp − gap^linear > 0.05`) | **~35%** — 4000 train samples, 136k probe parameters |
| MLP probe floor (init spread) exceeds 0.01 on ≥ 1 cell | **~50%** |
| `|R^mlp|` pooled exceeds 0.05 on either arm (reading withheld) | **~15%** |

By-product, not a prediction: the S72 HAR-OFF arm's **linear** decomposition
— reader share on the bridging benchmark's citable arm — does not exist yet
and is produced here by the same script. It enters the ledger as a new row
under its own name, not as a revision of the `har` (shared-window) share.

## 8. E20-B — the 66–88% band under uniform arms (ruled 2026-09-16)

**The premise, from artifacts.** The band's four arms are E5-family checkpoints
on `--benchmark har` (shared-window): `ckpt_e5_off_seed*` (OFF), `ckpt_e5_seed*`
(v1-ON), `ckpt_e5d_v2on_seed*`, `ckpt_e5d_v2ctl_seed*` (`docs/E6B_prereg.md`
§ audit table; `runs/e11_e6b/decomp.json`). E11-era: no `arm` field, era OFF.
Under Ruling A the OFF member's citable version is the S72 relaunch, which is
on **`har_subject`** — a different construction as well as a different
provenance.

**Why the band cannot be captioned as mixed.** A recorded, era-ON OFF share
already exists on the *same* `har` construction: `runs/e16_decomp/har_mafc_off.json`
reads **80.4%** (F_enc 0.076, F_read 0.310) against the E11-era **66.2%** for
the same arm. Provenance alone moves the band's floor by 14pp. A mixed band
would carry that as an artifact inside its own range.

**Ruling.** All four arms move together, on `har_subject`, under S72's flag set
(`--era-checkpoints --fp32-shadow`, 4 threads): OFF is done (`e10off_ec`);
v1-ON, v2-ON and v2-control relaunch with the flags `jobs_e5diag` / `jobs_e5d`
recorded (`--warmup-epochs 1 --warmup-mode {adapter,control}` for v2), 3 seeds
+ a floor pair each — `jobs_s72_band`, 15 runs. The linear decomposition runs
on all four through the same driver; `use_input_adapters` is **read from each
arm's artifact**, never assumed (the ON arms decompose with their adapter on
the deployed path). The E11-era band on `har` stays in the ledger as the
measurement it was — superseded by provenance *and* construction, not
withdrawn.

**What is known before the run, stated so it is not "predicted":** the S72 OFF
share is implied by the screen's own fields (`acc_refit` 0.7723, `acc_orig`
0.4665, `acc_ceiling + R`): F_read ≈ 0.306, F_enc ≈ 0.087, share ≈ **78%**. The
driver's number must agree with that to 1e-6 or the two scripts do not
describe one run.

**Predictions, filed before the band launches:**

| prediction | odds |
|---|---|
| all three ON arms' shares fall inside 66–88 on `har_subject` | **~55%** |
| the band's width on `har_subject` is narrower than 22pp | **~50%** |
| the E11-era ordering OFF < v1-ON < v2-ON < v2-control is preserved | **~40%** — two of the three gaps were < 2pp on `har` |
| v2-control ≥ 85% (the shared-window value was 87.9%) | **~45%** |
| every ON-arm floor pair is bit-identical (25/25), as OFF and LwF λ = 1.0 were | **~60%** — adapters add a path; determinism is per-config |

**Ledger consequence, pre-committed.** The one-liner's "66–88% across four arms
of a scratch-trained LSTM on UCI HAR" is replaced by the `har_subject` band
whatever it reads, with the construction named; the `har` band is cited beside
it as superseded. If an ON arm's floor pair is not bit-identical, its share is
read against the measured share floor (MEMO_e17's lesson: floors attach to
quantities), never against the AVG floor.

**CLAUSE → JOB → ARTIFACT, E20-B:**

| clause | job | artifact |
|---|---|---|
| v1-ON / v2-ON / v2-ctl, 3 seeds + floor pair each | `jobs_s72_band` (`s72band`) | `runs/e10{v1on,v2on,v2ctl}_ec_seed{s}/`, `…_floor4tec_{a,b}/`, `ckpt_…` + `_fp32` |
| checkpoint audit | `scripts/audit_checkpoints.py` | `runs/e20/band_ckpt_audit.json` |
| linear decomposition, four arms | `e16_decompose.py --arm s72_{off,v1on,v2on,v2ctl} --bench har_subject --probe linear` | `runs/e20/har_s72{off,v1on,v2on,v2ctl}_linear.json` |
| floor pairs decomposed (share floor per arm) | same driver, `--arm …_floor` entries added when the runs exist | `runs/e20/har_s72{…}_floor_linear.json` |
| the band | `scripts/e20_band.py` (shares, share floors, ordering, the two constructions side by side) | `runs/e20_band.json` |

## 9. Compute

Two Modal CPU analysis jobs per arm (linear regression + MLP probe), each
minutes: 12 cells × 2 eras × 3 init seeds = 72 lbfgs MLP fits per arm on
4000 × 256 features. Under 0.5 CPU-hour total. Local: the synthetic control,
seconds. **E20-B:** 15 `har_subject` runs at ~5 min each ≈ 1.3 CPU-hours, one
batch (~15 min wall), plus four linear decompositions (minutes).
