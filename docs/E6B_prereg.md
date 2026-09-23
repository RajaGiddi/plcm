# Pre-Registration: E6b — Three-Channel Forgetting Decomposition

**Status:** **v2, locked.** Six review catches applied (§8). Analysis-only, and
the premise is audited this time as well as the artifacts. One afternoon.

**Question:** What carries HAR's forgetting? E6 measured encoder in-S drift as real
(absorption-fingerprinted, H-B3′ 2.85×, ρ = 1.000) but **modest** (5.4% of drift
energy in a subspace holding 7.8% of representation energy), while classifier
rotation is large everywhere (39.7–76.5°) and largest exactly where forgetting is
worst (v2-control: encoder in-S drift −29%, reader rotation 76.5°, forgetting
0.4745 — worst of all four arms). E6b converts that correlation into a
decomposition.

**The channels, named:**
- **C-enc** — encoder-side damage: old-task representations at θ₄ are less
  linearly separable than at their own era.
- **C-read** — reader-walk: representations stay separable, but the classifier no
  longer points at them.
- **C-adapter** — the harmless channel (absorption by per-task input maps);
  ~zero on HAR per E5d/E6, operative on MNIST.

---

## 1. Method (recipe-controlled — see §8 catch 2)

Per arm × seed × old task k ∈ {0..3}, four quantities:

| quantity | encoder | head |
|---|---|---|
| `acc_ceiling(k)` | θ_{k+1} (end of task k) | deployed |
| `acc_refit_ceiling(k)` | θ_{k+1} | **fresh refit** |
| `acc_refit_θ4(k)` | θ₄ | **fresh refit** |
| `acc_orig(k)` | θ₄ | deployed |

**Decomposition:**

    R(k)      = acc_refit_ceiling - acc_ceiling      # the INSTRUMENT'S OWN BIAS
    F_enc(k)  = acc_refit_ceiling - acc_refit_θ4     # encoder effect, recipe held constant
    F_read(k) = acc_refit_θ4      - acc_orig         # reader effect, encoder held constant
    F_total(k)= acc_ceiling       - acc_orig         # deployed forgetting

    IDENTITY:  F_enc + F_read - R = F_total          # exact; printed as a column

**Why the fourth quantity exists.** v1 computed `F_enc = acc_ceiling − acc_refit`,
which moves the encoder *and* the head recipe at once. A head fit to convergence on
the full train set generally beats an online-trained head, so `acc_refit` is
inflated → `F_enc` understated → **`F_read` overstated** — biasing toward the
hypotheses. `acc_refit_ceiling` holds the recipe constant and turns the bias into a
measured quantity, `R`.

**Negative terms are permitted and interpretable.** `F_enc < 0` means θ₄'s encoder
is *better* for task k than its own era's — plausible under continued training on
related data. Reported as the fact it is, not clipped.

**Refit recipe (identical in every cell, no per-cell tuning).** Fresh
`nn.Linear(hidden, n_classes)`, Adam, **lr 1e-3, 5 epochs**, batch 256, seeded
`RECIPE_SEED`; fit on task-k **TRAIN** representations, evaluated on task-k
**TEST**. Inherited from `scripts/brittleness.py:probe_accuracy`, which validated
it, with the class count parameterized (HAR 6, MNIST 10).

**Representation** = final-layer LSTM cell state `c_n[-1]`, the same convention
E4b/E6/brittleness use. Everything after the LSTM is reader-side by definition, so
the refit head replaces the output gate *and* the classifier.

**Arms:** HAR {v1-ON, OFF, v2-ON, v2-control} + MNIST {ON, OFF}.

## 2. Audit — artifacts AND premises (§8 catches 1, 6)

### 2a. Checkpoint inventory (all task boundaries, not just θ₄)

| arm | θ₄ | θ₁..θ₄ boundaries | files/run | status |
|---|---|---|---|---|
| HAR v1-ON (`runs/ckpt_e5_seed*`) | 3/3 | 3/3 | 50 | ✅ |
| HAR OFF (`runs/ckpt_e5_off_seed*`) | 3/3 | 3/3 | 50 | ✅ bit-identical to E5 originals |
| HAR v2-ON (`runs/ckpt_e5d_v2on_seed*`) | 3/3 | 3/3 | 50 | ✅ |
| HAR v2-control (`runs/ckpt_e5d_v2ctl_seed*`) | 3/3 | 3/3 | 50 | ✅ |
| MNIST ON (`fullrank_ref` 42, `e4_on_*`) | 3/3 | 3/3 | 50 | ✅ |
| MNIST OFF (`e4_off_*`) | 3/3 | 3/3 | 50 | ✅ |

### 2b. Architectural premises — cited, not remembered

| premise | established by | value |
|---|---|---|
| every arm uses a **shared** classifier | `config.model.use_task_heads` in each θ₄ checkpoint | **False** (no per-task heads in any arm) |
| ON arms carry per-task adapters 0–4 | `task_adapters.*` keys in state dict | present |
| OFF arms carry none | same | absent |
| HAR readout rank | `classifier.weight` shape | 6×256 |
| MNIST readout rank | same | 10×256 |

**v1's H-C3/H-C4 assumed MNIST ON used per-task frozen heads. It does not — no arm
ever has.** Both hypotheses are struck; see §8 catch 1.

## 3. Hypotheses

- **H-C1 (reader channel is real on HAR):** `F_read ≥ 0.10` absolute on HAR OFF
  (mean over tasks, n=3). *Derivation:* total forgetting is ~0.37–0.42, and E6's
  5.4% in-S drift energy makes it implausible that C-enc alone carries it; 0.10 is
  ~25% of total — too large to footnote.
- **H-C2 (reader dominates in the natural-experiment cell):** on **v2-control**,
  `F_read > F_enc` **AND** `F_read ≥ 0.10`. The compound bar matters: a channel
  that "dominates" a *negative* competitor has not shown it carries forgetting.
  v2-control is the cell whose encoder drifted least (in-S 0.3152, −29% vs OFF) and
  whose reader rotated most (76.5°); if reader-walk does not dominate **here**, the
  channel story is wrong.
- **H-C3′ (instrument control):** `|R| ≤ 0.05`, evaluated **per dataset**.
  *This replaces v1's H-C3 and is strictly better:* it measures refit-advantage
  **directly** rather than inferring it through an architectural argument, needs no
  premise about heads, and makes branch (D) fire on the instrument's actual bias
  instead of physics-vs-artifact guesswork.

*(H-C4 struck. Its honest residue, for the memo's discussion: **per-task heads
remain an untested cure candidate for the reader channel — motivated by the channel
decomposition itself, not by any MNIST contrast, which does not exist.**)*

## 4. Branches

| branch | condition | reading |
|---|---|---|
| **(A)** | H-C1 ∧ H-C2 ∧ H-C3′ | Three-channel accounting confirmed and sized. Framework claim: forgetting = where the shift and the drift get absorbed — adapter (harmless, high-dim shifts), encoder in-S (fingerprinted, modest on HAR), reader (large on shared-classifier architectures). **E7 = HAR with per-task heads**, its design citing *the decomposition*. One-liner gains its mechanism clause. |
| **(B)** | H-C1 ∧ ¬H-C2 | Reader channel real but not dominant even in the natural-experiment cell. Report the split; a cure must address both; **no single-mechanism E7.** |
| **(C)** | ¬H-C1 | Reader-walk is angle without consequence (rotation inside the classifier's tolerance). HAR's forgetting is encoder-side after all despite 5.4% energy — meaning **small in-S drift is disproportionately damaging**, itself a finding. E6's within-S-rotation candidate is promoted for the next diagnostic. |
| **(D)** | ¬H-C3′ | Instrument confounded (refit advantage ≠ reader-walk). **No channel claims.** Diagnose the refit protocol before any reuse. |

## 5. Predictions on record (post-correction)

H-C1 ~75% (unchanged — never rested on the false premise) ·
H-C2 ~60% (down from 70: the absolute floor is a real bar and negative-`F_enc`
trivial passes are excluded) ·
H-C3′ ~70% (**least certain**: 5 epochs of converged fitting against an online head
is exactly where a few points could live) ·
joint (A) ~40%.

*Calibration:* mechanism predictions here run directionally right, magnitude wrong;
two 75% misses (E5's retention bar, E5b's branch-(B)) are the reference.

## 6. Locks

- **Analysis only.** §2's audit is the complete input inventory — artifacts and
  premises both.
- **Deployed-pathway lock (equal prominence to the read order):** on ON arms the
  refit *and* its evaluation apply **task-k's own adapter**, exactly as deployment
  does. Otherwise `F_read` silently absorbs an input-pathway swap instead of
  measuring reader-walk.
- Refit recipe identical in every cell; stated in the script header and the memo.
- **Read order — hard stop:** identity column (`F_enc + F_read − R = F_total` to
  numerical precision) → **H-C3′ per dataset** → HAR hypotheses.
  **If H-C3′ fails, STOP; no HAR reading is interpreted.**
- Per-task tables for every arm, not just means; **task 3** (E6's damage-dominant
  transition) gets its own row.
- Numbers reach the memo only via the script's recorded output — no transcription
  (the rule E4b's memo violated).
- Timebox: one afternoon. Blocked → report state, stop.

## 7. Reporting

Memo: decomposition table (`F_total`/`F_enc`/`F_read`/`R` per arm per dataset, with
the identity column), per-task breakdown, branch taken, the H-C4 residue sentence,
**claim-ledger one-liner edited first**, and — only on (A) — the E7 contract drafted
before any training run.

## 8. Review log — six catches

1. **BLOCKING — H-C3/H-C4 assumed per-task frozen heads that exist in no arm.**
   Verified: `use_task_heads=False` everywhere. Branch (D) would have discarded a
   working instrument on a false premise. → both struck; §2b premise audit added;
   promoted to standing rule (catch 21).
2. **BLOCKING — `F_enc` conflated encoder degradation with the head-recipe change**,
   biasing toward the hypotheses. → `acc_refit_ceiling` added; `R` measured.
3. **Negative terms could make H-C2 vacuous.** → compound bar (`F_read > F_enc`
   AND `F_read ≥ 0.10`).
4. **Refit had to use the deployed input pathway.** → adapter lock in §6.
5. **Inherited recipe hardcoded 10 classes.** → parameterized; recipe recorded.
6. **Audit covered θ₄ only**; the corrected protocol needs every task boundary. →
   §2a widened to all 50 files/run.

## 9. Amendment v3 — instrument repair (post branch-(D), approved before rerun)

**Branch (D) fired on first execution.** H-C3′ measured `R = −0.6498` (HAR) and
`−0.4280` (MNIST) against a ±0.05 bar. Identity held exactly (0.00e+00), so the
failure was the instrument, not the arithmetic. Per §6 the run STOPPED and no HAR
reading was interpreted.

**Diagnosis (branch (D)'s required step, before any reuse).** The confound ran
*opposite* to the anticipated direction — the refit **under**-performed the
deployed head rather than beating it. On one checkpoint:

| fit | accuracy |
|---|---|
| deployed classifier | **0.8555** |
| refit on `o_t·tanh(c_t)` (the deployed feature) | 0.3505 |
| refit on `tanh(c_t)` | 0.3465 |
| refit on raw `c_t` (v2's choice) | 0.2370 |
| refit on standardized `c_t` | 0.2130 |
| raw `c_t`, 20 / 100 epochs | 0.2605 / **0.1840** |

Two stacked confounds: **(i)** the SGD recipe underfit by ~50pp and *diverged*
with more epochs on unbounded features (max |c| = 75.3); **(ii)** v2 refit on raw
`c_t` while the deployed head reads `o_t·tanh(c_t)` — a different function class,
so `F_read` absorbed a representation change as well.

**Two locks amended:**

1. **Solver lock.** Multinomial logistic regression, lbfgs, fit to convergence
   (`max_iter` 5000, convergence checked and warned on), features standardized on
   TRAIN statistics. This removes lr, epochs, batch size and seed from the
   protocol, so `R` measures the genuine **deployed-vs-optimal** gap — the
   quantity H-C3′ was written to bound.
2. **Feature lock.** The refit operates on the **exact deployed feature**
   (`o_t·tanh(c_t)`), through the deployed pathway including task-k's adapter on
   ON arms. **The refit must differ from the deployed head in fitting procedure
   only.**

**Predictions on the amended instrument:** `R ∈ [0, +0.03]` (slightly positive —
the direction the original design assumed) ~70%; H-C3′ passes ~75%. H-C1 ~75%,
H-C2 ~60%, joint (A) ~40% carry over. Read order unchanged.

**Lesson promoted to standing rules (catch 21, final form):** *"validated" is
context-bound — a recipe's certificate names the features, scale, and question it
was validated for, and transfers to none other without re-verification.*
