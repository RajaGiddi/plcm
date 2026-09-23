# Pre-Registration: E7 — Per-Task Heads on HAR (the Reader-Channel Cure)

**Status:** Draft for sign-off. Drafted **before any training run**, as branch (A)
of `docs/E6B_prereg.md` requires.

**What licenses this experiment.** E6b's decomposition, not any MNIST contrast:

| channel | HAR share of forgetting |
|---|---|
| C-read (reader-walk) | **75–90%** (even charging the full instrument bias against it) |
| C-enc (encoder damage) | 0.042–0.076 absolute, ~10–25% |
| C-adapter | **~zero** (E5d: adapters never left identity; E6: ON/OFF in-S drift ratio 0.970) |

A **frozen per-task head cannot walk.** Walking is what 80–90% of HAR's forgetting
turns out to be. That is the whole hypothesis.

---

## 1. The sharpest form of the prediction, stated up front

If `F_read` is **causal** rather than merely correlated, freezing the reader should
collapse HAR's forgetting toward `F_enc`'s **0.042–0.076** — which would put HAR
near MNIST-ON territory (total 0.0686) **with the adapters contributing nothing**.

> **E7's pass would mean HAR never needed input adapters — it needed a reader that
> holds still.**

The E5 arc would then resolve as an **architecture confound inherited from the MAFC
configuration**, not as a property of sensor data, a modality boundary, or shift
dimensionality. The paper's framework becomes the three-channel accounting with each
channel's cure demonstrated.

This is the strongest claim the design can support, so it is the one that gets
falsified if the numbers disagree.

## 2. Premise provenance (catch 21) — verified, not remembered

| premise | established by | value |
|---|---|---|
| every prior arm used a **shared** classifier | `config.model.use_task_heads` in every E6b checkpoint | **False** |
| the config default | `configs/mafc_phase1.yaml` → `model.use_task_heads` | **False** |
| **what forces it** | `scripts/train.py:391` — `if build_type == "mafc": config["model"]["use_task_heads"] = False` | unconditional |
| per-task heads are created when enabled | `src/models/plcm.py:292` — `if self.use_task_heads and task_key not in self.task_classifiers` | lazily, per task |
| frozen after their task | `plcm.py:259-261` — outgoing head's params `requires_grad=False` | yes |

**Consequence for implementation:** `train.py:391` unconditionally disables per-task
heads for `mafc`. E7 requires a **new arm flag** (`--task-heads`) that overrides it.
The original suppression exists because MAFC's bank-anchored readout term is vacuous
under frozen heads (MAFC prereg §1) — E7 runs at `--mafc-arm lambda0` (λ=0), where
that term is off, so the two are **not in conflict**. This must be verified in the
produced checkpoints before any hypothesis is read (§6).

## 3. Arms and design

HAR, frozen shift config `3de66e205eb7`, 5 tasks × 10 epochs, seeds {42, 1337, 2024}.

| arm | adapters | per-task heads | purpose |
|---|---|---|---|
| **E7-heads** | off | **on** | the cure, isolated |
| **E7-heads+adapt** | on | **on** | does the adapter add anything once the reader is fixed? |
| v1-ON (existing) | on | off | E5/E6b baseline |
| OFF (existing) | off | off | E5/E6b baseline |

**6 new runs.** Checkpoints on for every arm (H-C3′ re-check and the per-task
decomposition are unmeasurable without them).

*Why `E7-heads` has adapters OFF:* §1's claim is that the reader, not the input
path, is the operative channel. Testing the cure **without** adapters is the
version that can embarrass us — if forgetting collapses there, the adapters are
shown to be unnecessary on HAR rather than merely insufficient.

## 4. Hypotheses

- **H-E1 (the cure works):** E7-heads forgetting **≤ 0.15**.
  *Derivation:* E6b puts HAR's encoder channel at 0.042–0.076; 0.15 is roughly
  2× the top of that range, allowing for the reader channel not vanishing entirely
  (per-task heads still share an encoder and an output gate). Against v1-ON's 0.392
  and OFF's 0.374 this is a ~60% reduction.
- **H-E2 (`F_read` collapses):** E7-heads `F_read` **≤ 0.10**, versus OFF's 0.2939.
  This is the mechanism check — H-E1 could pass for the wrong reason (e.g. per-task
  heads acting as extra capacity), and H-E2 is what distinguishes cure from
  coincidence.
- **H-E3 (adapters add nothing once the reader is fixed):** |E7-heads+adapt −
  E7-heads| AVG **< 5pp**. *If this passes, §1's sharpest claim stands.* If the
  adapter arm is materially better, the channels interact and the framework needs
  the interaction term.
- **H-E4 (deposition/erosion, from E6b's reconciliation):** per-task `F_read` under
  frozen heads is **flat in tasks-since-training** (max−min across T0–T3 ≤ 0.10),
  versus the monotone decline the shared-reader arms show (HAR/OFF: 0.311, 0.308,
  0.352 → 0.205). This is E6b's accumulation prediction, tested directly.

**Instrument re-check (mandatory, not a hypothesis):** **H-C3′ re-run at the same
bar, |R| ≤ 0.05, per dataset and reported PER SEED.** E6b's HAR instrument passed
pooled at +0.0496 while 4 of 12 cells exceeded the bar — a marginal instrument
re-earns its standing at every use. **If HAR's pooled R exceeds 0.05, the per-task
decomposition is not read; H-E1/H-E3 (which need only accuracies) still are.**

## 5. Branches (ties resolve downward)

| branch | condition | reading |
|---|---|---|
| **(A)** | H-E1 ∧ H-E2 ∧ H-E3 | **Reader-walk is causal and per-task heads cure it.** HAR never needed input adapters. E5's null resolves as an architecture confound. Framework = three channels, each with a demonstrated cure. |
| **(B)** | H-E1 ∧ H-E2 ∧ ¬H-E3 | Cure works but adapters still add value → the channels interact; framework gains an interaction term; both cures reported jointly. |
| **(C)** | H-E1 ∧ ¬H-E2 | Forgetting drops but **not** via the reader channel — per-task heads worked as capacity, not as walk-prevention. `F_read`'s causal status is **unestablished**; report as such. |
| **(D)** | ¬H-E1 | Freezing the reader does **not** cure HAR despite the reader carrying 80–90% of measured forgetting. `F_read` is **correlational**, the decomposition measures something other than a repairable channel, and E6b's framework claim is softened at full prominence. |

**Any branch:** H-E4 reported regardless — it tests E6b's accumulation account
independent of whether the cure works.

## 6. Locks

- **Premise verification before any hypothesis is read:** confirm produced
  checkpoints contain `task_classifiers.{0..4}` and `use_task_heads=True`. The
  premise this experiment turns on is the one catch 21 says must be cited.
- Frozen shift config fingerprint checked at launch (`3de66e205eb7`).
- **Reproduction floor measured before any delta is interpreted** — relaunch one
  arm-pair, **both arms** (catch 19). HAR's CPU path has shown same-config spread
  to 5.5pp historically, though the recent OFF regeneration reproduced
  bit-identically on all three seeds.
- Adapter arm uses the per-step 9×9 geometry (`adapter_mode: per_step`), unchanged.
- n=3; **no claim at n=1**; mechanical verification per the standing definition.
- All numbers emitted by the analysis scripts; none transcribed.
- Timebox: **one day.** Blocked → report state, stop.

## 7. Predictions on record

**Updated at launch:** H-E1 (heads-arm forgetting collapses toward the F_enc
floor) ~65% · H-E2 ~60% · H-E3 (heads beat shared by >= +15pp AVG) ~60% ·
H-E4 (flat per-task F_read under frozen heads) ~70% — *the one that converts the
deposition/erosion reconciliation from narrative into measurement* ·
**branch (D) ~15%** — priced low because the v2-control natural experiment and the
decomposition agree, but **not lower**, because this project's mechanism
predictions have missed at 75% twice.

*Calibration:* mechanism predictions in this project run **directionally right,
magnitude wrong**; the reference misses are E5's 75% on the retention bar and
E5b's ~75% on branch (B). H-E3 is priced lowest because it is the claim that would
most rewrite the program's story, and this project's biggest-sounding claims have
been the ones that broke.

**Named risk:** per-task heads at eval require task identity, i.e. the
**task-incremental** setting. The adapter result already carried this assumption
(`task_hint` selects the adapter), so E7 does not weaken the setting further — but
if E7 becomes the headline cure, the task-free story weakens correspondingly and
the paper must say so rather than let the reader assume otherwise.

## 8. Two notes the memo must carry (ruled in at launch)

**(a) The margins read the way they do because we accidentally hardened our own
instrument.** E6b's counterfactual showed lbfgs produces a *larger* `R` than Adam
on identical features (+0.0602 vs +0.0432 on HAR/OFF) — a closer-to-optimal fit has
a bigger advantage over the deployed head, so `R = optimal − deployed` grows. The
solver amendment therefore made H-C3′ **stricter**, and HAR's marginal passes are
partly an artifact of that. **This is the good direction to be wrong in**, and
saying so is what makes the margins legible rather than alarming.

**(b) The bias-charged bound is the PAPER number.** Reader share is reported as
**75.5–88.2%** — computed after charging the *entire* instrument bias against
`F_read` — in the prose. The uncorrected 79.5–89.7% goes in the appendix table.
Same pattern as printing both E4 bars: the reader sees the worst-case arithmetic
and the conclusion survives it.

**Pre-named ledger edit, if branch (A) fires:** the one-liner's HAR clause becomes
*"...and where the input path fails to engage, freezing the reader — not adapting
the input — removes the forgetting."* The E5 arc then resolves as: **HAR never
needed adapters; it needed a reader that holds still.**

## 9. Reporting

Memo: forgetting per arm, the E6b decomposition re-run on E7 checkpoints
(`F_total`/`F_enc`/`F_read`/`R`, per seed), H-E4's per-task table, branch taken,
the measured reproduction floor, and the claim-ledger one-liner edited **first**.
