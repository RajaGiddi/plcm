# PLCM — Persistent Latent Cell Memory

## What this project is

A research implementation of PLCM: an LSTM augmented with a persistent external memory
of cell state vectors that survives across tasks, enabling continual learning without
catastrophic forgetting. The core idea is that cell states live in a "thought space" and
are composed using a Gated Geometric Composition (GGC) operator instead of additive
blending, which would wash out information.

## Architecture overview

```
Input xₜ → LSTM → cₜ, hₜ
  → Read Controller queries Memory Bank M with context embedding
  → Retrieves top-k relevant past cell states c̃
  → GGC composes: cₜ' = γ⊙(cₜ⊙μ) + (1-γ)⊙ι
  → Output uses cₜ' instead of cₜ
  → Write Controller scores cₜ' for storage into M
  → Consolidation runs periodically (clustering, eviction)
```

## Key math — Gated Geometric Composition (GGC)

```
γ = σ(Wᵧ · [cₜ; c̃; cₜ⊙c̃] + bᵧ)     # composition gate
μ = σ(Wμ · c̃ + bμ)                      # modulation signal
ι = tanh(Wι · [cₜ; c̃] + bι)            # injection signal
cₜ' = γ ⊙ (cₜ ⊙ μ) + (1-γ) ⊙ ι        # composed state
```

Three pathways: modulation (multiplicative reshaping), injection (new info),
preservation (gate decides per-dimension).

## Project structure

```
src/models/composition.py   ← GGC operator (and future Möbius variant)
src/models/memory_bank.py   ← Key-value memory store with capacity management
src/models/controllers.py   ← Read (attention retrieval) and Write (importance scoring)
src/models/consolidation.py ← Periodic memory compression/eviction
src/models/plcm.py          ← Full PLCM model combining all components
src/models/lstm_base.py     ← Vanilla LSTM baseline for comparison
src/training/trainer.py     ← Continual learning training loop
src/training/ewc.py         ← Elastic Weight Consolidation baseline
src/training/metrics.py     ← Forgetting, forward/backward transfer metrics
src/data/permuted_mnist.py  ← Permuted MNIST benchmark (5 sequential tasks)
src/data/split_cifar.py     ← Split CIFAR-10 benchmark
scripts/train.py            ← Main entry point
scripts/evaluate.py         ← Post-training evaluation
scripts/ablation.py         ← Ablation studies (toggle components)
tests/                      ← Unit tests for math correctness
```

## Conventions

- Python 3.10+, PyTorch 2.x
- Type hints on all function signatures
- Docstrings on all public classes and methods
- Config via YAML files in configs/
- All tensor shapes documented as comments: `# [batch, hidden]`
- No magic numbers — constants live in config or as class attributes
- Device-agnostic: `.to(device)` everywhere, never hardcode cuda/cpu

## Running

```bash
# Train PLCM on permuted MNIST
python scripts/train.py --config configs/default.yaml --model plcm

# Compare all baselines
python scripts/train.py --config configs/default.yaml --model all

# Run ablation (toggle GGC vs additive, read/write controllers)
python scripts/ablation.py --config configs/default.yaml

# Tests
pytest tests/ -v
```

## Phase plan

1. **Phase 1 (current)**: Euclidean thought space + GGC composition
2. **Phase 2 (next)**: Hyperbolic thought space + Möbius composition
3. **Phase 3**: Full research — scaling, theoretical analysis, paper

## Key design decisions to remember

- Memory Bank has FIXED capacity with importance-weighted eviction (not unbounded growth)
- Read Controller uses multi-head attention over memory keys
- Write Controller uses gradient magnitude + gate activation as importance signal
- Consolidation merges similar vectors via k-means periodically
- GGC was chosen over additive blending because additive converges to centroid
- The composition gate receives cₜ⊙c̃ (Hadamard interaction) as explicit input

## Design principle — adapter shape

- **Match the adapter to the space where the shift lives.** Visible only once a
  second modality forced the choice. Permuted MNIST's permutation mixes pixels
  across rows, so the adapter must span the flattened image (784×784). UCI HAR's
  hardware-revision shifts are 9×9 channel maps applied uniformly over time, so a
  per-timestep 9×9 adapter represents them **exactly** — 81 params/task, 324 bytes
  at fp32. A flattened HAR variant would be 1152×1152 = 1.33M params with **no
  additional expressivity for that shift class** (16,000× larger; footnote, not
  experiment). Consequence for the paper: the `O(T·d²)` storage objection is
  **regime-dependent, not fundamental** — adapter cost tracks the shift's intrinsic
  dimensionality, not the input's. Implemented as `adapter_mode: "flat" | "per_step"`.

## Standing rules

- **Contracts catch design errors before runs; baselines catch claim errors
  after them. Eleven catches, two mechanisms, neither sufficient alone.**
  Pre-registration caught unsatisfiable hypotheses, reversed KL, and
  contradictory readers — all before compute was spent. But the two most
  consequential corrections (variance-collapse retracted as mechanism evidence;
  adapters cost 10x MORE storage than the buffer they beat) came from running
  the ER *baseline*, not from any contract review. Run both.
- **Nothing enters a results table at n=1.** Every configuration first measured
  at a single seed proved optimistic on seeding — 4/4, by 1.4pp and 3.57pp.
- **A new claim cannot rest on n=1.** If a single-seed run fires a
  claim-adding branch, backfill to n=3 before the claim is written down.
- **Selecting a hyperparameter at n=1 is selecting on noise.** The EWC λ=200
  "optimum" was picked at a single seed from {50,100,150,200,300}; seeding
  revealed ±0.044 spread on that configuration, which is larger than the entire
  spread across λ values. The whole sweep was noise. Any hyperparameter choice
  that feeds a reported comparison must be selected at n>=3, or the selection
  itself is a finding-shaped artifact.
- **n=1 error runs both ways.** Five instances were optimistic (worst −6.2pp,
  EWC), one pessimistic (+3.3pp, low-rank r64). Seeding is not a haircut applied
  to good news; it is a correction of unknown sign.
- **Amendments have blast radii, and every amendment must RE-VERIFY what it did
  not touch.** In E4b, redefining the subspace S from the probe's row space to the
  classifier's row space was correct for H-SUB1/2 — and silently made H-SUB3
  vacuous (`W·P_S = W`, so the check reads 0.000pp by construction and passes
  always). Distinct from every other catch: not a bad decision, but a good
  decision with an unexamined side effect.
  **Final form — an amendment must not SPEND A PROOF an earlier pass bought.**
  E12 needs `task_hint` to select the readout head (E7 proved it never did), a
  correct and necessary change. Made unconditionally it would alter the deployed
  path of *every prior arm* and destroy E11's demonstration that `PLCM.forward`
  reproduces every recorded accuracy matrix **exactly** — the equipment's own
  proof of correctness. So the change ships **opt-in, default off**, with a
  regression assert that the flag-off path is **bit-identical** on an E10
  checkpoint, and the experiment records that it sets the flag on.
  *A fix to enable the next experiment must not spend the proof the last one
  bought.* The component every experiment shares is the one where this rule
  binds hardest.

- **An "analysis-only, runs on existing checkpoints" claim requires a checkpoint
  AUDIT TABLE inside the contract — arm × seed × exists — verified before
  sign-off.** (Catch 20, and the second instance of its own error class.) E4's
  contract claimed existing checkpoints without auditing them and the audit
  found the only OFF-arm candidate differed in **two** variables (catch 15).
  One project later, E6's contract claimed "analysis only; existing checkpoints
  are the complete input" — and the audit found the HAR OFF arm had **no
  checkpoints at all**, because no job ever passed `--no-adapters` together
  with `--save-checkpoints`. Two hypotheses (the ON/OFF `drift_S` ratio and the
  per-task increment correlation) had no input and would have been discovered
  unmeasurable *after* sign-off.
  **The cheap signal again stood in for the expensive one:** "we have
  checkpoints" is not "we have the checkpoints this experiment needs." Write
  the table, run the audit, then sign.
  **Corollary — save checkpoints for BOTH arms whenever an arm might later be
  a comparison baseline.** The marginal cost at run time is disk; the cost
  discovered later is a re-run plus whatever reproduction noise it reintroduces.
- **Any contract clause of the form "arm X has property P" must cite the config
  field or verification output that establishes P.** (Catch 21 — the premise
  analogue of catch 20.) E6b's H-C3/H-C4 were built on "the MNIST winning
  configuration uses per-task frozen heads." **No arm in this program has ever
  had per-task heads**: `--mafc-arm lambda0` sets `use_task_heads=False`, which
  is precisely *why* E4b could define S from the shared classifier and why
  E5d's catch 4 ("no new head exists") was flagged three contracts earlier.
  The false premise was imported from memory into a load-bearing clause **in
  the same document that dutifully carried catch 20's checkpoint-audit table** —
  the audit covered *artifacts*, nothing covered *premises*.
  **What made it dangerous:** H-C3's failure branch was "the instrument is
  confounded, diagnose the refit protocol." A false premise would have fired it
  and discarded a working instrument. Premises get the same provenance standard
  as reported numbers: cite the field, or verify it before sign-off.
  **Final form — "validated" is context-bound.** A recipe's validation
  certificate names the **features, scale, and question** it was validated for,
  and transfers to none other without re-verification. E6b inherited
  `brittleness.py`'s probe ("validated") — Adam, lr 1e-3, 5 epochs — which was
  certified on **bounded 784-dim pixel features for a transfer question**. On
  **unbounded LSTM cell states (max |c| = 75.3)** it underfit by ~50pp and
  *diverged* as epochs rose (0.2370 → 0.1840 from 5 → 100 epochs). "Reuse the
  validated recipe" is itself a provenance claim and gets checked, not
  remembered. Prefer knob-free procedures (a convex solver fit to convergence)
  wherever a measurement's meaning would otherwise depend on optimizer settings.
- **Reproduction noise is a measurable quantity, and no delta below it may be
  reported. Relaunch, don't verify.** (Catch 19.) Relaunching six E5 configs
  identically gave: seed 42 **bit-identical in both arms**, ON/1337 identical,
  OFF/1337 −0.21pp, **ON/2024 +5.50pp, OFF/2024 +5.12pp**. The pipeline is
  reproducible for some seeds and not others.
  **First diagnosis was WRONG and is retained here as the correction.** ON/2024
  had been preempted-and-restarted, so the divergence was attributed to
  preemption and written up as "a preempted run can pass every completeness
  clause and still be wrong." The OFF arm then diverged on the same seed **with
  no preemption reported at all** — killing that explanation. The pattern is
  **seed-correlated, not preemption-correlated**: seed 2024 is chaotic in both
  arms, seed 42 is stable in both. Preemption was a coincidence sitting on top
  of ordinary run-to-run nondeterminism, and a plausible causal story survived
  one round of evidence before the control arrived.
  **Operational rule:** relaunch a subset of every experiment's configs and
  measure the reproduction spread *before* interpreting any delta. E5's arm
  delta was **0.95pp against reproduction noise reaching 5.5pp** — the
  comparison could never have resolved it, and no amount of seeding fixes a
  measurement whose noise floor sits above its effect. (H-G1a's 24pp failure is
  far outside that floor and stands.)
  **Relaunch both arms, never one.** An asymmetric relaunch cleans one side of a
  comparison and leaves the other dirty, biasing the delta in an unknown
  direction — worse than leaving both dirty, because it looks fixed.

- **Every certificate or gate ships with a POSITIVE CONTROL — a case where it
  is known it must fail — demonstrated before its first live use.** (Catch 25;
  third instance of the "cannot fail" class.) E4b's H-SUB3 could not fail by
  *algebra* (`W·P_S = W`, so the check read 0.000pp by construction). Catch 22's
  verdict strings could not disagree with the contract by *construction* (they
  were written to mirror it, not derived from the printed values). E10's P3
  certificate could not fail by *threshold* — 100× a 1e-6 reconstruction error
  is 1e-4, six orders of magnitude below the distance between any two distinct
  windows.
  **A gate that has only ever passed is indistinguishable from a gate that
  cannot fail.** Before trusting one, run it on a known-positive: E10's
  generative-honesty certificate is validated by firing it on E8's shared-window
  construction, where the generator provably reconstructs — it MUST fail there.
  Prefer scale-free criteria (compare against the data's own within-class
  spread) over absolute thresholds, which encode assumptions about scale that
  drift between datasets.
  **Corollary — a certificate's NORMALIZER is a degree of freedom, and its
  invariance to nuisance variation must be checked at design time** (same family
  as catch 23's DOF rule). E10's P3-v1 normalized pseudo→real distance by
  *per-task* within-group spread. The numerator was near-constant across tasks
  (9.25–10.79); the denominator varied **2.8×** (5.27–14.98) purely with how
  tightly each subject group clusters. So the gate scored **group tightness, not
  generative honesty**, and failed 2 of 4 tasks whose pseudo data sat five orders
  of magnitude from reconstruction. Ask of every normalizer: *what else does this
  vary with, and is that thing part of what I am testing?*
  **Third form — ANY GUARD ON A RATIO'S DENOMINATOR APPLIES PER-CELL, NEVER
  POST-POOLING.** (Catch 26.) `cure_screen.py` scored rho as a ratio of pooled
  means and checked `RHO_MIN_DENOM` against the pooled denominator, which is
  healthy by construction. E10's per-cell denominators spanned **0.0026 to 0.556
  (200x)**, with **4 of 24 cells at or below the guard** — none of which ever
  tripped it. **Pooling is itself a way of hiding a failed cell**: a guard
  evaluated after the average is applied where the failure has already been
  averaged away. Report mean-of-ratios over valid cells, print per-cell
  denominators, and count the exclusions in the table — an exclusion rate is a
  reportable fact about the benchmark, not a footnote. (`scripts/rho_percell.py`.)
  **Corollary — re-grade the OLD numbers before citing them again.** The same
  recompute moved E8's C3 from the published **1.037 to 0.798**, across H-R3's
  0.80 bar. The provenance rule does not grandfather our own favorite results.
- **Catch 35 — a STATISTIC CAN BE UNABLE TO VARY, and it looks exactly like a
  number while it cannot.** (2026-09-21, E25 R2; a variant of 25, one level
  below a gate.) Catch 25 is about a gate that cannot fail. This is about the
  *quantity the gate reads* being pinned by the design rather than measured.
  Two forms, both found in one afternoon on the same family:
  **(a) A spread over a factor with ONE level is zero by construction.** E25's
  R2 replicated E21's floor, `max(spread over realizations, mean per-cell
  jitter spread)`. At eps = 0, E21 runs **one** realization (`e21_row.py:135` —
  the perturbation is identical for every r), and the swap search is
  `torch.no_grad()` and deterministic, so **both** components are 0.000 and the
  level-0 floor is zero *by arithmetic*. It printed "A4 BEATS A3, floor
  0.0000": any non-zero delta clears a floor that cannot be non-zero. Fix: a
  level with one replicate carries **no** floor and is not read. *Before
  trusting a floor, ask how many values its spread is taken over; if the answer
  is one, it is not a floor.*
  **(b) A POOLED SIGN PATTERN IS NOT A CONSISTENCY STATEMENT.** E21's memo
  records A5 − A3 "positive in **9 of 9** realizations" and registers a paired
  read blocked on the wiring as the follow-up. Disaggregated, at m = 1 A5 beats
  A3 in **11 of 36 cells** — the three realization *means* are unanimous while
  the majority of individual cells go the other way, because a few cells with
  large gains carry each mean. A4 − A3 is worse: **3/3 realizations, 8/36
  cells**. Both statements are true and they support opposite readings.
  **This is catch 26's rule applied to signs rather than to denominators:
  pooling is a way of hiding a failed cell, and a sign count taken after the
  average has already averaged the failures away.** Any "positive in k of k"
  claim prints the per-cell count beside it, or it is a claim about means
  wearing a claim about consistency.
  **What this licenses and what it does not.** A method may legitimately win
  big on a few cells and lose slightly on many; that is a real result and the
  mean may be the quantity a practitioner wants. It is not licence to *report*
  it as consistency. Report both, and say which one the reading rests on.
- **Catch 28 — path-identity is a premise, not a property.** An instrument
  claiming to read the deployed path carries that claim as an unverified premise
  until proven by exactness: `head(captured_feature) == deployed_logits`, exact,
  on live inputs, every arm type, asserted permanently at the top of every
  consuming script. A docstring asserting path-identity verifies nothing — the v3
  feature lock asserted it for two experiments while reading the wrong tensor
  (raw `c_t` instead of the post-GGC composed `c_t'`). The assert must be
  NON-TAUTOLOGICAL: it must be able to fail if the wrong module is hooked.
  (Corollary of catch 21, applied to instruments; caught by a consistency check
  nobody required.)
  **Corollary — a suppression an instrument needs must be expressible THROUGH
  the deployed path, never around it.** C0 requires the deployed adapter
  suppressed so the analytic one can substitute; a hand-rebuilt bypass would
  reintroduce catch 28 in the cure that can least afford it. If the model's API
  cannot express "deployed path, adapter suppressed," that is a model-API change
  made under the assert's protection — not an instrument workaround.
  **Why the habit pays:** its value is not thoroughness, it is printing
  comparisons that usually agree, so the one time they do not is loud.
- **Catch 32 — a hardened helper's value is lost the moment a new script
  re-implements its job.** `channel_decomp.load_task_data` had already solved
  label alignment, with the fix written into its docstring: *"two separate
  comprehensions over a shuffle=True train loader draw two DIFFERENT
  permutations... the probe then trains on randomly relabelled data and
  collapses to chance. That bug produced R ~ -0.63 and was misread as optimizer
  underfitting."* E12's decomposition wrote a fresh extraction path that did not
  inherit it, extracted era features in one pass and theta_T features in
  another, and paired the second pass's features with the first pass's labels.
  **The corrupted result inverted the experiment's conclusion**: refit at
  theta_T read chance (0.198 vs 0.200), so F_enc came out 0.75 and F_read ~0,
  and E12 was reported as an encoder-side architecture boundary — branch (C),
  paper 1 retreating to LSTM-family framing. Corrected, refit reads 0.82-0.92,
  F_read 0.53-0.84, reader share ~85%: *inside* the LSTM family's 66-88% range.
  **RULE:** a new data path routes through the audited loader, or the contract
  states why it cannot; any parallel implementation ships a bitwise equivalence
  test against the original on a shared input. Same family as catch 28 — a
  premise carried in a docstring rather than an assert — different surface.
  **What caught it:** one number that did not fit. The class-IL arm's deployed
  head, restricted to a task's own classes, scored 0.7075 on theta_T features
  while a probe *fit to convergence* on those same features scored 0.1982. A
  freshly fit linear probe cannot lose to a fixed linear head by 50pp on
  identical inputs. The corrupted story was more dramatic and more
  publishable-sounding than the truth, and it survived two rounds of scrutiny, a
  full memo draft, and an odds recalculation. *Chase the number that does not
  fit before writing around it.*
- **A control rules out only the defects that would BREAK it. Before citing a
  control, state the failure mode it would catch and confirm the suspected
  defect is in that set.** Catch 32's label misalignment was dismissed on the
  strength of "the same recipe on era checkpoints gets 0.95" — presented as
  ruling out a probe bug. It could not: the era refit's features and labels came
  from the *same* pass, so an upstream misalignment cancels there and shows up
  only at theta_T. It was a control against *encoder-quality* explanations,
  never against *label-alignment* explanations, and it was allowed to stand for
  both. Catch 25's shape inside a control that had been explicitly endorsed.
- **Catch 29 — a reloaded checkpoint is not the model that was trained until its
  non-parameter state is proven present.** `PLCM.task_stats` is a plain dict, so
  `state_dict()` omits it, and `forward()` gates coordinate alignment on it being
  non-empty — every checkpoint-loading analysis since E4b silently evaluated a
  model missing that path. Evidence: `runs/e11_ckpt_fidelity_mnist_s42.json` —
  bare reload missed its own matrix by +0.0040/+0.0097/+0.0336/+0.0270 on E4/OFF,
  all four positive; restoring stats reproduced all 8 cells exactly.
  *General form: serialization completeness is a premise like path identity —
  audit what `state_dict()` omits (plain attributes, RNG state,
  buffers-by-convention) whenever behavior gates on non-parameter state; the
  fidelity check is reproduction of the recorded matrix, not successful loading.*
  **Process lesson attached:** this catch sat un-ruled for three exchanges
  because ledger traffic displaced it. *A ruling requested and answered is not a
  ruling landed until it is in the file.*
- **Catch 30 — the strength of a gate's pass is not evidence the gate ran on the
  right configuration. Arm identity is verified independently of how comfortable
  the verdict is.** E12's `E12_CFG` omitted `--task-heads` and `--no-adapters`,
  and `scripts/train.py` derives both from CLI flags and **overwrites the config
  after reading it** — so the runs labelled "base arm" were adapters-ON with one
  shared 5-way head scoring all 20 tasks: neither contracted arm, and not task-IL
  at all. P2(a) then passed at **5× its bar (0.716 vs 0.15)**.
  **The defect did not produce a suspicious number, it produced a COMFORTABLE
  one** — a single shared head over 20 disjoint 5-class problems can only do so
  badly, so catastrophic forgetting was near-guaranteed by construction. A defect
  yielding marginal readings gets caught by margin discipline; a defect yielding
  emphatic readings sails, **because emphatic is what passing looks like.**
  Re-run on the contracted arm, the same control moved 0.0173 → **0.0040** and
  the seed-42 wobble at 81% of the bar dissolved with the arm that produced it.
  **Sibling of the P3b catch, from the opposite direction:** there, asserts were
  vacuously green on a misconfigured amendment; here, a gate was resoundingly
  passed on a misconfigured arm. Together: *the verdict's quality tells you
  nothing about the premise's.*
  **Where the existing rules did not reach:** catch 20 audits artifacts, catch 21
  audits contracts, catch 28 audits instruments — **nothing audited the object in
  memory at train time**, which is the only place an arm actually exists. Verify
  arm identity from the run's own artifacts (E12's evidence was
  `adapter_dist_from_identity` present in every epoch dict and `head_weight_norm`
  absent from all of them), not from the config file the launcher overwrote.
- **AN OBSERVABILITY FEATURE THAT RELOADS, EVALUATES, OR REFITS INSIDE THE
  TRAINING LOOP IS PART OF A RUN'S CONFIGURATION — not a passive annotation.**
  `--era-checkpoints` does not merely write a file: `_save_era_checkpoint`
  reloads through `PLCM.load_era`, evaluates, runs `_refit_ceiling`, and fires
  the P3 gate, all inside training. Measured effect on final AVG at seed 42:
  **MNIST +2.13pp, HAR -1.67pp** — same flag, **opposite signs by benchmark**,
  so no blanket correction exists.
  **Confirmed, not conjectured:** re-running the floor pairs WITH the flag
  reproduced the sweep runs **bit-identically, 15/15, AVG gap 0.0000, on both
  benchmarks**. Unlike the two mechanisms that died this week, this one made a
  falsifiable prediction and it held exactly.
  **RULE:** era-checkpoint status is an arm-identity field (it is already in
  `arm_provenance`), every comparison must be UNIFORM on it, and uniformity is
  checked ACROSS the arms of a comparison — checking within each arm shows
  "uniform" and hides the mismatch, which is how this was nearly missed.
  *General form for the paper: anyone adding checkpoint-time diagnostics to a
  continual-learning pipeline and comparing against runs without them carries an
  unlabeled ~2pp confound of benchmark-dependent sign.*
- **A plausible mechanism attached to a true observation is the week's most
  repeated error; the observation survives, the story does not.** Two died in
  one week, identical in shape — each generalized ONE configuration beyond its
  evidence: *"the delta is certainly code state"* (killed by a 1.44pp same-seed
  spread) and *"the nondeterminism is BLAS reduction order"* (killed by HAR
  being 4-thread-deterministic and **1-thread-NONdeterministic**, the exact
  inverse of MNIST's `plain_lstm` — no thread-count-monotone story can produce
  both, and single-threaded BLAS has a fixed reduction order).
  **Final form, which survives because it never contained a mechanism:**
  *determinism belongs to (arm x config x platform x thread-count), established
  PER-PAIR, mechanism unknown until demonstrated.*
  **Operational corollary: PINNING IS AN AXIS, NOT A STABILIZER.** A floor is
  measured at the threading the reported runs use, or it bounds a configuration
  nobody reported. The earlier "floor pairs run pinned" rule is withdrawn.
- **The era presumption is a PRIOR, not a verdict, and re-running is its only
  rebuttal channel.** HAR OFF is the first rebuttal: three seeds reproduced the
  unrecorded-config `e5_har_noadapt_*` runs **bit-identically, 15/15, across a
  two-week code gap**. The flag was right to fire (lineage was unverifiable) and
  the lineage turned out clean — those runs are now citable *because of* the
  verification, not despite the flag. Cost: one hour.
- **A rule that fires mostly on false positives will be ignored where it
  matters; scope the rule to the hazard.** Wave 1's naming audit began as "grep
  the corpus for `vanilla`/`baseline` near a number" and returned ~20 hits, of
  which **two** were real. The corpus uses "baseline" in **two senses**: an ARM
  baseline (`plain_lstm`, `mafc_off` — the hazard, two arms one point apart) and
  a METHOD baseline (tier-0, the snapshot at rho 1.237, era-head/era-prototypes
  — *the thing a method must beat*, unambiguous and correct). Narrowed form:
  ***"vanilla" is never an arm name; "baseline" is fine for a method and never
  for an arm.*** A blanket ban would have trained everyone to skip the warning.
- **Catch 34 is not an incident, it is a PATTERN FROM A SPECIFIC ERA of this
  repo — pre-arm-recording, pre-run-directory discipline.** Two independent
  instances, both found in Wave 1, both structurally identical: *one value from
  an exploratory file + two unrecorded-config run dirs, averaged into a
  citation.*
    * adapters **0.9331** = mean(`fullrank_ref` 0.9470, `e4_on_seed1337`,
      `e4_on_seed2024`) -> verified **0.9183**
    * EWC **0.4330** = mean(**0.4950 read from inside `plcm_ewc_sweep.json`**,
      `ewc_seed1337`, `ewc_seed2024`)
  The second was found by **audit**, not by accident — which is the graduation
  the process exists to produce.
  **Scope statement, stated once and applied everywhere:** *every number whose
  lineage passes through that era is presumptively UNVERIFIED until re-run or
  traced to an artifact recording its own configuration.* The pipeline is
  bit-deterministic, so this is not caution about noise — it is that those runs
  certainly measured some other code state, and nothing in them says which.
- **"vanilla" is not an arm name. Artifacts name arms; tables cite artifact
  names.** Two rows in the drift memo sit **one point apart** and both read as
  "vanilla": **plain LSTM 0.4399** (`lstm_seed*` + `plcm_ewc_sweep.baseline_lstm`)
  and **MAFC with adapters off 0.4304** (`e4_off_seed*`). The attribution uses the
  second — `0.9331 - 0.4304 = +50.3pp` **exactly**, while the first gives +49.3pp.
  **~1pp of the headline rides on which one a reader or a future audit picks, and
  no output inspection would catch the swap** — the defining property of catch
  30's family. The fix is naming, not care: tables print the arm as the artifact
  records it (`plain_lstm` / `mafc_off`), and every caption states which arm each
  comparison uses.
- **A bit-deterministic pipeline does not wobble; the CODEBASE does. Only the
  artifact's own config field pins which codebase a number belongs to.** The
  permuted-MNIST CPU path reproduces **15/15 cells, max |delta| 0.000000** at a
  fixed code state — yet `e4_off` seed 1337 moved **+1.93pp** between 07-31 and
  today, and seed 42 of the adapters arm moved **-2.91pp**. Those are pure
  code-state deltas across two weeks of amendments, **measured, not inferred**.
  Consequence: on a pipeline proven deterministic, an unrecorded-config run is
  not merely unverified — it is *certainly* measuring some other code state, and
  the old runs prove their own obsolescence (no `arm` field, and arm-recording
  postdates them). *A zero floor raises the bar for citation rather than
  lowering it.*
- **Catch 34 — a reconstruction that MATCHES a cited value is consistent with
  being its source, never evidence that it is.** `runs/fullrank_ref` (0.9470),
  `e4_on_seed1337` (0.9279) and `e4_on_seed2024` (0.9243) average to **0.9331 at
  four decimals** — the exact number the adapter row cites. That match was read
  as identification, the provenance question was declared "closed by
  archaeology", and the re-run that would have tested it was argued against as
  buying nothing. It bought everything: under the recorded config seed 42 reads
  **0.9179**, and the verified 3-seed mean is **0.9183**, not 0.9331.
  **Matching is a necessary condition wearing a sufficient one's clothes.**
  Distinct from catch 32, which was a parallel *implementation* escaping an
  audited path; this is a parallel *identification*.
  **The visible warning was the NAME.** `fullrank_ref` sits beside
  `lowrank_r32`/`lowrank_r64` and says "the low-rank study's reference arm", not
  "e4_on seed 42". *Names carry experiment lineage; read them as evidence.*
  **RULE:** a number enters a caption only from artifacts that RECORD their own
  configuration. Where the artifact has no `arm` field, the run is re-run, not
  reconstructed.
- **Companion to 34 — when the deciding instrument is already running, interim
  evidence is for generating hypotheses, not verdicts.** The same episode
  produced the error TWICE, in opposite directions: arithmetic match -> "closed"
  (too credulous), then mixed-sign cross-seed deltas -> "noise, my configuration
  reading was too confident" (too quick to recant) — while the floor pair,
  argued into existence *for exactly this question*, was minutes from answering
  it. It came back **15/15 cells bit-identical, max |delta| 0.000000**, which
  disposes of the noise reading entirely: a deterministic pipeline cannot
  produce a 2.91pp gap by chance.
  **The tell that should trigger the rule:** noticing you are updating
  confidently on evidence that an in-flight measurement will strictly dominate.
  Catch 34 is cheap signals replacing provenance; this is cheap signals
  replacing patience.
  **Corollary — a zero floor RAISES the standard rather than relaxing it.** Once
  the pipeline is measured bit-deterministic, unrecorded-config runs are not
  merely risky, they are *certainly* measuring some other code state. The old
  runs prove their own obsolescence: no `arm` field, and arm-recording postdates
  them.
- **A hyperparameter with a default in the launcher is a hyperparameter selected
  outside the measurement.** The EWC lambda=200 lesson made structural: E16's
  `jobs_e16head_*` take the lambda as a REQUIRED argument and raise without it,
  because a default would let the launcher, rather than `runs/e16_sweep.json`,
  pick the reported configuration. Same family as the n>=3 selection rule, one
  level down — that rule governs how a value is chosen, this one governs where
  the choice is allowed to live.
- **A verification must EXERCISE the failure mode it claims to rule out.**
  `ast.parse` was used to check a launcher edit; it passed, and `import` then
  failed with a `NameError` because the JOBS dict referenced functions defined
  200 lines below it. A syntax check cannot see name resolution. Same family as
  the `;`-instead-of-`&&` chain, the vacuously-green assert, and the unanchored
  `.replace()`: in each, a cheap check stood where an expensive one belonged.
  *State the failure mode, then confirm the check can detect it.*
- **Catch 33 — every contract ships a CLAUSE -> JOB -> ARTIFACT table, verified
  before the headline runs launch.** E14's contract stated P2 as two clauses —
  task-0 drop >= 0.15 **plus** a repeat-task control <= 0.05. Only clause (a) had
  a launcher entry. E12 has `jobs_e12rptb`; E14 had **no `jobs_e14rpt` at all**,
  so P2 would have entered the memo marked PASS **on half its definition**.
  **Absence is invisible to every check that inspects outputs.** Catch 20 audits
  artifacts, catch 21 audits premises, catch 28 audits instruments, catch 30
  audits arm identity at train time, the build gate audits the build — all of
  them check *things that exist*. A clause with no job produces no artifact to
  look wrong, no verdict to read badly, and no file to fail an audit.
  **What hid it:** P2(a) passed at **4.4x its bar**, so the P2 row read
  emphatically green — catch 30's lesson in a new place, except here the missing
  half was not misconfigured, it was *absent*, and absence emits nothing.
  This is the only check in the family that operates on the **contract** rather
  than on the run: walk the preconditions section, name the launcher entry for
  each clause, require a path. Cheap, mechanical, and it must run **before** the
  headline jobs, not before the memo — caught pre-memo it cost 15 GPU-minutes;
  caught post-ledger it would have been a retraction.
  *(P2(b) subsequently read 0.0010 vs its 0.05 bar — the clause passed. A
  precondition that would have passed anyway is still a precondition that was
  never run, and the rule is about the second fact, not the first.)*
- **A one-sided conditional is a bet that the surprise will come from one
  direction.** E14's contract wrote branch (C) for "`F_enc` materially LARGER
  than the ViT's 0.060" and no mirror; it came in **7.9x smaller**. Write
  branches two-sided whenever the quantity can move either way. **Why it cost
  nothing here:** because the branch was written *explicitly*, the asymmetry
  became visible instead of being absorbed — an unwritten expectation would have
  quietly accommodated either outcome. The failure mode to fear is not the
  missing branch, it is the missing branch on an unstated expectation.
- **For every resource a method's spec assumes, price the TRIVIAL USE of that
  resource first. If trivial use dominates the method, the design measures the
  resource, not the method.** (Catch 24 — twice in two contracts.) E8's C3
  assumed a known shift map `M_k`; on a permutation-family benchmark that
  assumption hands back the old data exactly, so C3's ceiling-matching ρ
  measured **the benchmark's invertibility**, not the cure. E9's T1 assumed a
  stored era snapshot θ_{k+1} to estimate a transport map; but simply *running
  old tasks through the stored snapshot* scores **ρ = 1.237 — above every
  ceiling in the program**, because a snapshot has neither encoder damage nor
  reader mismatch. The estimation procedure would have been approximating what
  its own stored object already delivers exactly.
  **Test:** write down the assumed resource, ask "what is the dumbest thing I
  could do with exactly this?", and measure that first. It is either the
  baseline the method must beat, or proof the design is mis-scoped.
  *(Corollary, earned: the trivial baseline is often a real result. Per-task
  snapshot storage at ρ=1.237 / ~1MB per task is the degenerate upper-right
  anchor of the storage-accuracy frontier — the "just keep every model"
  endpoint every storage-honest method is implicitly priced against.)*
- **A printed verdict must be DERIVED from the values printed beside it, never
  written to mirror the contract's expected outcome.** (Catch 22 — second
  instance of its class.) E4's `drift_probe.py` printed a pass/fail string that
  fell through to the wrong branch while the ratios beside it said otherwise.
  Then E8's C3 scope check printed "EXACT reconstruction ... recovered for
  free" directly above its own numbers reading **17.5** and **1.0** — the
  conclusion was right, but it had been written from the expected result rather
  than computed from the measurement, and the check that was shown did not
  establish it (the train loaders were shuffled, so it compared different row
  orderings; the sequential test split proved the point later).
  **The failure mode is a verdict that agrees with the contract while
  disagreeing with the data on the same screen.** Compute the verdict from the
  same array you print, in the same expression where possible; when a printed
  claim and a printed number disagree, the number wins and the script is wrong.

- **An EXCLUSION is a claim with a context, and it transfers to no other
  construction without re-verification.** (2026-09-15, the C0deg finding —
  catch 21's rule applied to a label that says "not a cure" instead of one that
  says "validated".) `cure_screen.py` marked C0deg — re-layout old inputs into
  the CURRENT frame with the known map, score with the current model — a
  "CEILING ARTIFACT, never a cure", correctly, on E8's shared-window HAR where
  it reproduced task 4's test set. E10 rebuilt the benchmark around the exact
  property the label depended on, the column was computed on E10 in
  `runs/e11_e10/cures_e10.json`, and nobody read it, because the label said not
  to. Read, it is the best storage-honest method in the program (OFF forgetting
  **−0.0247** vs bridging's 0.0886, intervals separate), using a strict subset
  of bridging's resources — catch 24 on the method the paper was built around.
  **Every "not a cure / artifact / oracle / excluded" label in a screen names
  the construction it was earned on, and a benchmark repair re-reads every
  label, not only every number.** An inherited exclusion hides a result the way
  an inherited inclusion fakes one; the audit habit has only ever looked for the
  second.
- **A registered hypothesis names its formula by DISPLAYED EQUATION in the
  contract text. File citations are witnesses, not definitions.** (2026-09-15.)
  E10's H-X2 was registered as "the standard forgetting formula";
  `hx2_forgetting.py`'s docstring cited `src/training/metrics.py` for
  `mean_j(R[j,j] − acc_j)`, and `metrics.py` implements
  `mean_j max(0, max_l R[l,j] − acc_j)`. The two share a word and differ by the
  backward transfer the peak form folds in; on the same 11 cells they put C3 on
  opposite sides of the bar (diag 0.0886 MET, peak-clipped 0.1198 NOT MET). The
  verdict "H-X2 MET" sat in the ledger under a formula the row did not name.
  Ruled: diag is the paper's form (§3.1's `F_total` is diag by construction);
  the registration was defective; the row carries both and the paper does not
  say "pre-registered and met" without the qualifier. Found the way catch 32
  was: a control printed −0.0497 for an arm the ledger listed at 0.02507.
- **A must-fail's bar is priced against the benchmark before launch, and it
  does not move after.** (2026-09-15, CC.) The wrong-map control for C0deg
  pre-committed "below the right map in every cell" with the wrong map being the
  neighbouring task's. It failed 4/24: every failing cell differed from the
  right map by a gain/offset or a rotation only, to which the encoder is
  partially invariant; every permutation-differing cell collapsed. The control
  could be broken by a property of the model rather than a defect in the
  instrument, so it was a measurement wearing a control's clothes — *a control
  rules out only the defects that would BREAK it*, applied to one's own control.
  Recorded as FAIL with its cause; **the replacement (CC′, permutation-only,
  bar stated first) passed 12/12 × 2 × 2, and the cause became a scope finding
  (CD: the permutation component carries 84% of the recovery)**. Rewriting the
  criterion to a mean-gap form would have passed, and would have been the first
  time in this program a control's bar was moved to meet its outcome.
- **fp32 shadow is a standing requirement on every era-checkpoint run that
  enters a comparison at the 1e-6 gate; a benchmark CONSTRUCTION is an
  arm-identity field.** (2026-09-15, S72.) Era checkpoints are fp16 and reload
  one window off the recorded matrix (+0.00034 measured on E16, +0.0058 in a
  smoke); the consistency gate that proves "the screen and the matrix describe
  one run" is 1e-6, so an fp16-only run fails it for a reason that is not a
  discrepancy — witness imprecision producing a false one. `--fp32-shadow`
  reloads at exactly 0.0 and goes on **every arm of a comparison or none** (it
  reloads and refits inside the training loop, like the era feature itself).
  Same wave, third instance of a construction drifting under a name: the E16
  contract pinned "UCI HAR (subject-disjoint, E10 construction)" and the
  launcher wrote `--benchmark har` (shared-window); the arm field's
  `benchmark` value caught it before the row was cited. Verify the construction
  from the artifact, not the contract — and record every construction flag
  (`disjoint_content`, `no_shift`, angles) in `arm`, so the next one is
  catchable the same way.

- **A positive control for a STRONGER instrument must be a case the WEAKER
  instrument fails.** (2026-09-16, E20.) The MLP probe's control v1 used
  `(y mod 2) XOR [PC1 > 0]`; the *linear* probe scored 0.9635 on it — with ten
  separated class clusters PC1 is a function of the class, so the XOR reduced
  to a dichotomy of ten means, always linearly separable in 256-d. A control
  the weaker instrument passes cannot detect a broken stronger one. v2 built
  the second bit on the within-class residual, where the linear probe reads
  0.51 and the MLP 0.84. The v1 artifact stays on disk
  (`runs/e20/positive_control_synthetic_v1_FAIL.json`): the control caught its
  own author, which is what the synthetic run before any live read is for.
  **Final form (seven designs, `runs/MEMO_e20.md` §1):** v2–v7 all passed on
  synthetic features shaped like the live ones and **failed live** the same
  way — the linear probe read 0.67–0.71 on every target built to be
  nonlinear, the MLP +0.10 above it every time. On multimodal features in
  256-d, nearly every smooth target is a function of mode identity, and any
  labeling of separated modes is linearly separable. **Both halves of a
  positive control — the weaker instrument fails, the stronger passes — are
  printed numbers on the live features, never a synthetic pass and never an
  argument about geometry.** Where the live half cannot be built, say so,
  certify the recipe on a matched synthetic, and gate the live read with
  something non-vacuous (R1: the stronger instrument must not fall below the
  weaker one by more than the measurement's resolution).
- **A gate finer than the measurement's own resolution is not a gate; it
  detects noise.** (2026-09-16, R3.) The R1 gate's tolerance was the probe's
  init spread — as small as 0.1pp — and it failed 5/24 primary cells on
  shortfalls of 0.1–0.6pp where the test set's 1.96·SE was 0.6–1.1pp. The
  init spread measures the instrument's variance and omits the test set's,
  which is a real component. Tolerance became `max(init spread,
  1.96·SE_binomial)`, **stated before it was applied**, and the test that
  it is a correction and not a relaxation is that it still fails the
  substantive cells (the secondary's 6.9pp and 6.6pp shortfalls). A rule that
  rescues everything is fitted; one that rescues only the noise-level failures
  has the right resolution.

- **A contract states a mechanism by CITING THE LINE THAT IMPLEMENTS IT, not
  by describing it.** (2026-09-17; the same error twice in two contracts.)
  E18's draft defined the "invert the known map" arm in the non-deployable
  direction (old encoder + old head = SNAP); E21's draft wrote that bridging's
  head is fit "through $\hat M_k^{-1}$ on the corrupted distribution" — the
  generator maps *forward* into task k's frame
  ([cure_screen.py:280](scripts/cure_screen.py#L280), `har_relayout(xcur_tr,
  CUR, k, maps)`), the era teacher labels an ε-off frame, and the head is
  trained ε-off and tested true. Both times the mechanism was written from a
  model of the method and the repo said otherwise on one line. Every
  mechanism sentence in a contract carries a `file:line`; a "where the
  perturbation enters" table is written per arm from those lines.
- **A FINGERPRINT IS ONLY SENSITIVE TO THE AXES ITS INPUTS ENCODE, and the
  axis you are about to vary is the one to check it against.** (2026-09-20,
  E23-B.) `PermutedMNISTBenchmark.content_fingerprint` hashes the concatenated
  per-task index tensors to prove which content split produced an artifact.
  Chunking ONE permutation into 5 pieces or into 20 gives the same byte
  stream, so **T=5 (12k images/task) and T=20 (3k/task) both read
  `c1a570a02d35` at seed 42 while sharing no per-task content at all** — the
  guard that exists to catch construction drift was blind to the exact axis
  the sequence-length control varies, and a consumer that rebuilt the wrong T
  would have passed the assert.
  **Same class as the C0deg mislabel:** a check valid for the constructions it
  was written against, carried to one it does not cover. The difference is
  that an exclusion label announces itself and a hash does not — it returns
  twelve plausible hex digits either way.
  **Fix shape, when artifacts already record the old value:** keep the old
  hash byte-compatible (every E18 artifact cites it; changing it makes their
  screens unrunnable), add the corrected one BESIDE it
  (`construction_fingerprint`, which hashes the split's shape too: 3/3
  distinct where the old gives 2/3), and strengthen the CONSUMER's assert
  (`screen_e18` now checks `num_tasks` and `content_chunks`). Amending the
  witness in place would have spent the proof E18's artifacts bought.
  **Test:** before a construction parameter is varied, build the benchmark
  both ways and assert the fingerprints DIFFER. A fingerprint that cannot
  distinguish the two arms of the comparison it is guarding is not guarding
  it.
- **A contract's first paragraph is the grep of `src/data/` for the
  construction it claims does not exist.** (2026-09-18; the fifth contract in
  a row whose opening premise came from a model of the program rather than
  from a file.) E23 v1 opened "every pretrained result has no map" and
  proposed ResNet-50 on upsampled MNIST to close a confound; the repo has
  `cifar100_permuted` (E12's B6: pretrained ViT-B/16, twenty tasks, a
  per-task permutation on the model's actual input, exactly invertible,
  verified bitwise) with **six runs on it** — never decomposed only because
  the launcher's own docstring chose to skip era checkpoints and wrote "if it
  ever becomes a comparison baseline it needs a re-run." The premise error
  had the same shape as E22's (1a already existed as `har_subject` no-shift),
  W's (the quantity already sat in `runs/MEMO_e18.md`), and D's. Each time
  the corrected experiment was **better** than the drafted one, not merely
  cheaper: E23 v2 closes the confound within one benchmark instead of adding
  three variables. The rule is mechanical because the failure is: before a
  construction is declared absent, the grep is run and its output is pasted
  into the contract.
- **No count enters a contract unless it is READ from the appendix table at
  the moment of writing.** (2026-09-17.) "Nineteen entries, six misses"
  (E18–E20 draft), then "twenty prior entries scored" (E21 draft) — the
  third fabricated tally in three contracts, each matching no artifact. The
  prediction ledger (`docs/appendix.tex`, `\label{app:predictions}`) is the
  only source; a contract quotes its current line verbatim, and the table is
  brought current *before* the quote, not after.

## Gotchas / hard-won lessons

- **"A run finished" has a mechanical definition — directory existence is NOT it.**
  Results dirs are created by `os.makedirs` at job start, *before* training, so a
  directory proves a job **launched**. Under preemption-and-restart it can also
  hold a truncated result. Counting directories is the infrastructure cousin of
  the n=1 problem: a cheap signal standing in for the expensive one it does not
  contain (catch 18). A run is finished iff **(a)** the accuracy matrix is
  `num_tasks x num_tasks` with no NaN, **(b)** `task_history` has all tasks with
  their full epoch counts, and **(c)** the file mtime postdates any
  preemption/restart event. Use `scripts/verify_runs.py 'runs/<glob>'` — it prints
  each run's headline numbers on pass, so the verification output *is* the raw
  table and no transcription step sits between "verified" and "reported".
- **"A checkpoint exists" is not "a checkpoint loads."** The same cheap-signal
  error one level down from the directory-counting gotcha above. E11's audit
  found **14 of 33 checkpoint dirs unusable** — the whole E5 family missing or
  truncated by a partial `modal volume get`, one file **4.9MB and still
  unreadable** (`PytorchStreamReader: failed finding central directory`), plus
  **68 of 160 result dirs empty**. Size is not a criterion and neither is
  presence: audit by **opening every checkpoint through the same loader the
  analysis uses** (`scripts/audit_checkpoints.py`, criterion = `PLCM.load_era`
  succeeds). Discovered at read time these are 14 loud failures; undiscovered
  they are 14 silent ABSENTs mid-recompute.
- **A restarted container can produce a complete-looking but WRONG result — a
  nastier failure than truncation.** Truncation is loud: the matrix has NaNs or
  the wrong shape and every check fires. A preempted-and-restarted run instead
  yields a full `num_tasks x num_tasks` matrix, a complete `task_history`, and a
  fresh mtime — it passes the entire finish definition while carrying different
  numbers (E5 ON/2024: 0.6035 contaminated vs 0.6585 clean, a 5.5pp gap on the
  ON arm's headline). The contamination mechanism was not diagnosed; what
  matters operationally is that no *file-level* check can detect it, because the
  file is well-formed. Only re-running the config detects it. See catch 19.
- **Bank replay must filter `task_id >= 0`.** `rebalance()` frees excess slots by
  setting their `task_id` to -1 (they stay within `[:size]`). Sampling those for
  any task-labeled loss (e.g. training a task router on replayed states) feeds -1
  as a class label → `cross_entropy` raises `IndexError: Target -1 is out of bounds`.
  Always mask to `task_ids[:n] >= 0` before using stored task ids as labels.
- **Validation synthetic tasks must spread signal across timesteps.** PLCM
  classifies from the LSTM **cell state** `c_n[-1]` (an accumulator), not the
  hidden output. A toy task with signal only in the last row is learnable from a
  *hidden-state* readout but NOT reliably from the cell state, so it fails to train
  PLCM and its ceiling sits at chance — making it useless for validating anything
  downstream (routers, retrieval, etc.). Put signal in multiple rows / across
  timesteps so the cell-state accumulator is genuinely informative. (This cost a
  full validation cycle to learn.)
- **Untrained retrieval index (failure mode #5).** `memory_bank.key_proj` and the
  `WriteController` never receive gradients, so retrieval indexes through a random
  projection. A trained linear probe recovers task identity from cell states at
  ~0.995, but random-key retrieval routes at chance. Retrieval quality is bounded
  by index quality, not by what the thought space contains.
