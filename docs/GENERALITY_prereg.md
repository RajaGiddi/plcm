# Pre-Registration: Benchmark Generality Package (ICLR additions)

**Status:** **v2, locked.** Five amendments approved pre-launch (§7). No
new-benchmark code existed at v1 lock.

**Purpose:** the novelty review identified two ICLR-gating additions — (1) at
least one non-MNIST input-shift benchmark, (2) an encoder-drift probe that
converts the mechanism claim from interpretation to evidence. Scope-preserving:
**input-space shifts only**; semantic shift remains paper 2.

---

## 1. What is being tested

**G1 (generality across data regimes):** the adapter arm's accuracy is
approximately invariant across datasets and shift types, with delta differences
attributable to baseline (no-adapter) difficulty — extending the measured 1.7pp
ON-arm spread ({LSTM, MLP} × {permuted, rotated} on MNIST) beyond the MNIST family.

**G2 (mechanism):** the shared encoder in the adapter arm exhibits near-zero
representational drift on old-task inputs relative to the vanilla arm — the direct
test of "input-path gradient isolation."

---

## 2. Experiments

### E4 — Encoder-drift probe (FIRST; the mechanism gate)

**AMENDED (v2, Finding 1):** this does **not** run on existing checkpoints.
Audit at lock time:

| checkpointed run | seeds | adapters | task_heads |
|---|---|---|---|
| `fullrank_ref` | **42 only** | True | False |
| `rbst5` | 42/1337/2024 | False | **True** |

The only OFF-arm candidate (`rbst5`) differs in **two** variables — per-task frozen
heads vs shared trainable readout — so a drift comparison against it would be
contaminated by readout architecture. **E4 = 5 new runs (~1.7h)**: ON seeds
1337/2024, OFF seeds 42/1337/2024, all checkpointed, arms differing in
`adapters.enabled` only; ON/42 reuses `fullrank_ref` (config verified identical).

**Metrics**, per arm, task-0 test inputs, θ₀ = end-of-task-0, θ₄ = end-of-task-4:
- **Drift** = 1 − mean cos(f_θ0(x̃), f_θ4(x̃)), where x̃ is the input *as the encoder
  sees it* — post-adapter in the ON arm (A₀ is frozen after task 0, so identical at
  both times), raw in the OFF arm.
- **CKA** between the two representation sets (scale-insensitive check).
- **Probe retention**: train a linear probe on θ₀ task-0 representations, evaluate on
  θ₄ representations of the same inputs — the functional form of drift.

### E1 — Fashion-MNIST (permuted + rotated), near-free
Identical pipeline (28×28 grayscale, 5 tasks × 10 epochs), arms ON/OFF, LSTM,
seeds {42, 1337, 2024}, both shift types. **Honestly scoped:** escapes MNIST
*digits*, not the 28×28 grayscale regime. Supporting evidence, not the answer to
the reviewer objection.

### E5 — UCI HAR: applied sensor time series (priority above E2)
9 channels × 128 timesteps, 6 classes — LSTM-native, unlike E2's row-wise image
consumption. Tasks = 5 simulated **hardware-revision shifts**, all input-space,
frozen and recorded pre-run: task 0 = original calibration; tasks 1–4 = per-task
combinations of (a) fixed per-channel gain/offset miscalibration, (b) axis rotations,
(c) channel reordering. Adapter: per-timestep linear 9→9 or a small full linear on the
flattened window, chosen by the identity-init rule, recorded pre-run.

**AMENDED (v2, Finding 4):** timebox restated honestly — **E5 = data engineering
(majority: download, unpack, window, shift simulation) + training (minority).**
UCI HAR is not in torchvision. Priority unchanged; if the loader fights back past a
few hours, **flag rather than grind** — timebox discipline applies to plumbing too.

**Pre-named caveat:** simulated calibration shifts remain simulated. Supported claim
is *"the mechanism handles realistic shift classes on real sensor data,"* not
*"validated on deployed drift."*

### E2 — CIFAR-10 with input-space shifts
32×32×3; LSTM consumes rows as timesteps (32 steps × 96 dims). Adapter: linear
3072→3072 on the flattened image pre-encoder, identity-init, frozen per task.
Shift: **pixel permutation applied identically across channels** (preserves the
tested shift class; channel-permutation explicitly out of scope). Rotation variant
uses the Rotated-MNIST angles. Arms ON/OFF, seeds {42, 1337, 2024}, both shifts.

**AMENDED (v2, Finding 3) — capacity rule, not an entry condition.** Search hidden
size over **{256, 512, 1024}**, take the best clean-CIFAR-10 single-task accuracy,
**record it, and proceed regardless of whether 55% is reached.** Both arms always
share identical capacity, fixed before any continual run, never per-arm tuned.
*(v1 wrote a ≥55% gate; that contradicted the point of normalized retention, which
makes the hypothesis testable at any ceiling.)*

### E3 — CIFAR-10-C-style corruptions (OPTIONAL)
5 tasks = clean, Gaussian noise, blur, contrast, JPEG at fixed recorded severity —
genuinely non-linear input maps, the strongest extension of the Rotated finding.
Dropped without amendment if the timebox tightens; absence does not gate submission.

---

## 3. Hypotheses and thresholds (fixed before any number exists)

**Normalized retention** ≡ AVG / DIAG, per arm.
**AMENDED (v2, Finding 5):** the denominator is **DIAG** — mean diagonal accuracy,
i.e. each task measured immediately after training it. v1 said "mean single-task
accuracy at matched capacity"; these are **the same quantity**, and DIAG is already
measured per-arm in every run at no extra cost. Unified here so no future reader
treats them as different. MNIST reference: 0.9331 / 0.9779 = **0.954**.

- **H-G1a (ON-arm invariance):** within each new dataset (E1; E2; E5), ON-arm
  normalized retention ≥ **0.90**.
- **H-G1b (delta structure):** ON − OFF ≥ **+15pp** AVG on every new
  benchmark-shift pair.
- **Story-changing threshold:** if ON-arm normalized retention across all tested
  benchmark-shift pairs spans **> 5pp**, the claim is reframed from "invariance" to
  "consistent large gains" and the title's declarative register is re-examined.
- **H-G2 (mechanism):** ON-arm task-0 drift ≤ **⅓** of OFF-arm drift (cosine), **AND**
  the θ₀-probe retains ≥ **80%** of its original accuracy on θ₄ representations.

**Falsification branch (pre-named):** if the encoder drifts substantially in the ON
arm *yet retention stays high*, the "input-path gradient isolation" mechanism story is
**FALSIFIED as stated**. The paper softens to the empirical claim — *"routing task
variation to the input prevents forgetting; the protective mechanism is not reduced
drift"* — and the subtitle is revisited. **Reported with the same prominence as a pass.**

**Venue decision rule:**
- ICLR gate = H-G2 pass (or falsify-with-soften) ∧ H-G1 pass on **≥1 non-MNIST
  dataset (E2 or E5)**. A strong E5 + E1 + E4 suffices even if E2 is weak or
  unfinished — two data modalities including applied sensor data answers the
  dataset-family objection at least as well as a second image dataset.
- G1 passes (any non-MNIST), G2 falsifies → ICLR viable as an empirical paper;
  mechanism language softened throughout before submission.
- G1 fails on **both** E2 and E5 → ICLR off; scope the finding to demonstrated
  regimes in the CoLLAs/TMLR manuscript, plainly.

---

## 4. Locks

- Seeds {42, 1337, 2024}; mean ± std; **no claim at n=1**; hyperparameters fixed by
  the pre-registered capacity rule before any continual run, never per-arm tuned.
- The adapter is always: linear, input-dimension square, identity-init, trained with
  its task, frozen after. **No nonlinear or multi-layer adapters** in this package.
- **No mock simulations for outcome prediction** (standing rule, 4/4 failure record).
  Shape/gradient tests only.
- Priority: **E4 → E1 → E5 → E2 → E3 (optional)**. E4 is first *despite* no longer
  being cheap, because it is the only experiment whose outcome changes the *language*
  of everything else: if the falsification branch fires, the mechanism framing softens
  before the writing hardens around it. If the timebox forces a cut, **E2 is cut
  before E5**.
- Timebox: E4 ≈ 1.7h + analysis. E1 = 1 day. E5 = half a day (mostly data
  engineering). E2 = 2 days incl. capacity calibration. E3 only if all others complete
  inside 4 days. **Hard stop at the Aug 25 go/no-go regardless.**

## 5. Predictions on record (before numbers)

- **E1:** ON arm ~90–94% raw AVG on both shifts; baselines drift-prone; adds a row,
  changes nothing.
- **E2:** ON-arm normalized retention ≥ 0.90 (raw AVG 45–65% depending on capacity);
  OFF arm collapses under permutation far below its ceiling. Most likely to surprise.
  **Confidence ~60%.**
- **E2 storage (AMENDED v2, Finding 2 — pre-named so it is our boundary, not a
  reviewer's gotcha):**

  | | full-rank adapters | ER 100/task | ratio |
  |---|---|---|---|
  | MNIST (d=784) | 12.3 MB | 1.57 MB | 7.8× |
  | **CIFAR (d=3072)** | **188.7 MB** | 6.14 MB | **30.7×** |

  The `O(T·d²)` penalty worsens **4×** at CIFAR scale — 47M adapter parameters against
  a ~1M-parameter encoder. This also makes the **low-rank question
  architecture-coupled**: rank-32 at d=3072 is a **96× compression**, against a
  deviation whose effective rank at d=784 was already 56–74 and where 24× compression
  cost −25.9pp. Expect low-rank to be *less* viable at CIFAR scale — paper 2's
  efficiency thread inherits a measured constraint, not a paper-1 problem.
- **E5 (HAR):** ON-arm normalized retention ≥ 0.90, **confidence ~75%** — higher than
  E2 because the backbone fits the data. OFF arm degrades under calibration shifts but
  possibly less dramatically than image permutation; **the Rotated/MLP hedge applies** —
  the delta may be modest because the *baseline* is easier, and ON-arm normalized
  retention is the valid number.
- **E4:** ON-arm drift ≪ OFF-arm drift. **Confidence ~70%** — the falsification branch
  is genuinely live, because this project's mechanism stories have been wrong twice
  before (`P_k⁻¹` inversion; variance-collapse as isolation evidence).

## 6. Reporting

One memo regardless of outcomes: the generality table (all benchmark × shift × arm
cells), normalized retention alongside raw, drift/CKA/probe per arm, branch taken per
hypothesis, deviations, supersessions visible. **If any story-changing threshold fires,
the paper edits it triggers are listed in the memo BEFORE the edits are made.**

## 7. Amendment log

- **v1:** initial draft.
- **v2 (five amendments, all approved pre-launch):**
  1. **E4 does not run on existing checkpoints** — audit showed ON arm at 1 seed and
     no matched OFF arm (`rbst5` differs in `task_heads`, a second variable).
     Corrected to 5 new runs, ~1.7h, matched arms. *Fifteenth catch; the confound
     class this project has caught four times in other designs.*
  2. **E2 storage prediction added** (30.7× at d=3072) and the low-rank question noted
     as architecture-coupled.
  3. **E2 capacity gate → measurement rule** ({256,512,1024}, record, proceed). The v1
     ≥55% entry condition contradicted normalized retention's purpose.
  4. **E5 timebox restated** as data-engineering-majority.
  5. **Normalized-retention denominator locked = DIAG**, and the two v1 phrasings
     unified as the same quantity.

- **v3 (post-E5b; amendment 6 — measurement discipline, not a hypothesis change):**
  6. **Reproduction-floor requirement, added to §4 Locks.** Relaunching six E5
     configs identically produced within-configuration spread of up to **5.5pp**
     (seed 42 bit-identical in both arms; seed 2024 +5.50pp ON and +5.12pp OFF).
     The pattern is **seed-correlated, not preemption-correlated** — catch 19's
     first diagnosis blamed a preemption and was retracted when the un-preempted
     OFF arm diverged on the same seed.

     **Consequences, binding on every remaining experiment:**
     - **No delta below the measured floor may be reported as signed.** Report it
       as null. E5's −0.95pp arm delta sits *below* its own noise floor and was
       never resolvable; seeding does not fix this, because the noise is
       within-configuration rather than across seeds.
     - **Every experiment measures its own floor before its deltas are
       interpreted**: relaunch a subset of configs identically, report the spread.
       Floors are not assumed to transfer between benchmarks.
     - **Relaunch both arms, never one.** An asymmetric relaunch cleans one side
       of a comparison and biases the delta in an unknown direction.

  7. **E2 capacity calibration extended with a noise-floor stage (this amendment's
     operative change).** The §2 E2 rule stands — search hidden ∈ {256, 512, 1024},
     take the best clean single-task CIFAR-10 accuracy, record it, proceed
     regardless of whether 55% is reached, both arms always at identical capacity.
     **Added: relaunch the winning capacity once and record the reproduction
     spread before any continual run starts.**

     *Why this is worth one run:* if d=3072 with an unsaturated encoder yields a
     floor of ~8–10pp, then **H-G1b's +15pp bar sits barely above noise** and E2
     cannot cleanly answer the gate *even if it passes*. Learning that costs one
     run instead of twelve. If the floor forbids a clean read, that is reported as
     a property of the experiment — not worked around.

  **Retroactive audit required before submission.** Every reported delta is
  re-checked against the floor. Triage: headline adapter deltas (+50.3, +35.2,
  +43.6, +25.1, +29.2pp) are 5–10× the floor and safe; **freezing's +4.1pp is at
  the floor and must be re-examined**; the memory machinery's −1.0pp is below it —
  which does not change that claim's substance ("contributes nothing") but does
  change its phrasing from a point estimate to **"indistinguishable from zero."**

  **Process-appendix line (verbatim, for the manuscript):** *within-configuration
  reproduction spread was measured at up to 5.5pp; deltas below that are reported
  as null rather than signed.*
