# Pre-Registration: Adapter-Anchored + Bank-Anchored Functional Consistency (MAFC)
## Component One of the Persistent Cell Memory Program

**Status:** Locked contract, **v2**. Amendments require explicit sign-off and go
on the record (§9). No mechanism code existed at v1 lock.

**Naming (v2):** the unifying label "memory-anchored" is *earned only if H4
passes*. Until then the mechanism is named for what it provably is: a
**two-term** loss with two distinct anchors — **adapter-anchored** (encoder) and
**bank-anchored** (readout).

**Question:** can coupling encoder evolution to memory legibility (requirements
3+6) recover continual-learning accuracy while the encoder stays unfrozen?

**Deliberately NOT answered:** task-free routing (component two) or
retrieval-composition gains (component three).

---

## 1. Mechanism under test (v2 — two terms, two anchors)

Training on task t, with per-task input adapters A_k, a **shared** readout φ,
and parameter snapshots at each prior task boundary:

    L = L_task(θ, φ, A_t)  +  λ · [ L_enc  +  L_read ]

**Encoder term — ADAPTER-anchored (requirement 3, constrains θ and φ):**

    L_enc = (1/|K|) Σ_{k<t} E_{x~batch} [ KL( p*_k(x) ‖ g_φ(f_θ(A_k(x))) ) ]

    p*_k(x) = ( g_φ(k) ∘ f_θ(k) ∘ A_k )(x),  frozen end-of-task-k snapshot

Probe inputs are current-task batch inputs passed through OLD adapters. Zero
raw data stored. Targets are generated on the fly from snapshots — **the bank is
not and cannot be the source here** (see §1.1).

**Readout term — BANK-anchored (requirement 6, constrains φ only):**

    L_read = (1/|B|) Σ_{(c,γ,z*) ∈ bank} KL( z* ‖ g_φ(γ ⊙ tanh(c)) )

over the bank's committed triples (stored cell state `c` = `values`, stored
output gate `γ` = `gate_values`, stored logits `z*` = `logits`). This is the
genuinely bank-sourced term: the memory's own committed predictions pin the
reader.

Both KLs are **forward** KL (teacher ‖ student) at temperature T = 2 —
cross-entropy against soft targets, standard LwF form (v2 Amendment 4).

**What trains:** θ (encoder, deliberately UNFROZEN), φ (**shared** readout —
per-task heads are DISABLED for this component, else φ cannot drift and L_read
is vacuous), A_t (current adapter). Old adapters A_k frozen.

### 1.1 Why the decomposition is forced (v2, Amendment 1)

A stored bank vector is a **constant with respect to θ**; the gradient path from
any loss on bank contents to the encoder is severed. Therefore no bank-sourced
loss can ever constrain encoder drift. Requirements 3 and 6 do not share an
anchor:

| requirement | drifting object | anchor | term |
|---|---|---|---|
| 3 — encoder drift | θ (and φ) | old adapters + snapshots | `L_enc` |
| 6 — readout drift | φ | **bank** committed triples | `L_read` |

**Honesty clause (v2).** `L_enc` is Learning-without-Forgetting with one delta:
probe inputs are mapped through the system's own per-task adapters rather than
raw inputs (H3 tests this). `L_read` is the only place the memory system
supplies training constraint (H4 tests this). If H3 and H4 both fail, the
mechanism is plain LwF and the program's framing does not survive.

---

## 2. Hypotheses (fixed, falsifiable)

DIAG = mean diagonal accuracy; RET = mean final accuracy on tasks 0..T-2;
AVG = mean final accuracy over all tasks.

**Phase 0 (micro, 2 tasks, 3 epochs, seed 42, ~10 min):**
- **H0a:** ∃λ where Task-0 retention after Task-1 exceeds the λ=0 control by ≥ 15pp.
- **H0b:** at that λ, Task-1 diagonal within 3pp of the λ=0 diagonal.
- Gate: H0a ∧ H0b → Phase 1. ¬H0a → mechanism dead; stop, memo. H0a ∧ ¬H0b →
  proceed only if RET gain ≥ 2× DIAG loss.

**Phase 1 (full, 5 tasks, 10 epochs, seeds {42, 1337, 2024}):**
- **H1:** best-λ MAFC AVG ≥ 70%. **(v2 Amendment 3: if M-λ0 alone ≥ 55%, the
  bar becomes M-λ0 + 15pp, fixed now rather than after seeing the number.)**
- **H2:** best-λ MAFC ≥ **re-run EWC-λ200-at-this-commit-with-adapters-and-hints**
  by ≥ 10pp. (v2 Amendment 2; the historical 49.5% is cross-commit context only.)
- **H3:** MAFC ≥ M4-LwF (raw probe inputs, no adapter mapping) by ≥ 5pp AVG.
  Isolates the adapter anchor.
- **H4 (v2, now real):** full MAFC (both terms) ≥ encoder-term-only by ≥ 3pp AVG
  **or** measurably lower across-seed retention std. Isolates the bank's
  contribution. Failure ⇒ the mechanism is adapter-anchored only and the
  "memory-anchored" name is not earned.

**Decision rule:** H1 ∧ H2 ∧ H3 → component one validated; proceed to component
two; 85% becomes the COMBINED system's target. H1 ∧ H2 ∧ ¬H3 → LwF-in-disguise;
useful, novelty reduced. ¬H2 → stop the program branch. H2 ∧ ¬H1 → run §6
diagnostic before building anything further.

---

## 3. Why 70% and not 85%

Unchanged from v1. Component one controls encoder+readout drift only; it does
not include routing or retrieval-composition. 70% ≈ midway between the 47.6%
floor and the 97.7% task-hint ceiling, minus margin. 85%+ is the FULL stack's
pre-registered target. Exceeding 85% here is a bonus finding, reported as such.

---

## 4. Arms and baselines (identical seeds/data)

| arm | encoder term | readout term | probe inputs | purpose |
|---|---|---|---|---|
| **M-MAFC** | ✓ | ✓ | A_k(x) | the mechanism |
| **M-MAFC-enc** | ✓ | — | A_k(x) | H4 control (bank's contribution) |
| **M4-LwF** | ✓ | ✓ | raw x | H3 control (adapter anchor) |
| **M-λ0** | — | — | — | λ=0 floor |
| **M-EWC200** | weight-space Fisher | — | — | H2, re-run at this commit |

λ ∈ {0.1, 0.5, 1, 2, 5}; Phase 0 sweeps {0.5, 2, 5}.

**v2 Amendment 3:** M-λ0 is a FRESH run — unfrozen encoder + adapters + hints +
λ=0. The v1 reuse clause is struck; no existing run qualifies (every adapter run
froze the encoder). M-λ0 is an unmeasured configuration and its value is the
true floor H1 is judged against.

Component one is evaluated **task-incrementally (hints allowed)**; task-free
routing is component two. All arms share this setting so comparisons are valid.

---

## 5. Protocol locks

Real Permuted MNIST only. **No mock simulations of learning dynamics** (4/4
historical failure rate); mocks permitted ONLY for shape/gradient-flow unit
tests. Seeds: Phase 0 {42}; Phase 1 {42, 1337, 2024}, mean ± std. Checkpointing
per epoch. λ values fixed. KL temperature 2.0, **forward** direction. Snapshot
storage ~2.5 MB × (T−1). Timebox: Phase 0 half a day incl. harness; Phase 1
2 days. Blocked past timebox → report state and stop.

---

## 6. Diagnostic decomposition (run only under H2 ∧ ¬H1)

Measure separately at final horizon: readout drift on bank states (≈0 expected —
sanity), encoder drift E‖f_θ(T)(A_k(x)) − f_θ(k)(A_k(x))‖ per k, and residual.
Distinguishes capacity interference (shared θ) from anchor coverage (probe
distribution ≠ true old distribution). Gates any next step; licenses none.

---

## 7. Reporting

One-page memo regardless of outcome: λ-tradeoff curve (RET vs DIAG, all arms),
decision branch, deviations. Negative → paragraph in paper one's discussion (a
seventh documented failure mode). Positive → component two pre-registration
drafted BEFORE component two code.

---

## 8. Pre-stated risks

- **R1:** probe distribution is benchmark-favorable for *permutation* shifts;
  does not transfer to semantic shifts. Scope-limited claim.
- **R2:** zero raw data, but not zero stored parameters (snapshots). LwF shares
  this. Disclosed.
- **R3:** if the λ curve is V-shaped between adjacent values, ONE refinement
  sweep (max 3 values) is pre-authorized; logged as protocol note.
- **R4:** shared-θ interference may impose a ceiling no λ escapes → §6.
- **R5 (v2):** with per-task heads disabled, the shared readout is a weaker
  classifier than the per-task-head configuration; part of any MAFC-vs-frozen
  gap may be attributable to this rather than to drift. Noted so it is not
  misread later.

---

## 9. Amendment log

- **v1:** initial lock.
- **v2 (all four signed off pre-code, after contract review):**
  1. **H4 was not instantiable.** `p*_k(x)` must come from snapshots (x is a
     current-task input that did not exist at task k), and no loss on stored
     bank vectors can constrain θ (constants w.r.t. θ). Delta (ii) of v1 was
     structurally unavailable. Resolution: split into `L_enc` (adapter-anchored,
     encoder) + `L_read` (bank-anchored, readout); H4 now tests the bank term's
     real contribution; doc renamed to reflect the dual structure.
  2. **H2's 49.5% baseline was not reproducible at this commit** (pre-Path-3;
     current code gives 0.4271 unfrozen / 0.3707 selective-λ200) **and was
     confounded** (MAFC has adapters+hints, the EWC run had neither).
     Resolution: re-run EWC-λ200 at this commit with adapters+hints.
  3. **M-λ0 reuse clause struck** — no existing run is unfrozen+adapters.
     Fresh run required; H1's bar shifts to M-λ0+15pp if M-λ0 ≥ 55%.
  4. **KL direction was reversed** (student‖teacher). Typo, not design.
     Corrected to forward KL, matching the "standard distillation" claim.

- **v3 (escalation, signed off after Phase 0, before Phase 0′ code/run):**

  **Trigger.** Phase 0 returned formal **¬H0a** — but *unsatisfiably*, not by
  failure: the λ=0 control retained **95.9%** after one transition, leaving
  <5pp of headroom against a +15pp bar (bar = 110.9%, impossible). Same
  inconclusive-by-insufficient-drift shape as the RBST short chain; the 2-task
  micro-gate cannot test a cure when the patient is not sick.

  **Phase 0 did validate** (10 min, seed 42): harness end-to-end; the loss
  engages without wrecking the diagonal; RET rises monotonically in λ
  (+0.65/+0.75/+0.81pp) while DIAG falls monotonically (−0.06/−0.02/−0.43pp),
  so the tradeoff curve exists and bends the correct way; forgetting hits
  **exactly 0.0000** in all three MAFC arms vs 0.0068 for control. Faint, right
  shape.

  **Phase 0′ (one escalation, pre-specified).** Run the FULL 5-task, 10-epoch
  **M-λ0 control alone, one seed (42), BEFORE any MAFC arm.** It answers the
  question everything now hinges on: how much forgetting does
  unfrozen+adapters actually exhibit at scale? (Historical unfrozen runs
  *without* adapters retained 29–59% on Task 0; the Phase-0 control's 95.9%
  suggests adapters may absorb most drift pressure by routing each task's
  gradient through its own input map.)

  **Pre-written branches (fixed before the number exists):**
  - **(a) M-λ0 retention ≤ 70%** → drift gap is real at scale; Phase 1 proceeds
    exactly as contracted, H1 measured against this control per the v2
    Ruling-3 amendment.
  - **(b) M-λ0 retention ≥ 85%** → MAFC is solving a solved problem. The
    finding becomes *"per-task input adapters alone largely prevent
    encoder-mediated forgetting"* — a result in its own right, goes into
    paper one; the program re-derives what component one should be.
  - **(c) intermediate (70–85%)** → Phase 1 proceeds, with the honest note that
    the addressable gap is the **measured** one, not the imagined 47.6→97.7
    span.

- **v3 — BRANCH (b) EXECUTED** (Phase 0′ result, seed 42, 5 tasks × 10 epochs):

      M-λ0 control:  RET = 93.95%   AVG = 94.70%   DIAG = 97.76%   forgetting = 0.0383

  RET ≥ 85% ⇒ branch (b). **Component one as conceived is dissolved.** The
  addressable gap is 94.70 → 97.67 (≈3pp), not the theorized 47.6 → 97.7.

  *Consequence for H1:* under the v2 Ruling-3 amendment the bar becomes
  94.7 + 15 = **109.7%** — unsatisfiable, the same ceiling failure as H0a one
  phase earlier. Phase 1 as contracted cannot discriminate and is NOT run.

  *Finding promoted to paper one:* **per-task input adapters alone prevent
  nearly all encoder-mediated forgetting on input-space shifts; freezing the
  encoder contributes only ≈3pp.** Mechanism: each task's gradients reach the
  shared encoder only through that task's own input map, so the encoder sees
  near-identically-distributed streams and has little task-specific structure to
  overwrite — gradient isolation at the input, not weight protection at the core.
  This corrects a misattribution in the project's own headline result (the win
  was credited to the frozen core; it belongs to the per-task input paths).

  *Requirement 3 is re-derived:* "couple encoder evolution to memory legibility"
  is satisfiable by **input-path isolation**, not only by anchoring.

  *Open, promoted to the front of the queue:* why is the residual ≈3pp there at
  all, and does the bank-anchored readout term (`L_read`, the one honest memory
  mechanism in the design) account for any of it?

  *Outstanding confound (logged, not yet resolved):* the 46.4% no-adapter
  comparison differs from M-λ0 in three ways at once — adapters, task hints, and
  per-task heads vs shared readout. A single-variable control (M-λ0 config with
  `adapters.enabled=false`, everything else identical) is required before the
  "adapters are the active ingredient" claim is made in print.

*Process note: three contracts, eight spec errors caught pre-run, zero
discovered post-hoc.*
