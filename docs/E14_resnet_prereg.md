# Pre-Registration: E14 — ResNet-50 on Split-CIFAR-100
## Is reader-dominance transformer-specific?

**Status: DRAFT for sign-off.** New training on a new architecture family.
Numbered E14; E13 is the tier-0 stack / cross-regime experiment.

**The one claim this experiment buys:** *reader-dominance is not
transformer-specific among pretrained backbones.* Nothing more. What it
deliberately does **not** buy is stated in §8 and handed to paper 2.

**Why it is worth a day.** E12 measured reader share 91.8% on a pretrained
ViT-B/16 — above the 66–88% of small scratch-trained encoders. The reviewer
objection that costs us the generality claim is *"this is a ViT quirk."* One
convnet answers it. Every outcome is informative: replication widens the claim to
pretrained backbones, and failure is a genuine architecture-family boundary with
a mechanism attached, not a void.

---

## 0. Preamble — every new backbone re-earns the checks

E12's contract review found the program's three canonical error classes
**re-instantiating on first contact with a new architecture**, and execution then
found three more (catches 30–32). The standing rules transferred the *lessons*
and none of the *checks*.

**A rule learned on a ViT is a hypothesis about a ResNet until re-derived there.**
In particular P3a/P3b are architecture-fresh premises: the ResNet readout path
(global average pool → fc) has nothing in common with CLS-post-norm except that
both are called "the feature", and that word is not a verification.

---

## 1. Setup

- **Backbone:** ResNet-50, ImageNet-pretrained (timm), **trunk fully trainable**.
- **Benchmark:** Split-CIFAR-100, **20 tasks × 5 classes** — the same registered
  split, class order, and fingerprints as E12 (`4da9de3736a6` / seed-keyed).
- **Setting:** **task-IL only.** Class-IL is not re-run: E12 established it as
  corroborative-not-extending, and the boundary question does not need it.
- **Seeds:** 3 (42, 1337, 2024 — the registered set; controls in E12 used these).

### Arms

| arm | trunk | adapters | seeds | purpose |
|---|---|---|---|---|
| **base** | trainable | off | 3 | the decomposition (H-R1) |
| **frozen-probe** | frozen | off | 3 | P1 reference + the trivial-use baseline (catch 24) |

**No adapt arm.** H-V2 was untestable at E12's placement and this experiment does
not re-open it; adding an untestable arm would spend a third of the budget on a
question the contract cannot answer. Stated as a scope decision, not an omission.

### TASK COUNT — a deviation from the verbal scope, flagged for ruling

The verbal scope said "5-task split suffices." **This contract specifies 20
tasks**, because the number it must be compared against — E12's 91.8% — was
measured at 20. A 5-task ResNet answers "is reader-dominance present in a
convnet" but not "does it hold at the scale where we measured it," and closing
that gap later costs a ViT re-run at 5 tasks. ResNet-50 (~25M params) is cheaper
per image than ViT-B/16 (~86M), so 20 tasks is affordable at 3 seeds.
**If the ruling prefers 5, the claim narrows accordingly and the contract says so.**

---

## 1b. Build plan — each item with its verification

**No training run enters the record until all are green.**

| # | item | verification |
|---|---|---|
| C1 | ResNet-50 weights, hash-pinned | baked at image-build; **checkpoint hash recorded and asserted at container start** — the hash, not file presence (the E12/B2 rule) |
| C2 | `PLCM.backbone == "resnet"` path | forward-parity: the reimplemented feature path is **bitwise** timm's own `forward_features`→`global_pool`; shapes correct; trunk grads flow; loss moves |
| C3 | readout binding | P3a/P3b green on every arm × task cell with **both positive controls fired** (§2) |
| C4 | data reuse | B1's CIFAR-100 module used unchanged; class-order fingerprint **gated**, not printed |

C4 is deliberately a reuse item rather than a build item: **the audited loader is
used, not re-implemented** (catch 32).

---

## 2. Preconditions

**P0 — build gate.** C1–C4 green; fingerprint set (class order, ResNet weight
hash) computed in the execution environment.

**P1 — competence, scale-free.** Frozen-probe per-task DIAG must clear **0.85**
(else the pairing is void, the E2/CIFAR rule); the trainable trunk must land
**within 5pp** of the frozen reference. The absolute bar alone is close to
unfailable on a pretrained backbone — the reference is what makes the gate real.

**P2 — forgetting exists and is sequence-caused.** Task-0 deployed accuracy falls
**≥ 0.15** below its own ceiling, **plus** a repeat-task control (5 tasks of
identical data, fresh heads) at **≤ 0.05**.

**P3 — instrument exactness, RE-DERIVED for avgpool→fc.**

- **P3a — capture-point identity.** `head(captured_feature) == deployed_logits`,
  **max |Δ| = 0**, every cell. The deployed feature is the **post-global-average-
  pool, pre-fc** 2048-d vector.
  *Positive control:* capture the **pre-pool** spatial map (mean over channels,
  or the flattened map projected) — the assert must **FAIL**.
- **P3b — selection semantics.** On task *k* the readout module used **is head
  *k***, by module identity.
  *Positive control:* substitute the latest head — must **FAIL**.

**P4 — label alignment (catch 32).** Feature extraction uses a **sequential**
loader; a bitwise check that loader position *i* == `dataset[i]` prints in the
precondition block. **No probe number is read without this line.**

Any of P0–P4 failing → cells print ABSENT and no channel claim is read.

---

## 3. Measurements

Per seed × old task *k* at θ_T: `acc_ceiling`, `acc_orig`, `acc_refit`,
`acc_refit_ceiling`; then `R`, `F_enc`, `F_read`, `F_total` with the identity
`F_enc + F_read − R = F_total` printed per cell.

**Storage:** era checkpoints, **fp16, one per task boundary, audited by loading
at save time**; boundary-time values authoritative; the boundary-vs-reload delta
measured and reported (E12 measured 0.00000 — the expectation, not the
assumption). ResNet-50 fp16 ≈ 50MB × 20 × 3 ≈ **3GB**.

**Protocol locks inherited without amendment:** per-cell guard applied per cell,
never post-pooling; mean-of-ratios with CIs; pooled |R| ≤ 0.05 per arm; verdicts
computed from printed arrays; **reproduction floor measured on this path before
any delta is interpreted** — E12's floor was ~0.4pp aggregate but **~15pp
per-cell**, so pooled-with-CI is primary here too.

---

## 4. Hypotheses

- **H-R1 (PRIMARY).** Reader share `F_read/(F_enc+F_read)` **≥ 0.50** pooled on
  the base arm, CI lower bound above 0.50.
  *Derivation:* the same bar H-V1 used; 0.50 is where "the reader carries the
  majority" is the honest sentence.
- **H-R2 (descriptive, no gate).** Report ResNet's share against ViT's 91.8% and
  the LSTM/MLP 66–88%. Whether the arc continues, plateaus, or inverts is
  reported as measured — **no bar, because no prior measurement constrains it**
  (the dynamic-range rule: a bar without a demonstrated range is unreadable).
- **H-R3 (descriptive).** F_enc absolute magnitude vs ViT's 0.060 and the LSTM
  range 0.052–0.112.
- **H-R4 (descriptive, added post-E15).** Print the **rank-control-style span
  diagnostic** per task: build era-prototype prototypes, project the θ_T feature
  onto their span, and report (i) the projection's cost at the *era* checkpoint
  and (ii) its recovery at θ_T.
  *Why:* E15 established **span rotation under stable features** — the ViT's
  era-prototype span is information-preserving (−0.0015) and its features barely
  move (F_enc 0.06), yet projection recovers 0.4pp of a 67pp gap. That phenomenon
  currently has **one architecture's measurement**; a second costs nothing once
  the projection machinery is wired.
  **No bar, descriptive only**, per H-R2's register — no prior measurement
  constrains where a convnet should land.

---

## 5. Branches

- **(G) any precondition fails** → report which and stop.
- **(A) H-R1 MET** → reader-dominance is **not transformer-specific among
  pretrained backbones**; paper 1's generality claim spans two pretrained
  families plus two scratch-trained ones.
- **(B) H-R1 NOT MET** → a genuine **architecture-family boundary**: the
  diagnosis holds for transformers and small recurrent/feedforward encoders but
  not convnets. Reported at full prominence; paper 1 states the boundary and the
  ViT result stands scoped.
- **(C) H-R1 met but F_enc materially larger than ViT's 0.060** → both channels
  present; the *degree* of reader-dominance is architecture-dependent even where
  its sign is not. Its own paragraph.

---

## 6. Predictions on record

H-R1 MET: **~75%.** Reasoning: the mechanism is a frozen-era readout over a
drifted trunk, which has nothing transformer-specific in it, and the frozen-probe
literature works on convnets for the same reason it works on ViTs. Against that:
convnet features are more spatially local and less linearly separable at the
penultimate layer than a ViT's CLS token, so the refit could recover less.

Share ≥ 0.85: ~45%. F_enc ≤ 0.15: ~70%.

*Calibration:* this program's magnitude predictions in unfamiliar regimes run
directionally right and too narrow — five misses, then two hits, one of which
followed 11 visible cells. Treat the interval, not the point.

---

## 7. Locks

- Class order and weight hash frozen with fingerprints **computed in the
  execution environment**.
- One capacity/lr decision permitted for P1, made at n=3, recorded, applied
  identically to both arms.
- Audit by **loading**, not existence, before any analysis.
- Analysis runs where the checkpoints were built.
- Provenance: memo numbers from recorded script output only.
- Ledger edit drafted in the memo first, applied second.

---

## 8. Limitations — stated here, not discovered by a reviewer

**This experiment does not isolate pretraining from scale from architecture.**
E12's monotone arc (66–88% scratch-trained small → 91.8% pretrained ViT → 97.9%
shared-head) is *ordered along* the pretraining axis, but E14 varies architecture
family while holding pretraining fixed. A positive result therefore supports
*"reader-dominance is general among pretrained backbones"* and **not** *"pretraining
causes reader-dominance."*

**The honest answer to that confound is a factorial** — scratch-ViT,
scratch-at-scale, pretrained-small — which costs roughly a week and **belongs to
paper 2's opening**, where *what orders the arc?* is the research question
motivating absorption-control interventions rather than a loose end in a
diagnosis paper.

**Why not run the from-scratch ViT here instead.** It is the cleaner
discriminator on the pretraining axis, but it risks the E2/CIFAR competence
failure — ViTs are data-hungry and a from-scratch ViT on 5-way CIFAR-100 splits
may not clear P1's DIAG bar, returning **void** rather than evidence and burning
the day. ResNet's failure modes are all informative; that asymmetry decides it.
