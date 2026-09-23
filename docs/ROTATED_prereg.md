# Pre-Registration: Rotated MNIST — the invertibility test

**Status:** Locked before any Rotated run exists. This is the only remaining
experiment that can change what the paper *claims*, so the interpretation —
including the title — is fixed here, not after the numbers.

---

## 1. Why this benchmark

Permuted MNIST's task shift is a **permutation**: an exactly invertible linear
map on pixels, so a linear input adapter can represent `P_k⁻¹` **exactly**. That
is the most adapter-favourable shift that exists, and prereg risk R1 says the
+50.3pp result may be an artifact of it.

Rotation is also linear in pixel space (a resampling matrix) but **lossy**.
Measured on this benchmark: rotating +67.5° then −67.5° returns an image with

    mean absolute error = 0.02294  =  24.9% of the signal

A permutation round-trips to *exactly* 0. So no linear adapter can invert a
rotation, and this is a real test rather than a replication.

## 2. Design (identical to the attribution table, one variable)

- Config `configs/mafc_phase1.yaml` (5 tasks × 10 epochs, unfrozen encoder,
  shared readout, task hints at eval), `--benchmark rotated`.
- Angles: 0° / 22.5° / 45° / 67.5° / 90°; task 0 = identity, matching Permuted.
- Arms: `adapters ON` vs `adapters OFF` — **only `adapters.enabled` differs**.
- Seeds {42, 1337, 2024}, mean ± std. 6 runs.

**The number that matters is the DELTA between arms, not the adapter arm's
absolute score** — see the hedge in §4.

## 3. Outcomes → what the paper becomes (fixed now)

| outcome | criterion (Δ AVG, adapters on − off) | claim | title |
|---|---|---|---|
| **1** | ≈ +50pp (within ~5pp of Permuted's +50.3) | generalizes to input-space shifts broadly; R1 largely retired | declarative main-track: *"Forgetting Is an Input-Space Problem…"* |
| **2** | substantially positive but reduced (~+20–35pp) | mechanism efficacy is **gated by invertibility** — a law, not a number | characterization: *"When Input Adapters Suffice…"* |
| **3** | ≈ 0 (< ~10pp) | Permuted result was an artifact of exact invertibility | scope narrows to permutation-family shifts; CoLLAs register; the limitation section becomes load-bearing |

**Outcome 2 is plausibly the strongest paper.** A gating law — mechanism
efficacy as a function of shift invertibility — is a *finding*; "+50pp again" is
a replication. The 24.9% measured signal loss is outcome 2's interpretive
anchor: recovering, say, +30pp against a shift that destroys a quarter of the
signal under linear inversion is a quantified relationship, not merely a
smaller number.

## 4. Pre-registered interpretation hedge (task similarity)

Rotations at 22.5° increments make **adjacent tasks more similar to each other**
than Permuted's tasks are — random permutations are maximally dissimilar; small
rotations are not. Therefore:

- If the **no-adapter control forgets less** on Rotated than on Permuted
  (control AVG > 0.4304), that is **task similarity, not adapter magic**, and it
  must not be read as evidence for the mechanism.
- The comparison is therefore the **between-arm delta**, which cancels the
  shared task-similarity effect. Absolute scores are context only.
- Report both arms' absolute numbers alongside the delta so the reader can see
  the similarity effect rather than having it silently inflate the headline.

## 4a. AMENDMENT (post-result) — spec error nine

**Outcome 2's *criterion* was met (Δ = +29.2pp, in the 10–35pp band); outcome
2's *label* was wrong.**

The outcome→title mapping assumed a reduced delta could only arise from adapter
degradation. It did not: the ON arm moved **+0.35pp** across the two benchmarks
while the OFF arm moved **+21.4pp**. The delta reduction is fully attributable
to baseline relief, which §4 of this very document anticipated but which was
never propagated into the outcome definitions.

Consequences, ruled (iii):
1. Report the pre-registered verdict, then the decomposition, then the
   correction — the pre-registration is honored *and* corrected in public.
2. **"Gating law" is retired.** The finding is **invariance**: the adapter arm
   scores ~93.5% whether or not the shift is linearly invertible.
3. The motivating theory ("a linear adapter represents `P_k⁻¹` exactly") is
   falsified **as an explanation**, not as a result. The mechanism is per-task
   learned preprocessing + gradient isolation.
4. The optional third benchmark becomes a **boundary hunt** (elastic/nonlinear
   warp; predicted: ON arm flat ~93%), not a curve point.

## 5. Reporting

Same memo discipline: both arms n=3, mean ± std, the delta, the outcome branch
taken, and any deviations. Result lands in paper one's central figure as the
generality row.
