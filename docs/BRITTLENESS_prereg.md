# Pre-Registration: Architecture Brittleness Control (MLP backbone)

**Status:** Locked draft, awaiting sign-off. No MLP code exists at time of
writing. **This is the last remaining item that can kill a claim rather than
defend one**, which is why it runs first.

**INTERPRETATION FLAG (resolve at sign-off):** the spec "frozen-LSTM-vs-MLP
brittleness control" was never written out in full. This document drafts it as:
*swap the recurrent backbone for a feed-forward one, hold everything else fixed,
and test whether the +50.3pp adapter effect is a property of the mechanism or of
the LSTM.* If the intended control was narrower, amend before launch.

---

## 1. The claim under threat

The paper's central sentence is architecture-general in its wording:

> *forgetting is an input-space problem; per-task input adapters prevent nearly
> all encoder-mediated forgetting.*

Every measurement supporting it was taken on **one backbone** (a 256-unit
single-layer LSTM). If the effect is LSTM-specific, the sentence must be
rewritten to scope to recurrent models — a materially smaller paper. Nothing
else on the board can do that.

## 2. Design (single variable, matched to the attribution table)

MLP backbone replacing the LSTM; **everything else held fixed**: per-task input
adapters (full-rank, identity init), unfrozen backbone, shared readout, no
memory, 5 tasks × 10 epochs, task-incremental eval, seeds {42, 1337, 2024}.

- **MLP:** 784 → 256 → 256 → readout, ReLU. **Measured parameter counts
  (recorded before the run, not tuned):**

      LSTM backbone : 292,864   (total model 2,480,017)
      MLP  backbone : 266,752   (total model 2,453,905)

  Within 9% on the backbone and 1% on the total — close enough that a result
  cannot be attributed to capacity. Not tuned further; the topologies differ.
- **Arms:** `adapters ON` vs `adapters OFF` — only `adapters.enabled` differs,
  exactly as in the LSTM attribution control.
- **Reference (already measured, LSTM):** OFF 0.4304 ± 0.0109, ON 0.9331 ±
  0.0099, **Δ = +50.3pp**.

## 3. Pre-named branches (fixed before the number exists)

| branch | criterion (Δ AVG, MLP) | reading |
|---|---|---|
| **(a) generalizes** | ≥ +40pp | the mechanism is architecture-general. Central claim stands as written; the paper gains a second backbone and the "input-space problem" framing is earned rather than asserted. |
| **(b) attenuated** | +15 to +40pp | present but backbone-dependent. Claim must state the measured range across architectures; the LSTM number stops being *the* result and becomes the best case. |
| **(c) LSTM-specific** | < +15pp | **claim-killing.** The effect is a property of recurrent processing, not of input-space adaptation. Title and abstract rewrite to scope to RNNs; the "forgetting is an input-space problem" framing does not survive. |

**Commitment:** branch (c) is reported with the same prominence as a positive
result. This control exists precisely to be able to fire.

### 3a. Branch-(c) reinterpretation clause (added pre-launch)

Branch (c) is **not** interpretable in isolation. A small adapter delta on an
MLP admits two readings that this design alone cannot separate:

1. *the mechanism isn't architecture-general* (brittleness of the mechanism), or
2. *the mechanism isn't needed here* — the MLP backbone barely forgets, so there
   is little for adapters to buy.

**Discriminator:** arm B (frozen MLP transfer) of the frozen-feature control,
`docs/BRITTLENESS_frozen_prereg.md`, which runs FIRST for exactly this reason.
If frozen MLP features already transfer well, reading (2) is the default and
branch (c) must be reported as *"adapters add little where the backbone has a
free input layer AND low baseline forgetting"* — **not** as evidence that the
mechanism fails to generalize. Cite the measured arm-B number in any (c)
writeup.

**GRU held in reserve.** A third recurrent arm would distinguish
"recurrent-general" from "LSTM-specific", but it is only warranted if branch (b)
or (c) fires and that distinction becomes load-bearing. Not spent preemptively.

## 4. Pre-registered interpretation hedge (architecture changes the task)

An LSTM reads the image as 28 timesteps of 28 pixels, so a permutation scatters
information **across timesteps** — maximally destructive to a recurrent
accumulator. An MLP sees all 784 inputs simultaneously, so a permutation is a
reordering of input units that the first layer can partly absorb. **Therefore
the MLP no-adapter baseline may forget substantially less than the LSTM's
0.4304 for reasons that have nothing to do with adapters.**

Consequences, fixed now (same hedge that fired on Rotated):

- The comparison is the **between-arm delta within the MLP**, which cancels the
  shared architecture effect. Absolute scores are context only.
- If the MLP OFF arm is much higher than 0.4304, a smaller delta is **expected**
  and must not by itself be read as brittleness — the ceiling moved.
- Report both arms' absolute numbers alongside the delta so the architecture
  effect is visible rather than silently compressing the headline.

## 5. Pre-stated risks

- **R1:** an MLP first layer (784→256) is itself a per-input-unit linear map, so
  an input adapter is partially redundant with it in a way it is not for an
  LSTM. This is a real confound and is *why* the control is informative — but it
  means branch (c) would not prove adapters are useless generally, only that
  they add little where the backbone already has an unconstrained input layer.
  State this in any (c) writeup.
- **R2:** parameter counts cannot be exactly matched across topologies. Report
  both; do not tune.
- **R3:** single benchmark (Permuted MNIST). A (a)-branch result generalizes
  across *architectures*, not across shift types — Rotated covers that axis.

## 6. Reporting

Memo: both arms n=3, mean ± std, delta, branch, deviations. If (a) or (b), the
row joins the central figure as the second-backbone evidence. If (c), it becomes
the paper's scope section and the title changes.
