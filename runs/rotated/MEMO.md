# Rotated MNIST — Result Memo (the invertibility test)

**Status:** Complete. 6 runs, both arms n=3, single variable (`adapters.enabled`).
Reported per ruling **(iii)**: the pre-registered outcome rule is honored *and*
corrected in public.

---

## 1. Pre-registered verdict (reported first, as contracted)

The locked rule mapped the between-arm delta to an outcome band:

    Δ AVG = +29.2pp ± 0.7   (per-seed +29.9, +29.4, +28.3)
    Permuted MNIST was +50.3pp
    → falls in the 10–35pp band → OUTCOME 2 ("gating law")

## 2. …and why that label is wrong

| arm | AVG | RET | DIAG | forgetting | T0 end |
|---|---|---|---|---|---|
| adapters **ON** | 0.9366 ± 0.0101 | 0.9253 ± 0.0126 | 0.9796 | 0.0538 | 0.8237 |
| adapters **OFF** | 0.6446 ± 0.0100 | 0.5586 ± 0.0124 | 0.9881 | 0.4293 | 0.1945 |

**Decomposition of the delta reduction:**

| | adapters OFF | adapters ON | delta |
|---|---|---|---|
| Permuted | 0.4304 | 0.9331 | +50.3pp |
| Rotated | 0.6446 | 0.9366 | +29.2pp |
| **change** | **+21.4pp** | **+0.35pp** | −21.1pp |

**The adapter arm did not move (+0.35pp across two benchmarks). The entire
21.1pp delta reduction is attributable to the un-adapted baseline becoming
easier (+21.4pp).** The pre-registered task-similarity hedge (§4 of the
Rotated prereg) fired exactly as written: 22.5° neighbours are far more alike
than random permutations, so the control retains much more.

Efficacy was not gated. The baseline improved.

## 3. The finding: invariance

**The adapter arm scores ~93.5% regardless of whether the task shift is linearly
invertible.** Permuted MNIST's shift is a permutation — *exactly* invertible by
a linear map. Rotation is linear but lossy: measured on this benchmark, a
+67.5°/−67.5° round trip destroys **24.9% of the signal**, so no linear adapter
can invert it. The mechanism is indifferent to the difference.

## 4. Spec error nine — what died was the EXPLANATION, not the result

The design was motivated by: *"the task shift is a pixel permutation, which is a
linear operator, so a linear adapter A_k can represent the correction exactly."*

That theory is **falsified as an explanation.** The mechanism works equally well
where exact inversion is impossible. What it actually does is **per-task learned
preprocessing plus gradient isolation** — each task's gradients reach the shared
encoder only through its own input map. Corroborating evidence recorded earlier:
`adapter_dist_from_identity` grew steadily *even for task 0*, where identity was
already the optimal inverse — the adapters were never behaving as inverses.

**Sentence for the paper:** *we built it for the invertible case; it turned out
not to need invertibility.*

The pre-registration error itself: the outcome→title mapping assumed a smaller
delta could only mean adapter degradation. The §4 hedge anticipated baseline
improvement but that possibility was never propagated into the outcome
definitions. Caught by the decomposition, not by the rule.

## 5. Framing consequence

The mechanism evidence supports the **declarative register**. Headline sentence:
*"the adapter arm scores ~93.5% regardless of whether the shift is linearly
invertible."* The delta-band verdict is reported as the pre-registered check
that forced the close look which found the invariance — not as the result.

## 6. Board edit propagated

The earlier "gating law" language — and the proposed third invertibility point —
is now mis-aimed: the ON arm predicts a flat line, so plotting
efficacy-vs-invertibility measures nothing. A third benchmark remains worth ~1h
but its **purpose changes to a boundary hunt**: elastic deformation or a
nonlinear warp, with the pre-registered prediction that the ON arm stays ~93%
and the OFF arm lands wherever task similarity puts it. If the ON arm finally
drops, that locates the mechanism's true boundary — which the invertibility
theory failed to predict. **Recommended, not blocking.**
