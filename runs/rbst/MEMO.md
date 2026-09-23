# RBST Feasibility — Result Memo

**Status: COMPLETE.** Short chain (Tasks 0→1, seed 42) and escalated chain
(Tasks 0→4, seeds 42/1337/2024) both run. Verdict below; §§1–6 cover the short
chain, §7 the escalation and the final decision-rule branch.

---

## 0. VERDICT — ¬H1. Negative result. Stop.

Per the decision rule: *"¬H1 → recurrent state transport fails at this
granularity. Negative result; one section in the existing paper's future-work
discussion; stop."*

**Escalated chain, horizon 40 (mean ± std over 3 seeds):**

| condition | digit-probe acc |
|---|---|
| M0 — no transport | 0.346 ± 0.137 |
| M1 = M2 — transported | **0.158 ± 0.027** |
| M3 — snapshot | 0.129 ± 0.011 |
| UPPER — true era-t states | 0.785 ± 0.030 |

H1 required transport ≥ 0.70. It reached **0.158** — not marginal, a factor of
4.4 short, against a reachable ceiling of 0.785.

**The decisive result: transport lost to doing nothing in 12 of 12
seed×horizon cells**, by −5.8pp to −78.5pp. There is no cell, seed, or horizon
where the mechanism helped.

| seed | h10 | h20 | h30 | h40 |
|---|---|---|---|---|
| 42   | −62.8 | −19.7 | −30.6 | −10.5 |
| 1337 | −60.8 | −41.1 | −8.8 | −5.8 |
| 2024 | −78.5 | −55.8 | −19.0 | −40.1 |

*(M2 − M0 in pp; negative = transport actively harmful.)*

**H2′ ✗** (−1.7pp — the low-variance half is *worse* than the full set, the
opposite of the hypothesis). **H3 ✗** (ρ = −0.045 ± 0.026). Streaming vs
snapshot: +2.9pp for streaming, but both fail catastrophically, so this is not
a meaningful win for the streaming-novelty claim.

**One honest caveat on the H1 premise.** M0 at horizon 40 averaged 0.346, above
the < 0.25 the premise specifies — seeds 42 and 1337 essentially reached it
(0.245, 0.254) while seed 2024 did not (0.540, and non-monotonic: it read 0.287
at h30 before rising). So strictly, the "untransported falls to chance"
condition is met by 2/3 seeds, not all. This does **not** rescue transport:
drift did real damage (M0 fell 0.93 → 0.35 from short to long chain, and the
ceiling itself fell to 0.785), and transport still lost to no-op everywhere. The
verdict does not depend on where M0 landed.

Escalation was one run, as the contract specified. No further escalation.

**Headline:** ¬H1 on the short chain, but *not* for the reason the contract
anticipated. Two independent findings, one of which is robust and does not
depend on the escalation outcome.

---

## 1. Decision-rule branch taken

**M0 = 0.930 at horizon 10 (threshold: < 0.25) → inconclusive-by-insufficient-drift.**
Per Ruling 2, the short chain does not falsify transport; it fails to make
transport *necessary*. Escalation to Tasks 0→4 launched, thresholds unchanged.

**However, a second result is independent of that clause and does not need
the escalation to stand:**

**Transport is worse than doing nothing, at every horizon.**

| horizon | M0 (no transport) | M1 = M2 (transported) | M3 (snapshot) | UPPER (true era-t) |
|---|---|---|---|---|
| 3  | 0.928 | 0.206 | 0.145 | 0.961 |
| 5  | 0.911 | 0.271 | 0.200 | 0.947 |
| 8  | 0.904 | 0.215 | 0.249 | 0.942 |
| 10 | 0.930 | 0.140 | 0.233 | 0.928 |

*(digit-probe accuracy, era-t readers per amendment A1, seed 42; era-0 probe on
era-0 states sanity check = 0.979.)*

---

## 2. Why: R1 (the OOD gap) confirmed by direct measurement

The pre-registered risk R1 was the cause, and it is measurable rather than
inferred. For the **same map g_k**, in-distribution vs applied-to-memories:

| step | g_k R² on T1 holdout (in-dist) | cos(g_k(mem), truth) | cos(mem, truth) = **no-op** |
|---|---|---|---|
| 1  | 0.748 | 0.198 | **0.830** |
| 2  | 0.948 | 0.732 | **0.947** |
| 5  | 0.949 | 0.743 | **0.964** |
| 10 | 0.947 | 0.727 | **0.935** |

The drift map fits its own distribution well (R² ≈ 0.95) and moves Task-0
memories **further from their true current-era states than leaving them
untouched**. The learned map is strictly worse than the identity function on
the population it exists to serve. Chaining compounds it: single-step
application reaches 0.5–0.7 probe accuracy, the 10-step chain reaches 0.14.

**The OOD detector fired correctly:** 93–99% of memories exceeded τ_ood at
every step. The filter *knew* it was extrapolating.

---

## 3. Hypothesis outcomes

- **H1 (transport ≥ 70% where untransported < 25%):** ✗ — M2 = 0.140, and the
  premise fails too (M0 = 0.930, not < 0.25). Inconclusive per Ruling 2;
  escalation pending.
- **H2 (original):** vacuous by construction (0pp always) — flagged before
  results, superseded by H2′.
- **H2′ (low-σ² half beats all by ≥ 10pp):** ✗ — 0.144 vs 0.140 = +0.35pp.
- **H3 (Spearman ρ ≥ 0.3):** ✗ — ρ = −0.027.
  *Mechanically explained:* ~all memories were flagged OOD and received the same
  κ=10 inflation, so σ² is near-constant across memories and cannot discriminate.
  H2′ and H3 fail for the same underlying reason, consistent with the ruling
  that they are two views of one question.
- **Streaming vs snapshot (M2 − M3):** −9.3pp at horizon 10 — the streamed chain
  is *worse* than the single boundary map. No support for the streaming-novelty
  claim on this chain.

---

## 4. Drift actually observed (context for the M0 clause)

Cumulative geometric drift of memories is real and substantial, yet linear
readability is preserved — which is why M0 stays high:

| k | cos(f₀(x_old), f_k(x_old)) | M0 digit acc |
|---|---|---|
| 1  | 0.830 | — |
| 3  | 0.744 | 0.928 |
| 5  | 0.667 | 0.911 |
| 8  | 0.614 | 0.904 |
| 10 | 0.574 | 0.930 |

States move a long way (cos 0.57) while remaining ~93% linearly decodable. The
contract's < 25% expectation was transplanted from the 5-task frozen-head
experiment; over one transition, geometric drift and readability loss are not
the same quantity. Step 1 (the task transition itself) is the violent one
(within-step cos 0.41 on T1 inputs); subsequent epochs are gentle (≈0.94).

---

## 5. Protocol deviations

1. **Amendment A1** (era-t readers) — approved pre-results.
2. **H2′ substitution** — approved pre-results; both readouts reported.
3. **Escalation clause** — invoked; Tasks 0→4, 3 seeds, thresholds unchanged.
4. **Seed count:** this memo is seed 42 only. Escalated runs cover all three
   seeds (42/1337/2024); no multi-seed claim is made here.
5. **R4 caveat, as pre-registered:** transported object is raw `c_t`; the real
   bank stores `c_prime`. A pass would have been necessary-not-sufficient.

---

## 6. Reading

The negative result is stronger than "insufficient drift." A drift map trained
on current-task live pairs is not merely *insufficient* for transporting
old-task memories — it is *actively harmful*, because old memories lie off the
manifold the map was fit on, and the map's behavior there is unconstrained.
The Bayesian layer's mean cannot fix this (it propagates the same map), and its
variance cannot rank it (everything is OOD, so everything inflates equally).

The escalation will settle whether M0 collapses over 4 transitions. But unless
the OOD gap closes — which the mitigation noted under R1 (consistency
regularization on bank memories, making them in-distribution for the map) is
designed to address — transport is expected to remain below the no-op baseline
there too. That mitigation is the only branch of this design worth testing next,
and it would be a protocol deviation to be declared before running.

*(Pre-registered expectation recorded before the escalation ran. Outcome: M0 did
collapse — 0.93 → 0.35 — and transport did remain below the no-op baseline, in
every cell. The prediction held.)*

---

## 7. Escalated chain (Tasks 0→4, 3 seeds, horizons at task boundaries)

| horizon | M0 (no transport) | M1 = M2 | M3 (snapshot) | UPPER (ceiling) |
|---|---|---|---|---|
| 10 | 0.834 ± 0.062 | 0.161 ± 0.036 | 0.170 ± 0.030 | 0.923 ± 0.009 |
| 20 | 0.538 ± 0.079 | 0.149 ± 0.069 | 0.178 ± 0.046 | 0.868 ± 0.002 |
| 30 | 0.324 ± 0.046 | 0.129 ± 0.056 | 0.150 ± 0.063 | 0.809 ± 0.019 |
| 40 | 0.346 ± 0.137 | 0.158 ± 0.027 | 0.129 ± 0.011 | 0.785 ± 0.030 |

Three observations beyond the verdict:

1. **The ceiling itself decays** (0.923 → 0.785). Even *true* current-era states
   of Task-0 inputs lose linear readability as the encoder trains on Tasks 1–4.
   This is representational forgetting measured directly, and it caps what any
   transport method could achieve: the target is a moving, degrading one.

2. **Transport is geometrically farther from truth than doing nothing** at
   horizons 10/20/30 (cos 0.122/0.158/0.170 vs 0.526/0.326/0.201). At h40 it is
   nominally closer (0.227 vs 0.182), but both are near-orthogonal to truth, so
   that crossover is noise on a meaningless scale — probe accuracy still favors
   no-op 0.346 vs 0.158.

3. **Calibration shows a flicker, then dies.** ρ is weakly positive at early
   horizons (0.126–0.308; seed 2024 h10 briefly clears the 0.3 bar) and decays
   to ≈ 0 or negative by h30–40. Consistent with the short-chain mechanism: once
   nearly every memory is flagged OOD, σ² is uniform and cannot rank.

**Conclusion.** The failure is not a horizon artifact and not insufficient
drift. It is structural: a drift map fit on current-task live pairs is
unconstrained on the off-manifold region where old memories live, so applying it
is worse than leaving them alone — and chaining compounds that error. The
Bayesian layer cannot repair this: its mean *is* the naive map, and its variance
is uniformly inflated precisely because the OOD detector is correct.

**Disposition:** one section in the existing paper's future-work / failure
catalog. RBST as specified is not built. The single branch that could change
this verdict is the R1 mitigation (consistency regularization making bank
memories in-distribution for the map) — a new pre-registration, not a patch to
this one.
