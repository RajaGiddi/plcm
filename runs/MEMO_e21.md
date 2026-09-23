# MEMO — E21: Repair Under an Approximate Map

**Status (2026-09-18):** row read under the two rulings of 2026-09-18
(`docs/E21_prereg.md` §7a). 36/36 artifacts, all fits converged. Row:
`scripts/e21_row.py --a2-windows 2` → `runs/e21_row.json`. October / paper 2;
nothing here enters the September submission.

## 0. Rulings, and the one that had to be restated

| ruling | as landed |
|---|---|
| **C-ID, A2** | Bar **two test windows per cell** (`--a2-windows 2`: per-cell bar 2/n_test,k; max 2/409 = 0.0049). The ruling was on the *measured* floor — "max 0.0049 observed" — and that number is the ruling. My gloss "one window" was a description of the number that was wrong by a factor of two (task 0 has 409 windows, not ~204), and a description does not override the measurement it describes. A bar at one window would have sat below the floor: R3's defect. Carried as a floor on every A2/A7 delta (pooled 0.0053; below every such delta in the row, so no reading moves). A0/A1/A6 stay at 1e-6 and are exact, 36/36. |
| **C-DT** | **FAIL, recorded with cause, not re-barred.** 391/396. See §2. |
| **A4** | printed, **not read** — `docs/E21_mummadi.md` does not exist. Every A4 number below is unread. |

**Why floors are now stated in windows.** The row's inputs sat within a factor
of two of a bar that would have withheld the headline: at "one window",
`gain/level0` fails C-ID in two cells, A2 and A7 are not read, and the paper's
wrong-locus cost has no row. The result is unchanged under either bar; the
*gate* was one factor of two from withholding it. The deltas are now printed
in windows — **−1, −2, +2, −1** — which is a floor a reader can see, where
0.0049 was one they had to trust, and an integer is a witness that the row is
reading the right test set (a non-integer would have said it was not).

## 1. Amendment after the first sweep — bounded parameterization

11 of 13 gain-family jobs died (`linalg.inv … singular`, one by float
overflow): LBFGS's strong-Wolfe line search sent $e^{\phi_i}$ to zero.
Amended before the re-run (`docs/E21_prereg.md` §3): gain corrections
$\mathrm{diag}(e^{\ln 2 \cdot \tanh\phi})$ in $(\tfrac12, 2)$, offset
corrections $\sigma_k \odot \tanh\phi$ within $\pm 1\sigma$, swap unchanged;
$\phi = 0$ still the identity. **Both** continuous families re-run bounded so
A3/A4/A5 share one optimizer across families. The 15 unbounded artifacts
(offset complete, gain 2) are preserved, never read:
`runs/e21/_unbounded/`. Every bounded artifact records
`optimizer.parameterization = "bounded (2026-09-17 amendment) …"`; the poll
that pulled them kept a file only if it did.

## 2. Controls

| control | result |
|---|---|
| C-ID | A0/A1/A6 **0.0** in 36/36 cells. A2: `gain/level0` off in 4/12 cells by −1, −2, +2, −1 windows (s42/t2, s1337/t0, s2024/t0, s2024/t3); `offset/level0` and `swap/level0` exact. Across both sweeps, 3 of 5 x86 containers showed the same four cells at the same values; the screen was arm64. A2 is the only level-0 arm through `refit_probe`. **PASS at the ruled bar.** |
| C-EXACT | A6 spread **0.0** across every level and realization. PASS |
| C-WIT | path identity on every cell. PASS |
| C-CONV | 0 unconverged of 1296 fits per arm (432 cells × 3 jitters), all four optimized arms. |
| **C-DT** | **FAIL 391/396.** All five failures: seed 42, gain family, tasks 1–2 (L1r1 t1, L1r1 t2, L3r2 t1, L4r0 t1, L4r2 t1). Cause, from the fit records: the deranged objective is flat at $\phi \approx 0$ — LBFGS stops after 2–4 iterations, $|\phi|_{\max} \le 0.075$, so **A5d = A1 exactly** — while the *true* teacher walks the map to the bound ($|\phi|_{\max}$ 2.1–13.9) and loses 8–10pp (s42/t1: A5 0.811 vs A1 0.901, at every level and already at ε = 0). The bar assumed A5's refinement helps; where it hurts, a refinement that fails to move beats one that moves wrong. CC's shape: a control broken by a property of the arm, not a defect of the instrument. |
| C-DRIFT | §5. |
| C-FLOOR | per §5 of the contract: floor = max(spread over realizations of the pooled value, mean per-cell jitter spread, ruled window floor on A2/A7); both components printed in the row. |

## 3. Curves — forgetting, diag form, pooled over 12 cells, mean over realizations

Floors in brackets as [realization / jitter]; A2/A7 additionally carry 0.0053.

**P-gain** (realistic ≤ 0.05)

| ε | A0 none | **A1 C0deg** | **A2 bridging** | A3 | A4* | A5 | A5d | A7 | SNAP |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.386 | −0.038 | 0.096 | −0.063 [/.037] | −0.076 | −0.035 | 0.263 | 0.143 | 0 |
| 0.05 | 0.386 | −0.040 [.004] | 0.097 [.007] | −0.067 [.018/.027] | −0.080 | −0.037 | 0.246 | 0.130 | 0 |
| 0.10 | 0.386 | −0.034 [.013] | 0.105 [.014] | −0.062 | −0.080 | −0.036 | 0.256 | 0.150 | 0 |
| 0.20 | 0.386 | −0.023 [.018] | 0.118 [.015] | −0.054 | −0.068 | −0.034 | 0.228 | 0.136 | 0 |
| 0.30 | 0.386 | −0.034 [.064] | 0.139 [.039] | −0.050 [.060] | −0.072 | −0.037 | 0.214 | 0.150 | 0 |

**P-offset** (realistic ≤ 0.10σ)

| ε | A0 | **A1** | **A2** | A3 | A4* | A5 | A5d | A7 | SNAP |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.386 | −0.038 | 0.095 | −0.032 [/.060] | −0.043 | −0.003 | 0.450 | 0.168 | 0 |
| 0.05 | 0.386 | −0.040 [.010] | 0.100 [.017] | −0.051 | −0.051 | −0.009 | 0.444 | 0.190 | 0 |
| 0.10 | 0.386 | −0.026 [.013] | 0.103 [.016] | −0.021 [/.106] | −0.015 | −0.007 | 0.436 | 0.182 | 0 |
| 0.20 | 0.386 | −0.017 [.069] | 0.127 [.013] | +0.002 [.105/.111] | −0.015 | −0.005 | 0.492 | 0.178 | 0 |
| 0.30 | 0.386 | +0.009 [.100] | 0.160 [.114] | +0.040 | +0.046 | −0.008 | 0.430 | 0.194 | 0 |

**P-swap** (realistic = 0; m = transpositions)

| m | A0 | **A1** | **A2** | A3 | A4* | A5 | A5d | A7 | SNAP |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.386 | −0.038 | 0.095 | −0.029 | −0.040 | −0.050 | 0.379 | 0.127 | 0 |
| 1 | 0.386 | **0.108** [.081] | 0.207 [.176] | 0.008 [.049] | −0.024 | **−0.037** [.003] | 0.421 | 0.104 | 0 |
| 2 | 0.386 | 0.309 [.128] | 0.381 [.076] | 0.101 [.175] | 0.023 | −0.015 [.022] | 0.532 | 0.144 | 0 |
| 3 | 0.386 | 0.406 [.222] | 0.424 [.055] | 0.219 [.158] | 0.129 | 0.073 [.147] | 0.563 | 0.231 | 0 |

\* A4 unread.

## 4. Headline 1 — A1 vs A2: no crossing in any family

- **Gain:** A1 is flat — within its floor of its ε = 0 value at every level
  through a **30% per-channel gain error** (−0.038 → −0.034; the largest
  excursion, −0.023 at 0.20, is inside the 0.018 floor + window). A2 rises
  0.096 → 0.139. A1 below A2 at every level: **no regime.**
- **Offset:** A1 −0.038 → −0.040 → −0.026 → −0.017 → +0.009 at 0.3σ; A2
  0.095 → 0.160. Within the realistic range (≤ 0.1σ) A1 is inside its floor.
  A1 below A2 at every level: **no regime.**
- **Swap:** both collapse — A1 0.108 / 0.309 / 0.406, A2 0.207 / 0.381 /
  0.424 — inside floors of each other at every m (the swap "floor" is the
  spread over *which* channels were swapped, see §6). The third registered
  shape: *the error passes through both; the place of repair does not matter
  for this error class.*
- **A7 − A5, bridging on the refined map vs re-layout on the refined map,
  the matched-map-quality comparison (note 2):** gain +0.167…+0.188, offset
  +0.171…+0.201, swap +0.140…+0.177 — **14–20pp at every family and level**.
  At the exact map (A2 − A1, level 0) the gap is **13.4pp** (0.1338 / 0.1333
  / 0.1333 in the three families' level-0 files; S72 row 13.3). Two
  quantities, both reported; §7 of the paper cites the matched-map one and
  gives the exact-map one beside it.

**The registered reading:** *bridging has no regime under map error either;
the dominance result is unconditional on this benchmark.* The approximate-map
objection to C0deg has no regime in the continuous families at any level the
contract called realistic, nor at six times it. In the discrete family a wrong
wiring destroys both methods equally — bridging is not a fallback for a wrong
map, because bridging uses the same map.

## 5. Headline 2 — A5 vs A3/A4: the era teacher beats self-supervision nowhere (registered read)

- **Gain:** A5 is *worse* than A3 by 1.3–3.0pp at every level (floor
  0.027–0.060), worse than A4 (unread) by 3.4–4.4pp.
- **Offset:** worse at 0.05 (−4.3pp, floor 0.052); within floor at 0.1–0.3.
- **Swap:** A5 better than A3 by +0.045 / +0.116 / +0.146 — **not by more
  than the floor** (0.049 / 0.175 / 0.167).

**Registered verdict: A5 beats both A3 and A4 by more than the floor in no
family.** The conjunction fails on A3 alone, so it is scored without reading
A4. Per the contract's §6 sentence: *the era teacher is replaceable by
self-supervision — the fifth time a stored-self component reads as
redundant.* Reported as such.

**A measurement beside the verdict, not instead of it.** The swap family's
"floor" is the spread over three *realizations*, and in the discrete family a
realization is *which channels are transposed* — a treatment axis, not a
noise axis. Paired within realization (same wiring for both arms), A5 − A3 is
positive in **9 of 9** realizations (m=1: +0.049, +0.068, +0.018; m=2: +0.197,
+0.121, +0.030; m=3: +0.089, +0.169, +0.179) and A5 − A4 in 6 of 9. And A5
against the *wrong-map re-layout* A1: at m = 1 A5 reads −0.037 — the
exact-map value — against A1's 0.108: with the era teacher and an exhaustive
search over transpositions, **a one-transposition wiring error is fully
repaired**, a two-transposition one nearly (−0.015), a three-transposition one
partially (0.073). This is the first arm in the program where the stored self
does something the map alone cannot. It is not the registered read, the
registered read stands, and the bar is not moved: the contract's floor
normalizer included treatment variation (catch 25's normalizer corollary,
applied to a floor), which is a finding about the contract. **A paired read
with the wiring as a blocking factor is the follow-up, pre-registered before
it is run.**

**Note 1 (recorded at sign-off), one sentence as promised:** A5's teacher
reads the true frame and its accuracy bounds the supervision's quality; where
A5 lands below A1 (s42/t1, −9pp at every level), the mechanism is not
identified — A5 exceeds the teacher's own DIAG in 9/12 cells at ε = 0, so
agreement with a weaker teacher is not simply capped at the teacher's
accuracy.

## 6. C-DRIFT — do the objectives move a correct map? (two-sided, as registered)

At ε = 0, arm − A1 in accuracy, pooled; jitter floor beside; $\|\phi^*\|$:

| family | A3 | A4* | A5 |
|---|---|---|---|
| gain | +0.025 [0.037], ‖φ*‖ 4.0 | +0.038 [0.010], 3.3 | −0.003 [0.015], 3.8 |
| offset | −0.006 [0.060], 0.9 | +0.005 [0.049], 0.8 | **−0.035 [0.010]**, 1.2 |
| swap | **−0.009** [0; window 0.005], one transposition | +0.002, 1.3 | +0.013, 1.8 |

- **Registered reading fired (35%):** A3 reads below A1 by more than its floor
  at ε = 0 in the swap family — self-supervision moves a correct wiring in
  9/12 cells (4 lose 1–25pp, 5 gain 6–10pp; pooled −0.9pp against a 0.5pp
  window floor). *A finding about the objective, and a caveat on any A3 gain
  elsewhere.* In the continuous families every objective leaves the true map
  by a large $\|\phi^*\|$ and lands inside the jitter floor: the objectives'
  minima are broad, not at the true map.
- **The teacher's minimum is not where it should be in the offset family:**
  A5 moves a correct offset by −3.5pp against a 1.0pp floor. Registered
  sentence: *if A5 does [move a correct map], the teacher's minimum is not
  where it should be* — it does, for offsets.

## 7. Predictions — scored under the appendix's convention

The appendix (`app:predictions`) scores a **miss** only where the registered
side (≥ 50%) failed; a sub-50% prediction that does not happen is "did not
fire". The row script's `MISS` label was applied to every non-event and
produced the "two fired, three missed" I reported; that was the wrong
convention, and the count below is read from the table after the rows were
entered.

| prediction | odds | outcome |
|---|---|---|
| A1 within its floor of ε = 0 across the realistic range (gain ≤ 0.05, offset ≤ 0.10) | 80% | **fired** |
| A1 loses ≥ 20pp at one transposition | 85% | **miss** — 14.5pp (−0.038 → 0.108) |
| a crossing ε* exists in at least one family | 40% | did not fire |
| P-gain crossing | 15% | did not fire |
| P-offset crossing | 15% | did not fire |
| P-swap crossing | 40% | did not fire |
| if a crossing exists, it is in P-swap | 60% | not scored — condition unmet |
| A5 beats both A3 and A4 by more than the floor in at least one family | 45% | did not fire |
| A5's margin over A3 concentrates where the perturbation induces confident errors | 40% | not scored — no margin under the registered read (the swap-family paired margin is a measurement, §5) |
| C-DRIFT: A3 or A4 below A1 by more than its floor at ε = 0 in at least one family | 35% | **fired** (swap, A3) |
| A6 constant; A2 never above A6 | anchor | A6 exact; A2 above A6 at every level (bridging is never below C0deg on the exact map) |

Eight scored entries: two fired, one miss, five did not fire. **Note 3 at
sign-off:** the series' misses had all run "toward more structure and less
determinism than predicted." E21's one miss runs the other way — the encoder
absorbed more of a wrong wiring than the 85% cliff prediction allowed (14.5 vs
20pp) — and its two non-events (a crossing, a teacher advantage) are in the
direction every stored-self and known-map result in the program has taken.
Appendix F's sentence: the author's priors over-weight the stored self and
under-weight the map's trivial use, consistently, across E8, E9, E11, E16,
E18 and now E21.

## 8. For the paper

- **§7 (wrong-locus cost):** *Repairing at the readout what originated at the
  input costs 14–20 points at matched map quality (A7 − A5, every family and
  level), 13.4 at the exact map.* One benchmark, one architecture; W's row on
  the E18 constructions gives the exact-map gap on three more pairs (2.5 /
  17.6 / 4.5pp) from artifacts that exist.
- **§8 (limitation, narrowed):** *the map-known result holds under
  calibration error to 30% gain and 0.3σ offset; a wrong channel wiring
  destroys re-layout and bridging alike; a stored era model with a discrete
  search repairs a one-transposition wiring error to the exact-map value
  (measured, paired read pending).*
- **Appendix F:** the calibration sentence in §7.

## 9. Artifacts

`runs/e21/{gain,offset}/level{0..4}_real{r}.json` (bounded, 26),
`runs/e21/swap/level{0..3}_real{r}.json` (10), `runs/e21/_unbounded/` (15,
superseded), `runs/e21/smoke.json`, `runs/e21_row.json`,
`scripts/e21_perturb.py`, `scripts/e21_row.py`, `docs/E21_prereg.md` (§3
amendment, §7a rulings). Job logs: `/tmp/e21logs/`. Sweep apps:
`ap-4xDNb39EBUrpL1J7h8T3YN` (first, unbounded), `ap-w7irLjR77nRxeNqbtj6Vxy`
(bounded relaunch, 14:19–18:07).
