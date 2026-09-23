# MEMO — E18: known map versus estimated drift (first read, 2026-09-16)

**Status:** runs and screens in for the three MNIST constructions; **P3 (live +
must-fail), the C0deg controls on MNIST (G1′/CA′/CC′), tier-0, the storage row
and `e18_row.py` are still to run** — nothing below enters the ledger until
they do. Ports remain gated on the human read (condition 1). Probe subset
**seeded and recorded** (`probe_subset_seed = 20260916`) on every E18 screen
row; E18 is the first experiment whose refit draw is a recorded configuration.

**Build gate** (`runs/e18/build_gate.json`): flag-off loaders 60/60 hashes
identical to the pre-amendment reference; disjoint constructions correct
(12k/2k per task, pairwise disjoint, complete, identical across Permuted and
Rotated per seed); the bit-deterministic anchor `w2d_mafc_seed42_{a,b}`
reproduced **25/25** by a retrain on the amended code. PASS.

**Runs** (`jobs_e18_*`, 15, launched 11:36, finished 12:02, all verified):
every artifact records `disjoint_content: True`, its content fingerprint
(`c1a570a02d35` / `6d17d558e37f` / `19d72cb3db60` by seed), and fp32-shadow
delta 0.0 at every boundary. Determinism across three seed-42 launches:
Rotated-MLP all identical; Permuted-MLP floor pair identical, main run
differs (AVG 0.7876 vs 0.7831); Permuted-LSTM 0/25.

---

## 1. The screen — forgetting, diag form, intersection population, C0deg included

| construction | none | bridging (C3) | **C0deg** | SNAP | C0 (base layout) | C1 | n | ranking |
|---|---|---|---|---|---|---|---|---|
| Permuted, MLP | 0.2159 | 0.0241 | **−0.0007** | 0 | 0.3423 | 0.1524 | 11 | C0deg < C3, RESOLVED |
| Permuted, LSTM-OFF | 0.7395 | 0.1755 | **−0.0002** | 0 | 0.8157 | 0.7009 | 12 | RESOLVED |
| Rotated, MLP | 0.4557 | 0.0414 | **−0.0037** | 0 | 0.7760 | 0.4021 | 12 | RESOLVED |

Peak-clipped form: same ordering on every construction (C0deg 0.0018 / 0.0078
/ 0.0012). `runs/e18_*/hx2_c0deg{,_peak}.json`. Identity 0.0e+00 on every cell.

## 2. Per task — the crux cells

| construction | task k (map k → 4) | orig | ceiling | reset-with-data | bridging | **C0deg** | C3 / reset gain |
|---|---|---|---|---|---|---|---|
| Rotated, MLP | 0 (90°: **exact**) | 0.195 | 0.956 | 0.889 | 0.884 | **0.974** | 0.99 |
| | 1 (67.5°: lossy) | 0.332 | 0.972 | 0.925 | 0.923 | **0.969** | 1.00 |
| | 2 (45°: lossy) | 0.604 | 0.973 | 0.945 | 0.940 | **0.973** | 0.99 |
| | 3 (22.5°: lossy) | 0.915 | 0.969 | 0.958 | 0.958 | **0.968** | 1.00 |
| Permuted, MLP | 0–3 | 0.61–0.92 | 0.96 | 0.93–0.96 | 0.92–0.95 | **0.960–0.965** | 0.89–0.99 |
| Permuted, LSTM-OFF | 0–3 | 0.12–0.37 | 0.91–0.96 | **0.74–0.82** | 0.73–0.82 | **0.93–0.94** | 0.98–1.00 |

**Readings, against §4 of the contract (pre-P3, pre-controls):**

- **Applying beats estimating everywhere the map is exact** — and, on this
  benchmark, everywhere it is lossy too: the rotated lossy pairs sit within
  0.3pp of the exact one. The interpolation cost of a bilinear rotation is
  invisible to an MLP classifier at this resolution. The crossover the
  contract reserved 25% for does not appear on Rotated MNIST; if it exists it
  needs a lossier map than 28×28 bilinear rotation.
- **Bridging passes its own real test.** On disjoint content the pseudo-old
  data is novel content in old coordinates, and bridging recovers 96–100% of
  the reset-with-data gain on every construction. Ruling B stands unchanged:
  it is a measured intermediate, better than nothing, dominated by the map.
- **The Permuted-LSTM cell is the paper's cleanest sentence.** Reset with the
  real old data reaches only 0.74–0.82 against ceilings of 0.91–0.96 — the
  encoder's representation of task k's *layout* has drifted (F_enc ≈ 0.19),
  and no head refit recovers it. C0deg reads 0.93–0.94: the encoder's
  representation of the *content* is intact in the current layout, and
  re-layout reaches it. Encoder "damage" here is layout-specific, not
  content-specific — which is what §4 has to say about HAR and CIFAR being two
  phenomena with one signature.

## 3. Predictions (contract §9), scored on the screen alone — the ports' rows are open

| prediction | odds | outcome |
|---|---|---|
| Permuted-disjoint MLP: C0deg forgetting ≤ 0.02 (≈ DIAG) | 85% | **fired** (−0.0007) |
| Permuted-disjoint MLP: bridging recovers > 80% of the reset gain | 60% | **fired** (89–99% per task) |
| Permuted-disjoint: C0deg > bridging | 80% | **fired**, resolved |
| Permuted-disjoint: LSTM-OFF and MLP rankings agree | 65% | **fired** |
| Rotated-disjoint: C0deg on lossy pairs ≥ 5pp below the exact pair | 55% | **miss** — within 0.3pp |
| Rotated-disjoint: a port beats C0deg on a lossy pair | 25% | open (ports gated) |
| P3 live passes on both disjoint constructions | 75% | open |
| MLP-MNIST floors: a per-method forgetting floor > 2pp | 50% | open (floor sets screened; `e18_row.py` reads them) |
| HAR rows (ports, tier-0 re-run, crux) | — | open |

## 4. §4's two-origins paragraph — draft, against the Permuted-LSTM cell

*Written for: the paper's §4 (the decomposition section), to be placed after
the reader-share results and before the HAR/CIFAR comparison. Numbers are the
E18 first read; the seeded re-runs (R2) may move refit-derived values by
≤ 1.6pp per cell and none of the sentences below sit within that.*

> The decomposition charges forgetting to two channels, and on benchmarks
> with a known input map both channels are largely presentation drift rather
> than lost knowledge. The reader channel is the obvious case: refitting the
> head on the old data recovers it. The encoder channel is not obvious, and
> the Permuted-MNIST LSTM arm is the evidence. There, refitting the head on
> the *real* old-task data reaches only 0.74–0.82 against era ceilings of
> 0.91–0.96 — an encoder term of $F_{\mathrm{enc}} \approx 0.19$, measured
> exactly as §3.1 measures it — and a reviewer would read that as the encoder
> having lost nineteen points of task knowledge. It has not. Re-laying the
> same old-task inputs into the *current* task's coordinates and running the
> current model, with no refit, reads 0.93–0.94 on every old task: the
> encoder's representation of the *content* is intact; what it can no longer
> read is the old *layout*. The 0.19 the decomposition correctly attributes
> to the encoder is the encoder-as-reader-of-that-layout, not the encoder as
> a representation of digits. The same holds, more mildly, on Permuted-MNIST
> MLP ($F_{\mathrm{enc}}$ 0.07; re-layout 0.96 against a 0.96 ceiling) and on
> Rotated MNIST, where even the pairs whose map is a lossy bilinear rotation
> read within 0.3pp of the exact pair. On Split-CIFAR-100 there is no layout
> to re-lay: $F_{\mathrm{enc}} = 0.0076$ on the ResNet-50 is content, and
> the stranded head is the whole story. One instrument, two phenomena with
> one signature — layout drift with a known map, and a stranded head with no
> map — and Table 1 says which is which per row. The consequence for the
> reader is the one the rest of the paper is built on: where the map is
> known, apply it; where it is not, the head is what is broken.

Evidence table for the paragraph (three seeds, diag form, seeded draw pending):

| construction | ceiling | reset-with-data (F_enc = ceiling + R − reset) | C0deg (re-layout, no refit) |
|---|---|---|---|
| Permuted, LSTM-OFF | 0.91–0.96 | 0.74–0.82 (**F_enc ≈ 0.19**) | **0.93–0.94** |
| Permuted, MLP | 0.96 | 0.93–0.96 (F_enc 0.07) | 0.960–0.965 |
| Rotated, MLP (exact pair / lossy pairs) | 0.96–0.97 | 0.89 / 0.93–0.96 | 0.974 / 0.968–0.973 |
| Split-CIFAR-100, ResNet-50 (E14) | 0.945 | 0.931 (F_enc 0.0076) | — (no map) |

## 5. Remaining before the ledger

P3 on the live constructions and the shared-content must-fail (`ckpt_e17_mlp_*`);
C0deg controls on MNIST (content disjointness from the benchmark's index
sets; identity-permutation exactness; wrong-permutation must-fail 12/12);
tier-0 on the three sets and on S72; storage row; `e18_row.py` (finish, arm
identity across arms, floors per quantity from the floor sets, both formulas,
the main/appendix split of condition 2). Then the ports, after the read.
