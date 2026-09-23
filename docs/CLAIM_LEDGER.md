# Claim Ledger

**This file is authoritative.** From its creation forward, every ledger edit
happens here first; the conversation discusses the ledger, it does not hold it.

---

## Current one-liner

*(as of E14, 2026-08-10 — branch (A); supersedes the E12 one-liner. **Amended
2026-09-15** on the cure sentences only, per the C0deg controls — see History.)*

> Catastrophic forgetting can be eliminated at the input path when the shift is
> expensive to absorb and the encoder has something to lose. Where the input path
> fails to engage, forgetting is carried predominantly by reader–encoder
> mismatch, not representational damage: **75–89% across four arms of a
> scratch-trained LSTM on subject-disjoint UCI HAR** (three-seed means; per-seed
> shares 57–101%; uniform arms, recorded probe draw — the E11-era 66–88% on the
> shared-window construction is superseded by provenance and construction),
> **89.5% [88.5, 90.6] on a scratch-trained
> MLP on disjoint-content Permuted MNIST** (the shared-content E17 construction
> reads 82.2% [76.8, 87.4] seeded and cannot carry a re-layout column), and
> **91.8% [89.6, 94.1] on a
> pretrained ViT-B/16 and 98.8% [98.3, 99.4] on a pretrained ResNet-50, both
> under full fine-tuning across 20 tasks** — four architectures spanning
> recurrent, feedforward, transformer and convolutional families, one
> instrument, verified exact on all of them (scratch evidence is per-dataset
> and is never pooled across them) — **and the same
> split holds under class-incremental evaluation, where prior work independently
> attributes failure to readout bias**. Old-task features survive fine-tuning
> nearly intact (F_enc **0.06** on the ViT, **0.0076** on the ResNet; refit
> probes 0.888 vs 0.974 pretrained, and 0.936 vs a 0.943 ceiling); the readout
> is what breaks. The drift carrying the mismatch is **aimed at the readout
> subspace** (32–42% of drift energy in S against a 3.9% chance baseline), not
> scattered. Certified-honest generative refit (bridging) recovers approximately
> the era-checkpoint accuracy on sensor data (ρ ≈ 1: **1.02 [0.82, 1.22]**, the
> interval spans 1; forgetting **0.386 → 0.095, −75%**) and beats the field's
> training-time baseline (LwF **0.226**) — but is
> **dominated by trivial deployment of its own required snapshot and by the
> trivial use of its own required map**: re-laying old inputs into the current
> frame with the known transform recovers more than the era checkpoint had
> (forgetting **−0.038**, one encoder, no stored model, no refit). **Bridging
> has no regime where the map is known** — in that regime you apply the map —
> and is reported as a measured intermediate: better than regularization,
> worse than applying the map, with a possible regime (a self-contained head
> where the pipeline cannot acquire a preprocessing step) that the paper names
> and does not claim. It is not a method contribution. On sensor data the
> forgetting is presentation drift entirely, carried by the wiring component
> of the shift, and the fix is the map. Exact input correction into the *base*
> layout is harmful; into the *current* layout it is the best storage-honest
> method measured. The era-head cure family remains unevaluable at this
> benchmark's test-set sizes;
> the reader-channel claim's prior "regime boundary" is **RETRACTED** — the
> boundary was an instrument artifact (label misalignment), and its retraction is
> part of the record.

### Clause sourcing

| clause | evidence | status |
|---|---|---|
| input-path elimination when shift is expensive / encoder has something to lose | E5, E5b, E5c/d | held — untouched by E11 |
| **75–89%** of forgetting is reader–encoder mismatch — **`LSTM`, 4 arms, subject-disjoint HAR, uniform arms** (OFF **77.8**, v1-ON **86.7**, v2-ON **74.8**, v2-control **89.1**; per-seed 57–101%) | E20-B, `runs/e20_band.json` from `runs/e20/seeded/har_s72_*_linear.json` (era + fp32 shadow, 4 threads, arm identity and the v1/v2 witnesses read from every artifact, probe draw seeded 20260916); share floors over **three** seed-42 launches: OFF 0.0, v1-ON **22.4pp** (main run vs two mutually identical replicates that share 0/25 cells with it — different models, survives the seeded draw), v2-ON 9.5pp, v2-control 2.3pp. **ON-arm shares at seed 42 are not readable at better than ±10pp.** E11-era on `har`: `runs/e11_e6b/decomp.json` OFF 66.2, v1-ON 75.6, v2-ON 77.2, v2-control 87.9 — **superseded by provenance and construction, not withdrawn** | **revised three times.** From 75–90% (the OFF arm no longer supports a 75% floor); **NARROWED 2026-08-17** to four arms of one architecture on one dataset; **MOVED 2026-09-16 (E20-B, Ruling A + R2)** to the paper's construction under uniform arms with a recorded probe draw. Still majority everywhere. Three of five E20-B predictions missed, all toward more structure and less determinism (ordering, all-inside-66–88, floor pairs). |
| drift is **aimed at S**: 32–42% in-S energy vs 3.91% chance | `runs/e11_subspace/`, both HAR and MNIST | **new** — supersedes every "unaimed"/"scattered" sentence. Published space read 4–6%, indistinguishable from chance; that was catch 28. |
| generative refit recovers **≈ 1** of the reader channel — ρ **1.02 [0.82, 1.22]**, the interval spans 1, so "approximately the era-checkpoint accuracy", never "exceeds" (S72 arm, 12/12 cells, no exclusions, **recorded probe draw**) | `runs/e10ec_seeded/cures_e10.json` via `scripts/rho_percell.py --bench e10` (R2, 2026-09-16; the unseeded `runs/e10ec/` read 1.03 [0.83, 1.23] under an unrecorded draw — same ranking, retained as measured); P3-v3 licence carried from the same benchmark construction | **MOVED TO S72 2026-09-15 (Ruling A).** H-X1 clears the 0.80 bar. The E11-era value **0.974 [0.898, 1.050]** (`runs/e11_e10/cures_e10.json`, guarded 11 cells; forced-inclusion 0.893) is **superseded by provenance, not withdrawn**: a valid measurement of checkpoints that record no arm, no era status, and a code state two weeks removed. Same sign, same ranking. |
| C3 is **dominated by its own required snapshot** — **and by the trivial use of its other required resource, the known map** | `scripts/e11_evaluations.py` EV3: snapshot ρ **1.136 [1.051, 1.221]** vs C3 **0.974 [0.898, 1.050]** on OFF — intervals separate — at strictly less storage. ON dominates on the point estimate only (sd 1.725, unresolved). **2026-09-15:** C0deg (row below) beats C3 on both arms, both formulas, intervals separate, using the maps alone | **strengthened 2026-09-15** — converts the finding from deployment-shaped to mechanism-shaped; catch 24 twice over. Whatever bridging's regime is, it is not "the map is known": in that regime you apply the map |
| **re-layout into the CURRENT frame via the known map (`C0deg`) recovers more than the era checkpoint had** — HAR/OFF forgetting **−0.0379 [−0.047, −0.029]** (diag form, the paper's §3.1 definition) / **0.0372 [0.028, 0.046]** (peak-clipped, `metrics.py`) on the **S72 arm** (`runs/e10ec/hx2_c0deg{,_peak}.json`, 12/12 cells; Ruling A). E11-era: OFF **−0.0247 [−0.0346, −0.0148]** / 0.0129, ON **−0.0314 / 0.0356** — superseded by provenance, not withdrawn. Uses `M_k`, `M_4` only: no snapshot, no refit, no pseudo-labels, one encoder, one head | `runs/e11_hx2_c0deg.json`, `runs/e11_hx2_c0deg_peak.json` — same instrument and 11-cell intersection as the H-X2 row, ranking vs C3 **RESOLVED** in both forms; per cell C0deg > C3 in **11/12** OFF, **10/12** ON. Controls: `runs/e10/c0deg_controls.json` — G1 subject disjointness from `test_subjects` PASS; CA no-shift (M = I) **bit-identical** 12/12; CB stored column reproduced live to **0.0e+00**; **CC FAIL as pre-committed** (neighbouring-task wrong map below the right one in 9/12 OFF, 11/12 ON — bar does not move; cause below). `runs/e10/c0deg_cc2.json` — **CC′ permutation-only wrong maps PASS 12/12** × 2 wirings × 2 arms; CD component ablation | **new 2026-09-15.** **Scope:** E10's forgetting is presentation drift entirely — the no-shift arm reads **−0.0497** (diag), so content change alone produces none; the map recovers what the shift took. **The recovery is carried by the permutation component** (CD, descriptive, reading written after the run): OFF — correcting the permutation alone recovers **84%** of C0deg's gain (0.4212 → 0.8057 of → 0.8773), everything *but* the permutation **4%**; ON **71% / 2%**. The encoder partially absorbs gain/offset/rotation — *partially*: with the permutation already right (task 3) task-4's gain/rotation/offset still add **+8.6pp** OFF / **+13.4pp** ON. Consistent with E5d's `A_k ≈ I` and §5's engagement condition. **Label history:** excluded in `cure_screen.py` as a "ceiling artifact" on E8's shared-window construction, where relayout into task 4's frame reproduced task 4's test set; the label was carried to E10 unexamined — catch 21 applied to an *exclusion*. ON arm: learned adapter suppressed through the deployed path, analytic map substituted — a different arm, reported as such; the bridging benchmark is OFF |
| **The decomposition's probe subset was an unrecorded draw** — `load_task_data` took the first 4000 samples of a shuffled loader with nothing seeded. Measured on the E17 checkpoints: reload and deployed extraction reproduce to **0.0e+00**; refit fields move by up to **1.4–1.9pp per cell** (0.12pp pooled F_enc, 0.3pp on a share) between two draws | `runs/e20/regression.json`, `runs/e20/mnist_mlp_row.json` (`regression`); fix: `probe_subset_seed` in `channel_decomp.load_task_data`, opt-in, recorded in every artifact that uses it (`runs/e20/seeded/*`, `runs/e10ec_seeded/*`, every E18 screen row) | **new 2026-09-16 (R2).** Every decomposition and screen artifact before this date carries an unrecorded draw with the floor above; **the citable S72 and E20-B values are the seeded re-runs** (forgetting unchanged to 4 decimals; ρ 1.03 → 1.02). The witness rule now covers the draw. |
| **E20-A — the §3.1 lower-bound check, primary MLP-MNIST: TIGHT, in the favourable direction.** Paired ΔF_enc **−0.0073** pooled over 12 cells (|Δ| ≤ 0.01), 0/12 overfit flags, MLP init spread 0.005; reader share **82.2% → 84.0%** under the stronger probe — the linear probe understated survival slightly, and the direction of its error is the one §3.1 claimed | `runs/e20/mnist_mlp_row.json` (seeded artifacts `runs/e20/seeded/mnist_mlp_{linear,mlpprobe}.json`); gates: identity 0.0e+00, R^mlp −0.0002, R1 gate under **R3** tolerance `max(init spread, 1.96·SE_binomial)` **24/24** (five shortfalls of 0.1–0.6pp sit inside the test set's 0.6–1.1pp resolution and above the init spread — R3 was stated before it was applied and rescues only those); reload/extraction fields reproduce E17's artifact to 0.0e+00; recipe certified on the anisotropic synthetic; the seven-design positive-control record `runs/e20/positive_control_*` | **new 2026-09-16 (R1 (a), R3).** One sentence in §4; the seven designs and the observation that on these features every "nonlinear" target is linearly separable go to the appendix. |
| **E20-A, secondary HAR S72 OFF: WITHHELD** — the MLP probe **overfits** unbounded LSTM cell states: on the two failing cells it reaches train accuracy **1.000** (loss ≈ 5e-4) and tests **6.9pp and 6.6pp below** the convex fit, outside the test set's resolution; standardization is on train statistics in both probes (`channel_decomp.py:273, 316`), so this is not a preprocessing defect | `runs/e20/har_s72off_row.json`: R1/R3 gate **22/24 FAIL**, one unconverged cell; pooled ΔF_enc +0.0011 below the probe's own floor (0.030) | **gated, 2026-09-16.** The LSTM row's lower-bound claim rests on the logical argument, stated as such. **This is the paper's stated reason for linear probes with standardization**: a nonlinear probe memorizes 4000 recurrent states even standardized, and reads *below* the convex fit. The contract's alpha sweep (§2) is the remedy if ever pursued; not pursued here. |
| **E16 §7.2 CLOSED — the three-way row on the bridging benchmark, uniform arms.** `har_subject` OFF, era checkpoints + fp32 shadow, 4 threads, arm-recorded, 12/12 cells valid. Forgetting, diag (paper's form) / peak-clipped: **none 0.3864 / 0.4414 · LwF λ\*=1.0 0.2259 [0.212, 0.240] / 0.2831 · C3 (bridging) 0.0954 [0.084, 0.107] / 0.1506 · C0deg −0.0379 [−0.047, −0.029] / 0.0372 · SNAP 0 / 0.0550** | `runs/s72_row.json` (`scripts/s72_row.py`), from `runs/e10off_ec_seed*`, `runs/e16s_har_lam*_seed*`, `runs/e10ec/hx2_c0deg{,_peak}.json`; finish 21/21 by `verify_runs.check`; arm identity from artifacts with uniformity **across** arms; LwF witness (nonzero distillation loss every epoch, tasks ≥ 1); fp32-shadow reload delta 0.0 on every boundary. LwF sweep on *this* construction: λ\* = **1.0** on AVG (0.6145 vs 0.4988 at 0.25), competence margin **+0.8pp**, λ = 4 and 16 **excluded** (DIAG −14.8 / −19.3pp), across-λ spread 3.0× across-seed sd. **Floors, per quantity:** OFF pair **25/25 bit-identical**; LwF λ\* pair **25/25 bit-identical**; LwF λ = 0.25 pair **1/25** (AVG \|a−b\| 5.4pp) — determinism is per-λ; C3 and C0deg per-cell \|a−b\| **0.0000**. CC′ re-run on these checkpoints: **12/12 PASS**, both wirings (`runs/e10ec/c0deg_cc2.json`); CD here: permutation alone **71%** of the recovery, all-but-permutation 15% | **new 2026-09-15.** E16's pending prediction *"bridging > LwF on HAR/OFF, ~55%"* → **FIRED under both formulas**, intervals separate (LwF per-seed 0.159 / 0.333 / 0.186 — its worst seed still above C3's mean). **C0deg is ahead of both.** The pair as E16 planned it was never comparable: the LwF HAR arm ran `benchmark: 'har'` (shared-window) by its own `arm` field against a `har_subject` bridging number; both arms are new here. **Supersession question, for ruling:** the old `e10_off` runs (E11 checkpoints, no `arm` field, era OFF) and these differ by +14.3 / +6.9 / −9.7pp AVG by seed with 0–5 of 25 cells identical — era + shadow + two weeks of code state, unseparated. Under the unrecorded-config rule these runs are the bridging benchmark's citable OFF arm; the E11-era C3/C0deg rows stand as measurements of *those* checkpoints, same sign, same ranking |
| **Bridging has no regime where the map is known.** Under uniform arms on the bridging benchmark: C0deg −0.038 < bridging 0.095 < LwF 0.226 (diag). Bridging beats the field's training-time baseline and loses to the trivial use of its own resource by **13pp**, intervals separated | `runs/s72_row.json`; catch 24 on both of bridging's assumed resources (snapshot: EV3; map: C0deg) | **RULED 2026-09-15 (Ruling B).** Bridging's position in the paper: *a measured intermediate — better than regularization, worse than applying the map — with a possible regime the paper names and does not claim* (a self-contained head after one-time repair, where the validated pipeline cannot acquire a per-query preprocessing step; untested here). **Not a method contribution.** One row, this scope sentence, nothing more. Consequence: E18 redrafts around C0deg as the known-map arm in the known-vs-estimated question; bridging is one row in it |
| **Bridging has no regime under map error either.** Under $\hat M_k = M_k + \epsilon$ on the S72 arm: C0deg on the wrong map (A1) stays within its floor of its exact-map value through **30% per-channel gain** and **0.2σ offset** (−0.038 → −0.034 / −0.017; +0.009 at 0.3σ), bridging on the same wrong map (A2) rises 0.096 → 0.139 / 0.095 → 0.160; **A1 below A2 at every level, both families**. A wrong channel wiring collapses both (m=1: A1 0.108, A2 0.207; m=3: 0.406 / 0.424), within floor of each other. **Repairing at the readout what originated at the input costs 14–20pp at matched map quality** (A7 − A5: 0.140–0.201, every family and level) and **13.4pp at the exact map** (A2 − A1, level 0). The era teacher beats self-supervision by more than floor in no family (registered read) — the fifth stored-self component to read as redundant | `runs/e21_row.json` (`scripts/e21_row.py --a2-windows 2`), `runs/e21/{gain,offset,swap}/`, `runs/MEMO_e21.md`; contract `docs/E21_prereg.md` (signed v2, §3 bounded-parameterization amendment, §7a rulings). Controls: C-ID A0/A1/A6 exact 36/36, A2 at the ruled two-window refit floor (deltas −1, −2, +2, −1 windows, `gain/level0`); C-EXACT 0.0; C-WIT PASS; **C-DT FAIL 391/396 with cause** (deranged objective flat at φ≈0, true teacher walks the map and loses 8–10pp on s42/t1); 0/1296 unconverged fits per arm. Predictions: 8 scored, 2 fired, 1 miss (the 85% 20pp cliff read 14.5pp), 5 did not fire | **new 2026-09-18.** Extends the Ruling B row: the map-known result holds under calibration error to six times the realistic range; the discrete family destroys both methods equally. **Measured beside the verdict, not read as one:** in the swap family the era-teacher search (A5) repairs a one-transposition wiring error to the exact-map value (−0.037 vs A1's 0.108) and beats entropy (A3) in 9/9 paired realizations — the contract's floor normalizer included the wiring draw (treatment variation), so the registered read says "no"; a paired read with the wiring as a blocking factor is the pre-registered follow-up. A4 (Mummadi form) printed, unread. October / paper 2 |
| end-to-end forgetting after the best storage-honest cure — **HEADLINE MOVED TO S72 (Ruling A, 2026-09-15):** OFF uncured **0.3864** → bridging (C3) **0.0954 [0.084, 0.107]** (**−75%**) → C0deg **−0.0379**, diag form; peak-clipped 0.4414 → 0.1506 → 0.0372 | `runs/s72_row.json`, `runs/e10ec/hx2_c0deg{,_peak}.json`. **The best storage-honest method is C0deg**, both formulas. E11-era values — OFF C3 **0.0886 [0.0759, 0.1013]** diag / 0.1198 peak (`runs/e11_hx2{,_peak}.json`), ON 0.1113 / 0.1636, uncured 0.4812 / 0.5028 — superseded by provenance, not withdrawn; the shipped "89%" was true under the ledger when sent, and any future communication uses S72's 75% and explains the supersession if asked | **amended 2026-09-15 — H-X2 is AMBIGUOUSLY REGISTERED.** The prereg wrote "the standard forgetting formula"; the instrument's docstring cited `src/training/metrics.py` for the diag form `R[j,j] − acc`, which that file does not implement (`metrics.py:85-106` is peak-minus-final clipped at 0 — the form every recorded `forgetting` field and the P2 row use). Under the form the instrument computed and the paper's §3.1 defines (`F_total = A_k(h_k,θ_k) − A_k(h_k,θ_T)`, diag by construction), C3/OFF is **MET**; under the form the citation pointed to, **NOT MET**. **Ruled:** diag is the paper's form — peak-clipped references the maximum ever reached, not the era checkpoint, and would break the identity; the *registration* was defective. Both enter the row; the paper does not describe bridging as "pre-registered and met" without this qualifier. The previous cell read "OFF 0.4812 → 0.0886, H-X2 MET at the estimate; ON 0.1113 NOT MET" |
| ~~reconstruction benchmark read the cure at ~1.0; overstatement ~0.2~~ | E8 recomputes to **1.004**, E10 to **0.974** — gap **~0.03** | **RETRACTED.** The overstatement was an instrument artifact (catch 28), not a property of the reconstruction benchmark. E10's benchmark repair stands on its own merits; it does not cut this cure's score. |
| exact input correction at read time **into the BASE layout** is harmful | C0 negative in both benchmarks, both arms; recompute leaves it negative | **narrowed 2026-09-15** — the sentence was scoped by an inherited exclusion. Correction into the **CURRENT** layout (`C0deg`, row above) is the best storage-honest method measured in the program. The previous wording, "exact input correction at read time is harmful", was true only of the base-layout target |
| era-head cure family unevaluable at these test-set sizes | E10 C0 mean −2.949 is **one cell at ρ = −30.000 whose denominator is 0.0209** — legal by 0.0009 against the 0.02 guard | held, **mechanism added** |
| ~~H-C3′ instrument gate fails both arms (R = +0.0945 / +0.1375)~~ | corrected R = **+0.0058 / −0.0049**; pooled HAR R **+0.0534 → −0.0077**, cells over bar **28/72 → 0/48** | **RETRACTED** — a symptom of catch 28, not a property of the benchmark |
| E7 branch (D) reading | `runs/e11_e7_routing.json`: α_own **0.165**, own head top-weighted in **0/24 cells**, latest head dominant | **superseded** → "an attention-blended multi-head readout dominated by the latest head worsens forgetting"; the frozen-early design **was never executed by the deployed path and remains untested** |
| E10 partition as-executed | registered `5d047e4213d1` is the **laptop's** partition; every run used `1104af185c87` (subjects 2/5 swap, an exact tie broken differently on ARM vs x86) | documented **as-executed**; test subjects, disjointness and counts are exactly as registered; **reproduction requires x86** |
| **91.8% [89.6, 94.1] on a pretrained ViT-B/16** (task-IL, 20 tasks) | E12 H-V1, `runs/e12_hv1_v2.json`, 5 seeds × 19 old tasks = 95 cells/arm; base 91.8 [89.6, 94.1], adapt 92.4 [90.5, 94.4] (**intervals recomputed 2026-09-18 as 95% t over seeds** — the earlier [90.88, 92.67] / [91.49, 93.09] were ±1.96 SE over cells, which treats a seed's 19 cells as independent; the appendix's stated convention is t over seeds); instrument R +0.0168 PASS; identity residual 0.0e+00; P3a 0.0e+00 | **new** — the decomposition generalizes to a third architecture. The ViT sits **above** the LSTM range, so reader dominance *increases* with a pretrained trunk. |
| old-task features survive fine-tuning (F_enc **0.06**) | E12: refit probes **0.888** at θ_T vs **0.974** frozen-pretrained vs **0.224** deployed. Two independent instruments (probe sweep, exact decomposition) reach the same sentence from opposite directions. | **new** — the basis for the field-facing claim that freezing works by *pinning the reader's target*, not by protecting endangered features |
| same split holds in **class-IL** (97.9% [95.1, 100.6], t over 3 seeds — was [97.3, 98.5] per-cell; upper bound above 100 because F_enc can be negative) | E12 H-V3, **`runs/e12_decomp_cil_v4/`** (citation source moved from v3 on 2026-09-10: v4 is the re-fit of the same 57 cells on the same checkpoints with per-fit iteration counts and `sklearn_version` 1.9.0 recorded in every row — the first class-IL artifact with complete provenance; v3 retained as the reference it was audited against). Values are **identical** under the ledger's 1.96×SE method: 97.86% [97.27, 98.45], 100-way refit 0.6392 vs 0.6393. Probe convergence: 0/171 guard hits, max n_iter 564 of 5000. 57 cells; masked (within-task) column is the verdict quantity; 6/6 preconditions printed incl. the catch-32 alignment assert | **new, CORROBORATIVE ONLY** — the recency-bias literature independently predicts readout-dominated failure in class-IL, so this arm does not extend the claim. The claim rests on the task-IL row, measured where recency bias cannot be the mechanism (per-task frozen heads, P3b verified). |
| class-IL burden decomposition | joint-access reader **ceiling** 0.639 vs masked 0.941 (chance 0.010 / 0.200) | **descriptive** — ~⅓ of class-IL's difficulty is genuine cross-task discriminability, the rest readout misalignment a per-task oracle erases. **Never a repair target**: the 100-way probe is fit with simultaneous access to all 20 tasks' data, which no deployed system has. |
| ~~E12 "encoder-side architecture boundary" (branch (C))~~ | reported in-session from a probe trained on **misaligned labels** (catch 32): refit read 0.198 ≈ chance, F_enc 0.75, share −4.89% | **RETRACTED BEFORE LEDGER ENTRY** — never committed here. Corrected: refit 0.888, F_enc 0.06, share +91.8%. The hold-until-the-run-lands rule is what kept it out; `runs/MEMO_e12.md` §8.1 is the record. |
| adapter inertness under semantic shift (H-V2) | E12: adapt − base = −0.79pp (Welch t = −0.633); efficacy control on permuted-pixel +1.34pp (t = +1.14, n.s.) | **UNTESTABLE at this placement** — the adapter's best demonstrated effect is below H-V2's own 5pp bar. Consistent with the engagement condition; **never confirmation**. |
| tier-0 stack magnitude | **+0.075 [+0.052, +0.098]** on HAR/OFF — `runs/e13_stack_har.json`; 12/12 cells, guarded == forced, pooled R −0.0045, platform gate 0.0000 (Modal) | **UNFROZEN.** Supersedes the literal **0.347**, which was a **FROZEN LITERAL WITH NO COMPUTING ARTIFACT** — carried here on citation alone from the E11 audit until first computed in E13, and wrong by ~4.6×. |
| **adapters reference arm 0.9183** (n=3, permuted MNIST) | `runs/e4on_v2_seed{42,1337,2024}`, every artifact recording its own `arm`; current HEAD | **SUPERSEDES the cited 0.9331**, which averaged three runs recording no configuration — one of them `runs/fullrank_ref`, now positively excluded (see below). Shift **−1.48pp**. |
| **adapter attribution +48.4pp [±1.9pp]** | verified-minus-verified: ON 0.9183 − OFF 0.4340, `runs/e4{on,off}_v2_seed*`, 6/6 operands arm-recorded; per-seed 47.3 / 47.1 / 50.9pp | **SUPERSEDES +50.3pp.** Shift **−1.8pp**. The old point estimate carried no interval; the per-seed spread it concealed is 47.1–50.9pp. |
| **forgetting reduction 89.0% [83.6, 94.3]** | mean-of-ratios over 3 seeds, `1 − ON/OFF` forgetting = 0.0740 / 0.6713; per-seed 88.6 / 87.0 / 91.2 | **SUPERSEDES 92%** (old 91.7%). Shift **−2.7pp**. Interval is **±5.4pp**, far wider than the accuracy attribution's, because forgetting is a *difference of differences* and the variance compounds — computed, not adjusted. |
| `runs/fullrank_ref` is **not** the adapters arm at current HEAD | 2.91pp from a pipeline measured **bit-deterministic** (15/15 cells, max \|Δ\| 0.000000, seed 42) — so different code state or different experiment, no third option; no `arm` field; name sits beside `lowrank_r32`/`lowrank_r64` | **positively excluded**, marked in `runs/fullrank_ref/PROVENANCE.md` so no future audit re-adopts it. Its anomalies both flatter: highest accuracy *and* lowest forgetting in either set. **Open:** it may be legitimate in the low-rank storage-frontier table — same run, two tables, at most one real home. |
| **82.2% [76.8, 87.4] on a scratch-trained MLP, shared-content Permuted MNIST** (3 seeds; seeded re-run `runs/e20/seeded/mnist_mlp_linear.json`; the E17 artifact read 82.5% [74.1, 90.9] under an unrecorded draw). **Table 1's MLP row is the disjoint-content E18 construction, 89.5% [88.5, 90.6]** (row below): shared content is the construction on which re-layout reproduces the current task's own test set — the C0deg exclusion — so this row cannot carry the re-layout column | E17, `runs/e16_decomp/mnist_mlp.json` (an **E17** artifact under the shared E16 driver's path — see `runs/e16_decomp/README.md`); mean-of-ratios **82.51%**, t-CI [74.13, 90.90], widened by the **measured same-seed share floor 0.01pp** → **[74.11, 90.92]**; era `[True,True,True]`; identity 0.0e+00; capacity **266,752 params** = the arm `docs/BRITTLENESS_prereg.md` names | **new — and it is what restores "four architectures".** The entire uncertainty budget clears majority by **24pp**. Sits **inside** the LSTM band. **Scratch evidence is per-dataset: LSTM/HAR and MLP/MNIST are never pooled**, and no figure row pools them. |
| **FLOORS ATTACH TO QUANTITIES, not to runs** | the same two E17 replicate runs: **AVG floor 3.68pp**, **reader-share floor 0.01pp** (`runs/e16_decomp/mnist_mlp_floor.json`) | **new, methods.** Inferring the share caveat from the AVG floor errs by **two orders of magnitude in the pessimistic direction** — the rare failure where excess caution misreports. Observation, not mechanism: the nondeterminism perturbs both channels of the ratio together. Final floor tuple: **(arm × config × platform × threading × quantity × seed)**, per-pair, mechanism unclaimed. |
| full-rank storage cell **cites the adapters row 0.9183** | `w3c_fullrank_r0` (n=3, arm-recorded, `--adapter-rank 0`) reads 0.9133; Δ **0.50pp**, inside the arm's **1.49pp** measured per-seed floor | **disposition — one arm, one number.** Rule, two branches so preference cannot decide the next case: *a re-measurement INSIDE the arm's measured floor is a **floor observation**, and the canonical value stays the first verified measurement under recorded config; a re-measurement OUTSIDE the floor is not a choice but a **discrepancy investigation**.* Stated because the retained value is the higher one — the branch condition, not the outcome, is what makes it safe. |
| `runs/fullrank_ref` **superseded by MEASUREMENT** | the same configuration, verified and arm-recorded, reads **~0.918** against its 0.9470 | **strengthened** from "excluded by provenance". It was never this arm at any code state — the story closes. |
| adapters arm per-seed reproduction floor **1.49pp** | seeds 42 and 2024 bit-identical (15/15) across launches **weeks apart**; seed 1337 **0/15** | **new.** Catch 19's cleanest instance — same arm, config, platform, threading. Coincidence recorded, nothing resting on it: every floor pair in this program ran on **seed 42**, the seed separately known to run high on outcomes and now known to run deterministic on reproduction. |
| **LwF at its only competent λ is indistinguishable from doing nothing** | E16, matched config n=3: MNIST **−0.79pp [±4.42]**, HAR **+1.46pp [±4.75]**; floors **zero** on both (15/15, max \|Δ\| 0.000000) at the arm/config/threading the comparison uses; λ=0 positive control 15/15 with a live witness (`lwf_distill_loss` 6.18→14.90) | **new.** Reported at the instrument's resolution — **±4.4pp *is* the limit at n=3**, and distinguishing LwF from no-method here would need substantially more seeds. Not "loses to": the mixed-config −2.45/−2.86pp were computed against arms differing on the era-checkpoint axis. |
| **LwF's protection lands on the ENCODER — Li & Hoiem's stated mechanism is VINDICATED** | E16 H-D3, paired 12 cells/benchmark, `runs/e16_decomp/`: ΔF_enc **−0.0501 [−0.0821, −0.0180]** (MNIST), **−0.0599 [−0.0890, −0.0308]** (HAR) — **−20% / −79% of the encoder channel**, both CIs excluding zero | **new, and a PREDICTION MISS recorded at equal prominence** (~65% was on F_read). Their claim that preserving outputs "retain[s] the important shared structures" is confirmed, not refuted. |
| **…and it does not help, because that channel was not carrying the forgetting** | ΔF_total **−0.0022 [−0.0348, +0.0304]** (MNIST), **−0.0094 [−0.0524, +0.0337]** (HAR) — both straddle zero. The encoder gain is spent: ΔF_read **+0.0476 [+0.0140, +0.0812]** MNIST (**measured**), **+0.0447 [−0.0226, +0.1119]** HAR (**suggestive**) | **new** — *a method can work exactly as designed and still not help, when it operates on the channel that was not broken.* The thesis via the baseline's **success**, not its failure; it explains the null mechanistically rather than as an absence. Mechanism for the F_read worsening **labelled untested**. |
| **3 of 4 λ values fail competence on both benchmarks** | DIAG gaps vs a 5pp bar: MNIST +0.125 / +0.625 / +0.670; HAR +0.083 / +0.241 / +0.255. Across-λ AVG spread 0.2331 vs across-seed sd 0.0198 (**11×** — not noise) | **new, and the paper's cautionary exhibit.** Two counterfactuals: **forgetting-selection picks λ=16** (0.0100, a *99% reduction* from a model whose DIAG fell to **0.3007**, −67pp of competence); **AVG-without-floor picks λ=1.0** (a 9pp "win" sitting 12.5pp below competence). **The metric and the floor catch different wrong answers.** |
| **era-checkpointing is CONFIGURATION, not observability** | `_save_era_checkpoint` reloads, evaluates, refits and gates inside training. Final-AVG effect **MNIST +2.13pp, HAR −1.67pp** (opposite signs — no blanket correction); on the reference arms at n=3, **−1.66pp / −4.32pp**. Confirmed by prediction: matching the flag reproduced the sweep runs **15/15 bit-identical, AVG gap 0.0000**, both benchmarks | **new, methods contribution.** Every comparison must be uniform on this axis, and **uniformity is checked ACROSS the arms of a comparison** — checking within each arm reads "uniform" and hides the mismatch. |
| determinism map (5 cells) | MNIST `plain_lstm` 4t ✗ / 1t ✓; MNIST `mafc_off` 4t ✓(×2); MNIST `lwf` λ0.25 4t ✓ / 1t ✗; HAR `mafc_off` 4t ✓ / 1t ✗; HAR `lwf` λ0.25 4t ✓ | **new.** `plain_lstm` is the **sole outlier** — every other arm tested is 4t-deterministic and 1t-nondeterministic. The "BLAS reduction order" mechanism is **RETRACTED**: no thread-count-monotone story produces both directions, and single-threaded BLAS has a fixed reduction order. *Determinism belongs to (arm × config × platform × thread-count), per-pair, mechanism unknown until demonstrated. Pinning is an axis, not a stabilizer.* |
| `plain_lstm` **0.4319** (n=3, permuted MNIST) | `runs/w1_vanilla_seed{42,1337,2024}`, arm recorded in all three; `runs/MEMO_w1.md` §2 | **supersedes 0.4399.** Δ −0.80pp, **inside the arm's own 1.44pp 4-thread floor** — the old number was never in conflict with the new one. No drift to attribute. |
| `ewc_l200` **0.4329** (n=3) | `runs/w1_ewc_l200_seed*`; `ewc_lambda` 200.0 and `ewc_param_filter` None recorded in-artifact; witness `ewc_penalty_mean` nonzero on tasks ≥1, 3/3 seeds | **supersedes 0.4330 (Δ −0.01pp).** Replaces a mixed-provenance triple one member of which lived inside `plcm_ewc_sweep.json`. **Being right is not the same as being verifiable, and the audit's job was the second.** |
| **the EWC null survives verification** | EWC **0.4329** vs `plain_lstm` **0.4319** = **+0.10pp**, against a **measured 1.44pp floor** | **new, with alibi.** Three layers, all from artifacts: **active** (witness nonzero, growing **~150×** across tasks), **correctly configured** (λ + param filter recorded), **honestly bounded** (floor stated). Reported at its instrument's resolution; deliberately **not re-run to polish** — the claim is existence, not magnitude. EWC per-seed spread **9.3pp** on one config vs the sweep's 8.1pp across all five λ. |
| determinism is **(arm × config × platform × THREAD-COUNT)** | 2×2 (`runs/MEMO_w1.md` §4): `mafc` **15/15** at 4 threads across two independent pairs (4 runs); `plain_lstm` **0/15** at 4 threads (max \|Δ\| 0.1060) and **15/15 at 1 thread** | **new, and a RETRACTION.** "The permuted-MNIST CPU path is bit-deterministic" was generalized from one arm and is **withdrawn, not hedged** — with it, the "historical deltas are certainly code state" inference. `mafc`'s zero floor stands on four runs. Forward rule: **floor pairs run pinned; comparison triples may run at default threading carrying their measured floor.** |
| `fullrank_ref` exclusion **restated** | grounds are **no `arm` field** + **name lineage** (`lowrank_r32`/`r64` siblings), *not* the 2.91pp delta | **narrowed.** The "no third option" certainty claim is retracted — `plain_lstm` shows 1.44pp same-seed spread is possible. **Unverifiable configuration remains sufficient grounds.** `runs/fullrank_ref/PROVENANCE.md`. |
| storage-frontier table provenance | 3 of 4 cells are n=1 unrecorded-config; **low-rank siblings exist** (`lowrank_r{32,64}_seed{1337,2024}`), full-rank is an orphan | **flagged, price 3 runs not 9** (`jobs_w3c_fullrank`). The orphanhood resolves `fullrank_ref`'s identity as a side effect: a lone reference run beside two arms that have seed sets is a **study reference**, consistent with its name. |
| the two "vanilla" rows are **different arms** | plain LSTM **0.4399** (`lstm_seed*`, `plcm_ewc_sweep.baseline_lstm`) vs MAFC-off **0.4304** (`e4_off_seed*`); the attribution uses the second, exactly | **citation lock** — tables print artifact arm names (`plain_lstm` / `mafc_off`), never "vanilla". ~1pp rides on the swap and no output inspection would catch it. |
| tier-0 does **not** transfer cross-regime | E13 H-S2: MNIST/OFF stack **+0.120** vs the 0.30 bar (E8's own H-R4, reused unchanged); 12/12 cells, guarded == forced | **new** — branch (C). The family is weak in **both** regimes (HAR 0.075 < MNIST 0.120), so it is **not** HAR-scoped; it is weak everywhere measured. |
| **98.8% [98.3, 99.4] on a pretrained ResNet-50** (task-IL, 20 tasks; t over 3 seeds [98.27, 99.39] — numerically the same as the earlier per-cell [98.25, 99.38]) — reader-dominance is **not transformer-specific** | E14 H-R1, `runs/e14_hv1.json`, 3 seeds × 19 old tasks = 57 cells; `F_enc` **0.0076 [0.0042, 0.0110]**, `F_read` **0.6428**; instrument R **−0.0009**, **0/57** cells over bar; identity residual 0.0e+00; P3a 0.0e+00 with **both positive controls fired**; build gate 4/4 (`runs/e14_build_gate.json`, weight hash `4aa4ff84c7e0`); preconditions CLEARED (`runs/e14_preconditions.json`) | **new** — a second pretrained family, and a convnet. The claim spans **four architectures** (LSTM, MLP, ViT, ResNet) and two pretraining regimes. **Narrowed to three on 2026-08-17** when a figure audit found no MLP reader share existed; **restored to four the same day by E17, which measured one.** The narrowing is **superseded by measurement, not withdrawn** — it was correct on the evidence then available, and the chain *claimed → audited → narrowed → measured → restored* is the record, not an embarrassment in it. The MLP's two real roles were the E6 brittleness control (`runs/brittleness/summary.json`, transfer accuracy — and its own branch fired *DIES*) and an adapter-arm replication (`runs/mlp_adapt_seed*`); neither computed `F_read`/`F_enc`. `F_enc` is **7.9× smaller** than the ViT's — the lowest encoder-side damage in the program. **Pooled only**: E14's per-cell reproduction floor is **10.8pp** (`runs/e14floor_base_seed42_rep2`), so no per-cell or cell-granular cross-architecture delta is licensed. |
| **Disjoint-content MNIST decompositions (E18 constructions), the known-map rows that carry a re-layout column:** Permuted/MLP share **89.5% [88.5, 90.6]** (F_enc 0.0205, F_read 0.175, R +0.002, F_total 0.194), Permuted/LSTM-OFF **76.9% [69.8, 84.0]** (0.172 / 0.572 / +0.005 / 0.740), Rotated/MLP **91.5% [90.0, 93.0]** (0.039 / 0.417 / +0.0003 / 0.456); forgetting after re-layout with the known map on the same 12 cells: **−0.0007 / −0.0002 / −0.0037** | `runs/e18_{pmd_mlp,pmd_lstm,rmd_mlp}/decomp.json` — read out of the E18 screen (`cures.json`: audited loader, probe draw 20260916 recorded, path identity per cell, identity residual 0.0), not recomputed; re-layout from `hx2_c0deg.json`, identical on the 12 cells | **new 2026-09-18.** Table 1's known-map half is now four rows (S72 LSTM/HAR 77.8 [49.7, 103.7], re-laid −0.038; these three) and the map removes all of it in every row. The LSTM/HAR t-interval is wide because three seeds read 64 / 84 / 82: the per-arm share is poorly pinned and the finding rests on the direction across architectures, not on any row's precision |
| **On every scratch-trained Table 1 row the deployed readout is a SHARED head that trains through every task; `F_read` there contains head drift.** All four record `use_task_heads: False`; the era head h_k is a snapshot, deployed nowhere; the deployed head is W_T. Split per cell: `F_read = [acc_refit_T − acc(h_k Z_T)] + [acc(h_k Z_T) − acc(W_T Z_T)] = F_read^frozen + ΔH`. Pooled ΔH: **HAR/LSTM −0.3pp** (per cell −6.0 to +2.3; 5/12 negative), **P-LSTM +1.1**, **P-MLP +1.5**, **R-MLP +2.1** (per cell +0.2 to +4.8; 0/12 negative). Frozen-head shares **78.0 / 76.6 / 88.6 / 91.1** vs the reported 77.8 / 76.9 / 89.5 / 91.5. On the pretrained rows the premise holds literally: per-task heads are **bit-identical** between their own era checkpoint and the final one (ResNet-50 heads 0 and 10, ViT head 0, seed 42; `set_task` freezes the outgoing head, `plcm.py:322-324`), so there `F_read^frozen ≡ F_read` and `F_frozen ≡ F_total` | `runs/head_drift/{s72_off,e18_pmd_lstm,e18_pmd_mlp,e18_rmd_mlp}.json` (`scripts/head_drift.py`; C-ID: acc_orig and acc_ceiling reproduce the seeded decompositions to 1e-6 in 48/48 cells; identity residual 0.0; path identity per cell; HAR on x86, as-executed partition) | **new 2026-09-19** (reviewer weakness 5; D1 review B2). §3.1's "any change between the two terms arises from the encoder alone" is true on the per-task-head rows and false on the shared-head rows by 0.3–2.1pp of head drift; the paper states both cases and what `F_read` contains on each. No ordering in Table 1 moves; the shares move ≤ 0.9pp |
| **The structure of feature drift on the four known-map scratch arms (D1).** Fit a linear map between era features (frame k) and final features on the same samples re-laid into frame T. (i) **Encoder drift is shared across tasks once presentation is removed**: another task's repair costs ≤ 0.6 pp (adjacent) / ≤ 2.1 pp (distant) pooled on every arm; in the old frame it costs 10–62 pp — the presentation component is per-task by construction. (ii) **The drift is neither near-orthogonal nor linear**: spread log(σ₁/σ_d) 8–12 (bar 0.5), 54–66% of singular values below 0.5, repair residual 41–45% of the feature norm; era features all live. (iii) **The encoder term tracks the non-linear residual** (r +0.81, partial +0.75) more than collapse (r −0.55, partial −0.33). (iv) **Re-laid F_enc is zero within resolution on all four arms** (−0.017 / +0.011 / +0.007 / +0.004 against 0.087 / 0.172 / 0.021 / 0.039) — the map removes the encoder term, measured on F_enc itself | `runs/d1/` (`scripts/d1_fit.py`, `d1_diagnose.py`, `d1_row.py`; `runs/d1_row.json`; `runs/MEMO_d1.md`); contract `docs/D1_prereg.md` signed v2. Controls: C-ID 0.0e+00 48/48; C-SHUF must-fail 12/12 × 4 arms × 2 frames; C-TARGET own(k) within resolution of the era accuracy; C-λ verdicts stable; `D_out·Wᵀ` ≤ 4e−14 (the corrected identity). 10 predictions scored: 7 fired, 2 misses (near-orthogonal MLP at 65%; residual < 10% at 70%), 1 did not fire; D3 on the pretrained arms (frozen deployed heads, F_frozen ≡ F_total, 57 cells each): neither projection norm tracks the loss (R² 0.01 / 0.06; β_in miss at 60%), caveat n_train 2,500 ≈ d; a paired linear repair read by the frozen head recovers 0.89 of a 0.94 ceiling on both backbones | **new 2026-09-19.** For §4: drift is shared after re-lay and per-task before it; the encoder term is what a linear map cannot capture; the map removes it. For §9 of the contract: the self-supervised re-alignment method has no foundation — its sharing constraint holds only with the map applied, and with the map applied there is no encoder term left to recover |
| **On pretrained backbones, re-laying into the current frame recovers nothing and costs (E23).** ViT-B/16 and ResNet-50 on `cifar100_permuted` (B6: one known patch-consistent permutation per task, era checkpoints, 20 tasks, 3 seeds) vs the same backbones on unpermuted B1: F_enc raw → re-laid **0.080 → 0.099** (ViT), **0.012 → 0.025** (ResNet); deployed accuracy re-laid **−6.3 / −2.4 pp**; (E2) fails on both, (E1) holds at the floor with B6's encoder term above B1's on 3/3 seeds each. On every scratch arm the same operation takes F_enc to zero within resolution (D1). **The known-map discriminator is scratch-scoped on the evidence.** Arm B: the MLP on the S72 construction reads share **100.2** [85.6, 114.2] vs the LSTM's 77.8, F_enc −0.001 — the sensor result is not LSTM-specific, it is more reader-dominated on the MLP | `runs/MEMO_e23.md`, `runs/e23_row.json`, `runs/e23/e23_{vit,rn}/decomp_seed{s}.json` (+ `decomp_floor42`), `runs/e20/seeded/har_e23_mlp_linear.json`; contract `docs/E23_prereg.md` (signed v2, option (b)). Controls: C-WIT uniform; relay exact 0.0 every seed; fp16 reload ≤ 0.008; P2(a) 0.70/0.71/0.17; P2(b) permuted repeat control 0.005/0.002 (opt-in fixed perm, regression clean); floors: ViT relaunch 2.9 pp AVG, 3.4 pp re-laid F_enc, ResNet 0.1/1.0, arm B identical. **P1 FAIL on all three arms, ruled** (a)/(c): the contracted reference cannot read the construction (frozen trunks 0.72/0.70, linear-on-windows 0.55; arms above it by 6–24 pp) — a defective gate, not a failed arm; competence gap 12–19 pp printed beside the crux. **Wrong-permutation must-fail FAIL with cause** (right and wrong indistinguishable when the right map recovers nothing). 10 predictions: 5 fired, 4 misses, 1 did not fire | **new 2026-09-20.** §4.2 narrows to *encoders overwritten to the current frame*; §8 already carried the branch. **Confound CLOSED 2026-09-20 (E23-B, row below):** a twenty-task scratch LSTM re-lays to +71.7 pp deployed and F_enc → 0, so the scope is the training regime, not the sequence length. Reading offered: a trunk trained across twenty permutations is spread over twenty layouts, which predicts both the competence gap and "nothing to re-lay into" |
| **Sequence length does not explain it (E23-B).** Scratch LSTM on disjoint-content Permuted MNIST at three points, one factor at a time — (T=5, 12k/task) = E18's arm, **(T=5, 3k)** and **(T=20, 3k)** built with an opt-in `content_chunks` so the last two share the first five tasks' content and permutations exactly. Re-laid F_enc: **+0.011 / −0.005 / −0.006** against floors 0.011 / 0.033 / 0.030, from raw **+0.172 / +0.126 / +0.169**; deployed accuracy under re-layout **0.195 → 0.935, 0.159 → 0.844, 0.161 → 0.878**, improving in **12/12, 12/12, 57/57** cells. At the same twenty tasks and twenty layouts the pretrained trunks instead lose 6.3 / 2.4 pp and improve in 13/57 and 15/57 | `runs/MEMO_e23b.md`, `runs/e23b_row.json`, `runs/e23b/*/decomp_seed{s}.json`; contract `docs/E23B_seqlen_prereg.md` (registered before launch). Controls: C-ID vs `runs/e18_pmd_lstm/decomp.json` **0.0e+00 12/12**; C-CONSTR (num_tasks, content_chunks, construction_fingerprint) on every artifact; relay identity and round trip exact; identity residual 0.0; **both floor replicates bit-identical**; **wrong-permutation must-fail PASS 4/4, 4/4, 19/19×3** — the same control that FAILED on both pretrained arms, because it can only discriminate where the right map recovers something. 6 predictions: 5 fired, 1 miss (3k vs 12k images/task moves raw F_enc by 4.6 pp, not ≤ 2) | **new 2026-09-20.** **§4.2 and §8's scope sentence: training regime.** The discriminator works on encoders overwritten to the current frame — measured at 5 and 20 tasks, 12k and 3k images/task, LSTM and MLP, MNIST and HAR — and not on pretrained trunks fine-tuned across the same twenty layouts. Mechanism the pair supports: a scratch encoder ends in the last task's frame; a pretrained trunk ends in none in particular (it reads task-k content better in P_k than in P_T), so there is no current frame to carry anything into |
| feature stability does **not** confer span stability — **second architecture** | E14 H-R4: era control **+0.0001** (max +0.0100 vs the 0.05 readability bar, rank 5/2048); θ_T span projection recovers **+4.0%** of the reader gap vs E15's ViT **+0.5%** | **extended** — and the sharper instance, because it is measured on the arm with the *smallest* `F_enc` in the program: features that barely moved, a span that rotated anyway. |
| E14 carries **no deployment recommendation** | frozen ResNet-50 trunk + per-task linear heads: **0.9552**, forgetting **0 by construction**, vs **0.3257** for the fine-tuned trunk it diagnoses — the trivial use of the resource wins by **62.95pp** | **scope lock** (catch 24). The diagnosis — *why* fine-tuning forgets — is well-posed regardless; the setting is not one to recommend. Priced on **both** pretrained backbones before either was reported. |
| arc ordering is **not** attributed to pretraining | E14 varies architecture family with pretraining held fixed (`docs/E14_resnet_prereg.md` §8) | **explicit non-claim.** The 66–88% → 91.8% → 98.8% arc is *ordered along* the pretraining axis and **uncontrolled on it**. A speculative alternative ordering — linear separability of the penultimate representation — is flagged in `runs/MEMO_e14.md` §10.3 as a paper-2 hypothesis, not a finding. |
| tier-0 anatomy | projection alone **>** era head alone in both regimes (HAR +0.063 > +0.033; MNIST +0.119 > +0.062); on MNIST the era head adds **+0.001** | **new** — the ordering does not flip across regimes; visible only because the composite rule measures each cure alone |
| tier-0 composition with adapters | E13 H-S3: MNIST/ON guarded **+0.270** vs forced **−0.084** — sign disagreement under its own bookkeeping (EV1 requirement); separately, 6/12 cells guard-excluded, which is *half*, not more than half, so the evaluability threshold did not fire | **UNEVALUABLE at this residual scale**; both columns recorded |
| stored-prototype reader repair **fails universally**, and worst where features are most stable | E13 + E15: HAR ρ **0.075**, MNIST **0.120**, ViT **0.005 [−0.001, +0.011]** (95/95 cells, guarded == forced). Mechanism measured, not argued: at the ViT's θ_T the era-prototype span is **information-preserving** (rank control −0.0015 vs era ceiling, rank 5/768), old-task information **persists** (refit 0.888), yet projection onto the stored span recovers **0.4pp of a 67pp gap**. | **new** — **feature stability does not confer span stability.** Discriminative directions rotate out of stored subspaces even when representation quality is preserved (F_enc 0.06). Reader repair requires **span stability**, which no measured regime provides. |
| cure section's load-bearing result | **E10's C3 alone** — certified pseudo-refit, ρ 0.974, end-to-end forgetting −81.6% | tier-0 demoted to **weak baseline, anatomized**; C3 untouched by E13/E15. **Why C3 works where projection fails:** it **refits** the reader to current geometry rather than **projecting** onto stored geometry, sidestepping span staleness entirely. The failed cure explains the successful one. |

---

## History

> **INCOMPLETE — REQUIRES BACKFILL.** The ledger was maintained conversationally
> for the whole program prior to this file. Only the entries below were present in
> the exchange available when this artifact was instantiated. Every earlier
> version exists in prior recorded exchanges and is quotable from them, but was
> **not** available here, and reconstructing them from memory would reproduce
> exactly the provenance failure this file exists to close (catch 21). Backfill by
> quoting, not by recall.

### E9 era — repair clause (fragment)

> partially repairable... ρ = 0.347, ~3K floats/task

Triggering result: E9 transport estimate, tier-0 stack registered as baseline.
Superseded 2026-08-02 by the freeze — the 0.347 was never computed in-repo.

### E23-B — sequence length ruled out — 2026-09-20 — current

- **The confound E23's ruling named is closed.** A twenty-task scratch LSTM
  behaves like every other scratch arm under re-layout (F_enc → 0, +71.7 pp
  deployed, 57/57 cells), so E23's pretrained result is about the training
  regime. §4.2 and §8 can be written.
- **Two construction defects found and fixed before the read**, one per half
  of a witness: `content_fingerprint` was blind to the chunk boundaries
  (T=5 and T=20 collide), and the fix for it was dropped by
  `arm_provenance`'s whitelist. Re-ran the eight jobs rather than argue a
  conditional witness. Both rules in CLAUDE.md.
- Appendix table **67 scored, 14 misses**.

### E23 — the pretraining confound closes the other way — 2026-09-20 — current

- **E23 run and read.** Build gate on both backbones; 18 runs; the relay
  verified bitwise against the dataset's own rendering; one crash (per-image
  `apply_perm` handed a batch) fixed and respawned. Result: on ViT-B/16 and
  ResNet-50, re-laying task content into the current frame recovers nothing
  and costs 6.3 / 2.4 pp deployed; F_enc rises. The map is the discriminator
  on the scratch arms (D1: F_enc → 0, 4/4) and not on the pretrained ones.
- **P1 failed on all three arms and was ruled a defective gate** — the
  frozen/linear references cannot read the permuted or flattened inputs
  while the arms beat them by 6–24 pp; read under (a)/(c) with the
  competence gap printed. Wrong-permutation must-fail FAIL with cause.
- **Arm B** (MLP on HAR): share 100.2, F_enc −0.001 — the shared-head MLP
  opt-in (`mlp_input_dim`) shipped with a bit-identity regression.
- Appendix table **61 scored, 13 misses**; E23's four are the ledger's
  largest single calibration event (priors transferred across a regime
  boundary). Open: the sequence-length confound (5 vs 20 tasks) — a
  twenty-task scratch LSTM control before §4.2/§8 are rewritten.

### D1 — the structure of feature drift — 2026-09-19 — current

- **D1 v2 signed and run** on the four scratch arms (Modal x86, 4 CPU jobs,
  minutes). Shared drift after re-lay, per-task before it; neither orthogonal
  nor linear; the encoder term tracks the linear residual; re-laid F_enc zero
  within resolution on 4/4 arms (E23's (E1) measured early on the scratch
  half). The method the contract was scoped for has no foundation; the
  diagnostics stand as §4 findings. Appendix table: **49 scored, 8 misses**,
  now four in each direction. Two D3 rows on the pretrained arms pending
  `jobs_d1_pretrained`. Recorded: my C-λ flag used a 0.5 pp bar under a
  3.3 pp resolution (R3's error, in my own check); corrected before reading.

### Head drift on the shared-head rows — 2026-09-19 — current

- **§3.1's frozen-readout premise, checked per arm.** Per-task heads on the
  pretrained rows are bit-identical across era checkpoints (verified on the
  checkpoints, not the code). Every scratch row is shared-head; its `F_read`
  splits into a frozen-era-head part and a head-drift part ΔH measured at
  −0.3 / +1.1 / +1.5 / +2.1pp pooled (HAR-LSTM / P-LSTM / P-MLP / R-MLP), with
  HAR's sign varying by cell. The paper's §3.1 now covers both cases. Found by
  the D1 review (B2), which also caught a false displayed identity
  (`W(T−I)(I−P_W) = 0` reads 16.2 numerically; the identity that holds is
  `(T−I)(I−P_W)Wᵀ = 0`) — H-X2's class, in the clause deciding which
  outcomes are possible.
- **E23 §4** now states the frame of every era term explicitly (x_k in P_k,
  never re-laid), after D1 v1 paired a re-laid input with the era encoder.

### Table 1 conventions and the E18 rows — 2026-09-18 — current

- **Interval audit.** The appendix states "95% t-intervals over seeds"; the
  ViT, ResNet and class-IL shares had been printed with ±1.96 SE over
  *cells* (reproduced to the second decimal), the MLP with t over seeds — the
  widest convention on one row and the narrowest on three. **Ruled: t over
  seeds throughout, pooled-of-means point estimates**, the table made to match
  the stated convention rather than the convention changed to match the table.
  ViT 91.8 [89.6, 94.1]; ResNet 98.8 [98.3, 99.4]; class-IL 97.9 [95.1, 100.6];
  LSTM/HAR 77.8 [49.7, 103.7] — the last is what three seeds at 64/84/82 say,
  and it is information, not a defect.
- **Table 1's MLP row moves to the disjoint-content E18 construction** (89.5
  [88.5, 90.6]): the shared-content E17 construction is the one on which
  re-layout reproduces the current task's test set (the C0deg exclusion), so it
  cannot carry the re-layout column. The seeded E17 value (82.2 [76.8, 87.4])
  replaces the unrecorded-draw 82.5 [74.1, 90.9] where it is cited.
- **Re-layout column, measured not labelled.** Forgetting after re-layout with
  the known map, on each decomposition's own cells; "—" where the construction
  has no map. E23 tests whether the pretrained rows behave like the known-map
  half when given one.
- **B1 provenance correction.** The CIFAR decompositions draw no probe subset
  (full task train set, sequential loader); the "unrecorded draw" caveat was
  imported from the HAR/MNIST path and did not apply. E23 §2 corrected; no
  re-run.

### E21 — repair under an approximate map — 2026-09-18 — current

- **E21** (`docs/E21_prereg.md` signed v2; `runs/MEMO_e21.md`; `runs/e21_row.json`):
  three perturbation families on the S72 checkpoints, 36 artifacts. First sweep
  lost 11/13 gain jobs to an unbounded line search; the continuous families
  re-run under a **bounded** parameterization stated in §3 before the re-run
  (unbounded artifacts kept in `runs/e21/_unbounded/`, unread). **Rulings:** A2's
  C-ID at the *measured* refit floor, two test windows per cell (the ruling was
  on "max 0.0049 observed"; the "one window" gloss was wrong by a factor of two
  — task 0 has 409 windows — and a description does not override the
  measurement it describes); C-DT recorded FAIL with cause, not re-barred.
  **Read:** no C0deg/bridging crossing in any family; wrong-locus cost 14–20pp
  at matched map quality, 13.4 exact; era teacher beats self-supervision
  nowhere under the registered read, while repairing a one-transposition
  wiring error to the exact-map value in the paired measurement (follow-up
  pre-registered before it is read). Appendix table: **39 scored, 6 misses**
  under the convention now stated in the appendix (miss = above-even-odds
  prediction that did not occur; two 40% entries the old count carried as
  misses are did-not-fires). Misses run in two directions: four toward more
  structure / less determinism (E16 ×2, E20-B ×2), two toward more encoder
  invariance than predicted (E18's lossy rotations, E21's transposition).
- **E23 signed** (`docs/E23_prereg.md` v2): the pretrained known-map cell is
  `cifar100_permuted` (E12's B6), relaunched with era checkpoints on both
  pretrained backbones; the crux is a displayed equality
  F_enc(B6, re-laid) ≈ F_enc(B1). October.

### E18 first read, E20-B band, R1/R2 — 2026-09-16 — current

- **E18** (`docs/E18_prereg.md`, signed with four conditions; `runs/MEMO_e18.md`):
  disjoint-content Permuted (MLP, LSTM-OFF) and Rotated (MLP) MNIST built,
  gated (loader hashes 60/60, anchor pair 25/25 on the amended code), run
  (15/15) and screened with a **recorded** probe draw. Forgetting, diag: none
  0.216 / 0.740 / 0.456 → bridging 0.024 / 0.176 / 0.041 → **C0deg −0.001 /
  −0.000 / −0.004**; ranking resolved on every construction, both formulas.
  Bridging passes its now-real test (96–100% of the reset gain) and is
  dominated everywhere. The rotated lossy pairs read within 0.3pp of the exact
  one — the map-fidelity crossover needs calibration error, not interpolation.
  **Permuted-LSTM:** reset-with-data caps at 0.74–0.82 (F_enc ≈ 0.19) while
  C0deg reads 0.93–0.94 — the encoder's damage is layout-specific, not
  content-specific; §4's two-origins paragraph extends to F_enc (drafted in
  the memo). P3, the MNIST C0deg controls, tier-0, storage and `e18_row` are
  still to run; ports gated on the human read.
- **E20-B**: the band relaunched on `har_subject` under uniform arms — row
  above. v1-ON's 22pp share swing at seed 42 is between different models
  (0/25 cells) and survives the seeded draw: a determinism finding for the
  appendix.
- **R1**: the MLP probe's live positive control could not be built on these
  features (seven designs; linear 0.67–0.71 on every "nonlinear" target; MLP
  +0.10 above it every time); recipe certified on a matched synthetic; live
  half replaced by a non-vacuous gate, which then failed on both arms — row
  above. E20-A withheld.
- **R2**: the unrecorded probe draw — row above. S72 and E20-B re-run seeded;
  cited values are the seeded ones.

### C0deg controls — 2026-09-15 — superseded in part by the entry above (amends the E11 cure rows; the E14 claim rows are untouched)

Origin: the E18–E20 contract review, finding B1. Full record `runs/MEMO_c0deg.md`.

- **Finding.** The deployable form of "just invert the known transform" —
  re-layout old inputs into the *current* frame, run the current model — was
  computed in `runs/e11_e10/cures_e10.json` as `C0deg` and never read, because
  `cure_screen.py` labelled it a "ceiling artifact" on E8's shared-window
  construction and the label survived E10's benchmark repair unexamined. Read
  through the ledger's own H-X2 instrument it is the best storage-honest method
  in the program: OFF forgetting **−0.0247 [−0.0346, −0.0148]** against
  bridging's 0.0886, intervals separate, using a strict subset of bridging's
  resources.
- **Controls, ruled sequence.** (a) no-shift positive control bit-identical
  12/12; (b) subject-disjointness asserted from `test_subjects`; artifact
  consistency 0.0e+00. **CC** (neighbouring-task wrong map, "below the right map
  in every cell") **FAILED as pre-committed** — 9/12 OFF, 11/12 ON; every
  failing cell differed from the right map by gain/offset or rotation only.
  **Ruled: the bar does not move.** CC′ (permutation-only wrong maps, bar stated
  before launch) **PASS 12/12** on both wirings, both arms. CD ablation:
  permutation alone carries 84% (OFF) / 71% (ON) of the recovery.
- **Two rulings.** (1) Bridging is not a method contribution as framed; C0deg is
  the finding; bridging's regime is not "the map is known". (2) H-X2 is
  *ambiguously registered*: the prereg said "the standard forgetting formula",
  the instrument's docstring cited `metrics.py` for the diag form, `metrics.py`
  implements peak-clipped. Diag is the paper's form (§3.1's definition of
  `F_total`); the registration was defective; the row carries both. Standing
  rule: a registered hypothesis names its formula by displayed equation; file
  citations are witnesses, not definitions.
- **For §4 of the paper.** E10's forgetting is presentation drift entirely
  (no-shift arm −0.0497 diag). On HAR the readout share measures layout
  misalignment with a known map; on Split-CIFAR it measures a stranded head with
  no map. Same signature under the decomposition, different repair.
- **Found while closing E16 §7.2:** the E16 LwF HAR arm and its floor pair
  ran `benchmark: 'har'` (E5 shared-window) per their own `arm` field, while
  bridging is `har_subject`. The E16 contract pinned the E10 construction; the
  launcher wrote `har`. The §7.2 pair therefore differed on two axes; both arms
  were relaunched on `har_subject` with uniform flags (era + fp32 shadow, 4
  threads), the LwF λ sweep moving with it. Nothing in the E16 rows above is
  affected — every within-E16 comparison was uniform on `har`.
- **E16 §7.2 closed the same day** (row in clause sourcing): bridging > LwF
  **fired** under both formulas; C0deg ahead of both; λ\* = 1.0 on this
  construction, not 0.25; LwF's reproduction floor is per-λ (0.25: 1/25
  cells; 1.0: 25/25). Competence floors are per-construction; §7 says so.
- **Ruling A — the headline moves to S72.** The unrecorded-config rule applied
  to the paper's most-quoted number: the E11-era OFF arm has no `arm` field,
  no era status, and a code state two weeks removed; the relaunch has all
  three plus bit-identical floors. Bridging's reduction reads **0.386 → 0.095,
  −75%** (was −81.6%); ρ **1.03 [0.83, 1.23]** (was 0.97); C0deg **−0.038**
  (was −0.025). E11-era rows retained, superseded by provenance, not
  withdrawn. The shipped 89% was true under the ledger when sent; future
  communication uses S72 and explains the supersession if asked.
- **Ruling B — bridging has no regime where the map is known.** C0deg −0.038 <
  bridging 0.095 < LwF 0.226 under uniform arms. Bridging's position: a
  measured intermediate with a named, unclaimed regime (a self-contained head
  where the pipeline cannot acquire a preprocessing step). Not a method
  contribution. **Consequence: E18 redrafts around C0deg** — the known-map arm
  in the known-vs-estimated question (`docs/E18_prereg.md`): nothing / SDC-port
  / LDC-port / C0deg / bridging / reset-with-data.
- **Standing requirement:** fp32 shadow on every era-checkpoint run entering
  a comparison at the 1e-6 gate; fp16 reloads miss by one window.

### E14 — ResNet-50 on Split-CIFAR-100 — current

Memo: `runs/MEMO_e14.md`. Branch (A). **The program's last training run.**

- **H-R1 MET** — reader share **98.8% [98.3, 99.4]** (t over seeds, 2026-09-18; per-cell was [98.25, 99.38]), CI lower bound 48pp
  above the 50% bar. Task 0 across seeds: ceiling 0.9447 → deployed 0.2807 →
  **refit 0.9313**. Reader-dominance is **not transformer-specific**.
- **H-R2 (descriptive)** — the arc continues and steepens: 66–88% → 91.8% →
  **98.8%**. Reported as an **ordering**, not a measured gap (floor, below).
- **H-R3 (descriptive)** — `F_enc` **0.0076**, **7.9× below** the ViT's 0.0599
  and an order of magnitude below the LSTM/MLP range. Lowest in the program.
  The contract's branch (C) anticipated the opposite direction; the asymmetry is
  recorded, not resolved away.
- **H-R4 (descriptive)** — span rotation replicates on a second architecture:
  era control +0.0001 (rank 5/2048), θ_T projection recovers **+4.0%** vs the
  ViT's +0.5%.
- **Reproduction floor 10.8pp per cell** (5/210 cells bit-identical) — the least
  reproducible arm in the program, **measured before the claims were written**.
  Every E14 claim is pooled-with-CI for this reason; the headline moves 0.27pp.
  fp16 reload delta max 0.00400, **27× below the floor** — clause closed.
- **Preconditions CLEARED** — P0 4/4 green; P1 0.9552 frozen / 0.9446 trainable
  (gap +0.0106); P2(a) 0.6620; **P2(b) 0.0010**; P3a 0.0e+00 with both positive
  controls fired; P4 alignment assert printed.
- **Catch 33** — P2(b) had **no launcher job**; discovered after the headline
  runs, built and passed before the memo closed. See `CLAUDE.md`.
- **Catch 24 priced** — the frozen trunk (0.9552, zero forgetting) beats the
  fine-tuned trunk it diagnoses (0.3257) by **62.95pp**. No deployment
  recommendation follows.

### E12 — ViT-B/16 on Split-CIFAR-100 — superseded by E14

Memo: `runs/MEMO_e12.md`. Full hypothesis set adjudicated; branch (A).

- **H-V1 MET** — reader share **91.77% [90.88, 92.67]**, CI lower bound 40pp
  above the 50% bar. Task 0: ceiling 0.9236 → deployed 0.2240 → **refit 0.8536**.
- **H-V2 UNTESTABLE at this placement** — the efficacy control's 1.34pp ceiling
  sits below H-V2's own 5pp bar (new standing rule: *a hypothesis bar must sit
  inside the instrument's demonstrated range, and the range must be demonstrated
  before the bar is set*).
- **H-V3 MET, corroborative** — within-task reader share **97.9% [95.1, 100.6]** (t over seeds, 2026-09-18; per-cell was [97.27, 98.45]);
  within-task refit 0.9411 against a ceiling of 0.9448 while the deployed system
  reads 0.0017.
- **H-V4 descriptive** — F_enc 0.060 (task-IL) / 0.021 (class-IL) vs the LSTM/HAR
  range 0.052–0.112. Encoder damage does not grow with capacity.
- **The monotone arc:** 66–88% (scratch-trained, small) → 91.8% (pretrained ViT,
  per-task heads) → 97.9% (pretrained ViT, shared head).

**Instrument:** P3a max |Δ| **0.0e+00**, identity residual **0.0e+00**, fp16
reload delta **0.00000**, R inside bar on every arm, 95/95 and 57/57 cells kept.
**Reproduction floor:** aggregate ~0.4pp, **per-cell ~15pp** — pooled-with-CI is
primary for the whole ViT section.

**Catches logged:** 30 (a gate's emphatic pass is not evidence of the right arm),
31 (a gate that cannot read — control without a counterfactual), 32 (a hardened
helper's value is lost when a new script re-implements its job), plus the
control-scoping rule and the dynamic-range rule.

### E11 — instrument correction — superseded by E12

Memo: `runs/MEMO_catch28_supersession.md`. Triggering results, all from recorded
script output, all recomputed on Modal (the environment the originals ran in):

- **catch 28** — the instrument hand-rebuilt the readout from raw `c_t` while the
  deployed classifier reads the composed `h′ = o⊙tanh(c_t′)`. Corrected, it
  reproduces the deployed path **exactly** (0.0000 in 8/8 MNIST cells, 8/8 E10
  cells). The old instrument was wrong by **0.1368 / 0.1046** mean absolute on
  E10 and 0.0439 / 0.0319 on MNIST.
- **catch 29** — `task_stats` is absent from `state_dict()`, so every reloaded
  checkpoint since E4b ran with coordinate alignment silently disabled. Restoring
  it reproduces the recorded matrix exactly (8/8 cells).
- **P-B** — the E7 task-head arms' deployed readout is a routed blend, not any
  single head (max |Δlogit| 7.804; up to 75% prediction disagreement).
- **H-X1 NOT MET → MET**: C3/OFF ρ **0.615 → 0.974**.
- **H-X2**: OFF **MET** at 0.0886 (CI upper 0.1013 marginally exceeds the bar);
  ON **NOT MET** at 0.1113. *[Annotated 2026-09-15: diag form. Under the
  peak-clipped form the prereg's citation pointed to, OFF reads 0.1198 — NOT
  MET. Ambiguously registered; see clause sourcing.]*
- **Branch**: (A) on OFF — H-X1 ∧ H-X2 — with the CI caveat recorded; ON fails
  H-X2.

**Calibration, this round:**

| prediction | outcome |
|---|---|
| C3's recomputed ρ within ±0.15 of 0.615 (~50%) | **MISSED by 0.36** — the program's worst quantitative miss. Priced as noise-like; it was bias-like, both floors moving the same direction. |
| MNIST 0.214 moves by >0.05 (~50%) | did not fire — moved 0.0099 |
| qualitative structure survives (~65%) | fired — split intact on both datasets |
| E5/E4 floors affected (~90%) | fired, and understated — E10 an order of magnitude larger than the MNIST anchor |

### E10 branch (C) — SUPERSEDED by E11

Full text above. Triggering results, all with recorded artifacts:

- **P1** DIAG 0.8139–0.8713 across nine cells, no capacity escape (`scripts/verify_runs.py`)
- **P2** 94.4% shift-attributable (no-shift 0.02507 vs shifted-OFF 0.44937) *[Annotated 2026-09-15: peak-clipped form. In the diag form the no-shift arm reads −0.0497 — content change alone produces no forgetting; the reading is strengthened, not weakened.]*
- **P3-v3** PASS — τ = 0.030151·D, control separation 2,128×, Control A fails / B fires at 9.98% / C passes
- **H-X1** NOT MET — C3/OFF ρ = 0.615, CI [0.548, 0.682] against the 0.80 bar
- **Branch (C)** formal: ¬H-X1 with ρ ≥ 0.50

---

## Appendix E exhibit — the reference-table supersession chain

Recorded verbatim for the methods appendix. It is the paper's own argument —
*floors measured, configs recorded, provenance over plausibility* — executed on
our own reference table, with two mid-stream errors caught by the instrument
rather than by luck.

> cited **0.9331** (unsourced average)
> → arithmetic archaeology: three runs averaging to it at four decimals,
>   **read as identification — this failed** (catch 34)
> → re-run under a recorded configuration
> → **floor pair proves the pipeline bit-deterministic** (15/15 cells,
>   max |Δ| = 0.000000), which disposes of the noise reading entirely
> → **0.9183**, every artifact self-describing
> → the OFF arm given the same treatment
> → attribution recomputed **verified-minus-verified: +48.4pp [±1.9pp]**

**Two documented errors, in opposite directions, both mine, both caught by the
measurement rather than by judgement:** an arithmetic match read as provenance
(too credulous), then mixed-sign cross-seed deltas read as noise (too quick to
recant) — while the floor pair, argued into existence for exactly that question,
was minutes from answering it. The companion rule in `CLAUDE.md`: *when the
deciding instrument is already running, interim evidence is for generating
hypotheses, not verdicts.*

**What the episode cost and bought:** one writing day; two catches (34 and its
companion); three superseded numbers, all moving 1.5–2.7pp in the direction that
removes one flattering run's influence; and a reference table verified on both
sides for the first time. **No qualitative conclusion changed.**

---

## Provenance

> maintained conversationally until 2026-08-02; instantiated as an artifact at
> E10 branch (C); history quoted from the recorded exchange; all future edits
> happen in this file first.

**Qualification on the history clause, required for accuracy:** the E10 entry is
quoted from the recorded exchange. The E9-era entry is a *fragment* quoted from
the same exchange. All earlier versions are **absent**, not quoted — see the
backfill notice above. The provenance sentence describes the intended standard;
the history section does not yet meet it.
