# MEMO — E17: the MLP reader share, and the deck's close

**Contract:** E17, signed as a claim-integrity repair — the deck-close lock's one
sanctioned exception. "Four architectures" sat in the ledger one-liner, the E14
row, Figure 1 rev 1, and both shipped emails, and no MLP decomposition artifact
existed.

**Outcome: the sentence is now true.** MLP reader share **82.51%**, full
uncertainty budget **[74.11, 90.92]**, entirely above majority with 24 points to
spare. The pre-commitment fired as written.

---

## 1. Preconditions

| clause | status |
|---|---|
| era-checkpoint status uniform | **PASS** — `[True, True, True]` (arm), `[True, True]` (floor) |
| identity `F_enc + F_read − R = F_total` | **0.0e+00**, both cells |
| script version pinned across cells | content hash of driver + `channel_decomp.py` |
| catch-32 label alignment | structural — `load_task_data` imported, never reimplemented |
| capacity is the brittleness arm | **266,752 params, 256-d** — matches `docs/BRITTLENESS_prereg.md` exactly, verified by construction |
| arm identity | `backbone: mlp`, `use_input_adapters: False`, recorded in all 5 artifacts |

**Precondition 0 was struck before launch, and its premise was FALSE AND
SHARED.**
The contract required a model-aware era-checkpoint audit because
`PLCM.load_era` "reconstructs the wrong class for non-PLCM models." The MLP is
not a non-PLCM model: it is a **`--backbone`** (`train.py`'s choices are
`{lstm, mlp, vit, resnet}`), builds as `PLCM(backbone="mlp")`, and
`runs/mlp_adapt_seed42` records `model_type: mafc`. Wave 1 struck era checkpoints
for `--model lstm`/`lstm_ewc`, which **are** a different class; the Wave-1
report generalised that to "non-PLCM models", and the E17 contract wrote the
generalisation in without checking `train.py`'s `--backbone` choices.

**One invented the category; the other contracted it.** That division is the
instructive part: *the error survived two authors precisely by passing between
them*, each treating the other's framing as already verified. It also violates a
rule stated three days earlier — *a contract clause that names an artifact must
be checked against the code that writes it, at contract time*. **This is the
catch family's social form**, and it is the reason the checking rule has to
apply to inherited framings and not only to one's own.

*A property of two model types read as a property of a category* — the week's
recurring shape, caught this time before it cost a run. Striking it removed the
wave's entire shared-trainer risk.

---

## 2. H-M1 — the result

| | value |
|---|---|
| per-seed shares | 86.3% · 81.2% · 80.0% |
| **mean-of-ratios** | **82.51%** |
| t-interval (n=3) | [74.13, 90.90] |
| **measured same-seed share floor** | **0.01pp** |
| **full uncertainty budget** | **[74.11, 90.92]** |

`F_enc` +0.0707 · `F_read` +0.3351 · `F_total` +0.4054 · instrument `R` +0.0004.

**Where it lands:** 82.5% sits **inside** the LSTM band's 66–88%. H-M2's
split-row branch does not fire; H-M3 (below 50%) is nowhere near.

**Predictions:** ≥66% at ~75% — **fired**. Within 66–88 at ~45% — **fired**.

**Scope, per the contract and not softened:** this does **not** restore dataset
breadth. Scratch evidence remains per-dataset — LSTM arms on HAR, MLP on
Permuted MNIST — and **no figure row pools across datasets.**

---

## 3. The wave's real discovery: floors attach to QUANTITIES

**The same two runs produced a 3.68pp AVG floor and a 0.01pp share floor.**

| quantity | replicate a | replicate b | floor |
|---|---|---|---|
| AVG | 0.6496 | 0.6864 | **3.68pp** |
| reader share | 82.03% | 82.02% | **0.01pp** |

**Inferring the share caveat from the AVG floor would have been wrong by two
orders of magnitude, in the pessimistic direction** — the rare failure mode where
excess caution misreports. Ruling 1 is what prevented it: *decompose the pair,
don't translate the floor.*

**Observation, not mechanism:** the two replicates' shares agree to 0.01pp
because whatever the nondeterminism perturbs, it perturbs **both channels of the
ratio together**. Consistent-with; not explained-by.

**The floor rule gains its penultimate term:** a floor belongs to
**(arm × config × platform × threading × quantity)**.

**Objection scoring, both halves.** I argued against this wave on measurement
risk. **Right about the arm** — the MLP is the noisiest scratch arm in the
program on AVG. **Wrong about the quantity** — its share floor is the tightest
measured anywhere. The pre-commitment absorbed both without flinching, because
it demanded the **measured** share-level budget rather than an inferred one.

---

## 4. The full-rank disposition, and the rule it needs

`w3c_fullrank_r0` (3 seeds, arm-recorded, `--adapter-rank 0` explicit) and the
ledger's adapters row `e4on_v2` are **the same arm** — every shared config field
identical; `e4on_v2` merely predates the thread-recording fix, so those fields
are *absent*, not different.

| seed | `w3c_fullrank_r0` | `e4on_v2` | Δ | cells identical |
|---|---|---|---|---|
| 42 | 0.9179 | 0.9179 | +0.00pp | **15/15** |
| 1337 | 0.8941 | 0.9090 | **−1.49pp** | **0/15** |
| 2024 | 0.9280 | 0.9280 | +0.00pp | **15/15** |
| mean | 0.9133 | **0.9183** | −0.50pp | |

### The rule, in two branches, so preference cannot decide the next instance

> **When a re-measurement of an arm falls WITHIN that arm's measured
> reproduction floor, the canonical value is the first verified measurement
> under recorded config, and the re-run is logged as a FLOOR OBSERVATION — not a
> replacement. When it falls OUTSIDE the floor, it is not a choice at all: it is
> a discrepancy investigation.**

Applied here: the 0.50pp mean delta sits inside the **1.49pp per-seed floor this
same wave measured**, so 0.9133 and 0.9183 are *the same number at this
instrument's resolution*. Switching would ripple through +48.4pp and both memos
for zero information.

**Stated because the kept value is the higher one:** the branch condition — not
the outcome — is what makes this safe to apply when the retained number happens
to flatter. Outside the floor, there would have been no choice to make.

**The alternative was rejected:** a footnote reconciling 0.9133 to 0.9183
publishes a discrepancy in order to explain it. One ledger row, cited by both
tables, with the arm's floor stated once in the appendix — a reader who re-runs
and lands 0.5pp away finds the explanation waiting instead of a puzzle.

### Two contributions, both kept

1. **`fullrank_ref` is now superseded BY MEASUREMENT, not merely excluded by
   provenance.** The same configuration, verified and arm-recorded, reads
   ~0.918 against its 0.9470. That closes the run's story completely: **it was
   never this arm at any code state.**
2. **The arm's per-seed reproduction floor is 1.49pp**, obtained free.

---

## 5. The determinism map, final form

| arm (MNIST, 4 threads) | reproduces? |
|---|---|
| `mafc_off` | ✓ 15/15 (×2 pairs) |
| `lwf` λ=0.25 | ✓ 15/15 |
| `plain_lstm` | ✗ 0/15 |
| `mlp` | ✗ 0/15 |
| adapters, seed 42 / 2024 | ✓ 15/15 across launches weeks apart |
| adapters, seed 1337 | ✗ 0/15 |

**"`plain_lstm` is the sole outlier" is dead — second revision.** The pattern is
about the **path**: on MNIST at 4 threads the arms routing through the MAFC
composition are bit-exact and the plain backbones are not. **Mechanism
unclaimed.**

**And the seed term is now empirical rather than defensive.** Catch 19 recorded
the class (*"reproducible for some seeds and not others"*); this is its cleanest
instance — same arm, config, platform, threading, weeks apart.

> **Determinism belongs to (arm × config × platform × threading × quantity ×
> seed), established per-pair, mechanism unclaimed.**

**One observed coincidence, written down because a future audit should find it
here rather than rediscover it:** every floor pair in this program ran on
**seed 42** — the seed we separately know runs **high on outcomes** and now know
runs **deterministic on reproduction**. Stated as coincidence. Nothing rests on
it.

---

## 6. LEDGER EDITS — three actions, one pass, applied on sign-off

| claim | evidence | status |
|---|---|---|
| **REVERT to four architectures** | the MLP row below supplies the missing artifact | **restores** the one-liner and the E14 row to "four architectures (LSTM, MLP, ViT, ResNet)". The 2026-08-17 narrowing to three is **superseded by measurement**, not withdrawn — it was correct on the evidence then available |
| **MLP reader share 82.5% [74.1, 90.9]** (Permuted MNIST, scratch, 3 seeds) | E17, `runs/e16_decomp/mnist_mlp.json`; mean-of-ratios 82.51%, t-CI [74.13, 90.90], **widened by the measured same-seed share floor 0.01pp → [74.11, 90.92]**; era `[True,True,True]`; identity 0.0e+00; capacity 266,752 params = the brittleness arm | **new.** Clears majority with the **entire** uncertainty budget, by 24pp. Sits inside the LSTM band. **Scratch evidence is per-dataset — LSTM/HAR and MLP/MNIST are not pooled.** |
| **floors attach to quantities** | same two runs: AVG floor **3.68pp**, share floor **0.01pp** | **new, methods.** Inferring one from the other errs by 2 orders of magnitude. Floor tuple: (arm × config × platform × threading × quantity × seed) |
| **full-rank storage cell cites the adapters row (0.9183)** | `w3c_fullrank_r0` mean 0.9133 vs `e4on_v2` 0.9183; Δ 0.50pp **inside** the arm's measured 1.49pp per-seed floor | **disposition.** One arm, one number. Re-run logged as a **floor observation**, per §4's two-branch rule |
| `runs/fullrank_ref` **superseded by measurement** | the same configuration, verified and arm-recorded, reads ~0.918 vs its 0.9470 | **strengthened** from "excluded by provenance" — it was never this arm at any code state |
| adapters arm per-seed reproduction floor **1.49pp** (seed 1337) | seeds 42/2024 bit-identical across launches weeks apart; 1337 0/15 | **new** |

---

## 7. What closes

Every artifact on the deck is read. Every number is one number. The program's
most-repeated sentence has an artifact behind every word of it.

**Consequences banked:** both shipped emails are true as sent — **no correction
owed**. The prediction ledger gains two hits. The arc reads **66–88 → 82.5 →
91.8 → 98.8**, which is **rising, not strictly monotone** (the MLP sits inside
the LSTM range), and the figure caption says exactly that.

**E17 is the deck's last experiment, and the right one to end on: the program's
final wave existed to make its first sentence true, and it did.**
