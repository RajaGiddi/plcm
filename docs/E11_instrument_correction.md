# E11 — Instrument Correction (catch 28)

**This is a correction pass, not a new experiment.** No new training, no new
hypotheses, no threshold changes. Every pre-registered bar stays exactly where it
is; only the instrument's readings change.

**Timebox:** one day. Blocked past it → report state, stop.

---

## The defect (catch 28, established)

`scripts/channel_decomp.py::features_and_logits` reconstructs the readout by hand
from the raw final cell state `c_n[-1]` (`c_t`):

```
lstm_out, (_, c_n) = model.lstm(xb)
c_t  = c_n[-1]
o_t  = sigmoid(model.output_gate(cat([lstm_out[:, -1, :], c_t], -1)))
feat = o_t * tanh(c_t)                    # asserted to be "the deployed feature"
```

The deployed classifier does not read this. `PLCM.forward()` runs
LSTM → memory read → **GGC composition** → classifier on the *composed* state
`c_t′`. The v3 feature lock got the form right (`o_t * tanh(·)`) and the argument
wrong (`c_t` instead of `c_t′`).

**The lock asserted path-identity in a docstring; nothing ever verified it.**
A premise carried in a comment, never checked against the thing it described —
catch 21's shape, this time inside the instrument.

### Evidence (OFF / seed 42, E10 checkpoints)

| task | matrix final row | `PLCM.forward` | screen `f&l` |
|---|---|---|---|
| 0 | 0.4010 | 0.4597 | 0.2372 |
| 1 | 0.1570 | 0.2442 | 0.4041 |
| 2 | 0.1556 | 0.2041 | 0.1684 |
| 3 | 0.8042 | 0.8329 | 0.5431 |
| **mean \|Δ\| vs matrix** | — | **0.056** | **0.171** |

The screen's error is 3× larger and **bidirectional** (−0.164, +0.247, +0.013,
−0.261). `PLCM.forward`'s is **systematically positive** — a separate, smaller
defect (step 3).

### Inherited scope

Every number computed through this path since E6b inherits the error: the F_read
decomposition shares, E8's floors/ceilings/ρ, E9's transport features, E10's
screen. `acc_orig` is the floor in every ρ, so a wrong floor is a wrong numerator
*and* a wrong denominator.

---

## Step 1 — Fix with proof, not assertion

Rewrite feature extraction to obtain `(feature, logits)` **from the deployed
forward path itself** — hook or refactor `PLCM.forward` so the exact tensor
entering the classifier is captured, rather than hand-reconstructed.

Then add the equality assert that should have existed since E6b:

```
assert allclose(captured_logits, model.forward(x))
```

on a live batch, for **every arm type** — ON (with adapters), OFF, and task-heads.

**This assert runs at the top of every screen/decomposition script permanently,
not once.** If any arm fails it: **stop and report. Do not special-case.**

---

## Step 2 — Scope confirmation

Run the three-column check (matrix final row / `PLCM.forward` / corrected
instrument) on **E5 and E4** checkpoints. One-line verdict per experiment:
**floor sound** / **floor affected**. This bounds the supersession chain before
any recompute.

---

## Step 3 — The residual

`PLCM.forward` sits **+0.056 systematically** above the matrix (OFF/s42).
Separate defect, one-directional. Candidates:

- test-loader construction differences, or
- `task4_epoch9.pt` holding a different state than the matrix's final row was
  written from.

**Diagnose by content** (parameter/prediction hash at the matrix-write point),
**not filename** — the catch-19 lesson. The floor is **not declared clean** until
this is closed or explained. If it traces to eval-loader differences, state which
loader is authoritative and why.

---

## Step 4 — Recompute chain, in dependency order

Each with its **old number printed beside its new one**.

1. **E6b decomposition** (F_total / F_enc / F_read, all arms, both datasets) —
   the program's central 75–90% claim. Per-cell, mean-of-ratios, CIs, H-C3′
   per-seed at the same **|R| ≤ 0.05** bar.
2. **E8 screen** (floors, ceilings, R, all cures) — per-cell protocol, per-cure
   denominator assignment as ruled (C3 raw, era-head cures adjusted).
3. **E10 screen** — same, then **H-X1 re-verdict at the unchanged 0.80 bar**, and
   H-X2's cure-corrected row (now single-instrument by construction, which
   retroactively fixes its half-and-half defect).
4. **E9** — recomputed **only if** step 2 shows the transport features materially
   change. Report the feature-space delta first; if small, note and skip — E9's
   verdict was geometric, not ρ-based.

---

## Step 5 — Supersession memo + ledger

One memo: `runs/MEMO_catch28_supersession.md` — the defect, the scope table,
every old→new number, which verdicts changed and which survived.

Then `docs/CLAIM_LEDGER.md` gets one edit: provisional markers resolved (cleared
or revised per the recompute), the mechanism named, history appended.

**Ledger edit text drafted in the memo first, applied second.**

---

## Read order

Step 1 assert green on all arms → Step 2 scope → Step 3 residual → recompute in
order → memo → ledger.

**No downstream number is read before its upstream dependency clears.**

---

## Predictions on record (for calibration)

| claim | probability |
|---|---|
| the assert passes after the fix | trivially — it is the definition of the fix |
| E5/E4 floors AFFECTED (same shared path) | ~90% |
| F_read-dominant structure survives recompute (reader channel still majority) | ~65% |
| C3's recomputed E10 ρ lands within ±0.15 of 0.615 | ~50% — genuinely uncertain; floor and ceiling move together but not identically |

Calibration record cited: the two 75% misses, and catch 28 itself.

---

## Locks

- **No threshold edits, no new cures, no arm additions.** Corrections only.
- Provenance rule in full force: memo numbers from recorded script output.
- Verdicts computed from printed arrays (catch 22).
- Any guard on a ratio's denominator applies per-cell (catch 26).
- Timebox: one day. Blocked past it → report state, stop.
