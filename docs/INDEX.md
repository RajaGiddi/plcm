# Experiment index

Generated from the files, not maintained by hand: run
`python scripts/make_index.py` after adding an experiment.

Each row is one experiment. **Contract** is the pre-registration signed
before it ran; **memo** is the write-up; **artifacts** counts the run
directories under `runs/` whose name begins with that key.

| Experiment | Contract | Memo | Scripts | Artifacts |
|---|---|---|---|---|
| **E4B** | `E4B_subspace_prereg.md` | — | — | — |
| **E5** | `E5_CONTRACT_NOTE.md` | — | — | 7 |
| **E5B** | `E5B_control_prereg.md` | — | — | — |
| **E5C** | `E5C_prereg.md` | — | — | — |
| **E5D** | `E5D_prereg.md` | — | — | 3 |
| **E6** | `E6_prereg.md` | — | — | 5 |
| **E6B** | `E6B_prereg.md` | — | — | 1 |
| **E7** | `E7_prereg.md` | — | 1 | 8 |
| **E8** | `E8_prereg.md` | — | — | 1 |
| **E9** | `E9_prereg.md` | — | — | 1 |
| **E10** | `E10_prereg.md` | — | — | 36 |
| **E11** | `E11_instrument_correction.md` | — | 1 | 4 |
| **E12** | `E12_prereg.md` | `MEMO_e12.md` | 9 | 39 |
| **E13** | **—** | `MEMO_e13.md` | 1 | — |
| **E14** | `E14_resnet_prereg.md` | `MEMO_e14.md` | 3 | 8 |
| **E15** | `E15_vit_tier0_prereg.md` | `MEMO_e15.md` | 1 | 1 |
| **E16** | `E16_lwf_prereg.md` | `MEMO_e16.md` | 1 | 53 |
| **E17** | **—** | `MEMO_e17.md` | — | 5 |
| **E18** | `E18_prereg.md` | `MEMO_e18.md` | 2 | 23 |
| **E20** | `E20_prereg.md` | `MEMO_e20.md` | 3 | 1 |
| **E21** | `E21_prereg.md` | `MEMO_e21.md` | 2 | 1 |
| **E23** | `E23_prereg.md` | `MEMO_e23.md` | 6 | 31 |
| **E23B** | `E23B_seqlen_prereg.md` | `MEMO_e23b.md` | — | 9 |
| **E25** | **—** | `MEMO_e25.md` | 2 | 1 |
| **E25A** | **—** | — | 2 | — |
| **E25B** | **—** | — | 2 | — |
| **E25C** | **—** | — | 1 | — |
| **E26** | **—** | `MEMO_e26.md` | 6 | 1 |
| **E27** | **—** | `MEMO_e27.md` | 2 | 1 |
| **E28** | **—** | `MEMO_e28.md` | 4 | 22 |
| **E29** | `E29_section0.md` | `MEMO_e29.md` | 7 | 1 |
| **E30** | **—** | — | 2 | 1 |
| **E31** | `E31_section0.md` | — | — | — |

## Dangling citations

Paths cited by a memo or script that the repository does not contain.
Listed so a reader meets them here rather than in a dead link.

- `docs/E21_mummadi.md` — cited by `MEMO_e21.md`, `MEMO_e25.md`, `e21_perturb.py`, `e21_row.py`
- `docs/E25_prereg.md` — cited by `MEMO_e25.md`, `e25_reads.py`, `e25a_curve.py`, `e25b_row.py`
- `docs/E26_prereg.md` — cited by `MEMO_e26.md`, `e26_dc.py`, `e26_dc_row.py`, `e26_fs.py`
- `docs/E27_prereg.md` — cited by `MEMO_e27.md`, `e27_defect.py`, `e27_row.py`
- `docs/E28_prereg.md` — cited by `train.py`

These are provenance gaps, not disputed results: the predictions
those contracts registered are scored in `docs/appendix.tex`. The
contracts were signed in a working session and never committed.
