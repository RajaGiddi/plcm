# `runs/e16_decomp/` — decomposition outputs

**Contains E17 results as well as E16's.** `mnist_mlp.json` and
`mnist_mlp_floor.json` are **E17** (the MLP reader share and its floor pair);
every other file is E16.

They live here because `scripts/e16_decompose.py` is the shared driver and
writes to its own directory. **The paths are not renamed** — a rename breaks
provenance, and the ledger rows cite these exact paths, which is what catch 21
requires.

This file is the breadcrumb: a future `grep -r e17 runs/` will not find the
artifact by name, and this line is why. Two weeks of this program were spent on
what happens when artifacts and names drift apart (`fullrank_ref`); one line
prevents this becoming the next instance.

| file | experiment |
|---|---|
| `mnist_mlp.json` | **E17** — MLP reader share, 3 seeds |
| `mnist_mlp_floor.json` | **E17** — MLP floor pair, 2 replicates of seed 42 |
| `mnist_{lam0.25,lam4.0,lam16.0,mafc_off}.json` | E16 |
| `har_{lam0.25,lam4.0,lam16.0,mafc_off}.json` | E16 |
