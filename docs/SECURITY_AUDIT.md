# Security Audit — dataset ingestion, deserialization, and the Modal path

Scope: how this project acquires data, what it deserializes, and what it ships to
remote compute. Focus per request on the torchvision dataset downloads.

**One real finding, fixed. Two hardenings applied. One residual risk that cannot be
removed, only understood.**

---

## FINDING 1 (fixed) — unnecessary arbitrary-code-execution surface on checkpoint load

Three analysis scripts called:

    torch.load(path, weights_only=False, map_location="cpu")

`weights_only=False` runs the pickle unpickler, which **can execute arbitrary code**
during load. PyTorch made `weights_only=True` the default in 2.6 precisely because
this is the standard model-supply-chain attack: a checkpoint that runs code when
opened.

**Was it needed?** No — tested, not assumed. Our checkpoints contain only
`{config: dict, model_state: tensors, epoch, task_id, seed, model_type}`, all of
which `weights_only=True` supports:

    weights_only=True -> OK. keys: ['config','epoch','model_state','model_type','seed','task_id']
    PLCM.load_from_checkpoint under weights_only=True: OK
    adapters restored: ['0','1','2','3','4'] | mode per_step | dim 9

**Fixed** in `scripts/subspace_drift.py`, `scripts/drift_probe.py`,
`scripts/rbst_feasibility.py`. Verified the E5 adapter analysis still reproduces.

*Why it mattered even though the checkpoints are ours:* they round-trip through a
shared Modal Volume and are re-read by analysis scripts. Any write access to that
volume became code execution on the analyst's machine. The flag bought nothing.

## FINDING 2 (residual, cannot be eliminated) — CIFAR-10 is distributed as pickle

`torchvision.datasets.CIFAR10` reads the archive with:

    entry = pickle.load(f, encoding="latin1")

The CIFAR-10 "python version" *is* a pickle file. There is no safe-mode option:
using this dataset means unpickling it. Same for the batches' metadata.

**What actually protects us, stated precisely — because the usual "MD5 is broken"
reflex is the wrong analysis here:**

| control | value |
|---|---|
| transport | HTTPS, `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` |
| pinned digest | MD5 `c58f30108f718f92721af3b95e74349a`, plus per-batch MD5s |
| check | `check_integrity()` before extraction, refuses on mismatch |

MD5's practical break is **collision** resistance — an attacker crafting *two* files
that hash alike. That is not the threat model here. Substituting a *specific*
known-good file requires a **second-preimage**, for which MD5 remains ~2^123 — not
feasible. So the pin is meaningfully protective against a swapped archive, whether
by a compromised mirror or a TLS-stripping proxy.

**The residual risk is therefore not the hash — it is the trust chain.** We trust
that (a) the digest baked into torchvision corresponds to a benign file, and (b)
`pypi/torchvision` itself is not compromised. Nothing in our code can verify that.
Accepted, and recorded here rather than left implicit.

*Mitigating factor:* on Modal the dataset is downloaded **once at image build**, not
per run, so the network fetch happens in a sandboxed build container rather than in
each of the 14 training containers.

## FINDING 3 (safe by construction) — HAR archive extraction

`src/data/har_shift.py` reads a **user-supplied** zip. Checked for zip-slip /
path traversal, the standard archive vulnerability:

    outer = zipfile.ZipFile(path)
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
    raw = inner.read(f"UCI HAR Dataset/{split}/Inertial Signals/{ch}_{split}.txt")

**No `extractall`, no member-name iteration, nothing written to disk.** Every read
is an explicit, hardcoded member name into memory. A malicious archive cannot
write outside a target directory because nothing is ever written. Reading a nested
zip into memory is a theoretical zip-bomb surface, but the file is local and
operator-supplied, not fetched.

**Hardening applied:** the archive's SHA-256 is now recorded in
`runs/e5/FROZEN_CONFIG.json` — `c00b803081a5c797…` — so a rerun can prove it
consumed the same bytes. This is provenance for reproducibility as much as security.

## FINDING 4 (hardening applied) — the 58MB archive was not gitignored

`git check-ignore` showed the HAR zip **would have been committed**. Added to
`.gitignore` along with `*.zip`. Committing datasets bloats history irreversibly and
would have published a redistributed copy of the dataset under this repo.

## FINDING 5 (accepted, with a note) — dependency pinning is inconsistent

| | pinning |
|---|---|
| `requirements.txt` (local) | `torch>=2.0.0`, `torchvision>=0.15.0`, … all `>=` |
| Modal image (where results are produced) | `torch==2.13.0`, `torchvision==0.28.0` |

**The runs that produce results are exactly pinned**, which is what matters for
reproducibility and for supply-chain determinism. The local `>=` ranges mean a fresh
local install could silently pick up a different version than the one the results
were computed under. Not urgent — no result is produced locally — but the two should
be reconciled before release so a reader reproducing the work locally gets the same
stack. **Left as-is deliberately rather than changed mid-experiment**, since pinning
the local env now would risk perturbing a stack that is currently producing gate runs.

## Known-CVE scan

All **69 installed packages** queried against the OSV vulnerability database
(`api.osv.dev/v1/querybatch`, the same data source `pip-audit` uses):

    scanned 69 installed packages against OSV
      NO KNOWN VULNERABILITIES

*Method note:* `pip-audit` itself could not run here — its requirements mode
provisions a throwaway venv via `ensurepip`, which aborts under this uv-managed
Python 3.13. Querying OSV directly avoids the venv entirely and checks the
**actually installed** versions rather than the `>=` ranges in `requirements.txt`,
which is the stronger check (see Finding 5).

## Checked and clean

- **No secret material in the repo** — scanned for `.env*`, `*.pem`, `*.key`,
  `id_rsa*`, `*credential*`, `.netrc`, `*.token`. None found. This matters because
  `add_local_dir(".", …)` ships the working tree to Modal with only
  `runs/ checkpoints/ data/ .venv/ .git/ *.ipynb` excluded — any credential file in
  the tree **would** be uploaded. Worth re-checking before any future launch rather
  than assuming it stays true.
- **No `eval`/`exec`/`os.system`/`shell=True`** on external input; the Modal runner
  builds `subprocess.run([...])` argv lists, never a shell string.
- **No network access at training time** — datasets are baked into the image.

## Summary

| # | issue | severity | status |
|---|---|---|---|
| 1 | `weights_only=False` on checkpoint load | **medium** | **fixed** |
| 2 | CIFAR-10 is inherently a pickle | low (residual) | accepted, documented |
| 3 | HAR zip extraction | none | safe by construction; SHA-256 recorded |
| 4 | 58MB archive not gitignored | low | fixed |
| 5 | local vs image pinning mismatch | low | accepted, flagged for release |
