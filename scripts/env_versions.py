"""Dump the container's installed versions of everything the image leaves UNPINNED.

Why this exists: the contract's first prerequisite is "pin scipy". But scipy sits
in `modal_runner.py`'s pip_install beside `numpy`, `pyyaml`, `tqdm` and
`scikit-learn`, ALL unpinned, and editing that list invalidates the image cache.
The rebuild then refetches every unpinned package at whatever version is current
that day -- so a change whose stated purpose is to FIX one version would float
four others, including the scikit-learn that `refit_probe` runs on and that every
decomposition in this program reads through.

Pinning one package in an unpinned list is not a narrowing, it is a re-roll.
This prints what is actually installed so the pins can reproduce the container
that produced the existing results, rather than whatever today's index serves.
"""
import json, os, sys
import importlib.metadata as md

PKGS = ["scipy", "numpy", "scikit-learn", "torch", "torchvision", "timm", "pyyaml", "tqdm"]
out = {}
for p in PKGS:
    try:
        out[p] = md.version(p)
    except Exception as e:
        out[p] = f"ABSENT ({type(e).__name__})"
out["python"] = sys.version.split()[0]
print("=" * 72 + "\nCONTAINER VERSIONS (what a pin must reproduce)\n" + "=" * 72)
for k, v in out.items():
    print(f"  {k:<16} {v}")
print("\n  suggested pip_install line, reproducing THIS container:")
print("    " + ", ".join(f'"{k}=={v}"' for k, v in out.items()
                         if k != "python" and not str(v).startswith("ABSENT")))
os.makedirs("/runs/e30", exist_ok=True)
json.dump(out, open("/runs/e30/env_versions.json", "w"), indent=2)
print("\n  wrote /runs/e30/env_versions.json")
