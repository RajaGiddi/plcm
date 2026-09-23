"""E18 P0 build gate (docs/E18_prereg.md sec 6): three checks, one verdict.

  (1) flag-off loader hashes identical to the pre-amendment reference
      (scripts/e18_loader_hash.py, runs/e18/loader_hash_reference.json);
  (2) flag-on constructions: per-task index sets pairwise disjoint, complete,
      12k/2k, identical across Permuted and Rotated for a seed;
  (3) the bit-deterministic MNIST anchor pair runs/w2d_mafc_seed42_{a,b}
      reproduced 25/25 by a retrain on the amended code with the flag OFF
      (modal_runner.jobs_e18_gate -> runs/e18_gate_w2d_mafc_seed42).

Verdict derived from the printed values. Nothing in E18 launches unless all
three read PASS. Writes runs/e18_build_gate.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.permuted_mnist import PermutedMNISTBenchmark
from src.data.rotated_mnist import RotatedMNISTBenchmark

REF = "runs/e18/loader_hash_reference.json"
ANCHORS = ["runs/w2d_mafc_seed42_a/mafc_results.json", "runs/w2d_mafc_seed42_b/mafc_results.json"]
GATE_RUN = "runs/e18_gate_w2d_mafc_seed42/mafc_results.json"


def main():
    out = {}
    print("=" * 100 + "\nE18 BUILD GATE\n" + "=" * 100)
    # (1)
    r = subprocess.run([sys.executable, "scripts/e18_loader_hash.py", "--check", REF],
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if "flag-off loaders" in l]
    print("  (1)", line[0] if line else r.stdout[-300:])
    out["loader_hash"] = {"pass": r.returncode == 0, "line": line[0] if line else None}
    # (2)
    ok2 = True
    fps = {}
    for seed in (42, 1337, 2024):
        for cls in (PermutedMNISTBenchmark, RotatedMNISTBenchmark):
            b = cls(num_tasks=5, batch_size=128, seed=seed, disjoint_content=True)
            for sets, n in ((b.train_index_sets, 60000), (b.test_index_sets, 10000)):
                cat = torch.cat(sets)
                ok2 &= int(cat.numel()) == n and int(cat.unique().numel()) == n
                ok2 &= all(int(t.numel()) == n // 5 for t in sets)
            fps.setdefault(seed, set()).add(b.content_fingerprint())
        ok2 &= len(fps[seed]) == 1
    print(f"  (2) disjoint constructions: sizes/disjointness/completeness {'OK' if ok2 else 'FAIL'}; fingerprints {dict((k, list(v)) for k, v in fps.items())}")
    out["disjoint"] = {"pass": bool(ok2), "fingerprints": {str(k): sorted(v) for k, v in fps.items()}}
    # (3)
    if Path(GATE_RUN).exists():
        g = json.load(open(GATE_RUN)); mg = np.array(g["accuracy_matrix"])
        ident = [int((mg == np.array(json.load(open(a))["accuracy_matrix"])).sum()) for a in ANCHORS]
        arm = g.get("arm", {})
        ok3 = all(i == 25 for i in ident) and arm.get("disjoint_content") is False
        print(f"  (3) anchor retrain: cells identical vs a/b {ident} /25 each; arm.disjoint_content={arm.get('disjoint_content')} -> {'PASS' if ok3 else 'FAIL'}")
        out["anchor"] = {"pass": bool(ok3), "identical": ident}
    else:
        print(f"  (3) anchor retrain {GATE_RUN} not present -> PENDING")
        out["anchor"] = {"pass": False, "pending": True}
    out["all_pass"] = bool(out["loader_hash"]["pass"] and out["disjoint"]["pass"] and out["anchor"]["pass"])
    print(f"\n  BUILD GATE {'PASS -- E18 runs may launch' if out['all_pass'] else 'NOT PASS -- nothing launches'}")
    Path("runs/e18").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("runs/e18/build_gate.json", "w"), indent=2)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
