"""
E11 — is the E10 partition the one that was REGISTERED, in the environment that
actually ran the jobs?

Found while recomputing the E10 screen: the partition fingerprint reads
5d047e4213d1 on the laptop (matching docs/E10_prereg.md:18) and 1104af185c87 in
the Modal container that trained every E10 checkpoint. The groups differ by a
single swap -- subjects 2 and 5 trade places between task 0 and task 3 -- while
the five TEST subjects are identical.

So the pre-registration check performed at the start of E11 ("fingerprint MATCH")
established that THIS LAPTOP reproduces the registered value. It did not
establish that the jobs did. Catch 21's shape again, and catch 25's corollary: a
certificate is valid for the context it was verified in and transfers to no other
without re-verification.

This script prints the partition CONTENT plus the environment that produced it,
so the two can be compared directly instead of through a hash that only reports
inequality.

Usage:
    python scripts/partition_provenance.py
    modal run modal_runner.py::analysis --argv "scripts/partition_provenance.py"
"""

import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.har_subject import HARSubjectBenchmark
from src.data.har_shift import spec_fingerprint

REGISTERED_PARTITION = "5d047e4213d1"     # docs/E10_prereg.md:18
REGISTERED_SHIFT = "3de66e205eb7"         # docs/E10_prereg.md:17


def main():
    print("=" * 90)
    print("E10 PARTITION PROVENANCE — content and environment, not just the hash")
    print("=" * 90)
    print(f"  python {platform.python_version()}   numpy {np.__version__}   "
          f"torch {torch.__version__}")
    print(f"  platform {platform.platform()}   machine {platform.machine()}")

    b = HARSubjectBenchmark(num_tasks=5, root=".", batch_size=256)
    pf, sf = b.partition_fingerprint(), spec_fingerprint()
    print(f"\n  partition fingerprint {pf}  vs registered {REGISTERED_PARTITION}  "
          f"-> {'MATCH' if pf == REGISTERED_PARTITION else 'MISMATCH'}")
    print(f"  shift fingerprint     {sf}  vs registered {REGISTERED_SHIFT}  "
          f"-> {'MATCH' if sf == REGISTERED_SHIFT else 'MISMATCH'}")

    print(f"\n  groups (train | test), per task:")
    for k, (g, t) in enumerate(zip(b.groups, b.test_subjects)):
        train = [s for s in g if s not in t]
        print(f"    task {k}: train {train}   test {list(t)}")

    # Window counts drive the greedy balancing, so near-ties are where an
    # environment-dependent sort order can change the assignment.
    counts = {}
    for k in range(b.num_tasks):
        Xtr, _ = b.raw_windows(k, True)
        Xte, _ = b.raw_windows(k, False)
        counts[k] = (int(Xtr.shape[0]), int(Xte.shape[0]))
    print(f"\n  per-task window counts (train, test):")
    for k, (a, c) in counts.items():
        print(f"    task {k}: {a:5d} train  {c:5d} test")

    print(f"\n  test-subject identity is what the EVALUATION depends on: "
          f"{[list(t) for t in b.test_subjects]}")

    # The calibration constants come from TASK 0's TRAIN windows only -- exactly
    # where subjects 2 and 5 swap. If they differ, then every task's tensors
    # differ numerically, and the local-vs-Modal accuracy gap is the PARTITION,
    # not the BLAS. This is the discriminating measurement.
    import hashlib
    print(f"\n  calibration from task-0 train windows:")
    print(f"    mu[:4] {np.round(b._mu[:4], 8).tolist()}")
    print(f"    sd[:4] {np.round(b._sd[:4], 8).tolist()}")
    print(f"    mu sha1 {hashlib.sha1(b._mu.tobytes()).hexdigest()[:12]}   "
          f"sd sha1 {hashlib.sha1(b._sd.tobytes()).hexdigest()[:12]}")

    print(f"\n  per-task TEST tensor hashes (what every reported accuracy is "
          f"measured on):")
    for k in range(b.num_tasks):
        Xte, yte = b.raw_windows(k, False)
        print(f"    task {k}: X sha1 {hashlib.sha1(Xte.numpy().tobytes()).hexdigest()[:12]}"
              f"   y sha1 {hashlib.sha1(yte.numpy().tobytes()).hexdigest()[:12]}"
              f"   shape {tuple(Xte.shape)}")


if __name__ == "__main__":
    main()
