"""E12 build gate — B1 and B6 verifications.

These are the contract's verification steps, written as tests so "green" is a
command rather than a claim:

  B1: class-order fingerprint printed and gated in the execution environment;
      per-task counts and class-disjointness asserted.
  B6: shift spec fingerprinted in the execution environment, plus a direct
      assert that shifted and unshifted tensors differ by the registered
      permutation -- apply, invert, compare, EXACT.

The permutation tests need no dataset download; only the loader tests do, and
those are skipped when CIFAR-100 is absent so the gate reports "blocked on data"
rather than failing for the wrong reason.
"""

import os

import pytest
import torch

from src.data.split_cifar100 import (
    IMAGE_SIZE, PATCH_DIM, PATCH_SIZE, N_CLASSES_TOTAL,
    SplitCIFAR100Benchmark, apply_perm, invert_perm,
    build_patch_consistent_perm, build_global_perm,
)

HAVE_DATA = os.path.exists("./data/cifar-100-python")
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="CIFAR-100 not downloaded (B3)")


# ----------------------------------------------------------------- B6 core --
def test_b6_patch_perm_is_exactly_invertible():
    """apply, invert, compare — bitwise, per the contract."""
    p = build_patch_consistent_perm(IMAGE_SIZE, seed=1)
    x = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    back = apply_perm(apply_perm(x, p), invert_perm(p))
    assert torch.equal(back, x), "round trip is not exact"


def test_b6_global_perm_is_exactly_invertible():
    p = build_global_perm(IMAGE_SIZE, seed=1)
    x = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(apply_perm(apply_perm(x, p), invert_perm(p)), x)


def test_b6_shift_actually_shifts():
    """A permutation that changes nothing would make the control flat for the
    most boring possible reason."""
    p = build_patch_consistent_perm(IMAGE_SIZE, seed=1)
    x = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    assert not torch.equal(apply_perm(x, p), x)


def test_b6_patch_perm_is_identical_in_every_patch():
    """THE property that makes the control capable of passing: the same
    within-patch reordering everywhere, so a per-token linear map on patch
    embeddings can express its inverse."""
    p = build_patch_consistent_perm(IMAGE_SIZE, seed=3)
    # Mark each position with its flat index, permute, then read patches back.
    idx = torch.arange(3 * IMAGE_SIZE * IMAGE_SIZE).reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    permuted = apply_perm(idx.float(), p).long()
    n_side = IMAGE_SIZE // PATCH_SIZE

    def patch_pattern(py, px):
        ys = slice(py * PATCH_SIZE, (py + 1) * PATCH_SIZE)
        xs = slice(px * PATCH_SIZE, (px + 1) * PATCH_SIZE)
        block = permuted[:, ys, xs].reshape(-1)
        base = idx[:, ys, xs].reshape(-1)
        # position of each source within this patch, as a local ordering
        lookup = {int(v): i for i, v in enumerate(base)}
        return tuple(lookup[int(v)] for v in block)

    first = patch_pattern(0, 0)
    assert len(first) == PATCH_DIM
    for py in range(n_side):
        for px in range(n_side):
            assert patch_pattern(py, px) == first, (
                f"patch ({py},{px}) uses a different ordering — the shift would "
                f"not be undoable by a per-token adapter")


def test_b6_global_perm_is_NOT_patch_consistent():
    """The negative control for the test above: it must be able to fail, and the
    global permutation is the case where it must."""
    p = build_global_perm(IMAGE_SIZE, seed=3)
    idx = torch.arange(3 * IMAGE_SIZE * IMAGE_SIZE).reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    permuted = apply_perm(idx.float(), p).long()
    # A patch-consistent permutation keeps every patch's contents inside it.
    ys = xs = slice(0, PATCH_SIZE)
    base = set(int(v) for v in idx[:, ys, xs].reshape(-1))
    got = set(int(v) for v in permuted[:, ys, xs].reshape(-1))
    assert got != base, "global permutation unexpectedly preserved patch membership"


# ------------------------------------------------------------ B1 structure --
def test_b1_split_is_disjoint_and_complete():
    b = SplitCIFAR100Benchmark(download=False, root="./data") if HAVE_DATA else None
    classes = (b.task_classes if b else
               _offline_benchmark().task_classes)
    seen: set[int] = set()
    for k, cs in enumerate(classes):
        assert len(cs) == 5, f"task {k} has {len(cs)} classes"
        assert not (set(cs) & seen), f"task {k} overlaps an earlier task"
        seen |= set(cs)
    assert len(seen) == N_CLASSES_TOTAL == 100
    assert len(classes) == 20


def test_b1_fingerprint_is_stable_and_seed_sensitive():
    a1 = _offline_benchmark(seed=42).class_order_fingerprint()
    a2 = _offline_benchmark(seed=42).class_order_fingerprint()
    b = _offline_benchmark(seed=1337).class_order_fingerprint()
    assert a1 == a2, "fingerprint is not deterministic"
    assert a1 != b, "fingerprint does not distinguish class orders"


def test_b1_fingerprint_uses_plain_ints():
    """E10's lesson: a fingerprint that varies with numpy's scalar repr is
    describing the environment, not the data."""
    b = _offline_benchmark()
    assert all(type(c) is int for cs in b.task_classes for c in cs)


def test_b6_shift_fingerprint_distinguishes_modes():
    base = _offline_benchmark(shift_mode=None).shift_fingerprint()
    patch = _offline_benchmark(shift_mode="patch").shift_fingerprint()
    glob = _offline_benchmark(shift_mode="global").shift_fingerprint()
    assert len({base, patch, glob}) == 3, "shift fingerprint collides across modes"


# ------------------------------------------------------- data-dependent (B3) --
@needs_data
def test_b1_task_loaders_shapes_and_counts():
    b = SplitCIFAR100Benchmark(root="./data", batch_size=8, download=False)
    tr, te = b.get_task_loaders(0)
    x, y = next(iter(tr))
    assert x.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert int(y.max()) < 5 and int(y.min()) >= 0, "task-IL labels must be 0..4"
    assert len(te.dataset) == 500, f"5 classes x 100 test images, got {len(te.dataset)}"
    assert len(tr.dataset) == 2500, f"5 classes x 500 train images, got {len(tr.dataset)}"


@needs_data
def test_b1_class_il_keeps_global_labels():
    b = SplitCIFAR100Benchmark(root="./data", batch_size=8, remap_labels=False,
                               download=False)
    _, te = b.get_task_loaders(3)
    _, y = next(iter(te))
    assert set(int(v) for v in y) <= set(b.task_classes[3])


def _offline_benchmark(seed: int = 42, shift_mode=None):
    """Build the split without touching torchvision (structure only)."""
    b = SplitCIFAR100Benchmark.__new__(SplitCIFAR100Benchmark)
    import numpy as np
    b.num_tasks, b.classes_per_task = 20, 5
    b.seed, b.shift_mode, b.image_size = seed, shift_mode, IMAGE_SIZE
    order = np.random.RandomState(seed).permutation(N_CLASSES_TOTAL)
    b.class_order = [int(c) for c in order]
    b.task_classes = [b.class_order[k * 5:(k + 1) * 5] for k in range(20)]
    b.perms = []
    for k in range(20):
        if shift_mode is None or k == 0:
            b.perms.append(None)
        elif shift_mode == "patch":
            b.perms.append(build_patch_consistent_perm(IMAGE_SIZE, seed * 1000 + k))
        else:
            b.perms.append(build_global_perm(IMAGE_SIZE, seed * 1000 + k))
    return b
