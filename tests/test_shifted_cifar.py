"""E2's CIFAR-10 input-shift loader — shift math, verified on synthetic tensors.

No dataset download: the properties that can actually be wrong are geometric,
and synthetic inputs test them BETTER than real images because the correct
output is known exactly rather than eyeballed. Real CIFAR is baked into the
Modal image, where the runs happen.

The three things that would silently corrupt E2:
  1. The spatial permutation must be applied IDENTICALLY across the three colour
     channels. Contract §2 E2 puts channel permutation explicitly out of scope,
     so a per-channel-independent permutation would test a different shift class
     than the one pre-registered — and would still "look fine" in every summary
     statistic.
  2. Task 0 must be the exact identity under both shift classes, or the ON arm's
     A_0 has something to undo and the arms stop being comparable at task 0.
  3. The [3,32,32] -> [32,96] sequence layout must match split_cifar.py's
     convention, or the LSTM reads scrambled rows and the encoder never sees the
     structure the benchmark claims to present.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.shifted_cifar import (
    CIFAR_FLAT, CIFAR_PIXELS, CIFAR_ROWS, CIFAR_ROW_FEATURES,
    ShiftedCIFAR10, ShiftedCIFAR10Benchmark,
)

N = 8


def _synth(n: int = N) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 [n,3,32,32] with every pixel value distinct within an image.

    Distinctness matters: it makes a permutation detectable elementwise rather
    than only in aggregate, so a wrong-axis permute cannot hide behind matching
    means or matching sorted values.
    """
    g = torch.Generator().manual_seed(0)
    imgs = torch.stack([
        torch.randperm(3 * 32 * 32, generator=g).reshape(3, 32, 32) % 256
        for _ in range(n)
    ]).to(torch.uint8)
    return imgs, torch.arange(n) % 10


def _as_chw(sample: torch.Tensor) -> torch.Tensor:
    """Undo the [32,96] sequence layout back to [3,32,32] for comparison."""
    return sample.reshape(CIFAR_ROWS, 32, 3).permute(2, 0, 1)


class TestSequenceLayout:
    def test_shape_is_rows_by_row_features(self):
        imgs, labs = _synth()
        s, y = ShiftedCIFAR10(imgs, labs)[0]
        assert s.shape == (CIFAR_ROWS, CIFAR_ROW_FEATURES) == (32, 96)
        assert isinstance(y, int)

    def test_layout_roundtrips(self):
        """[3,32,32] -> [32,96] must be exactly the documented reshape."""
        imgs, labs = _synth()
        ds = ShiftedCIFAR10(imgs, labs)
        s, _ = ds[0]
        expected = (imgs[0].float() / 255.0 - ds._mean) / ds._std
        assert torch.allclose(_as_chw(s), expected, atol=1e-5)

    def test_adapter_dim_matches_flattened_image(self):
        assert CIFAR_FLAT == 3 * CIFAR_PIXELS == 3072


class TestPermutationIsChannelIdentical:
    """The pre-registered shift class: pixels MOVE, they do not change colour."""

    def test_same_permutation_applied_to_every_channel(self):
        imgs, labs = _synth()
        g = torch.Generator().manual_seed(1)
        perm = torch.randperm(CIFAR_PIXELS, generator=g)
        base = _as_chw(ShiftedCIFAR10(imgs, labs)[0][0]).reshape(3, CIFAR_PIXELS)
        got = _as_chw(ShiftedCIFAR10(imgs, labs, permutation=perm)[0][0]).reshape(3, CIFAR_PIXELS)
        assert torch.allclose(base[:, perm], got, atol=1e-5)

    def test_channels_stay_aligned_after_permutation(self):
        """A pixel's (R,G,B) triple must survive intact, just relocated.

        This is the test that fails if the permutation is ever applied to the
        flattened 3072 vector instead of the 1024 spatial positions.
        """
        imgs, labs = _synth()
        g = torch.Generator().manual_seed(2)
        perm = torch.randperm(CIFAR_PIXELS, generator=g)
        base = _as_chw(ShiftedCIFAR10(imgs, labs)[0][0]).reshape(3, CIFAR_PIXELS)
        got = _as_chw(ShiftedCIFAR10(imgs, labs, permutation=perm)[0][0]).reshape(3, CIFAR_PIXELS)
        for dest, src in enumerate(perm.tolist()[:64]):
            assert torch.allclose(got[:, dest], base[:, src], atol=1e-5)

    def test_permutation_preserves_pixel_multiset(self):
        imgs, labs = _synth()
        g = torch.Generator().manual_seed(3)
        perm = torch.randperm(CIFAR_PIXELS, generator=g)
        base = _as_chw(ShiftedCIFAR10(imgs, labs)[0][0]).flatten().sort().values
        got = _as_chw(ShiftedCIFAR10(imgs, labs, permutation=perm)[0][0]).flatten().sort().values
        assert torch.allclose(base, got, atol=1e-5)

    def test_permutation_actually_changes_the_image(self):
        imgs, labs = _synth()
        g = torch.Generator().manual_seed(4)
        perm = torch.randperm(CIFAR_PIXELS, generator=g)
        a, _ = ShiftedCIFAR10(imgs, labs)[0]
        b, _ = ShiftedCIFAR10(imgs, labs, permutation=perm)[0]
        assert not torch.allclose(a, b, atol=1e-5)

    def test_labels_are_untouched_by_any_shift(self):
        """Input-space shift only: the label space must never move."""
        imgs, labs = _synth()
        g = torch.Generator().manual_seed(5)
        perm = torch.randperm(CIFAR_PIXELS, generator=g)
        for i in range(N):
            assert ShiftedCIFAR10(imgs, labs)[i][1] == \
                   ShiftedCIFAR10(imgs, labs, permutation=perm)[i][1] == \
                   ShiftedCIFAR10(imgs, labs, angle=30.0)[i][1]


class TestIdentityTaskZero:
    def test_no_shift_is_exact_identity(self):
        imgs, labs = _synth()
        a, _ = ShiftedCIFAR10(imgs, labs)[0]
        b, _ = ShiftedCIFAR10(imgs, labs, permutation=None, angle=0.0)[0]
        assert torch.equal(a, b)

    def test_rotation_of_zero_degrees_is_a_noop(self):
        imgs, labs = _synth()
        a, _ = ShiftedCIFAR10(imgs, labs)[0]
        b, _ = ShiftedCIFAR10(imgs, labs, angle=0.0)[0]
        assert torch.equal(a, b)

    def test_nonzero_rotation_changes_the_image(self):
        imgs, labs = _synth()
        a, _ = ShiftedCIFAR10(imgs, labs)[0]
        b, _ = ShiftedCIFAR10(imgs, labs, angle=45.0)[0]
        assert not torch.allclose(a, b, atol=1e-5)


class TestBenchmarkShiftSchedule:
    """Task/angle/permutation bookkeeping — no image data needed."""

    @staticmethod
    def _bench(monkeypatch, shift="permuted", num_tasks=5, seed=42):
        imgs, labs = _synth()

        def _stub(self, root, train, download):
            return imgs, labs

        monkeypatch.setattr(ShiftedCIFAR10Benchmark, "_load_split", _stub, raising=False)
        return ShiftedCIFAR10Benchmark(num_tasks=num_tasks, shift=shift,
                                       seed=seed, download=False)

    def test_task_zero_has_no_shift(self, monkeypatch):
        b = self._bench(monkeypatch)
        assert b.permutations[0] is None and b.angles[0] == 0.0

    def test_later_tasks_have_distinct_permutations(self, monkeypatch):
        b = self._bench(monkeypatch)
        keys = {p.numpy().tobytes() for p in b.permutations[1:]}
        assert len(keys) == b.num_tasks - 1, "each task needs its own permutation"

    def test_permutations_are_seed_reproducible(self, monkeypatch):
        a = self._bench(monkeypatch, seed=7)
        c = self._bench(monkeypatch, seed=7)
        assert all(torch.equal(x, y) for x, y in zip(a.permutations[1:], c.permutations[1:]))
        assert a.shift_fingerprint() == c.shift_fingerprint()

    def test_different_seeds_give_different_fingerprints(self, monkeypatch):
        assert self._bench(monkeypatch, seed=1).shift_fingerprint() != \
               self._bench(monkeypatch, seed=2).shift_fingerprint()

    def test_rotated_angles_span_the_range(self, monkeypatch):
        b = self._bench(monkeypatch, shift="rotated")
        assert b.angles == [0.0, 22.5, 45.0, 67.5, 90.0]

    def test_shift_classes_have_different_fingerprints(self, monkeypatch):
        assert self._bench(monkeypatch, shift="permuted").shift_fingerprint() != \
               self._bench(monkeypatch, shift="rotated").shift_fingerprint()

    def test_declared_dims(self, monkeypatch):
        b = self._bench(monkeypatch)
        assert (b.input_size, b.adapter_dim, b.num_classes) == (96, 3072, 10)

    def test_rejects_unknown_shift(self, monkeypatch):
        with pytest.raises(AssertionError):
            self._bench(monkeypatch, shift="rotate")
