"""Adapter geometry: the "flat" vs "per_step" application modes.

E5 (UCI HAR) introduced per_step because the HAR shifts are 9x9 channel maps
applied identically at every timestep — 81 params/task instead of 1152x1152.
These tests pin down the two properties that make that substitution safe:

  1. per_step must be an EXACT identity at init, on the same footing as flat.
     Every branch reading in the adapter contracts depends on identity-init, so
     a mode that starts elsewhere would make a failure uninterpretable
     (capacity vs initialization).
  2. flat must be BIT-IDENTICAL to its pre-refactor behaviour. Both application
     sites were collapsed into one helper (_apply_adapter); if that changed flat
     even slightly, every MNIST-family result would silently stop being
     comparable to the ones already in the tables.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plcm import PLCM


def _model(mode: str, input_size: int, adapter_dim: int, **kw) -> PLCM:
    torch.manual_seed(0)
    return PLCM(
        input_size=input_size, hidden_size=16, num_classes=6, memory_capacity=16,
        key_dim=8, top_k=2, num_heads=2, write_threshold=0.2,
        use_input_adapters=True, adapter_dim=adapter_dim, adapter_mode=mode,
        use_task_heads=False, **kw,
    )


class TestPerStepGeometry:
    """The shape claim: the adapter lives in channel space, not flattened space."""

    def test_per_step_adapter_is_channel_sized(self):
        """9x9 = 81 params/task, NOT (128*9)^2. This is the whole storage claim."""
        m = _model("per_step", input_size=9, adapter_dim=9)
        m.set_task(0)
        n = sum(p.numel() for p in m.task_adapters["0"].parameters())
        assert n == 81, f"per_step adapter should be 9x9=81 params, got {n}"

    def test_per_step_overrides_stale_adapter_dim(self):
        """A caller passing the MNIST default must not silently get a 784x784 map.

        train.py sets adapter_dim from the benchmark, but a config file carrying
        the old default is exactly the kind of thing that would produce a
        1.33M-param adapter and a storage claim off by 16,000x.
        """
        m = _model("per_step", input_size=9, adapter_dim=784)
        assert m.adapter_dim == 9, "per_step must derive adapter_dim from input_size"
        m.set_task(0)
        assert sum(p.numel() for p in m.task_adapters["0"].parameters()) == 81

    def test_rejects_unknown_mode(self):
        with pytest.raises(AssertionError):
            _model("per_timestep", input_size=9, adapter_dim=9)   # near-miss name


class TestIdentityInit:
    """Both modes must start as an EXACT identity, not merely close to one."""

    @pytest.mark.parametrize("mode,in_size,dim,shape", [
        ("per_step", 9, 9, (4, 128, 9)),
        ("flat", 28, 784, (4, 28, 28)),
    ])
    def test_adapter_is_exact_identity_at_init(self, mode, in_size, dim, shape):
        m = _model(mode, input_size=in_size, adapter_dim=dim)
        m.set_task(0)
        x = torch.randn(*shape)
        out = m._apply_adapter(x, "0")
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-6), f"{mode} must be identity at init"

    def test_per_step_lowrank_is_exact_identity_at_init(self):
        """Residual low-rank form must also start at identity under per_step."""
        m = _model("per_step", input_size=9, adapter_dim=9, adapter_rank=4)
        m.set_task(0)
        x = torch.randn(4, 128, 9)
        assert torch.equal(m._apply_adapter(x, "0"), x)


class TestSharedWeightsAcrossTime:
    """per_step applies the SAME map at every timestep — that is the point."""

    def test_same_map_every_timestep(self):
        m = _model("per_step", input_size=9, adapter_dim=9)
        m.set_task(0)
        with torch.no_grad():
            m.task_adapters["0"].weight.copy_(torch.randn(9, 9))
        # A constant-over-time input must stay constant over time after adapting.
        x = torch.randn(4, 1, 9).expand(4, 128, 9).contiguous()
        out = m._apply_adapter(x, "0")
        assert torch.allclose(out, out[:, :1, :].expand_as(out), atol=1e-6), \
            "per_step must share one map across timesteps"

    def test_per_step_can_represent_a_channel_shift_exactly(self):
        """The expressivity claim: a 9x9 map inverts a 9x9 channel shift exactly.

        This is why the flattened 1152x1152 variant buys nothing for this shift
        class — it is a footnote, not an experiment.
        """
        torch.manual_seed(0)
        M = torch.linalg.qr(torch.randn(9, 9))[0]        # invertible channel map
        m = _model("per_step", input_size=9, adapter_dim=9)
        m.set_task(0)
        with torch.no_grad():
            m.task_adapters["0"].weight.copy_(torch.linalg.inv(M))
        x = torch.randn(4, 128, 9)
        shifted = x @ M.T
        assert torch.allclose(m._apply_adapter(shifted, "0"), x, atol=1e-4)


class TestFlatModeUnchanged:
    """Guard the refactor: flat must behave exactly as before the helper existed."""

    def test_flat_matches_explicit_reshape_path(self):
        m = _model("flat", input_size=28, adapter_dim=784)
        m.set_task(0)
        with torch.no_grad():
            m.task_adapters["0"].weight.copy_(torch.randn(784, 784) * 0.01)
        x = torch.randn(4, 28, 28)
        expected = m.task_adapters["0"](x.reshape(4, -1)).reshape(4, 28, 28)
        assert torch.equal(m._apply_adapter(x, "0"), expected)

    def test_default_mode_is_flat(self):
        """Every existing config omits `mode`; they must keep the old geometry."""
        m = PLCM(input_size=28, hidden_size=16, num_classes=10, memory_capacity=16,
                 key_dim=8, top_k=2, num_heads=2, use_input_adapters=True,
                 adapter_dim=784)
        assert m.adapter_mode == "flat"
        assert m.adapter_dim == 784

    def test_missing_adapter_key_is_a_noop(self):
        """task 0 before set_task, and the M4-LwF raw-probe arm, rely on this."""
        m = _model("per_step", input_size=9, adapter_dim=9)
        x = torch.randn(2, 128, 9)
        assert torch.equal(m._apply_adapter(x, "7"), x)


class TestCheckpointRoundTrip:
    """A reloaded model must rebuild the SAME adapter geometry.

    from_config drives load_from_checkpoint, so a dropped `mode` key would
    reload an E5 checkpoint as a flat model and fail on shape — or worse,
    succeed with the wrong map.
    """

    def test_mode_survives_from_config(self):
        cfg = {
            "model": {"input_size": 9, "hidden_size": 16, "num_classes": 6,
                      "num_layers": 1, "dropout": 0.0},
            "memory": {"capacity": 16, "key_dim": 8, "top_k": 2, "num_heads": 2,
                       "write_threshold": 0.2, "consolidation_interval": 50,
                       "consolidation_clusters": 8},
            "composition": {"mode": "ggc"},
            "adapters": {"enabled": True, "dim": 9, "mode": "per_step"},
        }
        m = PLCM.from_config(cfg)
        assert m.adapter_mode == "per_step" and m.adapter_dim == 9
        m.set_task(0)
        assert sum(p.numel() for p in m.task_adapters["0"].parameters()) == 81

    def test_forward_runs_under_per_step(self):
        """End-to-end shape check on HAR-shaped input."""
        m = _model("per_step", input_size=9, adapter_dim=9)
        m.set_task(0)
        out = m(torch.randn(4, 128, 9), store_memories=False)
        assert out["logits"].shape == (4, 6)
