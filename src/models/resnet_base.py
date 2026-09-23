"""
ResNet-50 backbone — E14's encoder (C1).

Pre-registration: docs/E14_resnet_prereg.md

DELIBERATELY PARALLEL TO vit_base.py, NOT REFACTORED WITH IT. The two branches
duplicate structure, and that is the correct engineering debt to carry into a
deadline: unifying three backbones now would be an amendment with maximal blast
radius at the worst possible time, and the regression that just proved the shared
path bit-identical to E11's matrices — after four experiments' worth of
amendments — is exactly the proof such a refactor would spend. Paper 2 pays it
down.

NOT A DROP-IN, for the same reason the ViT is not. `MLPEncoder` returns
LSTM-shaped tensors so the whole PLCM stack runs on top; here the contract
registers the deployed feature as **post-global-average-pool, pre-fc** (2048-d),
so the native readout is used and memory / GGC / output-gate are bypassed. E14
measures forgetting in a plain fine-tuned ResNet, with PLCM contributing task
bookkeeping and head routing only.

THE CAPTURE POINT IS A FRESH PREMISE. "The feature" means something different on
a convnet than on a transformer, and the word is not a verification —
`assert_matches_timm` proves the reimplemented path is bitwise timm's own, and
C2's positive control proves the capture point is not vacuously right by failing
on a different reduction of the same layer.
"""

import torch
import torch.nn as nn

# Pinned variant, not a family name. timm ships several ResNet-50 weight sets
# (`a1_in1k` timm's own recipe, `tv_in1k` the torchvision port, `gluon_*`, ...)
# and "resnet50" alone resolves by timm's default, which can move between
# releases. The E10 lesson applied to weights: the hash certifies what ran.
RESNET_MODEL = "resnet50.a1_in1k"
RESNET_EMBED_DIM = 2048


class ResNetEncoder(nn.Module):
    """timm ResNet-50 exposing the post-pool, pre-fc feature the head reads."""

    def __init__(self, model_name: str = RESNET_MODEL, pretrained: bool = True):
        super().__init__()
        import timm                       # lazy: the repo imports without timm
        self.model_name = model_name
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.net.num_features          # 2048

    # ------------------------------------------------------------- internals --
    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        """The pre-pool feature map, [B, 2048, H, W]."""
        return self.net.forward_features(x)

    def forward(self, x: torch.Tensor, adapter=None) -> torch.Tensor:
        """The DEPLOYED FEATURE: post-global-average-pool, pre-fc.

        `forward_head(..., pre_logits=True)` is timm's own pooling, stopping
        before the classifier — exactly the tensor P3a names.
        """
        if adapter is not None:
            raise NotImplementedError(
                "E14 registers no adapt arm; the ResNet branch has no adapter "
                "injection point. An arm with a pre-known unreadable verdict is "
                "budget spent narrating a hole (contract sec 1).")
        return self.net.forward_head(self._spatial(x), pre_logits=True)

    # ---------------------------------------------------------- verification --
    @torch.no_grad()
    def wrong_capture_point(self, x: torch.Tensor) -> torch.Tensor:
        """C2's POSITIVE CONTROL: same layer, same shape, WRONG reduction.

        Spatial MAX over the pre-pool map instead of the deployed AVERAGE. It is
        [B, 2048] so the head accepts it, and it is a real tensor from the real
        forward — so P3a fails on it only because the capture point genuinely
        matters. A control built from noise would prove nothing.
        """
        return self._spatial(x).amax(dim=(2, 3))

    @torch.no_grad()
    def assert_matches_timm(self, x: torch.Tensor) -> float:
        """C2: the reimplemented path IS timm's, bitwise.

        Non-tautological — a wrong order, a skipped pooling, or reading the
        spatial map instead of its pooled summary makes this unequal.
        """
        mine = self.forward(x)
        theirs = self.net.forward_head(self.net.forward_features(x), pre_logits=True)
        d = float((mine - theirs).abs().max())
        assert torch.equal(mine, theirs), (
            f"reimplemented ResNet feature path is not timm's "
            f"(max |diff| = {d:.3e})")
        return d
