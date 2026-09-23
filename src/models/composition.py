"""
Composition operators for the Thought Space.

Defines how current cell states are composed with retrieved memory vectors.
The key insight: additive blending (weighted sum) converges to the centroid
of stored vectors, destroying task-specific information. These operators
preserve information through multiplicative and gated pathways.

Phase 1: Gated Geometric Composition (GGC) — Euclidean space
Phase 2: Möbius Composition — Hyperbolic (Poincaré ball) space
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class GatedGeometricComposition(nn.Module):
    """
    Gated Geometric Composition (GGC) operator.

    Composes current cell state cₜ with retrieved memory c̃ through three
    parallel pathways controlled by learned gates:

        γ = σ(Wᵧ · [cₜ; c̃; cₜ⊙c̃] + bᵧ)     — composition gate
        μ = σ(Wμ · c̃ + bμ)                      — modulation signal
        ι = tanh(Wι · [cₜ; c̃] + bι)            — injection signal
        cₜ' = γ ⊙ (cₜ ⊙ μ) + (1-γ) ⊙ ι

    Modulation path: retrieved memory acts as a multiplicative lens on the
    current state, amplifying/suppressing dimensions without dragging toward
    the mean.

    Injection path: introduces genuinely new information from the joint
    representation through nonlinear projection.

    Composition gate: decides per-dimension how much to trust modulation vs
    injection. Takes the Hadamard product cₜ⊙c̃ as input to capture
    element-wise agreement/disagreement between current and retrieved states.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Composition gate: takes [cₜ; c̃; cₜ⊙c̃] → 3x input
        self.W_gamma = nn.Linear(hidden_size * 3, hidden_size)

        # Modulation signal: takes c̃ only
        self.W_mu = nn.Linear(hidden_size, hidden_size)

        # Injection signal: takes [cₜ; c̃]
        self.W_iota = nn.Linear(hidden_size * 2, hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize gates with slight biases toward preservation.

        The composition gate bias is initialized positive so γ ≈ 0.7 at start,
        meaning the model defaults to modulation (preserving existing state)
        rather than injection (overwriting with new info). This prevents
        early training instability where random injections corrupt the state.
        """
        nn.init.xavier_uniform_(self.W_gamma.weight)
        nn.init.constant_(self.W_gamma.bias, 0.5)  # bias toward modulation

        nn.init.xavier_uniform_(self.W_mu.weight)
        nn.init.constant_(self.W_mu.bias, 0.0)

        nn.init.xavier_uniform_(self.W_iota.weight)
        nn.init.constant_(self.W_iota.bias, 0.0)

    def forward(
        self,
        c_t: torch.Tensor,
        c_retrieved: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Compose current cell state with retrieved memory.

        Args:
            c_t: Current cell state [batch, hidden]
            c_retrieved: Retrieved memory vector [batch, hidden]
                (already aggregated by the read controller)
            return_diagnostics: If True, return gate activations for analysis

        Returns:
            c_prime: Composed cell state [batch, hidden]
            diagnostics: Optional dict with gate values for debugging/analysis
        """
        # Element-wise interaction captures agreement/disagreement
        interaction = c_t * c_retrieved  # [batch, hidden]

        # Composition gate
        gate_input = torch.cat([c_t, c_retrieved, interaction], dim=-1)  # [batch, 3*hidden]
        gamma = torch.sigmoid(self.W_gamma(gate_input))  # [batch, hidden]

        # Modulation pathway: retrieved memory reshapes current state
        mu = torch.sigmoid(self.W_mu(c_retrieved))  # [batch, hidden]
        modulated = c_t * mu  # [batch, hidden]

        # Injection pathway: new information from joint representation
        joint = torch.cat([c_t, c_retrieved], dim=-1)  # [batch, 2*hidden]
        iota = torch.tanh(self.W_iota(joint))  # [batch, hidden]

        # Gated composition
        c_prime = gamma * modulated + (1.0 - gamma) * iota  # [batch, hidden]

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "gamma_mean": gamma.mean().item(),
                "gamma_std": gamma.std().item(),
                "mu_mean": mu.mean().item(),
                "modulation_ratio": gamma.mean().item(),
                "norm_change": (
                    (c_prime.norm(dim=-1) / (c_t.norm(dim=-1) + 1e-8)).mean().item()
                ),
            }

        return c_prime, diagnostics


class AdditiveComposition(nn.Module):
    """
    Additive (weighted sum) composition — the baseline to beat.

    cₜ' = α·cₜ + (1-α)·c̃

    This converges to the centroid of stored vectors over repeated application.
    Included as a comparison to demonstrate why GGC is necessary.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Learned blending weight (scalar per dimension)
        self.alpha_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        c_t: torch.Tensor,
        c_retrieved: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Additive composition baseline.

        Args:
            c_t: Current cell state [batch, hidden]
            c_retrieved: Retrieved memory [batch, hidden]

        Returns:
            c_prime: Blended cell state [batch, hidden]
            diagnostics: Optional gate values
        """
        joint = torch.cat([c_t, c_retrieved], dim=-1)
        alpha = torch.sigmoid(self.alpha_proj(joint))  # [batch, hidden]

        c_prime = alpha * c_t + (1.0 - alpha) * c_retrieved

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "alpha_mean": alpha.mean().item(),
                "norm_change": (
                    (c_prime.norm(dim=-1) / (c_t.norm(dim=-1) + 1e-8)).mean().item()
                ),
            }

        return c_prime, diagnostics


class MobiusComposition(nn.Module):
    """
    Möbius composition in the Poincaré ball — Phase 2.

    Operates in hyperbolic space where:
    - Points near origin = general/shared representations
    - Points near boundary = task-specific representations
    - Möbius addition is naturally non-commutative and norm-preserving

    cₜ ⊕_M c̃ = ((1 + 2⟨cₜ,c̃⟩ + ‖c̃‖²)cₜ + (1 − ‖cₜ‖²)c̃)
                / (1 + 2⟨cₜ,c̃⟩ + ‖cₜ‖²‖c̃‖²)

    Cell states must be projected into the Poincaré ball (‖x‖ < 1)
    before composition and projected back to Euclidean space after.
    """

    # Maximum representable norm inside the Poincaré ball. Points are kept
    # strictly inside (‖x‖ < 1); the small margin avoids the boundary where
    # the metric diverges and atanh(‖x‖) blows up.
    MAX_NORM: float = 1.0 - 1e-5

    def __init__(self, hidden_size: int, curvature: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.curvature = curvature

        # Euclidean → Poincaré projection
        self.to_poincare = nn.Linear(hidden_size, hidden_size)
        # Poincaré → Euclidean projection
        self.from_poincare = nn.Linear(hidden_size, hidden_size)
        # Learned gate for blending Möbius result with direct path
        self.blend_gate = nn.Linear(hidden_size * 2, hidden_size)

    def _exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from origin of Poincaré ball.

        Maps a Euclidean vector to the Poincaré ball.
        exp_0(v) = tanh(‖v‖) · v/‖v‖

        The result norm equals tanh(‖v‖), which saturates to 1.0 in floating
        point for large ‖v‖. We clamp the target magnitude to MAX_NORM so the
        mapped point stays strictly inside the ball.
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        magnitude = torch.tanh(v_norm).clamp(max=self.MAX_NORM)  # target norm
        return magnitude * v / v_norm

    def _log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map to origin of Poincaré ball.

        Maps a point in Poincaré ball back to Euclidean tangent space.
        log_0(y) = arctanh(‖y‖) · y/‖y‖
        """
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=self.MAX_NORM)
        return torch.atanh(y_norm) * y / y_norm

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition in the Poincaré ball.

        x ⊕ y = ((1 + 2⟨x,y⟩ + ‖y‖²)x + (1 − ‖x‖²)y)
                / (1 + 2⟨x,y⟩ + ‖x‖²‖y‖²)
        """
        x_sq = (x * x).sum(dim=-1, keepdim=True)  # ‖x‖²
        y_sq = (y * y).sum(dim=-1, keepdim=True)  # ‖y‖²
        xy = (x * y).sum(dim=-1, keepdim=True)     # ⟨x,y⟩

        num = (1 + 2 * xy + y_sq) * x + (1 - x_sq) * y
        den = 1 + 2 * xy + x_sq * y_sq

        # Clamp to stay inside the ball
        result = num / den.clamp(min=1e-8)
        result_norm = result.norm(dim=-1, keepdim=True)
        result = torch.where(
            result_norm > self.MAX_NORM,
            result * self.MAX_NORM / result_norm,
            result,
        )
        return result

    def forward(
        self,
        c_t: torch.Tensor,
        c_retrieved: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Möbius composition in hyperbolic thought space.

        Args:
            c_t: Current cell state [batch, hidden]
            c_retrieved: Retrieved memory [batch, hidden]

        Returns:
            c_prime: Composed cell state [batch, hidden]
            diagnostics: Optional hyperbolic metrics
        """
        # Project to Poincaré ball
        c_t_hyp = self._exp_map_zero(self.to_poincare(c_t))
        c_r_hyp = self._exp_map_zero(self.to_poincare(c_retrieved))

        # Möbius addition (non-commutative: current state is the base)
        composed_hyp = self._mobius_add(c_t_hyp, c_r_hyp)

        # Project back to Euclidean space
        composed_euc = self.from_poincare(self._log_map_zero(composed_hyp))

        # Gated blend with direct path for stability
        gate = torch.sigmoid(self.blend_gate(
            torch.cat([c_t, composed_euc], dim=-1)
        ))
        c_prime = gate * composed_euc + (1 - gate) * c_t

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "hyp_norm_ct": c_t_hyp.norm(dim=-1).mean().item(),
                "hyp_norm_cr": c_r_hyp.norm(dim=-1).mean().item(),
                "hyp_norm_composed": composed_hyp.norm(dim=-1).mean().item(),
                "blend_gate_mean": gate.mean().item(),
            }

        return c_prime, diagnostics


def build_composition(mode: str, hidden_size: int, **kwargs) -> nn.Module:
    """Factory for composition operators.

    Args:
        mode: One of "ggc", "additive", "mobius"
        hidden_size: Cell state dimension

    Returns:
        Composition module
    """
    if mode == "ggc":
        return GatedGeometricComposition(hidden_size)
    elif mode == "additive":
        return AdditiveComposition(hidden_size)
    elif mode == "mobius":
        return MobiusComposition(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown composition mode: {mode}")
