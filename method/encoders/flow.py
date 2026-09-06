"""Conditional invertible flow f_phi (method-spec.md §3).

`f_phi` maps a transition's target `y_i,t = (r_i,t, o_i,t+1)`, conditioned
on `x_i,t = (o_i,t, a_i,t)`, to a code `c_i,t` in R^d. This is a small
conditional affine-coupling flow (RealNVP-style): each coupling layer
splits its input in half, and uses an MLP conditioned on the other half
*and* the external condition `x` to produce a scale/shift, so the
transform is invertible in closed form for any conditioner network.

bruno-sac (`nn_bijective_layers.py`) uses a conditional MAF instead — same
role (conditional invertible flow), different coupling scheme. Rewritten
here in PyTorch rather than reused, since bruno-sac is TF1 and can't be
mixed into this project's PyTorch/TorchRL stack. Affine coupling is used
instead of MAF purely because it's simpler to get right in a single pass;
swap in a MAF/IAF stack later if flow expressiveness turns out to matter.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConditionerMLP(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim * 2),
        )

    def forward(self, half: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([half, condition], dim=-1))
        log_scale, shift = out.chunk(2, dim=-1)
        # Bound log_scale for numerical stability (standard RealNVP trick).
        log_scale = torch.tanh(log_scale)
        return log_scale, shift


class ConditionalAffineCoupling(nn.Module):
    """One affine-coupling layer, conditioned on an external `condition`."""

    def __init__(self, dim: int, cond_dim: int, hidden_dim: int, flip: bool):
        super().__init__()
        self.dim = dim
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        self.flip = flip
        # The conditioner for the transformed half always outputs d2 params
        # when not flipped, d1 when flipped (whichever half is transformed).
        out_dim = self.d2 if not flip else self.d1
        in_dim = self.d1 if not flip else self.d2
        self.conditioner = _ConditionerMLP(in_dim, cond_dim, out_dim, hidden_dim)

    def forward(self, y: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y1, y2 = y[..., : self.d1], y[..., self.d1 :]
        if not self.flip:
            log_scale, shift = self.conditioner(y1, condition)
            z2 = y2 * torch.exp(log_scale) + shift
            z = torch.cat([y1, z2], dim=-1)
        else:
            log_scale, shift = self.conditioner(y2, condition)
            z1 = y1 * torch.exp(log_scale) + shift
            z = torch.cat([z1, y2], dim=-1)
        log_det = log_scale.sum(dim=-1)
        return z, log_det

    def inverse(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[..., : self.d1], z[..., self.d1 :]
        if not self.flip:
            log_scale, shift = self.conditioner(z1, condition)
            y2 = (z2 - shift) * torch.exp(-log_scale)
            return torch.cat([z1, y2], dim=-1)
        else:
            log_scale, shift = self.conditioner(z2, condition)
            y1 = (z1 - shift) * torch.exp(-log_scale)
            return torch.cat([y1, z2], dim=-1)


class ConditionalFlow(nn.Module):
    """f_phi: stack of conditional affine-coupling layers.

    `forward(y, condition) -> (code, log_det_jacobian)`, `inverse(code,
    condition) -> y`. `dim` must be >= 2 (coupling needs two halves); pad
    with a constant if your (r, o') target is 1-D.
    """

    def __init__(self, dim: int, cond_dim: int, n_layers: int = 4, hidden_dim: int = 32):
        super().__init__()
        if dim < 2:
            raise ValueError("ConditionalFlow needs dim >= 2 for coupling to split the input")
        self.dim = dim
        self.layers = nn.ModuleList(
            [ConditionalAffineCoupling(dim, cond_dim, hidden_dim, flip=(i % 2 == 1)) for i in range(n_layers)]
        )

    def forward(self, y: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)
        z = y
        for layer in self.layers:
            z, log_det = layer(z, condition)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        y = z
        for layer in reversed(self.layers):
            y = layer.inverse(y, condition)
        return y
