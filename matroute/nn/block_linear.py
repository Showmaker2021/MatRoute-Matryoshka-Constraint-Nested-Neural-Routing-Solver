'''
BlockLinear module for MatRoute.
Implements block-causal linear transformations with structural masks and slicing.
'''

from typing import FrozenSet, Optional, Set, Union
import torch
import torch.nn as nn
from matroute.layout import BlockLayout


class BlockLinear(nn.Module):
    """
    Block-causal linear layer respecting the constraint hierarchy:
    - CORE only reads CORE: Pa(CORE) = {CORE}
    - Constraint block g reads CORE and g: Pa(g) = {CORE, g}
    """
    def __init__(
        self,
        layout: BlockLayout,
        bias: bool = True,
        init_constraint_factor: float = 0.1,
        soft_core: bool = False,
        stopgrad_core: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.total_dim = layout.dim()
        self.soft_core = soft_core
        self.stopgrad_core = stopgrad_core

        # Full weight matrix
        W = torch.empty(self.total_dim, self.total_dim)
        nn.init.xavier_uniform_(W)

        # Scale down initial weights of constraint blocks for smooth initialization from TSP
        core_idx = layout.idx("CORE")
        for b in layout.names:
            if b != "CORE":
                b_idx = layout.idx(b)
                W[b_idx, :] *= init_constraint_factor

        self.weight = nn.Parameter(W)
        if bias:
            b_tensor = torch.zeros(self.total_dim)
            self.bias = nn.Parameter(b_tensor)
        else:
            self.register_parameter("bias", None)

        # Build causal block mask
        mask = torch.zeros(self.total_dim, self.total_dim)
        for b in layout.names:
            parents = {"CORE"} if b == "CORE" else {"CORE", b}
            b_idx = layout.idx(b)
            for p in parents:
                p_idx = layout.idx(p)
                mask[b_idx[:, None], p_idx[None, :]] = 1.0

        self.register_buffer("mask", mask)

        # Optional soft-gating parameter for CORE reading constraint blocks
        if self.soft_core:
            # Scalar gate initialized to 0.0 -> tanh(0) = 0 (exact read-only core initially)
            self.core_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("core_gate", None)

    def forward(
        self,
        h: torch.Tensor,
        S: Union[Set[str], FrozenSet[str], None],
    ) -> torch.Tensor:
        """
        Forward computation on active subset S.
        h: Tensor [..., active_dim(S)]
        """
        idx = self.layout.active_idx(S)

        # Apply causal mask
        eff_mask = self.mask.clone()
        if self.soft_core and self.core_gate is not None:
            # Allow CORE to softly read constraint blocks in S
            core_idx = self.layout.idx("CORE")
            gate_val = torch.tanh(self.core_gate)
            active_names = self.layout.active_names(S)
            for b in active_names:
                if b != "CORE":
                    b_idx = self.layout.idx(b)
                    eff_mask[core_idx[:, None], b_idx[None, :]] = gate_val

        w_eff = self.weight * eff_mask

        # Sliced weights corresponding to active blocks
        w_active = w_eff[idx][:, idx]

        h_in = h
        if self.stopgrad_core:
            # Stop gradient flowing back into CORE from constraint blocks
            core_dim = self.layout.sizes["CORE"]
            if h.shape[-1] > core_dim:
                h_core = h[..., :core_dim]
                h_constraints = h[..., core_dim:]
                h_in = torch.cat([h_core.detach(), h_constraints], dim=-1)

        out = h_in @ w_active.T

        if self.bias is not None:
            b_active = self.bias[idx]
            out = out + b_active

        return out
