'''
BlockFFN module for MatRoute.
Implements block-causal Feed-Forward Networks respecting constraint hierarchy:
CORE reads only CORE, while constraint blocks read CORE and themselves.
'''

from typing import FrozenSet, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from matroute.layout import BlockLayout


class BlockFFN(nn.Module):
    """
    Block-causal Feed-Forward Network:
    - CORE: linear(core_dim, 4 * core_dim) -> relu -> linear(4 * core_dim, core_dim)
    - Constraint block g: linear(core_dim + g_dim, 4 * g_dim) -> relu -> linear(4 * g_dim, g_dim)
    """
    def __init__(
        self,
        layout: BlockLayout,
        expansion_factor: int = 4,
        soft_core: bool = False,
        stopgrad_core: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.soft_core = soft_core
        self.stopgrad_core = stopgrad_core

        core_dim = layout.sizes["CORE"]
        core_hidden = expansion_factor * core_dim

        # Core FFN
        self.core_w1 = nn.Linear(core_dim, core_hidden)
        self.core_w2 = nn.Linear(core_hidden, core_dim)

        # Constraint block FFNs
        self.constraint_w1 = nn.ModuleDict()
        self.constraint_w2 = nn.ModuleDict()

        for b in layout.names:
            if b != "CORE":
                b_dim = layout.sizes[b]
                b_hidden = expansion_factor * b_dim
                # Input dimension is core_dim + b_dim because g reads CORE and g
                self.constraint_w1[b] = nn.Linear(core_dim + b_dim, b_hidden)
                self.constraint_w2[b] = nn.Linear(b_hidden, b_dim)

                # Initialize constraint block outputs with small scale
                self.constraint_w2[b].weight.data.mul_(0.1)

        if self.soft_core:
            self.core_gate = nn.Parameter(torch.zeros(1))
            self.soft_projections = nn.ModuleDict()
            for b in layout.names:
                if b != "CORE":
                    self.soft_projections[b] = nn.Linear(layout.sizes[b], core_dim)
        else:
            self.register_parameter("core_gate", None)

    def forward(
        self,
        h: torch.Tensor,
        S: Union[Set[str], FrozenSet[str], None],
    ) -> torch.Tensor:
        """
        h: Tensor [batch, num_nodes, active_dim(S)]
        Returns:
            Tensor [batch, num_nodes, active_dim(S)]
        """
        active_names = self.layout.active_names(S)
        core_dim = self.layout.sizes["CORE"]
        h_core = h[..., :core_dim]

        # 1. CORE forward
        core_hidden = F.relu(self.core_w1(h_core))
        out_core = self.core_w2(core_hidden)

        out_blocks = [out_core]
        offset = core_dim

        # Optional soft gating from constraint blocks into CORE
        if self.soft_core and self.core_gate is not None:
            gate_val = torch.tanh(self.core_gate)
            soft_in = 0.0

        for b in active_names:
            if b == "CORE":
                continue
            b_dim = self.layout.sizes[b]
            h_b = h[..., offset : offset + b_dim]
            offset += b_dim

            # Constraint block reads CORE and h_b
            h_core_in = h_core.detach() if self.stopgrad_core else h_core
            input_b = torch.cat([h_core_in, h_b], dim=-1)

            hidden_b = F.relu(self.constraint_w1[b](input_b))
            out_b = self.constraint_w2[b](hidden_b)
            out_blocks.append(out_b)

            if self.soft_core and self.core_gate is not None:
                soft_in = soft_in + self.soft_projections[b](h_b)

        if self.soft_core and self.core_gate is not None and len(active_names) > 1:
            out_blocks[0] = out_blocks[0] + gate_val * soft_in

        return torch.cat(out_blocks, dim=-1)
