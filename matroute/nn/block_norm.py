'''
BlockInstanceNorm module for MatRoute.
Applies instance normalization independently per block across the node sequence.
Prevents cross-block statistical contamination and preserves prefix properties.
'''

from typing import FrozenSet, Set, Union
import torch
import torch.nn as nn
from matroute.layout import BlockLayout


class BlockInstanceNorm(nn.Module):
    """
    Independent instance normalization per block:
    Computes mean and variance over the node dimension for each block separately.
    """
    def __init__(self, layout: BlockLayout, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.layout = layout
        self.affine = affine
        self.eps = eps

        if affine:
            self.gammas = nn.ParameterDict()
            self.betas = nn.ParameterDict()
            for b in layout.names:
                b_dim = layout.sizes[b]
                self.gammas[b] = nn.Parameter(torch.ones(b_dim))
                self.betas[b] = nn.Parameter(torch.zeros(b_dim))
        else:
            self.gammas = None
            self.betas = None

    def forward(
        self,
        h: torch.Tensor,
        S: Union[Set[str], FrozenSet[str], None],
    ) -> torch.Tensor:
        """
        h: Tensor [batch, num_nodes, active_dim(S)]
        Returns:
            Normalized tensor of same shape [batch, num_nodes, active_dim(S)]
        """
        active_names = self.layout.active_names(S)
        out_blocks = []
        offset = 0

        for b in active_names:
            b_size = self.layout.sizes[b]
            h_b = h[..., offset : offset + b_size]
            offset += b_size

            # Compute instance norm over node dimension (dim=-2)
            # h_b shape: [batch, num_nodes, b_size]
            mean = h_b.mean(dim=-2, keepdim=True)
            var = h_b.var(dim=-2, unbiased=False, keepdim=True)
            norm_b = (h_b - mean) / torch.sqrt(var + self.eps)

            if self.affine:
                norm_b = norm_b * self.gammas[b] + self.betas[b]

            out_blocks.append(norm_b)

        return torch.cat(out_blocks, dim=-1)
