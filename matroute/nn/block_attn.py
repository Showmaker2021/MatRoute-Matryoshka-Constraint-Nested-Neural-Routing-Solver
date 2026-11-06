'''
BlockAttention module for MatRoute.
Implements multi-head attention separated by constraint blocks with block-causal projections,
head-wise pair biases (D, D^T, and PDP relation), and optional kNN sparsity.
'''

import math
from typing import FrozenSet, Optional, Set, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from matroute.layout import BlockLayout
from matroute.nn.block_linear import BlockLinear


class BlockAttention(nn.Module):
    """
    Multi-head attention with dedicated head groups per block:
    - Queries, Keys, Values projected via BlockLinear
    - Block-wise distance / relation pair biases
    - Core attention computes independently of constraint blocks
    """
    def __init__(
        self,
        layout: BlockLayout,
        soft_core: bool = False,
        stopgrad_core: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.head_dim = layout.head_dim
        assert self.head_dim == 8, f"Head dimension must be 8, got {self.head_dim}"

        # Q, K, V projections
        self.q_proj = BlockLinear(layout, bias=False, soft_core=soft_core, stopgrad_core=stopgrad_core)
        self.k_proj = BlockLinear(layout, bias=False, soft_core=soft_core, stopgrad_core=stopgrad_core)
        self.v_proj = BlockLinear(layout, bias=False, soft_core=soft_core, stopgrad_core=stopgrad_core)

        # Output projection
        self.out_proj = BlockLinear(layout, bias=False, soft_core=soft_core, stopgrad_core=stopgrad_core)

        # Per-head learnable bias parameters
        # CORE heads: half D, half D^T
        n_core_heads = layout.n_heads("CORE")
        self.core_alpha = nn.Parameter(torch.ones(n_core_heads, 1, 1))

        # PD heads: relation bias
        n_pd_heads = layout.n_heads("PD")
        self.pd_beta = nn.Parameter(torch.ones(n_pd_heads, 1, 1))

        # Other constraint block heads: distance bias
        self.constraint_alphas = nn.ParameterDict()
        for b in layout.names:
            if b not in ("CORE", "PD"):
                n_b_heads = layout.n_heads(b)
                self.constraint_alphas[b] = nn.Parameter(torch.ones(n_b_heads, 1, 1))

    def _compute_pair_bias(
        self,
        block_name: str,
        dist_norm: torch.Tensor,
        dist_norm_t: torch.Tensor,
        log_scale: float,
        relation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes [batch, n_heads, N, N] pair bias for block_name.
        """
        batch_size, n_nodes, _ = dist_norm.shape

        if block_name == "CORE":
            n_heads = self.layout.n_heads("CORE")
            half = n_heads // 2

            # Negative scaled distance: smaller distance -> less negative -> higher attention
            scaled_d = -log_scale * dist_norm      # [B, N, N]
            scaled_dt = -log_scale * dist_norm_t  # [B, N, N]

            # First half uses D, second half uses D^T
            bias_d = self.core_alpha[:half] * scaled_d.unsqueeze(1)    # [B, half, N, N]
            bias_dt = self.core_alpha[half:] * scaled_dt.unsqueeze(1)  # [B, n_heads - half, N, N]
            return torch.cat([bias_d, bias_dt], dim=1)

        elif block_name == "PD":
            n_heads = self.layout.n_heads("PD")
            if relation is not None:
                # relation is 0 for associated pairs, 1 for non-associated
                # Negative scale relation: -beta * R
                # Associated pairs get 0 bias (boosted), unassociated get negative bias
                r_bias = -1.0 * relation.unsqueeze(1)  # [B, 1, N, N]
                return self.pd_beta * r_bias
            else:
                # Fallback to standard distance bias
                scaled_d = -log_scale * dist_norm
                return self.pd_beta * scaled_d.unsqueeze(1)

        else:
            # DEM, TW, PRZ, RTE
            alpha = self.constraint_alphas[block_name]
            scaled_d = -log_scale * dist_norm
            return alpha * scaled_d.unsqueeze(1)

    def forward(
        self,
        h: torch.Tensor,
        S: Union[Set[str], FrozenSet[str], None],
        dist: torch.Tensor,
        log_scale: Optional[float] = None,
        relation: Optional[torch.Tensor] = None,
        sparse_knn: Optional[int] = None,
    ) -> torch.Tensor:
        """
        h: Tensor [batch, num_nodes, active_dim(S)]
        dist: Tensor [batch, num_nodes, num_nodes]
        log_scale: float, log2(num_nodes)
        relation: Optional Tensor [batch, num_nodes, num_nodes] for PDP
        sparse_knn: Optional int for kNN sparse attention
        """
        batch_size, num_nodes, _ = h.shape
        active_names = self.layout.active_names(S)

        if log_scale is None:
            log_scale = math.log2(max(num_nodes, 2))

        # Distance normalization per instance (all_max style from URS)
        max_dist = dist.amax(dim=(1, 2), keepdim=True) + 1e-8
        dist_norm = dist / max_dist
        dist_norm_t = dist_norm.transpose(1, 2)

        # 1. Project Q, K, V
        q = self.q_proj(h, S)
        k = self.k_proj(h, S)
        v = self.v_proj(h, S)

        # Optional kNN sparse mask
        knn_mask = None
        if sparse_knn is not None and sparse_knn < num_nodes:
            # Keep top-k nearest neighbors plus global tokens (first 4 nodes)
            _, knn_indices = torch.topk(dist_norm, k=sparse_knn, dim=-1, largest=False)
            knn_mask = torch.zeros((batch_size, 1, num_nodes, num_nodes), dtype=torch.bool, device=h.device)
            knn_mask.scatter_(-1, knn_indices.unsqueeze(1), True)
            knn_mask[:, :, :, :min(4, num_nodes)] = True  # global tokens

        # 2. Compute attention independently per block
        out_blocks = []
        offset = 0

        for b in active_names:
            b_size = self.layout.sizes[b]
            n_heads = self.layout.n_heads(b)

            q_b = q[..., offset : offset + b_size].reshape(batch_size, num_nodes, n_heads, self.head_dim).transpose(1, 2)
            k_b = k[..., offset : offset + b_size].reshape(batch_size, num_nodes, n_heads, self.head_dim).transpose(1, 2)
            v_b = v[..., offset : offset + b_size].reshape(batch_size, num_nodes, n_heads, self.head_dim).transpose(1, 2)
            offset += b_size

            # Scaled dot-product: [B, n_heads, N, N]
            scores = (q_b @ k_b.transpose(-1, -2)) / math.sqrt(self.head_dim)

            # Add pair bias
            pair_bias = self._compute_pair_bias(b, dist_norm, dist_norm_t, log_scale, relation)
            scores = scores + pair_bias

            if knn_mask is not None:
                scores = scores.masked_fill(~knn_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            # Safe NaN check
            if torch.isnan(attn_weights).any():
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Context: [B, n_heads, N, 8] -> [B, N, b_size]
            ctx_b = (attn_weights @ v_b).transpose(1, 2).reshape(batch_size, num_nodes, b_size)
            out_blocks.append(ctx_b)

        # Concatenate active block outputs
        cat_out = torch.cat(out_blocks, dim=-1)

        # 3. Output projection
        return self.out_proj(cat_out, S)
