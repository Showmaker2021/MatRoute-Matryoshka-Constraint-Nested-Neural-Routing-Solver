'''
Encoder architecture for MatRoute.
Stacks L block-causal layers combining BlockInstanceNorm, BlockAttention, and BlockFFN.
'''

from typing import FrozenSet, Optional, Set, Union
import torch
import torch.nn as nn
from matroute.layout import BlockLayout
from matroute.nn.block_norm import BlockInstanceNorm
from matroute.nn.block_attn import BlockAttention
from matroute.nn.block_ffn import BlockFFN


class EncoderLayer(nn.Module):
    """
    Single block-causal encoder layer:
    Pre-norm:
      h = h + BlockAttention(BlockNorm_1(h))
      h = h + BlockFFN(BlockNorm_2(h))
    """
    def __init__(
        self,
        layout: BlockLayout,
        soft_core: bool = False,
        stopgrad_core: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.norm1 = BlockInstanceNorm(layout)
        self.attn = BlockAttention(layout, soft_core=soft_core, stopgrad_core=stopgrad_core)
        self.norm2 = BlockInstanceNorm(layout)
        self.ffn = BlockFFN(layout, soft_core=soft_core, stopgrad_core=stopgrad_core)

    def forward(
        self,
        h: torch.Tensor,
        S: Union[Set[str], FrozenSet[str], None],
        dist: torch.Tensor,
        log_scale: Optional[float] = None,
        relation: Optional[torch.Tensor] = None,
        sparse_knn: Optional[int] = None,
    ) -> torch.Tensor:
        # 1. Multi-head block attention with pre-norm
        h_norm1 = self.norm1(h, S)
        attn_out = self.attn(
            h_norm1,
            S=S,
            dist=dist,
            log_scale=log_scale,
            relation=relation,
            sparse_knn=sparse_knn,
        )
        h = h + attn_out

        # 2. Block-causal FFN with pre-norm
        h_norm2 = self.norm2(h, S)
        ffn_out = self.ffn(h_norm2, S)
        h = h + ffn_out

        return h


class Encoder(nn.Module):
    """
    MatRoute Encoder consisting of L block-causal layers.
    """
    def __init__(
        self,
        layout: BlockLayout,
        num_layers: int = 12,
        soft_core: bool = False,
        soft_core_top_layers: int = 4,
        stopgrad_core: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # If soft_core is enabled, apply to the top (last) layers
            is_soft = soft_core and (layer_idx >= num_layers - soft_core_top_layers)
            layer = EncoderLayer(
                layout=layout,
                soft_core=is_soft,
                stopgrad_core=stopgrad_core,
            )
            self.layers.append(layer)

    def forward(
        self,
        init_emb: torch.Tensor,
        S: Union[Set[str], FrozenSet[str], None],
        dist: torch.Tensor,
        log_scale: Optional[float] = None,
        relation: Optional[torch.Tensor] = None,
        sparse_knn: Optional[int] = None,
    ) -> torch.Tensor:
        """
        init_emb: [batch, num_nodes, active_dim(S)]
        dist: [batch, num_nodes, num_nodes]
        """
        h = init_emb
        for layer in self.layers:
            h = layer(
                h,
                S=S,
                dist=dist,
                log_scale=log_scale,
                relation=relation,
                sparse_knn=sparse_knn,
            )
        return h
