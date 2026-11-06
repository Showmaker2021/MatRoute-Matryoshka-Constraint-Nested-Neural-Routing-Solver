'''
Decoder and Funnel logits implementation for MatRoute.
Implements two-stage retrieval decoder:
- Stage 0 (Shortlist): candidate nominations by active blocks
- Stage 1 (Rerank): full dimension reranking over candidate union
'''

import math
from typing import Dict, FrozenSet, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from matroute.layout import BlockLayout
from matroute.features import DYNAMIC_QUERY_DIMS


def funnel_logits(
    q: Dict[str, torch.Tensor],
    k: Dict[str, torch.Tensor],
    S: FrozenSet[str],
    feasible: torch.Tensor,
    topk: Dict[str, int],
    zeta: float = 50.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Two-stage funnel logits:
    - Stage 0: Low-dimension scoring per active block on feasible nodes -> shortlist union C_t
    - Stage 1: Full-dimension scoring on C_t -> tanh clipping + masked softmax

    Args:
        q: dict of query tensors {b: [batch, pomo, d_b]} for active blocks
        k: dict of key tensors {b: [batch, num_nodes, d_b]} for active blocks
        S: set of active constraint blocks
        feasible: boolean tensor [batch, pomo, num_nodes] (where ninf_mask == 0)
        topk: dict of {b: k_b} candidate limits per block
        zeta: logit clipping scale (default 50.0)

    Returns:
        u_masked: [batch, pomo, num_nodes] logit tensor with -inf for unchosen/infeasible nodes
        scores: dict of raw block scores {b: [batch, pomo, num_nodes]}
        cand: boolean tensor [batch, pomo, num_nodes] of shortlisted candidates
    """
    batch_size, pomo_size, _ = feasible.shape
    num_nodes = feasible.shape[-1]

    # Compute dot-product scores for each active block: [batch, pomo, num_nodes]
    scores = {}
    for b in q.keys():
        d_b = q[b].shape[-1]
        # q[b]: [B, P, d_b], k[b]: [B, N, d_b] -> [B, P, N]
        s_b = torch.matmul(q[b], k[b].transpose(-1, -2)) / math.sqrt(d_b)
        scores[b] = s_b

    # Stage 0: Shortlist candidates across active blocks
    cand = torch.zeros_like(feasible)
    k_max = min(max(topk.values()), num_nodes)

    for b, s in scores.items():
        s_feas = s.masked_fill(~feasible, float("-inf"))
        k_b = min(topk.get(b, 50 if b == "CORE" else 20), num_nodes)
        if k_b > 0:
            top_indices = s_feas.topk(k_b, dim=-1).indices
            cand.scatter_(-1, top_indices, True)

    cand &= feasible

    # Ensure every trajectory with at least one feasible node has at least one candidate
    no_cand = (cand.sum(dim=-1, keepdim=True) == 0) & (feasible.sum(dim=-1, keepdim=True) > 0)
    if no_cand.any():
        cand = torch.where(no_cand, feasible, cand)

    # Stage 1: Rerank candidates with full multi-block sum
    u_full = sum(scores.values())
    u_clipped = zeta * torch.tanh(u_full)
    u_masked = u_clipped.masked_fill(~cand, float("-inf"))

    return u_masked, scores, cand


class MatRouteDecoder(nn.Module):
    """
    Decoder module managing per-block query and key projections,
    caching encoded keys, and running decode steps.
    """
    def __init__(self, layout: BlockLayout, zeta: float = 50.0):
        super().__init__()
        self.layout = layout
        self.zeta = zeta

        # Query projections per block: maps [h_first, h_last, dyn_state] -> d_b
        self.q_projections = nn.ModuleDict()
        self.k_projections = nn.ModuleDict()

        for b in layout.names:
            d_b = layout.sizes[b]
            dyn_dim = DYNAMIC_QUERY_DIMS[b]
            query_in_dim = 2 * d_b + dyn_dim  # [h_first, h_last, dyn_state]
            self.q_projections[b] = nn.Linear(query_in_dim, d_b, bias=False)
            self.k_projections[b] = nn.Linear(d_b, d_b, bias=False)

    def compute_keys(
        self,
        encoded_nodes: torch.Tensor,
        S: FrozenSet[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Projects encoder node representations into decoder key representations per block.
        encoded_nodes: [batch, num_nodes, active_dim(S)]
        Returns:
            dict[str, Tensor]: {b: [batch, num_nodes, d_b]}
        """
        keys = {}
        active_names = self.layout.active_names(S)
        offset = 0
        for b in active_names:
            d_b = self.layout.sizes[b]
            h_b = encoded_nodes[..., offset : offset + d_b]
            offset += d_b
            keys[b] = self.k_projections[b](h_b)
        return keys

    def compute_queries(
        self,
        h_first: torch.Tensor,
        h_last: torch.Tensor,
        dynamic_query_feats: Dict[str, torch.Tensor],
        S: FrozenSet[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Projects [h_first, h_last, dyn_state] into query vectors per active block.
        h_first: [batch, pomo, active_dim(S)]
        h_last:  [batch, pomo, active_dim(S)]
        Returns:
            dict[str, Tensor]: {b: [batch, pomo, d_b]}
        """
        queries = {}
        active_names = self.layout.active_names(S)
        offset = 0

        for b in active_names:
            d_b = self.layout.sizes[b]
            q_first_b = h_first[..., offset : offset + d_b]
            q_last_b = h_last[..., offset : offset + d_b]
            offset += d_b

            inputs = [q_first_b, q_last_b]
            if DYNAMIC_QUERY_DIMS[b] > 0:
                dyn_state = dynamic_query_feats.get(b)
                if dyn_state is not None:
                    inputs.append(dyn_state)
                else:
                    inputs.append(torch.zeros((*q_first_b.shape[:-1], DYNAMIC_QUERY_DIMS[b]), device=h_first.device))

            q_in = torch.cat(inputs, dim=-1)
            queries[b] = self.q_projections[b](q_in)

        return queries

    def forward_step(
        self,
        queries: Dict[str, torch.Tensor],
        keys: Dict[str, torch.Tensor],
        feasible: torch.Tensor,
        S: FrozenSet[str],
        level: str = "FULL",
        topk: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Single decoding step.
        level in {"FULL", "FUNNEL", "CORE_ONLY"}
        """
        if topk is None:
            topk = {b: 50 if b == "CORE" else 20 for b in queries.keys()}

        if level == "FULL":
            # Softmax over all feasible nodes with full multi-block sum
            scores = {}
            for b in queries.keys():
                d_b = queries[b].shape[-1]
                s_b = torch.matmul(queries[b], keys[b].transpose(-1, -2)) / math.sqrt(d_b)
                scores[b] = s_b

            u_full = sum(scores.values())
            u_clipped = self.zeta * torch.tanh(u_full)
            u_masked = u_clipped.masked_fill(~feasible, float("-inf"))
            cand = feasible.clone()
            return u_masked, scores, cand

        elif level == "FUNNEL":
            return funnel_logits(queries, keys, S, feasible, topk, zeta=self.zeta)

        elif level == "CORE_ONLY":
            # Softmax only using CORE scores
            scores = {}
            for b in queries.keys():
                d_b = queries[b].shape[-1]
                scores[b] = torch.matmul(queries[b], keys[b].transpose(-1, -2)) / math.sqrt(d_b)

            u_core = scores["CORE"]
            u_clipped = self.zeta * torch.tanh(u_core)
            u_masked = u_clipped.masked_fill(~feasible, float("-inf"))
            cand = feasible.clone()
            return u_masked, scores, cand

        else:
            raise ValueError(f"Unknown decoding level '{level}'. Expected: FULL, FUNNEL, CORE_ONLY.")
