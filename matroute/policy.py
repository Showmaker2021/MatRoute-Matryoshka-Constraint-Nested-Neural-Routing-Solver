'''
MatRoutePolicy module combining Encoder, Decoder, and Multi-level POMO Rollouts.
Implements FULL, FUNNEL, and CORE_ONLY decoding strategies with on-policy distillation.
'''

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from matroute.layout import BlockLayout
from matroute.features import (
    BlockInputEmbedding,
    build_raw_block_features,
    build_dynamic_query_features,
)
from matroute.encoder import Encoder
from matroute.decoder import MatRouteDecoder
from problem.ProblemSet import ProblemSet


@dataclass
class RolloutOutput:
    logp: torch.Tensor             # [batch, pomo]
    reward: torch.Tensor           # [batch, pomo]
    dist_loss: torch.Tensor        # scalar tensor
    trajectories: torch.Tensor     # [batch, pomo, num_steps]


def get_node_encoding(encoded_nodes: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    """
    encoded_nodes: [batch, num_nodes, dim]
    node_indices:  [batch, pomo]
    Returns:       [batch, pomo, dim]
    """
    batch_size, pomo_size = node_indices.shape
    dim = encoded_nodes.shape[-1]
    gathering_index = node_indices.unsqueeze(-1).expand(batch_size, pomo_size, dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)


class MatRoutePolicy(nn.Module):
    """
    Unified Policy network for MatRoute.
    Handles encoding of problem instances and multi-trajectory decoding.
    """
    def __init__(
        self,
        layout: BlockLayout,
        num_encoder_layers: int = 12,
        zeta: float = 50.0,
        soft_core: bool = False,
        soft_core_top_layers: int = 4,
        stopgrad_core: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.zeta = zeta

        self.input_embedder = BlockInputEmbedding(layout)
        self.encoder = Encoder(
            layout=layout,
            num_layers=num_encoder_layers,
            soft_core=soft_core,
            soft_core_top_layers=soft_core_top_layers,
            stopgrad_core=stopgrad_core,
        )
        self.decoder = MatRouteDecoder(layout=layout, zeta=zeta)

    def encode(
        self,
        reset_state,
        S: FrozenSet[str],
        variant: str,
        eta: Optional[torch.Tensor] = None,
        sparse_knn: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Embeds raw input features and runs L block-causal encoder layers.
        Returns:
            encoded_nodes: [batch, num_nodes, active_dim(S)]
        """
        device = next(self.parameters()).device
        problems = reset_state.problems.to(device)
        dist = reset_state.dist.to(device)
        relation = reset_state.relation.to(device) if reset_state.relation is not None else None
        eta = eta.to(device) if eta is not None else None

        raw_feats = build_raw_block_features(
            problems=problems,
            variant=variant,
            S=S,
            eta=eta,
            demand_max1=True,
        )
        embedded_blocks = self.input_embedder(raw_feats, S=S)
        init_emb = self.input_embedder.concat_embedded(embedded_blocks, S=S)

        encoded_nodes = self.encoder(
            init_emb=init_emb,
            S=S,
            dist=dist,
            log_scale=reset_state.log_scale,
            relation=relation,
            sparse_knn=sparse_knn,
        )
        return encoded_nodes

    def select_start_nodes(
        self,
        step_count: int,
        variant: str,
        batch_size: int,
        pomo_size: int,
        depot_num: int,
        start_node_from_env: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Determines starting node selection for POMO steps 0 and 1, strictly matching URS.
        """
        v_clean = variant.lower().strip()

        if step_count == 0:
            if v_clean in ("tsp", "atsp"):
                # TSP trajectories start from different nodes 0 .. pomo-1
                return torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)
            elif "md" in v_clean:
                # Multi-depot: divide trajectories equally across depots
                depots = torch.arange(depot_num, device=device).repeat_interleave(pomo_size // depot_num)
                return depots.unsqueeze(0).expand(batch_size, -1)
            else:
                # Single depot problems start at depot 0
                return torch.zeros((batch_size, pomo_size), dtype=torch.long, device=device)

        elif step_count == 1:
            if v_clean in ("tsp", "atsp") or pomo_size <= 1:
                return None  # Already picked in step 0 or not branching

            if "b" in v_clean and "bp" not in v_clean and start_node_from_env is not None:
                # Backhaul problems must start from linehaul nodes
                return start_node_from_env.to(device)
            elif "md" in v_clean:
                problem_nodes = pomo_size // depot_num
                selected = torch.arange(start=depot_num, end=problem_nodes + depot_num, device=device)
                selected = selected.repeat(batch_size, (pomo_size + problem_nodes - 1) // problem_nodes)
                return selected[:, :pomo_size]
            else:
                return torch.arange(start=1, end=pomo_size + 1, device=device)[None, :].expand(batch_size, pomo_size)

        return None

    def rollout(
        self,
        env,
        mask_fn,
        S: FrozenSet[str],
        level: str = "FULL",
        topk: Optional[Dict[str, int]] = None,
        greedy: bool = False,
        track_distill: bool = False,
        eta: Optional[torch.Tensor] = None,
        sparse_knn: Optional[int] = None,
    ) -> RolloutOutput:
        """
        Executes a complete episode rollout across parallel POMO trajectories.
        """
        reset_state, _, _ = env.reset()
        state, reward, done = env.pre_step()

        batch_size = state.batch_size
        pomo_size = state.pomo_size
        variant = env.problem_name
        device = env.device

        # 1. Encode instances
        encoded_nodes = self.encode(
            reset_state=reset_state,
            S=S,
            variant=variant,
            eta=eta,
            sparse_knn=sparse_knn,
        )

        # 2. Precompute decoder keys
        keys = self.decoder.compute_keys(encoded_nodes, S=S)

        step_count = 0
        h_first = None
        prob_list = []
        dist_loss_list = []

        while not done:
            # Check for predetermined start moves (steps 0 and 1)
            selected_fixed = self.select_start_nodes(
                step_count=step_count,
                variant=variant,
                batch_size=batch_size,
                pomo_size=pomo_size,
                depot_num=env.depot_num,
                start_node_from_env=state.START_NODE,
                device=device,
            )

            if selected_fixed is not None:
                selected = selected_fixed
                prob = torch.ones((batch_size, pomo_size), device=device)
                if step_count == 0:
                    h_first = get_node_encoding(encoded_nodes, selected.to(device))
            else:
                # Step >= 2 (or >= 1 for TSP)
                h_last = get_node_encoding(encoded_nodes, state.current_node.to(device))

                # Extract dynamic query features
                dyn_query_feats = build_dynamic_query_features(
                    step_state=state,
                    S=S,
                    batch_size=batch_size,
                    pomo_size=pomo_size,
                    device=device,
                )

                # Compute queries
                queries = self.decoder.compute_queries(h_first, h_last, dyn_query_feats, S=S)

                # Feasible mask: 0 in ninf_mask means feasible
                feasible = (state.ninf_mask.to(device) == 0)

                # Decode step
                logits, scores, cand = self.decoder.forward_step(
                    queries=queries,
                    keys=keys,
                    feasible=feasible,
                    S=S,
                    level=level,
                    topk=topk,
                )

                probs = F.softmax(logits, dim=-1)
                if torch.isnan(probs).any() or (probs.sum(dim=-1, keepdim=True) == 0).any():
                    probs = torch.nan_to_num(probs, nan=0.0)
                    zero_sum = (probs.sum(dim=-1, keepdim=True) == 0)
                    if zero_sum.any():
                        feas_f = feasible.float()
                        fallback = feas_f / (feas_f.sum(dim=-1, keepdim=True) + 1e-8)
                        probs = torch.where(zero_sum, fallback, probs)

                # On-policy distillation loss calculation
                if track_distill and len(scores) > 1:
                    with torch.no_grad():
                        u_full_unfiltered = sum(scores.values())
                        pi_full = F.softmax(self.zeta * torch.tanh(u_full_unfiltered).masked_fill(~feasible, float("-inf")), dim=-1)

                    step_kl = 0.0
                    for b, s_b in scores.items():
                        pi_b = F.softmax(self.zeta * torch.tanh(s_b).masked_fill(~feasible, float("-inf")), dim=-1)
                        # Masked KL divergence: sum(pi_full * log(pi_full / pi_b))
                        kl_b = (pi_full * (torch.log(pi_full + 1e-8) - torch.log(pi_b + 1e-8))).sum(dim=-1).mean()
                        step_kl = step_kl + kl_b

                    dist_loss_list.append(step_kl / len(scores))

                # Node selection
                if greedy:
                    selected = torch.argmax(probs, dim=-1)
                    prob = torch.gather(probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
                else:
                    flat_probs = probs.reshape(batch_size * pomo_size, -1)
                    # Safe multinomial sampling
                    selected = flat_probs.multinomial(1).squeeze(1).reshape(batch_size, pomo_size)
                    prob = torch.gather(probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)

            # Environment step (ensure action matches env internal device)
            env_dev = env.selected_node_list.device
            state, reward, done = env.step(selected.to(env_dev), mask_fn=mask_fn)
            prob_list.append(prob)
            step_count += 1

        # Accumulated log probability per trajectory
        all_probs = torch.stack(prob_list, dim=-1)  # [batch, pomo, steps]
        logp = all_probs.clamp_min(1e-8).log().sum(dim=-1)  # [batch, pomo]

        dist_loss = torch.stack(dist_loss_list).mean() if dist_loss_list else torch.tensor(0.0, device=device)
        reward_out = reward.to(device) if reward is not None else None

        return RolloutOutput(
            logp=logp,
            reward=reward_out,
            dist_loss=dist_loss,
            trajectories=env.selected_node_list,
        )
