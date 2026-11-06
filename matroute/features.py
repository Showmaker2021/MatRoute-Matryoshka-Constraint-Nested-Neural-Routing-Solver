'''
Feature extraction and projection into MatRoute block representations.
Transforms raw URS environment inputs into structured block features and embeddings.
'''

from typing import Dict, FrozenSet, Optional, Tuple
import torch
import torch.nn as nn
from problem.ProblemRep import get_problem_representations
from matroute.layout import BlockLayout
from matroute.spec import blocks_for


RAW_FEATURE_DIMS: Dict[str, int] = {
    "CORE": 4,  # [x, y, eta, is_depot]
    "DEM": 3,   # [demand, is_delivery, is_pickup]
    "TW": 3,    # [early_time, late_time, service_time]
    "PRZ": 2,   # [prize, penalty]
    "PD": 2,    # [is_pickup, is_delivery]
    "RTE": 2,   # [multi_route, open_route]
}

DYNAMIC_QUERY_DIMS: Dict[str, int] = {
    "CORE": 0,
    "DEM": 1,   # [load]
    "TW": 1,    # [time]
    "PRZ": 2,   # [collected_prize, tour_maxlength]
    "PD": 0,
    "RTE": 0,
}


def build_raw_block_features(
    problems: torch.Tensor,
    variant: str,
    S: FrozenSet[str],
    eta: Optional[torch.Tensor] = None,
    demand_max1: bool = True,
    relation_present: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extracts raw unprojected feature tensors for active blocks {CORE} U S.
    Blocks outside S are completely omitted (not included in the dictionary).

    Args:
        problems: Tensor [batch, num_nodes, 9] from reset_state.problems
            Columns: [x, y, demand, prize, fake_prize, penalty, tw_start, tw_end, service_time]
        variant: Name of VRP variant (e.g. 'cvrp', 'tsp')
        S: Set of active constraint blocks
        eta: Random identifier noise [batch, num_nodes, 1] for asymmetric problems
        demand_max1: Whether to normalize demand to [0, 1]
        relation_present: Whether paired PDP relation is present

    Returns:
        dict[str, Tensor]: Mapping from block name to feature tensor [batch, num_nodes, raw_dim]
    """
    batch_size, num_nodes, _ = problems.shape
    device = problems.device
    v_clean = variant.lower().strip()

    reps = get_problem_representations()
    lam = reps[v_clean]

    is_asym = bool(lam[0] == 1)
    depot_num = 0 if v_clean in ("tsp", "atsp") else (3 if "md" in v_clean else 1)

    features = {}

    # 1. CORE block: [x, y, eta, is_depot]
    core_feat = torch.zeros((batch_size, num_nodes, 4), device=device)
    if is_asym:
        if eta is None:
            core_feat[:, :, 2] = torch.rand((batch_size, num_nodes), device=device)
        else:
            core_feat[:, :, 2] = eta.squeeze(-1)
    else:
        core_feat[:, :, 0] = problems[:, :, 0]  # x
        core_feat[:, :, 1] = problems[:, :, 1]  # y

    if depot_num > 0:
        core_feat[:, :depot_num, 3] = 1.0  # is_depot
    features["CORE"] = core_feat

    # 2. DEM block: [demand, is_delivery, is_pickup]
    if "DEM" in S:
        dem_feat = torch.zeros((batch_size, num_nodes, 3), device=device)
        demand_raw = problems[:, :, 2]
        if demand_max1:
            norm_factor = demand_raw.amax(dim=-1, keepdim=True) + 1e-8
            dem_norm = demand_raw / norm_factor
        else:
            dem_norm = demand_raw

        # In URS Model_LIB.py, for PDCVRP demand is negated
        if "pd" in v_clean and "cvrp" in v_clean:
            dem_norm = -dem_norm

        dem_feat[:, :, 0] = dem_norm

        # Delivery and Pickup (Backhaul) flags
        if "b" in v_clean:
            # Backhaul: demand > 0 is delivery, demand < 0 is pickup
            dem_feat[:, :, 1] = (demand_raw > 0).float()
            dem_feat[:, :, 2] = (demand_raw < 0).float()
        else:
            if depot_num > 0:
                dem_feat[:, depot_num:, 1] = 1.0  # standard customers are delivery

        features["DEM"] = dem_feat

    # 3. TW block: [early_time, late_time, service_time]
    if "TW" in S:
        tw_feat = torch.zeros((batch_size, num_nodes, 3), device=device)
        # depot_tw_end is used as tw_norm_factor in URS
        tw_norm_factor = problems[0, 0, 7]
        if tw_norm_factor == 0 or torch.isinf(tw_norm_factor):
            tw_norm_factor = 1.0

        service_time_val = problems[0, -1, 8]  # service_time
        tw_feat[:, :, 0] = problems[:, :, 6] / tw_norm_factor  # early time window
        tw_feat[:, :, 1] = problems[:, :, 7] / tw_norm_factor  # late time window
        tw_feat[:, :, 2] = service_time_val                   # service time
        features["TW"] = tw_feat

    # 4. PRZ block: [prize, penalty]
    if "PRZ" in S:
        prz_feat = torch.zeros((batch_size, num_nodes, 2), device=device)
        if "spctsp" in v_clean:
            # SPCTSP uses fake_prize (col 4)
            prz_feat[:, :, 0] = problems[:, :, 4]
        else:
            prz_feat[:, :, 0] = problems[:, :, 3]  # prize
        prz_feat[:, :, 1] = problems[:, :, 5]      # penalty
        features["PRZ"] = prz_feat

    # 5. PD block: [is_pickup, is_delivery]
    if "PD" in S:
        pd_feat = torch.zeros((batch_size, num_nodes, 2), device=device)
        cust_cnt = num_nodes - depot_num
        half = cust_cnt // 2
        # Pickup nodes: 1st half of customers
        pd_feat[:, depot_num : depot_num + half, 0] = 1.0
        # Delivery nodes: 2nd half of customers
        pd_feat[:, depot_num + half :, 1] = 1.0
        features["PD"] = pd_feat

    # 6. RTE block: [multi_route, open_route]
    if "RTE" in S:
        rte_feat = torch.zeros((batch_size, num_nodes, 2), device=device)
        rte_feat[:, :, 0] = float(lam[11])  # multi_route
        rte_feat[:, :, 1] = float(lam[12])  # open_route
        features["RTE"] = rte_feat

    return features


def build_dynamic_query_features(
    step_state,
    S: FrozenSet[str],
    batch_size: int,
    pomo_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Extracts dynamic state features for active blocks to append to decoder queries.
    Returns:
        dict[str, Tensor]: Mapping from block name to Tensor [batch, pomo, dyn_dim]
    """
    query_feats = {}
    if "DEM" in S and hasattr(step_state, "load") and step_state.load is not None:
        query_feats["DEM"] = step_state.load.unsqueeze(-1).to(device)

    if "TW" in S and hasattr(step_state, "time") and step_state.time is not None:
        query_feats["TW"] = step_state.time.unsqueeze(-1).to(device)

    if "PRZ" in S:
        prz_dyn = torch.zeros((batch_size, pomo_size, 2), device=device)
        if hasattr(step_state, "collected_prize") and step_state.collected_prize is not None:
            prz_dyn[:, :, 0] = step_state.collected_prize
        if hasattr(step_state, "tour_maxlength") and step_state.tour_maxlength is not None:
            prz_dyn[:, :, 1] = step_state.tour_maxlength
        query_feats["PRZ"] = prz_dyn

    return query_feats


class BlockInputEmbedding(nn.Module):
    """
    Linear projection module that maps raw block features into their assigned block dimensions.
    """
    def __init__(self, layout: BlockLayout):
        super().__init__()
        self.layout = layout
        self.projections = nn.ModuleDict()
        for name in layout.names:
            raw_dim = RAW_FEATURE_DIMS[name]
            block_dim = layout.sizes[name]
            self.projections[name] = nn.Linear(raw_dim, block_dim, bias=False)

    def forward(self, raw_features: Dict[str, torch.Tensor], S: FrozenSet[str]) -> Dict[str, torch.Tensor]:
        """
        Projects each active block {CORE} U S into embedding space.
        Returns:
            dict[str, Tensor]: [batch, num_nodes, block_dim] for each active block.
        """
        embedded = {}
        active_names = self.layout.active_names(S)
        for name in active_names:
            if name not in raw_features:
                raise KeyError(f"Expected raw features for active block '{name}', but not found.")
            embedded[name] = self.projections[name](raw_features[name])
        return embedded

    def concat_embedded(self, embedded: Dict[str, torch.Tensor], S: FrozenSet[str]) -> torch.Tensor:
        """Concatenates embedded blocks along feature dim in canonical order."""
        active_names = self.layout.active_names(S)
        tensors = [embedded[name] for name in active_names]
        return torch.cat(tensors, dim=-1)
