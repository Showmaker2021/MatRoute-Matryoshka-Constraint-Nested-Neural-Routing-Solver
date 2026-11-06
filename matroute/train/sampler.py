'''
Task and Sandwich Level Sampler for MatRoute training.
Selects VRP variant, active block set S, and decoding level (FULL, FUNNEL, CORE_ONLY).
'''

import random
from typing import Dict, FrozenSet, List, Optional, Tuple
from problem.ProblemSet import ProblemSet
from matroute.spec import get_variant_blocks


class SandwichSampler:
    """
    Manages task sampling and sandwich level selection across training epochs.
    - Warmup (first 20% of epochs): 100% FULL level
    - Remaining 80%: FULL (0.5), FUNNEL (0.3), CORE_ONLY (0.2)
    """
    def __init__(
        self,
        problem_list: Optional[List[str]] = None,
        warmup_epoch_ratio: float = 0.20,
        lambda_dist_start: float = 0.10,
        lambda_dist_end: float = 0.02,
        total_epochs: int = 500,
    ):
        if problem_list is None:
            self.problem_list = sorted(list(ProblemSet.get("train_problem_list")), key=len)
        else:
            self.problem_list = problem_list

        self.warmup_epochs = int(total_epochs * warmup_epoch_ratio)
        self.total_epochs = total_epochs
        self.lambda_dist_start = lambda_dist_start
        self.lambda_dist_end = lambda_dist_end

        # Default level probabilities outside warmup
        self.level_probs = {
            "FULL": 0.5,
            "FUNNEL": 0.3,
            "CORE_ONLY": 0.2,
        }

    def sample_task(self) -> Tuple[str, FrozenSet[str]]:
        """Uniformly chooses a VRP variant from problem_list and retrieves its block set S."""
        variant = random.choice(self.problem_list)
        S = get_variant_blocks(variant)
        return variant, S

    def sample_level(self, current_epoch: int, current_step: int = 0) -> Tuple[str, Optional[Dict[str, int]]]:
        """
        Selects decoding level and candidate budget topk.
        Returns:
            level: 'FULL', 'FUNNEL', or 'CORE_ONLY'
            topk: dict of {block: k} or None
        """
        if current_epoch <= self.warmup_epochs:
            return "FULL", None

        # Sample from level probabilities
        r = random.random()
        if r < self.level_probs["FULL"]:
            return "FULL", None
        elif r < self.level_probs["FULL"] + self.level_probs["FUNNEL"]:
            # Sample k from {16, 32, 64}
            k_val = random.choice([16, 32, 64])
            topk = {
                "CORE": k_val,
                "DEM": k_val,
                "TW": k_val,
                "PRZ": k_val,
                "PD": k_val,
                "RTE": k_val,
            }
            return "FUNNEL", topk
        else:
            return "CORE_ONLY", None

    def get_lambda_dist(self, current_epoch: int) -> float:
        """Computes linear decay for distillation loss weight."""
        if current_epoch <= self.warmup_epochs:
            return self.lambda_dist_start

        progress = (current_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.lambda_dist_start + progress * (self.lambda_dist_end - self.lambda_dist_start)
