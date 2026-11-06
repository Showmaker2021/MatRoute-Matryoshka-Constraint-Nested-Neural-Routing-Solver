'''
Variant to block set mapping logic for MatRoute.
Maps each VRP variant to its active block subset S based on vector lambda.
'''

from typing import Dict, FrozenSet, Optional, Sequence
from problem.ProblemRep import get_problem_representations
from problem.ProblemSet import ProblemSet


_PROBLEM_REPRESENTATIONS: Optional[Dict[str, Sequence[int]]] = None


def get_cached_problem_representations() -> Dict[str, Sequence[int]]:
    global _PROBLEM_REPRESENTATIONS
    if _PROBLEM_REPRESENTATIONS is None:
        _PROBLEM_REPRESENTATIONS = get_problem_representations()
    return _PROBLEM_REPRESENTATIONS


def blocks_for(variant: str, lam: Sequence[int], relation_present: bool = False) -> FrozenSet[str]:
    """
    Given a variant name, its 13-bit attribute vector lam, and relation_present flag,
    determines the active constraint block set S subset of {DEM, TW, PRZ, PD, RTE}.

    Indices of lambda (order from paper):
        0: identifier
        1: coordinates
        2: demand
        3: prize
        4: penalty
        5: early_time
        6: late_time
        7: service_time
        8: depot
        9: pickup
        10: delivery
        11: multi_route
        12: open_route
    """
    S = set()
    if lam[2]:
        S.add("DEM")
    if lam[5] or lam[6] or lam[7]:
        S.add("TW")
    if lam[3] or lam[4]:
        S.add("PRZ")
    if relation_present:
        S.add("PD")  # Only active when paired pickup-delivery relation matrix is present
    if lam[11] or lam[12]:
        S.add("RTE")
    return frozenset(S)


def get_variant_blocks(variant: str, relation_present: Optional[bool] = None) -> FrozenSet[str]:
    """
    Convenience lookup to get active block set S for any recognized variant string.
    If relation_present is None, it defaults to True if 'pd' in variant.lower(), else False.
    """
    reps = get_cached_problem_representations()
    v_clean = variant.strip().lower()
    if v_clean not in reps:
        raise KeyError(f"Unknown variant '{variant}'. Must be one of the 110 supported problems.")

    lam = reps[v_clean]
    if relation_present is None:
        relation_present = ("pd" in v_clean)
    return blocks_for(v_clean, lam, relation_present=relation_present)
