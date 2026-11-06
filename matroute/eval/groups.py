'''
Evaluation group partitions for URS / MatRoute.
Matches Tables 11 & 12 of URS benchmark paper:
- seen: 11 training problems
- unseen44: 44 symmetric zero-shot problems (Table 11)
- unseen55: 55 asymmetric & PDCVRP zero-shot problems (Table 12)
'''

from typing import List
from problem.ProblemSet import ProblemSet


def get_seen_group() -> List[str]:
    """Returns the 11 seen training problems."""
    return sorted(list(ProblemSet.get("train_problem_list")), key=len)


def get_unseen44_group() -> List[str]:
    """
    Returns the 44 symmetric zero-shot problems.
    Computed as: (symmetric_list - train_problem_list) - {pdcvrp, opdcvrp}.
    """
    seen = set(ProblemSet.get("train_problem_list"))
    sym = set(ProblemSet.get("symmetric_list"))
    sym_zero_shot = sym - seen
    pd_sym_unseen = {"pdcvrp", "opdcvrp"}
    unseen44 = sym_zero_shot - pd_sym_unseen
    return sorted(list(unseen44), key=len)


def get_unseen55_group() -> List[str]:
    """
    Returns the 55 asymmetric and PDCVRP zero-shot problems.
    Computed as: (asymmetric_list - train_problem_list) + {pdcvrp, opdcvrp}.
    """
    seen = set(ProblemSet.get("train_problem_list"))
    asym = set(ProblemSet.get("asymmetric_list"))
    asym_zero_shot = asym - seen
    pd_sym_unseen = {"pdcvrp", "opdcvrp"}
    unseen55 = asym_zero_shot | pd_sym_unseen
    return sorted(list(unseen55), key=len)


def get_evaluation_group(group_name: str) -> List[str]:
    """Retrieves problem list for a group ('seen', 'unseen44', 'unseen55', or any ProblemSet name)."""
    name_clean = group_name.strip().lower()
    if name_clean in ("seen", "train", "train_problem_list"):
        return get_seen_group()
    if name_clean in ("unseen44", "table11"):
        return get_unseen44_group()
    if name_clean in ("unseen55", "table12"):
        return get_unseen55_group()
    if name_clean in ("zero_shot", "zero_shot_list", "unseen99"):
        return sorted(list(ProblemSet.get("zero_shot_list")), key=len)
    return ProblemSet.get(name_clean)
