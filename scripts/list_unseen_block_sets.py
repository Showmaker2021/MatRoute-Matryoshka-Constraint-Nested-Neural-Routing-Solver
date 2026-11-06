'''
Script to inspect active block sets (S) across all 110 VRP variants in URS.
Displays distribution of seen vs unseen variants across block sets.
'''

import os
import sys
from collections import defaultdict

# Set stdout to UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problem.ProblemSet import ProblemSet
from matroute.spec import get_variant_blocks


def format_block_set(S):
    if not S:
        return "Empty (CORE-only)"
    order = ["DEM", "TW", "PRZ", "PD", "RTE"]
    sorted_blocks = [b for b in order if b in S]
    return "{" + ", ".join(sorted_blocks) + "}"


def main():
    seen_list = ProblemSet.get("train_problem_list")
    zero_shot_list = ProblemSet.get("zero_shot_list")
    all_list = ProblemSet.get("all_evaluated_list")

    seen_set = set(seen_list)

    blocks_to_seen = defaultdict(list)
    blocks_to_unseen = defaultdict(list)

    for p in all_list:
        S = get_variant_blocks(p)
        key = format_block_set(S)
        if p in seen_set:
            blocks_to_seen[key].append(p)
        else:
            blocks_to_unseen[key].append(p)

    all_keys = sorted(list(set(blocks_to_seen.keys()) | set(blocks_to_unseen.keys())))

    print("=" * 80)
    print("   MATROUTE BLOCK SET (S) DISTRIBUTION ACROSS 110 VARIANTS")
    print("=" * 80)
    print(f"{'Block Set S':<28} | {'Seen':<5} | {'Unseen':<7} | {'Total':<6} | Sample Variants")
    print("-" * 80)

    for k in all_keys:
        s_cnt = len(blocks_to_seen[k])
        u_cnt = len(blocks_to_unseen[k])
        tot = s_cnt + u_cnt
        sample = (blocks_to_seen[k] + blocks_to_unseen[k])[:4]
        sample_str = ", ".join(sample) + ("..." if tot > 4 else "")
        print(f"{k:<28} | {s_cnt:<5} | {u_cnt:<7} | {tot:<6} | {sample_str}")

    print("=" * 80)
    print(f"Summary: 11 Seen, {len(zero_shot_list)} Unseen, {len(all_list)} Total Variants.")
    print("=" * 80)


if __name__ == "__main__":
    main()
