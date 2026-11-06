'''
Unit test for matroute/spec.py and matroute/eval/groups.py.
Verifies active block sets for 11 seen problems and correct partition of unseen44 / unseen55.
'''

import unittest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from matroute.spec import get_variant_blocks
from matroute.eval.groups import get_seen_group, get_unseen44_group, get_unseen55_group, get_evaluation_group
from problem.ProblemSet import ProblemSet


class TestSpecAndGroups(unittest.TestCase):

    def test_seen_block_sets(self):
        """Verifies active block sets for the 11 seen problems strictly match Table in Guide 3.3."""
        expected = {
            "tsp": frozenset(),
            "atsp": frozenset(),
            "op": frozenset({"PRZ"}),
            "pctsp": frozenset({"PRZ"}),
            "pdtsp": frozenset({"PD"}),
            "cvrp": frozenset({"DEM", "RTE"}),
            "acvrp": frozenset({"DEM", "RTE"}),
            "cvrpb": frozenset({"DEM", "RTE"}),
            "ocvrp": frozenset({"DEM", "RTE"}),
            "cvrptw": frozenset({"DEM", "TW", "RTE"}),
            "ocvrptw": frozenset({"DEM", "TW", "RTE"}),
        }

        for problem_name, exp_blocks in expected.items():
            actual_blocks = get_variant_blocks(problem_name)
            self.assertEqual(
                actual_blocks,
                exp_blocks,
                f"Mismatch for {problem_name}: expected {exp_blocks}, got {actual_blocks}"
            )

    def test_unseen_group_partitions(self):
        """Verifies unseen44 and unseen55 exactly partition the 99 zero-shot variants."""
        seen = set(get_seen_group())
        unseen44 = set(get_unseen44_group())
        unseen55 = set(get_unseen55_group())
        zero_shot = set(ProblemSet.get("zero_shot_list"))

        self.assertEqual(len(seen), 11, "Expected 11 seen problems")
        self.assertEqual(len(unseen44), 44, "Expected 44 problems in unseen44 (Table 11)")
        self.assertEqual(len(unseen55), 55, "Expected 55 problems in unseen55 (Table 12)")

        # Disjoint
        self.assertEqual(len(unseen44 & unseen55), 0, "unseen44 and unseen55 must be disjoint")
        self.assertEqual(len(seen & unseen44), 0, "seen and unseen44 must be disjoint")
        self.assertEqual(len(seen & unseen55), 0, "seen and unseen55 must be disjoint")

        # Union is exactly zero-shot list
        self.assertEqual(unseen44 | unseen55, zero_shot, "Union of unseen44 and unseen55 must equal zero_shot_list")

    def test_evaluation_group_helper(self):
        self.assertEqual(len(get_evaluation_group("seen")), 11)
        self.assertEqual(len(get_evaluation_group("unseen44")), 44)
        self.assertEqual(len(get_evaluation_group("unseen55")), 55)
        self.assertEqual(len(get_evaluation_group("zero_shot")), 99)


if __name__ == "__main__":
    unittest.main()
