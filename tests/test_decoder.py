'''
Unit tests for Giai đoạn 4 (matroute/decoder.py, matroute/policy.py).
Implements the 4 mandatory acceptance tests:
1. test_funnel_equals_full
2. test_feasibility (all 11 seen problems with FULL, FUNNEL, CORE_ONLY)
3. test_core_only_for_tsp
4. test_start_nodes
'''

import unittest
import sys
import os
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from matroute.layout import BlockLayout
from matroute.spec import get_variant_blocks
from matroute.decoder import MatRouteDecoder, funnel_logits
from matroute.policy import MatRoutePolicy
from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry
from problem.ProblemSet import ProblemSet


class TestDecoderAndPolicy(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(3407)
        self.layout = BlockLayout.preset("d64")
        self.batch_size = 2
        self.pomo_size = 2
        self.num_nodes = 15

    def test_funnel_equals_full(self):
        """
        Test 1:
        When k >= N for all blocks, funnel_logits MUST strictly equal FULL logits on all feasible nodes.
        """
        S = frozenset({"DEM", "TW"})
        q = {
            "CORE": torch.randn(self.batch_size, self.pomo_size, self.layout.sizes["CORE"]),
            "DEM": torch.randn(self.batch_size, self.pomo_size, self.layout.sizes["DEM"]),
            "TW": torch.randn(self.batch_size, self.pomo_size, self.layout.sizes["TW"]),
        }
        k = {
            "CORE": torch.randn(self.batch_size, self.num_nodes, self.layout.sizes["CORE"]),
            "DEM": torch.randn(self.batch_size, self.num_nodes, self.layout.sizes["DEM"]),
            "TW": torch.randn(self.batch_size, self.num_nodes, self.layout.sizes["TW"]),
        }

        # Random feasibility mask with at least 5 feasible nodes
        feasible = torch.zeros(self.batch_size, self.pomo_size, self.num_nodes, dtype=torch.bool)
        feasible[:, :, :8] = True

        decoder = MatRouteDecoder(self.layout)

        # FULL logits
        u_full, _, _ = decoder.forward_step(q, k, feasible, S=S, level="FULL")

        # FUNNEL logits with topk >= num_nodes
        topk_large = {b: self.num_nodes + 5 for b in q.keys()}
        u_funnel, _, _ = decoder.forward_step(q, k, feasible, S=S, level="FUNNEL", topk=topk_large)

        # Compare on feasible nodes
        diff = (u_full[feasible] - u_funnel[feasible]).abs().max().item()
        self.assertLess(
            diff,
            1e-6,
            f"Funnel with k >= N did not match FULL! Max difference: {diff:.8e}",
        )

    def test_core_only_for_tsp(self):
        """
        Test 3:
        For TSP (where S is empty), FULL and CORE_ONLY must produce identical logits.
        """
        S = frozenset()
        q = {"CORE": torch.randn(self.batch_size, self.pomo_size, self.layout.sizes["CORE"])}
        k = {"CORE": torch.randn(self.batch_size, self.num_nodes, self.layout.sizes["CORE"])}
        feasible = torch.ones(self.batch_size, self.pomo_size, self.num_nodes, dtype=torch.bool)

        decoder = MatRouteDecoder(self.layout)
        u_full, _, _ = decoder.forward_step(q, k, feasible, S=S, level="FULL")
        u_core, _, _ = decoder.forward_step(q, k, feasible, S=S, level="CORE_ONLY")

        diff = (u_full - u_core).abs().max().item()
        self.assertLess(diff, 1e-6, f"For TSP, FULL and CORE_ONLY differed by {diff:.8e}")

    def test_start_nodes(self):
        """
        Test 4:
        Verifies step 0 and step 1 start node selections match URS across problem families.
        """
        policy = MatRoutePolicy(self.layout, num_encoder_layers=2)

        # 1. TSP: trajectories start from nodes 0 .. pomo-1
        s0_tsp = policy.select_start_nodes(0, "tsp", self.batch_size, 4, depot_num=0)
        self.assertEqual(s0_tsp[0].tolist(), [0, 1, 2, 3])

        # 2. CVRP: step 0 all start at depot 0; step 1 branch to customers 1 .. pomo
        s0_cvrp = policy.select_start_nodes(0, "cvrp", self.batch_size, 4, depot_num=1)
        self.assertEqual(s0_cvrp[0].tolist(), [0, 0, 0, 0])

        s1_cvrp = policy.select_start_nodes(1, "cvrp", self.batch_size, 4, depot_num=1)
        self.assertEqual(s1_cvrp[0].tolist(), [1, 2, 3, 4])

        # 3. MD: 3 depots, pomo=6 -> step 0 interleaved depots [0, 0, 1, 1, 2, 2]
        s0_md = policy.select_start_nodes(0, "mdcvrp", self.batch_size, 6, depot_num=3)
        self.assertEqual(s0_md[0].tolist(), [0, 0, 1, 1, 2, 2])

    def test_feasibility_all_seen_problems(self):
        """
        Test 2:
        Executes rollouts across all 11 seen problems for FULL, FUNNEL, and CORE_ONLY.
        Verifies 100% solution feasibility (reward finite and not None).
        """
        seen_problems = ProblemSet.get("train_problem_list")
        policy = MatRoutePolicy(self.layout, num_encoder_layers=2)
        policy.eval()

        levels = ["FULL", "FUNNEL", "CORE_ONLY"]

        for problem_name in seen_problems:
            mask_fn = mask_registry.build_combined_mask(problem_name)
            S = get_variant_blocks(problem_name)
            capacity = 50.0 if "vrp" in problem_name else None

            for level in levels:
                env = UniVRPEnv()
                env.load_problems(
                    batch_size=2,
                    problem_name=problem_name,
                    problem_size=20,
                    pomo_size=2,
                    capacity=capacity,
                    device="cpu",
                )

                topk = {b: 4 for b in self.layout.names}
                out = policy.rollout(
                    env=env,
                    mask_fn=mask_fn,
                    S=S,
                    level=level,
                    topk=topk,
                    greedy=False,
                    track_distill=True,
                )

                self.assertIsNotNone(out.reward, f"Reward was None in {problem_name} ({level})")
                self.assertFalse(torch.isnan(out.reward).any(), f"NaN reward in {problem_name} ({level})")
                self.assertFalse(torch.isinf(out.reward).any(), f"Inf reward in {problem_name} ({level})")
                self.assertEqual(out.reward.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
