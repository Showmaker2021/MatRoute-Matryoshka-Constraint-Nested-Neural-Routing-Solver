'''
Unit tests for BlockLayout and Feature extraction in MatRoute.
Verifies exact parity with URS Model_LIB.py across all 11 seen problems.
'''

import unittest
import sys
import os
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from matroute.layout import BlockLayout, CANONICAL_BLOCK_NAMES, PRESET_CONFIGS
from matroute.spec import get_variant_blocks
from matroute.features import build_raw_block_features, BlockInputEmbedding
from env.UniVRPEnv import UniVRPEnv
from problem.ProblemSet import ProblemSet
from model.Model_LIB import (
    unified_node_position_construction,
    unified_node_attribute_construction,
)


class TestLayoutAndFeatures(unittest.TestCase):

    def test_block_layout_properties(self):
        """Verifies dimension calculations, indexing and presets in BlockLayout."""
        for preset_name, expected_sizes in PRESET_CONFIGS.items():
            layout = BlockLayout.preset(preset_name)
            self.assertEqual(layout.names, CANONICAL_BLOCK_NAMES)
            self.assertEqual(layout.sizes, expected_sizes)
            self.assertEqual(layout.dim(), sum(expected_sizes.values()))

            # Verify index ranges are contiguous and non-overlapping
            all_indices = []
            for name in layout.names:
                idx = layout.idx(name)
                self.assertEqual(len(idx), expected_sizes[name])
                self.assertEqual(layout.n_heads(name), expected_sizes[name] // 8)
                all_indices.extend(idx.tolist())

            self.assertEqual(all_indices, list(range(layout.dim())))

            # Test active_idx with various subsets
            S_sub = frozenset({"DEM", "TW"})
            active_idx = layout.active_idx(S_sub)
            expected_dim = expected_sizes["CORE"] + expected_sizes["DEM"] + expected_sizes["TW"]
            self.assertEqual(len(active_idx), expected_dim)
            self.assertEqual(layout.active_dim(S_sub), expected_dim)

    def test_feature_parity_with_urs_all_11_seen(self):
        """
        For all 11 seen problems, verifies that MatRoute block features strictly match
        the corresponding feature columns produced by URS Model_LIB.py (tolerance < 1e-6).
        """
        seen_problems = ProblemSet.get("train_problem_list")
        batch_size = 2
        problem_size = 20

        for problem_name in seen_problems:
            env = UniVRPEnv()
            capacity = 50.0 if "vrp" in problem_name else None
            env.load_problems(
                batch_size=batch_size,
                problem_name=problem_name,
                problem_size=problem_size,
                pomo_size=2,
                capacity=capacity,
                device="cpu",
            )
            reset_state, _, _ = env.reset()

            # 1. Compute URS features from Model_LIB.py
            urs_pos = unified_node_position_construction(reset_state.problems, problem_name)
            urs_attr = unified_node_attribute_construction(reset_state.problems, problem_name, demand_max1=True)

            # 2. Compute MatRoute features
            S = get_variant_blocks(problem_name)
            eta = torch.zeros((batch_size, env.problem_size + env.depot_num, 1))
            mat_raw = build_raw_block_features(
                reset_state.problems,
                problem_name,
                S=S,
                eta=eta,
                demand_max1=True,
            )

            # --- Check CORE ---
            self.assertIn("CORE", mat_raw)
            # x, y check (symmetric problems)
            if "a" not in problem_name:
                diff_xy = torch.abs(mat_raw["CORE"][:, :, :2] - urs_pos[:, :, 1:3]).max().item()
                self.assertLess(diff_xy, 1e-6, f"CORE x,y mismatch in {problem_name}")

            # depot flag check
            diff_depot = torch.abs(mat_raw["CORE"][:, :, 3] - urs_attr[:, :, 6]).max().item()
            self.assertLess(diff_depot, 1e-6, f"CORE is_depot mismatch in {problem_name}")

            # --- Check DEM ---
            if "DEM" in S:
                self.assertIn("DEM", mat_raw)
                # Demand
                diff_dem = torch.abs(mat_raw["DEM"][:, :, 0] - urs_attr[:, :, 0]).max().item()
                self.assertLess(diff_dem, 1e-6, f"DEM demand mismatch in {problem_name}")
                # Delivery & Pickup flags
                diff_del = torch.abs(mat_raw["DEM"][:, :, 1] - urs_attr[:, :, 8]).max().item()
                diff_pk = torch.abs(mat_raw["DEM"][:, :, 2] - urs_attr[:, :, 7]).max().item()
                self.assertLess(diff_del, 1e-6, f"DEM delivery flag mismatch in {problem_name}")
                self.assertLess(diff_pk, 1e-6, f"DEM pickup flag mismatch in {problem_name}")

            # --- Check TW ---
            if "TW" in S:
                self.assertIn("TW", mat_raw)
                # early, late, service_time match urs_attr[:, :, 3:6]
                diff_tw = torch.abs(mat_raw["TW"][:, :, :3] - urs_attr[:, :, 3:6]).max().item()
                self.assertLess(diff_tw, 1e-6, f"TW early/late/service mismatch in {problem_name}")

            # --- Check PRZ ---
            if "PRZ" in S:
                self.assertIn("PRZ", mat_raw)
                # prize, penalty match urs_attr[:, :, 1:3]
                diff_prz = torch.abs(mat_raw["PRZ"][:, :, :2] - urs_attr[:, :, 1:3]).max().item()
                self.assertLess(diff_prz, 1e-6, f"PRZ prize/penalty mismatch in {problem_name}")

            # --- Check PD ---
            if "PD" in S:
                self.assertIn("PD", mat_raw)
                # pickup, delivery match urs_attr[:, :, 7:9]
                diff_pd = torch.abs(mat_raw["PD"][:, :, :2] - urs_attr[:, :, 7:9]).max().item()
                self.assertLess(diff_pd, 1e-6, f"PD flags mismatch in {problem_name}")

            # --- Check RTE ---
            if "RTE" in S:
                self.assertIn("RTE", mat_raw)
                # multi_route, open_route match urs_attr[:, :, 9:11]
                diff_rte = torch.abs(mat_raw["RTE"][:, :, :2] - urs_attr[:, :, 9:11]).max().item()
                self.assertLess(diff_rte, 1e-6, f"RTE multi/open flags mismatch in {problem_name}")

    def test_block_input_embedding_projection(self):
        """Verifies linear projection and concatenation across active blocks."""
        layout = BlockLayout.preset("d128")
        embedding = BlockInputEmbedding(layout)

        batch_size = 2
        num_nodes = 21
        S = frozenset({"DEM", "TW"})

        # Create dummy raw features
        raw_feats = {
            "CORE": torch.randn((batch_size, num_nodes, 4)),
            "DEM": torch.randn((batch_size, num_nodes, 3)),
            "TW": torch.randn((batch_size, num_nodes, 3)),
        }

        embedded = embedding(raw_feats, S=S)
        self.assertEqual(set(embedded.keys()), {"CORE", "DEM", "TW"})
        self.assertEqual(embedded["CORE"].shape, (batch_size, num_nodes, layout.sizes["CORE"]))
        self.assertEqual(embedded["DEM"].shape, (batch_size, num_nodes, layout.sizes["DEM"]))
        self.assertEqual(embedded["TW"].shape, (batch_size, num_nodes, layout.sizes["TW"]))

        concat_emb = embedding.concat_embedded(embedded, S=S)
        expected_active_dim = layout.active_dim(S)
        self.assertEqual(concat_emb.shape, (batch_size, num_nodes, expected_active_dim))


if __name__ == "__main__":
    unittest.main()
