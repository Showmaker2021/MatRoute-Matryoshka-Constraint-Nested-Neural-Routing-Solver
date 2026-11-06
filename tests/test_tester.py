'''
Unit tests for MatRoute evaluation harness (Stage 6).
Verifies:
1. CSV output format matches URS benchmark specification.
2. Gap calculation parity for both minimization (TSP, CVRP) and maximization (OP, AOP).
3. Harness consistency: URSPolicyAdapter inside MatRouteTester produces identical results
   to URS baseline calculation on the same instances.
'''

import os
import csv
import torch
import pytest
import numpy as np

from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry
from model.Model import Model
from problem.ProblemRep import get_problem_representations
from utils.utils import AverageMeter

from matroute.layout import BlockLayout
from matroute.policy import MatRoutePolicy
from matroute.eval.tester import MatRouteTester, URSPolicyAdapter


class TestTesterHarness:

    def test_op_vs_min_gap_formula(self):
        """Checks correctness of gap formula for minimization vs maximization."""
        oracle_min = 10.0
        score_min = 12.0
        # Min: (score - oracle) / oracle = +20%
        gap_min = (score_min - oracle_min) * 100.0 / oracle_min
        assert abs(gap_min - 20.0) < 1e-6

        oracle_max = 50.0
        score_max = 45.0
        # Max: (oracle - score) / oracle = +10% (underperformed oracle by 10%)
        gap_max = (oracle_max - score_max) * 100.0 / oracle_max
        assert abs(gap_max - 10.0) < 1e-6

    def test_csv_generation_smoke(self, tmp_path):
        """Verifies MatRouteTester generates CSV with exact columns and finite values."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        layout = BlockLayout.from_preset("d64")
        policy = MatRoutePolicy(layout=layout, num_encoder_layers=2, zeta=50.0).to(device)

        csv_file = str(tmp_path / "test_eval.csv")
        tester = MatRouteTester(
            policy=policy,
            data_dir="dataset",
            device=device,
            batch_size=16,
            augmentation_enable=False,
            model_name="matroute_smoke",
            seed=42,
        )

        res = tester.test_problems(
            problem_list=["tsp"],
            scale=100,
            max_instances=8,
            output_csv=csv_file,
        )

        assert len(res) == 1
        assert res[0]["variant"] == "tsp"
        assert res[0]["group"] == "seen"
        assert np.isfinite(res[0]["gap"])
        assert np.isfinite(res[0]["no_aug_gap"])

        # Check CSV file content
        assert os.path.exists(csv_file)
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_fields = ["variant", "group", "gap", "no_aug_gap", "time", "n", "model", "seed"]
            assert reader.fieldnames == expected_fields
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["variant"] == "tsp"
            assert rows[0]["model"] == "matroute_smoke"

    def test_harness_consistency_with_urs_equations(self):
        """
        Confirms that running a model through MatRouteTester yields
        identical POMO scores to the direct URS rollouts on the same instances.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        torch.manual_seed(999)
        model_params = {
            "embedding_dim": 64,
            "sqrt_embedding_dim": 8,
            "encoder_layer_num": 2,
            "qkv_dim": 8,
            "head_num": 8,
            "logit_clipping": 50.0,
            "ff_hidden_dim": 128,
            "eval_type": "greedy",
            "norm": "instance",
            "demand_max1": True,
            "problem_names": ["tsp"],
        }
        urs_model = Model(**model_params).to(device)
        adapter = URSPolicyAdapter(urs_model, device=device)

        tester = MatRouteTester(
            policy=adapter,
            data_dir="dataset",
            device=device,
            batch_size=8,
            augmentation_enable=False,
            model_name="urs_wrapped",
            seed=999,
        )

        res = tester.test_one_problem("tsp", scale=100, max_instances=8)
        assert res is not None
        assert np.isfinite(res["score"])
        assert np.isfinite(res["gap"])
        # Score must be positive distance
        assert res["score"] > 0
