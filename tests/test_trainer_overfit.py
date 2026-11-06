'''
Overfit and Trainer functional tests for MatRoute (Giai đoạn 5 acceptance).
1. Overfit a fixed batch on TSP N=20: solution cost decreases by >= 15%.
2. Checkpoint resume parity: loaded model produces identical outputs.
'''

import unittest
import sys
import os
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from matroute.layout import BlockLayout
from matroute.policy import MatRoutePolicy
from matroute.spec import get_variant_blocks
from matroute.train.nested_reinforce import (
    RunningRewardStats,
    train_step,
    save_checkpoint,
    load_checkpoint,
)
from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry


class TestTrainerOverfit(unittest.TestCase):

    def test_overfit_fixed_batch_tsp(self):
        """
        Tests that policy can overfit a fixed batch of TSP instances (N=20, 16 instances).
        Verifies average tour cost decreases by at least 15% from initialization.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        torch.manual_seed(3407)
        layout = BlockLayout.preset("d64")
        policy = MatRoutePolicy(layout, num_encoder_layers=2).to(device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=0.0)

        variant = "tsp"
        S = get_variant_blocks(variant)
        mask_fn = mask_registry.build_combined_mask(variant)

        batch_size = 16
        problem_size = 20
        pomo_size = 20
        fixed_seed = 9999

        def make_batch_env():
            torch.manual_seed(fixed_seed)
            env = UniVRPEnv()
            env.load_problems(
                batch_size=batch_size,
                problem_name=variant,
                problem_size=problem_size,
                pomo_size=pomo_size,
                device=device,
            )
            return env

        # Initial greedy evaluation
        policy.eval()
        with torch.no_grad():
            eval_env = make_batch_env()
            eval_out = policy.rollout(eval_env, mask_fn, S=S, level="FULL", greedy=True)
            initial_cost = (-eval_out.reward).amin(dim=1).mean().item()

        print(f"\n[Overfit Test] Initial TSP-20 Cost: {initial_cost:.4f}")

        # Training loop over 150 steps
        policy.train()
        stats = RunningRewardStats()

        costs = []
        for step in range(1, 151):
            env = make_batch_env()
            metrics = train_step(
                policy=policy,
                optimizer=optimizer,
                env=env,
                mask_fn=mask_fn,
                variant=variant,
                S=S,
                level="FULL",
                topk=None,
                lambda_dist=0.0,
                stats=stats,
            )
            costs.append(metrics["cost_metric"])

        # Final greedy evaluation
        policy.eval()
        with torch.no_grad():
            eval_env = make_batch_env()
            eval_out = policy.rollout(eval_env, mask_fn, S=S, level="FULL", greedy=True)
            final_cost = (-eval_out.reward).amin(dim=1).mean().item()

        improvement = (initial_cost - final_cost) / initial_cost * 100.0
        print(f"[Overfit Test] Final TSP-20 Cost: {final_cost:.4f} (Improvement: {improvement:.2f}%)")

        self.assertGreaterEqual(
            improvement,
            15.0,
            f"Overfit failed! Improvement was {improvement:.2f}%, expected >= 15.0%",
        )

    def test_checkpoint_resume(self):
        """
        Tests checkpoint saving and loading parity.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        layout = BlockLayout.preset("d64")
        policy1 = MatRoutePolicy(layout, num_encoder_layers=2).to(device)
        optimizer1 = torch.optim.AdamW(policy1.parameters(), lr=1e-4)
        stats1 = RunningRewardStats()
        stats1.update("tsp", "FULL", torch.tensor([1.0, 2.0, 3.0], device=device))

        ckpt_path = "results/local/test_ckpt.pt"
        save_checkpoint(ckpt_path, policy1, optimizer1, None, stats1, epoch=3, step=75)

        policy2 = MatRoutePolicy(layout, num_encoder_layers=2).to(device)
        optimizer2 = torch.optim.AdamW(policy2.parameters(), lr=1e-4)
        stats2 = RunningRewardStats()

        epoch, step = load_checkpoint(ckpt_path, policy2, optimizer2, None, stats2, device=device)
        self.assertEqual(epoch, 3)
        self.assertEqual(step, 75)

        # Check weights are strictly identical
        for p1, p2 in zip(policy1.parameters(), policy2.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-7))

        self.assertAlmostEqual(stats1.get_std("tsp", "FULL"), stats2.get_std("tsp", "FULL"), places=4)

        # Clean up
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


if __name__ == "__main__":
    unittest.main()
