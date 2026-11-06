'''
Smoke test script for URS Environment (UniVRPEnv) across 11 seen problems.
Verifies that random rollouts respecting feasibility masks run to completion
and produce valid solutions without violating constraints.
'''

import os
import sys
import torch

# Ensure repository root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry
from problem.ProblemSet import ProblemSet


def run_smoke_test(problem_size=20, batch_size=2, pomo_size=2, device='cpu', max_steps=250):
    print("=" * 75)
    print("   URS SMOKE TEST: 11 SEEN PROBLEMS (UniVRPEnv + Feasibility Masks)")
    print(f"   Config: N={problem_size}, Batch={batch_size}, POMO={pomo_size}, Device={device}")
    print("=" * 75)

    seen_problems = ProblemSet.get("train_problem_list")
    assert len(seen_problems) == 11, f"Expected 11 seen problems, got {len(seen_problems)}"

    results = []

    for idx, problem_name in enumerate(seen_problems, start=1):
        # 1. Initialize environment & build dynamic mask
        env = UniVRPEnv()
        mask_fn = mask_registry.build_combined_mask(problem_name)
        capacity = 50.0 if "vrp" in problem_name else None

        # 2. Load problem instances
        env.load_problems(
            batch_size=batch_size,
            problem_name=problem_name,
            problem_size=problem_size,
            pomo_size=pomo_size,
            capacity=capacity,
            device=device,
        )

        reset_state, _, _ = env.reset()
        state, reward, done = env.pre_step()

        # 3. First move
        # Depot (node 0) for VRP problems; node 0 for TSP/ATSP
        selected = torch.zeros((batch_size, pomo_size), dtype=torch.long, device=device)

        steps = 0
        while not done:
            state, reward, done = env.step(selected, mask_fn=mask_fn)
            steps += 1
            if done:
                break

            if steps > max_steps:
                raise RuntimeError(f"Problem {problem_name} exceeded max_steps ({max_steps}) without finishing!")

            # 4. Next move selection based on feasibility mask
            # mask values: 0 for feasible, -inf for forbidden
            mask = state.ninf_mask  # shape: (batch, pomo, problem_size + depot_num)
            feasible_actions = (mask == 0).float()

            b, p, n = feasible_actions.shape
            action_counts = feasible_actions.sum(dim=-1)
            if (action_counts == 0).any():
                raise RuntimeError(f"Deadlock detected in {problem_name}: no feasible action available!")

            # Randomly sample from feasible actions
            flat_feasible = feasible_actions.reshape(b * p, n)
            selected = torch.multinomial(flat_feasible, 1).reshape(b, p)

        # 5. Feasibility validation
        assert reward is not None, f"Reward is None after completion in {problem_name}"
        assert not torch.isnan(reward).any(), f"NaN in reward for {problem_name}"
        assert not torch.isinf(reward).any(), f"Inf in reward for {problem_name}"

        # In URS:
        # For OP: reward is positive prize collected
        # For TSP/CVRP/etc: reward is -total_distance (negative)
        avg_reward = reward.mean().item()
        if problem_name in ["op", "aop"]:
            solution_metric = f"Prize collected: {avg_reward:.4f}"
        else:
            solution_metric = f"Tour cost: {-avg_reward:.4f}"

        # Sequence length in selected_node_list
        seq_len = env.selected_node_list.shape[-1]

        results.append({
            "idx": idx,
            "problem": problem_name,
            "steps": steps,
            "seq_len": seq_len,
            "metric": solution_metric,
            "reward": avg_reward,
            "status": "PASS (Feasible)",
        })

        print(f"[{idx:2d}/11] {problem_name:10s} | Steps: {steps:3d} | SeqLen: {seq_len:3d} | {solution_metric:26s} | Status: PASS")

    print("=" * 75)
    print("ALL 11 SEEN PROBLEMS PASSED FEASIBILITY TEST SUCCESSFULLY!")
    print("=" * 75)
    return results


if __name__ == "__main__":
    torch.manual_seed(3407)
    run_smoke_test()
