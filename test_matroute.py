'''
CLI Evaluation Entry Point for MatRoute.

Usage examples:
  # E1: Evaluate seen problems
  python test_matroute.py --group seen --model_load checkpoints/best.pt --output_csv results/e1_seen.csv

  # E2: Evaluate unseen zero-shot problems
  python test_matroute.py --group unseen44 --model_load checkpoints/best.pt --output_csv results/e2_unseen.csv

  # Local functional smoke test (4 problems, 64 instances)
  python test_matroute.py --problems tsp,cvrp,cvrptw,pdtsp --max_instances 64 --output_csv results/smoke_eval.csv
'''

import os
import sys
import yaml
import argparse
from pathlib import Path
from logging import getLogger, basicConfig, INFO

import torch

from matroute.layout import BlockLayout
from matroute.policy import MatRoutePolicy
from matroute.eval.tester import MatRouteTester, URSPolicyAdapter
from matroute.eval.groups import (
    get_seen_group,
    get_unseen44_group,
    get_unseen55_group,
    get_evaluation_group,
)

logger = getLogger("test_matroute")


def parse_args():
    parser = argparse.ArgumentParser(description="MatRoute Evaluation Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to evaluation config YAML")
    parser.add_argument("--group", type=str, default="seen", help="Evaluation group: seen, unseen44, unseen55, all")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated list of problem variants to evaluate")
    parser.add_argument("--model_load", type=str, default=None, help="Path to MatRoute model checkpoint")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root directory of datasets")
    parser.add_argument("--scale", type=int, default=100, help="Problem scale (default: 100)")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of instances per problem")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation chunks")
    parser.add_argument("--no_aug", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--aug_factor", type=int, nargs=2, default=[8, 128], help="Augmentation factors [sym, asym]")
    parser.add_argument("--level", type=str, default="FULL", choices=["FULL", "FUNNEL", "CORE_ONLY"], help="Decoder level")
    parser.add_argument("--sparse_knn", type=int, default=None, help="Optional sparse kNN attention threshold")
    parser.add_argument("--output_csv", type=str, default=None, help="Target CSV output path")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cuda or cpu")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--compare_urs", action="store_true", help="Run harness consistency test against URS Model")
    return parser.parse_args()


def load_policy_from_checkpoint(checkpoint_path: str, device: torch.device) -> MatRoutePolicy:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("config", {}).get("model", {})
    layout_name = model_cfg.get("layout", "d192")
    num_layers = model_cfg.get("num_layers", model_cfg.get("num_encoder_layers", 12))
    zeta = model_cfg.get("zeta", 50.0)

    layout = BlockLayout.from_preset(layout_name)
    policy = MatRoutePolicy(layout=layout, num_encoder_layers=num_layers, zeta=zeta).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded policy from {checkpoint_path} (preset={layout_name}, layers={num_layers})")
    return policy


def build_default_policy(device: torch.device, preset: str = "d192") -> MatRoutePolicy:
    layout = BlockLayout.from_preset(preset)
    policy = MatRoutePolicy(layout=layout, num_encoder_layers=6, zeta=50.0).to(device)
    logger.warning("No checkpoint loaded: using randomly initialized policy for testing/benchmarking.")
    return policy


def main():
    basicConfig(level=INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)

    # Device configuration
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Determine problems list
    if args.problems:
        problem_list = [p.strip().lower() for p in args.problems.split(",") if p.strip()]
    elif args.group.lower() == "seen":
        problem_list = get_seen_group()
    elif args.group.lower() == "unseen44":
        problem_list = get_unseen44_group()
    elif args.group.lower() == "unseen55":
        problem_list = get_unseen55_group()
    elif args.group.lower() == "all":
        problem_list = get_seen_group() + get_unseen44_group() + get_unseen55_group()
    else:
        problem_list = get_evaluation_group(args.group)

    logger.info(f"Target problems ({len(problem_list)}): {problem_list}")

    # Output CSV path
    if args.output_csv:
        output_csv = args.output_csv
    else:
        output_csv = f"results/{args.group}_n{args.scale}.csv"

    # Policy initialization
    if args.model_load and os.path.exists(args.model_load):
        policy = load_policy_from_checkpoint(args.model_load, device)
        model_name = Path(args.model_load).stem
    else:
        policy = build_default_policy(device, preset="d128" if device.type == "cpu" else "d192")
        model_name = "matroute_init"

    tester = MatRouteTester(
        policy=policy,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        aug_factor=args.aug_factor,
        augmentation_enable=not args.no_aug,
        level=args.level,
        sparse_knn=args.sparse_knn,
        model_name=model_name,
        seed=args.seed,
    )

    results = tester.test_problems(
        problem_list=problem_list,
        scale=args.scale,
        max_instances=args.max_instances,
        output_csv=output_csv,
    )

    logger.info(f"Evaluation complete. {len(results)} variants written to {output_csv}")


if __name__ == "__main__":
    main()
