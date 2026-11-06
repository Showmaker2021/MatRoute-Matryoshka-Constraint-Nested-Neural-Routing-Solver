'''
Entry point for training MatRoute models.
Supports:
- YAML configuration with CLI overrides
- Weights & Biases (wandb) experiment tracking
- Automatic resumption from checkpoint (--resume)
- Dynamic sandwich sampling and nested REINFORCE
'''

import os
import sys
import csv
import time
import yaml
import argparse
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matroute.layout import BlockLayout
from matroute.policy import MatRoutePolicy
from matroute.train.sampler import SandwichSampler
from matroute.train.nested_reinforce import (
    RunningRewardStats,
    train_step,
    save_checkpoint,
    load_checkpoint,
)
from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry
from problem.ProblemSet import ProblemSet


def parse_args():
    parser = argparse.ArgumentParser(description="MatRoute Neural Solver Training")
    parser.add_argument("--config", type=str, default="matroute/configs/smoke_local.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--max_epochs", type=int, default=None, help="Alias for --epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log / results output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--soft_core", action="store_true", help="Enable soft-gated CORE connections")
    parser.add_argument("--stopgrad_core", action="store_true", help="Stop gradients from constraints into CORE")

    # Weights & Biases options
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity / team")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode")

    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # 1. Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Device configuration
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_idx)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # 3. Load config
    cfg = load_yaml_config(args.config)
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    l_cfg = cfg.get("logging", {})

    # Directories
    results_dir_str = args.log_dir or l_cfg.get("results_dir", "results/default")
    results_dir = Path(results_dir_str)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir_str = args.checkpoint_dir or l_cfg.get("checkpoint_dir", str(results_dir))
    checkpoint_dir = Path(checkpoint_dir_str)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize BlockLayout and Policy
    layout_preset = m_cfg.get("layout_preset", "d128")
    layout = BlockLayout.from_preset(layout_preset)

    soft_core = args.soft_core or m_cfg.get("soft_core", False)
    stopgrad_core = args.stopgrad_core or m_cfg.get("stopgrad_core", False)

    policy = MatRoutePolicy(
        layout=layout,
        num_encoder_layers=m_cfg.get("num_encoder_layers", 12),
        zeta=m_cfg.get("zeta", 50.0),
        soft_core=soft_core,
        stopgrad_core=stopgrad_core,
    ).to(device)

    # 5. Optimizer and Scheduler
    lr = float(t_cfg.get("learning_rate", 1e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-6))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    lr_decay_epoch = t_cfg.get("lr_decay_epoch")
    lr_decay_factor = float(t_cfg.get("lr_decay_factor", 0.1))

    # 6. Problem selection & Sampler
    prob_setting = t_cfg.get("problems", "train_problem_list")
    if prob_setting == "train_problem_list":
        problem_list = sorted(list(ProblemSet.get("train_problem_list")), key=len)
    elif isinstance(prob_setting, list):
        problem_list = prob_setting
    else:
        problem_list = ProblemSet.get(prob_setting)

    total_epochs = args.epochs or args.max_epochs or t_cfg.get("total_epochs", 500)
    batches_per_epoch = t_cfg.get("batches_per_epoch", 2000)
    max_steps = args.max_steps if args.max_steps is not None else t_cfg.get("max_steps", None)

    sampler = SandwichSampler(
        problem_list=problem_list,
        warmup_epoch_ratio=t_cfg.get("warmup_ratio", 0.20),
        lambda_dist_start=t_cfg.get("lambda_dist_start", 0.10),
        lambda_dist_end=t_cfg.get("lambda_dist_end", 0.02),
        total_epochs=total_epochs,
    )

    stats = RunningRewardStats()

    # 7. Initialize Weights & Biases (optional)
    wandb_logger = None
    if args.use_wandb or l_cfg.get("use_wandb", False):
        try:
            import wandb
            project = args.wandb_project or l_cfg.get("wandb_project", "MatRoute")
            entity = args.wandb_entity or l_cfg.get("wandb_entity", None)
            name = args.wandb_name or f"{layout_preset}_{Path(args.config).stem}"
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=cfg,
                mode=args.wandb_mode,
            )
            wandb_logger = wandb
            print(f">> [W&B] Initialized successfully: project='{project}', run='{name}'")
        except ImportError:
            print(">> [W&B WARNING] wandb package is not installed. To track on W&B, run: pip install wandb")
            wandb_logger = None

    # 8. Resume if specified
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            filepath=args.resume,
            policy=policy,
            optimizer=optimizer,
            stats=stats,
            device=device,
        )
        print(f"Resumed successfully from {args.resume} (Epoch {start_epoch}, Step {global_step})")

    # 9. Training loop
    csv_path = results_dir / "train_log.csv"
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "epoch", "variant", "level", "loss_rl", "loss_dist", "total_loss", "cost_metric", "elapsed_s"])

    log_interval = l_cfg.get("log_interval", 50)
    ckpt_steps = l_cfg.get("checkpoint_interval_steps")
    ckpt_epochs = l_cfg.get("checkpoint_interval_epochs", 10)

    problem_size = t_cfg.get("problem_size", 100)
    batch_size = t_cfg.get("batch_size", 128)
    pomo_size = t_cfg.get("pomo_size", problem_size)
    default_capacity = t_cfg.get("capacity", 50.0)

    print("=" * 80)
    print(f"   STARTING MATROUTE TRAINING: {cfg.get('preset', 'custom')}")
    print(f"   Layout: {layout_preset} (dim={layout.dim()}) | Device: {device}")
    print(f"   Problems: {problem_list}")
    print(f"   N={problem_size}, Batch={batch_size}, POMO={pomo_size}")
    if wandb_logger is not None:
        print(f"   W&B Tracking: Active (Project: {args.wandb_project or 'MatRoute'})")
    print("=" * 80)

    start_time = time.time()
    done_training = False
    best_cost = float("inf")

    for epoch in range(start_epoch, total_epochs + 1):
        # Learning rate decay check
        if lr_decay_epoch is not None and epoch >= lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * lr_decay_factor

        current_lr = optimizer.param_groups[0]["lr"]

        for b_idx in range(batches_per_epoch):
            global_step += 1

            # 1. Sample task and level
            variant, S = sampler.sample_task()
            level, topk = sampler.sample_level(current_epoch=epoch, current_step=global_step)
            lambda_dist = sampler.get_lambda_dist(current_epoch=epoch)

            # 2. Setup environment for current batch
            env = UniVRPEnv()
            mask_fn = mask_registry.build_combined_mask(variant)
            capacity = default_capacity if "vrp" in variant else None

            env.load_problems(
                batch_size=batch_size,
                problem_name=variant,
                problem_size=problem_size,
                pomo_size=pomo_size,
                capacity=capacity,
                device=device,
            )

            # 3. Training step
            step_metrics = train_step(
                policy=policy,
                optimizer=optimizer,
                env=env,
                mask_fn=mask_fn,
                variant=variant,
                S=S,
                level=level,
                topk=topk,
                lambda_dist=lambda_dist,
                stats=stats,
            )

            # 4. Log progress
            elapsed = time.time() - start_time
            csv_writer.writerow([
                global_step,
                epoch,
                variant,
                level,
                f"{step_metrics['loss_rl']:.4f}",
                f"{step_metrics['loss_dist']:.4f}",
                f"{step_metrics['total_loss']:.4f}",
                f"{step_metrics['cost_metric']:.4f}",
                f"{elapsed:.2f}",
            ])
            csv_file.flush()

            # W&B step-level logging
            if wandb_logger is not None:
                wandb_logger.log({
                    "step": global_step,
                    "epoch": epoch,
                    "train/loss_total": step_metrics["total_loss"],
                    "train/loss_rl": step_metrics["loss_rl"],
                    "train/loss_dist": step_metrics["loss_dist"],
                    "train/cost_metric": step_metrics["cost_metric"],
                    "train/lr": current_lr,
                    "train/lambda_dist": lambda_dist,
                    f"tasks/{variant}/cost": step_metrics["cost_metric"],
                }, step=global_step)

            if global_step % log_interval == 0:
                print(
                    f"Epoch {epoch:3d} | Step {global_step:5d} | {variant:8s} | Level: {level:9s} | "
                    f"RL: {step_metrics['loss_rl']:7.4f} | Dist: {step_metrics['loss_dist']:7.4f} | "
                    f"Total: {step_metrics['total_loss']:7.4f} | Metric: {step_metrics['cost_metric']:7.4f}"
                )

            # Checkpoint by steps
            if ckpt_steps is not None and global_step % ckpt_steps == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(str(ckpt_path), policy, optimizer, None, stats, epoch, global_step, cfg)
                print(f">> Saved checkpoint: {ckpt_path}")

            if max_steps is not None and global_step >= max_steps:
                done_training = True
                break

        # Checkpoint by epoch
        if ckpt_epochs is not None and (epoch % ckpt_epochs == 0 or epoch == total_epochs):
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            latest_path = checkpoint_dir / "checkpoint_latest.pt"
            save_checkpoint(str(ckpt_path), policy, optimizer, None, stats, epoch, global_step, cfg)
            save_checkpoint(str(latest_path), policy, optimizer, None, stats, epoch, global_step, cfg)
            print(f">> [Epoch {epoch}] Saved latest checkpoint: {latest_path}")

        # W&B epoch-level logging
        if wandb_logger is not None:
            wandb_logger.log({
                "epoch/epoch": epoch,
                "epoch/lr": current_lr,
            }, step=global_step)

        if done_training:
            print(f"Reached max_steps ({max_steps}). Stopping training.")
            break

    # Save final best / completed checkpoint
    final_path = checkpoint_dir / "checkpoint_best.pt"
    save_checkpoint(str(final_path), policy, optimizer, None, stats, epoch, global_step, cfg)
    print(f">> Saved final model to: {final_path}")

    csv_file.close()
    if wandb_logger is not None:
        wandb_logger.finish()
    print("Training finished successfully.")


if __name__ == "__main__":
    main()
