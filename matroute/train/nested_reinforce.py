'''
Nested Reinforce training algorithms and running statistics for MatRoute.
Implements multi-task POMO shared baseline with reward normalization and on-policy distillation.
'''

import os
from typing import Dict, FrozenSet, Optional, Tuple
import torch
import torch.nn as nn
from matroute.policy import MatRoutePolicy, RolloutOutput


class RunningRewardStats:
    """
    Tracks running standard deviation of rewards per (variant, level) tuple
    using Exponential Moving Average (decay 0.99) for reward scale balancing.
    """
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    def update(self, variant: str, level: str, reward_tensor: torch.Tensor):
        key = (variant, level)
        batch_std = reward_tensor.float().std().item()
        if batch_std < 1e-4 or torch.isnan(torch.tensor(batch_std)):
            batch_std = 1.0

        if key not in self.stats:
            self.stats[key] = {"std": batch_std}
        else:
            old_std = self.stats[key]["std"]
            self.stats[key]["std"] = self.decay * old_std + (1.0 - self.decay) * batch_std

    def get_std(self, variant: str, level: str) -> float:
        key = (variant, level)
        if key in self.stats:
            return max(self.stats[key]["std"], 1e-4)
        return 1.0

    def state_dict(self) -> dict:
        # Convert tuple keys to string for clean serialization
        return {f"{k[0]}|{k[1]}": v for k, v in self.stats.items()}

    def load_state_dict(self, state_dict: dict):
        self.stats = {}
        for k, v in state_dict.items():
            parts = k.split("|")
            self.stats[(parts[0], parts[1])] = v


def train_step(
    policy: MatRoutePolicy,
    optimizer: torch.optim.Optimizer,
    env,
    mask_fn,
    variant: str,
    S: FrozenSet[str],
    level: str,
    topk: Optional[Dict[str, int]],
    lambda_dist: float,
    stats: RunningRewardStats,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    Executes one training step on a sampled batch of instances.
    """
    policy.train()
    optimizer.zero_grad()

    # 1. Rollout across parallel POMO trajectories
    out: RolloutOutput = policy.rollout(
        env=env,
        mask_fn=mask_fn,
        S=S,
        level=level,
        topk=topk,
        greedy=False,
        track_distill=True,
    )

    # 2. Update running reward statistics
    stats.update(variant, level, out.reward)
    scale_std = stats.get_std(variant, level)

    # 3. Compute standardized POMO advantage
    # Baseline is mean reward across POMO trajectories of each instance: [batch, 1]
    pomo_baseline = out.reward.float().mean(dim=1, keepdim=True)
    adv = (out.reward - pomo_baseline) / scale_std

    # 4. REINFORCE loss with shared baseline
    loss_rl = -(adv.detach() * out.logp).mean()

    # 5. Distillation loss
    loss_dist = out.dist_loss

    # Total loss
    total_loss = loss_rl + lambda_dist * loss_dist

    # 6. Backward and step
    total_loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)

    optimizer.step()

    # Solution metric (cost is -reward for routing, or positive prize for OP)
    avg_reward = out.reward.mean().item()
    if variant in ("op", "aop"):
        cost_metric = avg_reward
    else:
        cost_metric = -avg_reward

    return {
        "loss_rl": loss_rl.item(),
        "loss_dist": loss_dist.item() if isinstance(loss_dist, torch.Tensor) else float(loss_dist),
        "total_loss": total_loss.item(),
        "avg_reward": avg_reward,
        "cost_metric": cost_metric,
    }


def save_checkpoint(
    filepath: str,
    policy: MatRoutePolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    stats: RunningRewardStats,
    epoch: int,
    step: int,
    config: Optional[dict] = None,
):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "stats": stats.state_dict(),
        "config": config,
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    policy: MatRoutePolicy,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    stats: Optional[RunningRewardStats] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int]:
    checkpoint = torch.load(filepath, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if stats is not None and "stats" in checkpoint:
        stats.load_state_dict(checkpoint["stats"])

    epoch = checkpoint.get("epoch", 1)
    step = checkpoint.get("step", 0)
    return epoch, step
