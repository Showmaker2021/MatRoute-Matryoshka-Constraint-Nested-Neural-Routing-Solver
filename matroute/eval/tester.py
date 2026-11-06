'''
Evaluation module for MatRoute and baseline models.
Provides MatRouteTester conforming to URS testing protocols, with POMO,
augmentation (x8 symmetric, x128 asymmetric), oracle gap computation,
and CSV logging matching the paper specification:
`variant, group, gap, no_aug_gap, time, n, model, seed`
'''

import os
import csv
import time
import math
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from logging import getLogger

import torch
import torch.nn as nn

from env.UniVRPEnv import UniVRPEnv
from env.mask.mask_registry import mask_registry
from problem.ProblemRep import get_problem_representations
from problem.ProblemSet import ProblemSet
from data.DataFinder import DataFinder
from data.DataReader import get_saved_data
from utils.utils import AverageMeter

from matroute.spec import get_variant_blocks
from matroute.eval.groups import get_seen_group, get_unseen44_group, get_unseen55_group

logger = getLogger(__name__)


class URSPolicyAdapter:
    """
    Adapter wrapping URS Model to match MatRoute rollout signature.
    Used for harness consistency verification.
    """

    def __init__(self, urs_model: nn.Module, device: torch.device):
        self.model = urs_model
        self.device = device
        self.representations = {
            name: torch.tensor(rep, dtype=torch.float32, device=device)
            for name, rep in get_problem_representations().items()
        }

    def rollout(self, env, mask_fn, S=None, level="FULL", topk=None, greedy=True, **kwargs):
        self.model.eval()
        self.model.set_decoder_type("greedy" if greedy else "sampling")

        problem_name = env.problem_name
        prob_rep = self.representations[problem_name]

        reset_state, _, _ = env.reset()
        self.model.decoder.assign(prob_rep)
        self.model.pre_forward(reset_state, problem_name, prob_rep)

        state, reward, done = env.pre_step()
        while not done:
            cur_dist = env.get_local_feature()
            selected, _ = self.model(state, cur_dist)
            state, reward, done = env.step(selected, mask_fn=mask_fn)

        class DummyOut:
            def __init__(self, r):
                self.reward = r
        return DummyOut(reward)


class MatRouteTester:
    """
    Evaluator for MatRoute models on URS benchmark datasets.
    """

    def __init__(
        self,
        policy: Any,
        data_dir: str = "dataset",
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        aug_factor: List[int] = [8, 128],
        augmentation_enable: bool = True,
        level: str = "FULL",
        topk: Optional[Dict[str, int]] = None,
        sparse_knn: Optional[int] = None,
        model_name: str = "matroute",
        seed: int = 1234,
    ):
        self.policy = policy
        self.data_dir = data_dir
        self.data_finder = DataFinder(data_dir=data_dir)
        self.batch_size = batch_size
        self.aug_factor = list(aug_factor)
        self.augmentation_enable = augmentation_enable
        self.level = level
        self.topk = topk
        self.sparse_knn = sparse_knn
        self.model_name = model_name
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Pre-cache group sets for rapid classification
        self.seen_set = set(get_seen_group())
        self.unseen44_set = set(get_unseen44_group())
        self.unseen55_set = set(get_unseen55_group())

    def get_group_name(self, variant: str) -> str:
        v = variant.lower().strip()
        if v in self.seen_set:
            return "seen"
        elif v in self.unseen44_set:
            return "unseen44"
        elif v in self.unseen55_set:
            return "unseen55"
        return "other"

    def test_one_problem(
        self,
        problem_name: str,
        scale: int = 100,
        max_instances: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Runs evaluation on a single problem variant and scale.
        """
        p_name = problem_name.lower().strip()
        data_info = self.data_finder.get(p_name, scale)

        if data_info is None:
            logger.warning(f"No test dataset found for {p_name} at scale {scale}")
            return None

        # Determine number of episodes to load
        episodes_to_load = max_instances if max_instances is not None else 1000

        try:
            dataset, oracle_score = get_saved_data(
                filename=data_info["data_path"],
                problem_name=p_name,
                total_episodes=episodes_to_load,
                device=self.device,
                solution_name=data_info.get("solution_path"),
            )
        except Exception as e:
            logger.error(f"Error loading dataset for {p_name}: {e}")
            return None

        total_episodes = dataset["xy"].shape[0] if "xy" in dataset else episodes_to_load
        if max_instances is not None and total_episodes > max_instances:
            total_episodes = max_instances

        # Determine augmentation factor
        if not self.augmentation_enable:
            aug = 1
        elif "a" in p_name:
            aug = self.aug_factor[1]  # 128 for asymmetric
        else:
            aug = self.aug_factor[0]  # 8 for symmetric

        score_meter = AverageMeter()
        aug_score_meter = AverageMeter()

        mask_fn = mask_registry.build_combined_mask(p_name)
        tested_episodes = 0
        start_time = time.time()

        if hasattr(self.policy, "eval"):
            self.policy.eval()

        with torch.inference_mode():
            while tested_episodes < total_episodes:
                cur_batch = min(self.batch_size, total_episodes - tested_episodes)

                # Initialize fresh environment per batch
                env = UniVRPEnv()
                env.load_problems(
                    batch_size=cur_batch,
                    problem_name=p_name,
                    problem_size=scale,
                    pomo_size=scale,
                    validation_data=dataset,
                    aug_factor=aug,
                    start=tested_episodes,
                    device=self.device,
                )

                # Active block set S
                has_relation = getattr(env.reset_state, "relation", None) is not None
                S = get_variant_blocks(p_name, relation_present=has_relation)

                # Policy rollout in greedy mode
                out = self.policy.rollout(
                    env=env,
                    mask_fn=mask_fn,
                    S=S,
                    level=self.level,
                    topk=self.topk,
                    greedy=True,
                    sparse_knn=self.sparse_knn,
                )

                reward = out.reward
                # reward shape: (aug * cur_batch, pomo_size)
                pomo_size = env.pomo_size
                aug_reward = reward.reshape(aug, cur_batch, pomo_size)

                # Maximize over POMO trajectories
                max_pomo_reward, _ = aug_reward.max(dim=2)  # shape: (aug, cur_batch)

                # First slice (index 0) corresponds to unaugmented baseline
                if p_name in ("op", "aop"):
                    no_aug_score = max_pomo_reward[0, :].float().mean().item()
                else:
                    no_aug_score = -max_pomo_reward[0, :].float().mean().item()

                # Best over all augmentations
                max_aug_reward, _ = max_pomo_reward.max(dim=0)  # shape: (cur_batch,)
                if p_name in ("op", "aop"):
                    aug_score = max_aug_reward.float().mean().item()
                else:
                    aug_score = -max_aug_reward.float().mean().item()

                score_meter.update(no_aug_score, cur_batch)
                aug_score_meter.update(aug_score, cur_batch)

                tested_episodes += cur_batch

                # Clean up batch memory
                del env
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        # Calculate optimality gaps relative to oracle
        if p_name in ("op", "aop"):
            no_aug_gap = ((oracle_score - score_meter.avg) * 100.0 / oracle_score)
            aug_gap = ((oracle_score - aug_score_meter.avg) * 100.0 / oracle_score)
        else:
            no_aug_gap = ((score_meter.avg - oracle_score) * 100.0 / oracle_score)
            aug_gap = ((aug_score_meter.avg - oracle_score) * 100.0 / oracle_score)

        final_gap = aug_gap if self.augmentation_enable else no_aug_gap
        group_name = self.get_group_name(p_name)

        result = {
            "variant": p_name,
            "group": group_name,
            "gap": round(final_gap, 4),
            "no_aug_gap": round(no_aug_gap, 4),
            "time": round(elapsed_time, 2),
            "n": scale,
            "model": self.model_name,
            "seed": self.seed,
            "oracle": round(float(oracle_score), 4),
            "score": round(score_meter.avg, 4),
            "aug_score": round(aug_score_meter.avg, 4),
        }

        logger.info(
            f"[{p_name.upper()}-{scale}] Done in {elapsed_time:.1f}s | "
            f"Gap: {final_gap:.2f}% (No-Aug: {no_aug_gap:.2f}%) | "
            f"Score: {aug_score_meter.avg:.3f} (Oracle: {oracle_score:.3f})"
        )
        return result

    def test_problems(
        self,
        problem_list: List[str],
        scale: int = 100,
        max_instances: Optional[int] = None,
        output_csv: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs evaluation over a list of problems and logs to CSV.
        """
        results = []
        fieldnames = ["variant", "group", "gap", "no_aug_gap", "time", "n", "model", "seed"]

        if output_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            write_header = not os.path.exists(output_csv)
            f_csv = open(output_csv, mode="a" if not write_header else "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                f_csv.flush()
        else:
            f_csv = None
            writer = None

        try:
            for p in problem_list:
                res = self.test_one_problem(p, scale=scale, max_instances=max_instances)
                if res is not None:
                    results.append(res)
                    if writer is not None:
                        row = {k: res[k] for k in fieldnames}
                        writer.writerow(row)
                        f_csv.flush()
        finally:
            if f_csv is not None:
                f_csv.close()

        return results
