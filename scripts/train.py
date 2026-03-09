"""
rl_sage/scripts/train.py

Entry point for RL-SAGE training.

Usage:
    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --config config/training_config.yaml --resume checkpoints/iter_500
    python scripts/train.py --config config/training_config.yaml --debug   # DistilGPT-2, 50 iters
"""

import os
import sys
import gc
import argparse
import logging
from pathlib import Path

import torch
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.policy import load_policy_model, load_reference_model
from src.models.reasoning_scorer import ReasoningScorer
from src.modules.task_generator import TaskGenerator
from src.modules.solution_generator import SolutionGenerator
from src.modules.evaluator import Evaluator
from src.modules.reward_model import RewardModel
from src.modules.replay_buffer import ReplayBuffer
from src.modules.curriculum import CurriculumScheduler
from src.training.ppo_trainer import build_ppo_trainer, build_optimizer
from src.training.train_loop import RLSAGETrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_sage.train")


def parse_args():
    parser = argparse.ArgumentParser(description="RL-SAGE Training")
    parser.add_argument("--config",  type=str, default="config/training_config.yaml")
    parser.add_argument("--resume",  type=str, default=None, help="Checkpoint dir to resume from")
    parser.add_argument("--debug",   action="store_true",    help="Debug mode: tiny model, 50 iters")
    parser.add_argument("--no-wandb", action="store_true",   help="Disable Weights & Biases logging")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_debug_overrides(config: dict) -> dict:
    """Override config for rapid debugging runs."""
    logger.warning("DEBUG MODE — using DistilGPT-2, 50 iterations")
    config["model"]["base_model"] = "distilgpt2"
    config["model"]["quantization"]["enabled"] = False
    config["model"]["lora"]["enabled"] = False
    config["model"]["gradient_checkpointing"] = False
    config["training"]["total_iterations"] = 50
    config["training"]["rollout_size"] = 4
    config["training"]["update_batch_size"] = 2
    config["training"]["max_seq_length"] = 128
    config["training"]["max_new_tokens"] = 64
    config["logging"]["log_every"] = 5
    config["logging"]["eval_every"] = 25
    config["logging"]["checkpoint_every"] = 50
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.debug:
        config = apply_debug_overrides(config)
    if args.no_wandb:
        config["logging"]["use_wandb"] = False

    # ── GPU Diagnostics ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}  ({total_vram:.1f} GB VRAM)")
    else:
        logger.warning("No CUDA GPU detected — training on CPU will be very slow!")

    # ── W&B Setup ─────────────────────────────────────────────────────────────
    wandb_run = None
    if config["logging"].get("use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config["logging"].get("wandb_project", "rl-sage"),
                config=config,
                tags=config["logging"].get("wandb_tags", []),
                resume="allow" if args.resume else None,
            )
            logger.info(f"W&B run: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")

    # ── Load Models ───────────────────────────────────────────────────────────
    logger.info("Loading policy model...")
    policy_model, tokenizer = load_policy_model(config["model"])

    # ── Resume Logic (load adapter before building trainer components) ──────
    start_iteration = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        from peft import PeftModel
        try:
            policy_model = PeftModel.from_pretrained(policy_model, args.resume, is_trainable=True)
        except TypeError:
            policy_model = PeftModel.from_pretrained(policy_model, args.resume)

        # Extract iteration number from directory name if possible.
        try:
            dir_name = Path(args.resume).name
            start_iteration = int(dir_name.replace("iter_", "").split("_")[0]) + 1
            logger.info(f"Resuming from iteration {start_iteration}")
        except Exception:
            pass

    logger.info("Loading reference model...")
    try:
        from peft import PeftModel
        share_ref_weights = isinstance(policy_model, PeftModel)
    except Exception:
        share_ref_weights = False
    ref_model = load_reference_model(
        config["model"],
        policy_model=policy_model if share_ref_weights else None,
        share_weights=share_ref_weights,
    )

    # ── Reasoning Scorer (CPU) ────────────────────────────────────────────────
    scorer_cfg = config.get("reasoning_scorer", {})
    reasoning_scorer = None
    if not args.debug:
        try:
            reasoning_scorer = ReasoningScorer(
                model_name=scorer_cfg.get("model", "cross-encoder/nli-deberta-v3-small"),
                device=scorer_cfg.get("device", "cpu"),
                batch_size=scorer_cfg.get("batch_size", 8),
            )
        except Exception as e:
            logger.warning(f"Could not load reasoning scorer: {e}")

    # ── Seed Tasks from GSM8K ─────────────────────────────────────────────────
    logger.info("Loading GSM8K seed tasks...")
    from datasets import load_dataset
    gsm8k = load_dataset(
        config["datasets"]["gsm8k"]["path"],
        config["datasets"]["gsm8k"]["split"],
        split="train",
    )
    task_generator = TaskGenerator.from_dataset(
        policy_model, tokenizer, gsm8k, config.get("generation", {})
    )

    # ── Build Modules ─────────────────────────────────────────────────────────
    solution_generator = SolutionGenerator(policy_model, tokenizer, config.get("generation", {}))
    evaluator          = Evaluator(reasoning_scorer=reasoning_scorer)
    reward_model       = RewardModel(config.get("reward", {}))
    replay_buffer      = ReplayBuffer(
        capacity=config["training"].get("replay_buffer_capacity", 512),
        alpha=config["training"].get("replay_buffer_alpha", 0.6),
    )
    curriculum = CurriculumScheduler(config.get("curriculum", {}))

    # ── Build PPO Trainer ─────────────────────────────────────────────────────
    optimizer   = build_optimizer(policy_model, config)
    ppo_trainer = build_ppo_trainer(policy_model, ref_model, tokenizer, config, optimizer)

    # ── Main Trainer ──────────────────────────────────────────────────────────
    trainer = RLSAGETrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        ppo_trainer=ppo_trainer,
        task_generator=task_generator,
        solution_generator=solution_generator,
        evaluator=evaluator,
        reward_model=reward_model,
        replay_buffer=replay_buffer,
        curriculum=curriculum,
        config=config,
        optimizer=optimizer,
        wandb_run=wandb_run,
    )

    # ── Run Training ──────────────────────────────────────────────────────────
    logger.info("Starting training loop...")
    trainer.train(start_iteration=start_iteration)

    if wandb_run:
        wandb_run.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
