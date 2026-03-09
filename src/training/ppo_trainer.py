"""
rl_sage/src/training/ppo_trainer.py

PPO Trainer wrapper: configures and exposes the TRL PPOTrainer
for use in the RL-SAGE training loop.
"""

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig

logger = logging.getLogger(__name__)


def build_ppo_config(config: dict) -> PPOConfig:
    """
    Build a TRL PPOConfig from the training configuration dict.

    Args:
        config: Full training config dict (training_config.yaml parsed)

    Returns:
        PPOConfig
    """
    ppo_cfg  = config.get("ppo", {})
    train_cfg = config.get("training", {})

    return PPOConfig(
        # PPO hyperparameters
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        batch_size=train_cfg.get("update_batch_size", 16),
        mini_batch_size=max(1, train_cfg.get("update_batch_size", 16) // 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        num_ppo_epochs=train_cfg.get("ppo_epochs", 4),
        lam=ppo_cfg.get("lam", 0.95),

        seed=42,
    )


def build_ppo_trainer(
    policy_model,
    ref_model,
    tokenizer: AutoTokenizer,
    config: dict,
    optimizer=None,
) -> PPOTrainer:
    """
    Instantiate the TRL PPOTrainer with the policy and reference models.

    Args:
        policy_model: PEFT model with LoRA (AutoModelForCausalLMWithValueHead or wrapped)
        ref_model: Frozen reference model
        tokenizer: Tokenizer
        config: Full training config dict
        optimizer: Optional pre-built optimizer (paged AdamW 8-bit)

    Returns:
        PPOTrainer instance
    """
    ppo_config = build_ppo_config(config)

    # TRL >= 0.8 requires a train_dataset even if we feed data manually
    class DummyDataset:
        def __len__(self): return 1
        def __getitem__(self, idx): return {"query": "dummy"}

    trainer = PPOTrainer(
        args=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        reward_model=ref_model,  # Mock requirement, unused since we calculate rewards manually
        value_model=policy_model, # Mock requirement
        processing_class=tokenizer,
        optimizers=(optimizer, None) if optimizer else None,
        train_dataset=DummyDataset(),
        data_collator=lambda x: x,   # We pass tensors directly
    )

    logger.info("PPOTrainer initialized.")
    ppo_epochs = getattr(ppo_config, "num_ppo_epochs", getattr(ppo_config, "ppo_epochs", "n/a"))
    kl_coeff = getattr(ppo_config, "init_kl_coef", "n/a")
    clip_eps = getattr(ppo_config, "cliprange", "n/a")
    logger.info(f"  batch_size={ppo_config.batch_size}, ppo_epochs={ppo_epochs}")
    logger.info(f"  KL coeff={kl_coeff}, clip eps={clip_eps}")

    return trainer


def build_optimizer(model, config: dict):
    """
    Build a paged AdamW 8-bit optimizer (bitsandbytes) for memory efficiency.

    Falls back to standard AdamW if bitsandbytes is not available.
    """
    lr = config.get("training", {}).get("learning_rate", 1e-5)
    wd = config.get("training", {}).get("weight_decay", 0.01)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        logger.info("Using paged AdamW 8-bit optimizer (bitsandbytes).")
    except ImportError:
        from torch.optim import AdamW
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        logger.warning("bitsandbytes not found — using standard AdamW (higher VRAM usage).")

    return optimizer
