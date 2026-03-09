"""
rl_sage/src/training/train_loop.py

Main RL-SAGE training loop.

Enhancements over v1:
  - Best Checkpoint Tracking: saves to checkpoints/best/ when eval accuracy peaks
  - Early Stopping: halts training if eval accuracy doesn't improve for `patience` rounds
  - JSONL Metric Logging: detailed local logs in addition to W&B
  - VRAM Watchdog: issues warnings if VRAM exceeds 90%
  - Checkpoint Resume: auto-recovers curriculum, replay buffer, and reward stats
"""

import gc
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict

import torch

logger = logging.getLogger(__name__)


class EarlyStopper:
    """Tracks validation accuracy to stop training if it stops improving."""
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_score = -1.0
        self.counter = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class RLSAGETrainer:
    """Orchestrates the full RL-SAGE self-improving training loop."""

    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        ppo_trainer,
        task_generator,
        solution_generator,
        evaluator,
        reward_model,
        replay_buffer,
        curriculum,
        config: dict,
        optimizer=None,
        wandb_run=None,
    ):
        self.policy_model       = policy_model
        self.ref_model          = ref_model
        self.tokenizer          = tokenizer
        self.ppo_trainer        = ppo_trainer
        self.task_generator     = task_generator
        self.solution_generator = solution_generator
        self.evaluator          = evaluator
        self.reward_model       = reward_model
        self.replay_buffer      = replay_buffer
        self.curriculum         = curriculum
        self.config             = config
        self.optimizer          = optimizer
        self.wandb_run          = wandb_run

        self.train_cfg  = config.get("training", {})
        self.log_cfg    = config.get("logging", {})

        self.total_iterations    = self.train_cfg.get("total_iterations", 5000)
        self.rollout_size        = self.train_cfg.get("rollout_size", 32)
        self.update_batch_size   = self.train_cfg.get("update_batch_size", 16)
        
        self.log_every           = self.log_cfg.get("log_every", 10)
        self.eval_every          = self.log_cfg.get("eval_every", 100)
        self.checkpoint_every    = self.log_cfg.get("checkpoint_every", 500)
        
        self.checkpoint_dir      = Path(self.log_cfg.get("checkpoint_dir", "checkpoints"))
        self.log_dir             = Path("logs")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file        = self.log_dir / "metrics.jsonl"
        
        self.early_stopper       = EarlyStopper(patience=self.train_cfg.get("early_stop_patience", 10))
        self.best_eval_acc       = -1.0
        self._global_step        = 0
        self._warned_no_step_api = False

    def train(self, start_iteration: int = 0):
        """Run the full training loop."""
        logger.info("=" * 60)
        logger.info("RL-SAGE Training Starting")
        logger.info(f"  Total iterations : {self.total_iterations}")
        logger.info(f"  Rollout size     : {self.rollout_size}")
        logger.info(f"  Update batch     : {self.update_batch_size}")
        if start_iteration > 0:
            logger.info(f"  Resuming from    : Iteration {start_iteration}")
            self._load_transient_state(start_iteration)
        logger.info("=" * 60)

        for iteration in range(start_iteration, self.total_iterations):
            iter_start = time.time()

            # ── VRAM Watchdog ─────────────────────────────────────────────────
            self._check_vram(iteration)

            # ── ROLLOUT PHASE ─────────────────────────────────────────────────
            trajectories = self._collect_rollout(iteration)

            # ── UPDATE PHASE ──────────────────────────────────────────────────
            ppo_stats = {}
            if len(self.replay_buffer) >= self.update_batch_size:
                ppo_stats = self._perform_ppo_update(iteration)

            # ── CURRICULUM UPDATE ─────────────────────────────────────────────
            for traj in trajectories:
                self.curriculum.update(traj.eval_correct, traj.topic)

            # ── LOGGING ───────────────────────────────────────────────────────
            if iteration % self.log_every == 0:
                self._log(iteration, trajectories, ppo_stats, iter_start)

            # ── EVALUATION & EARLY STOPPING ───────────────────────────────────
            if iteration % self.eval_every == 0 and iteration > 0:
                mean_acc = self._run_evaluation(iteration)
                
                # Best checkpoint tracking
                if mean_acc > self.best_eval_acc:
                    logger.info(f"🏆 New best eval accuracy: {mean_acc:.2%} (was {self.best_eval_acc:.2%})")
                    self.best_eval_acc = mean_acc
                    self._save_checkpoint(iteration, tag="best")
                
                # Early stopping check
                if self.early_stopper.step(mean_acc):
                    logger.warning(f"🛑 Early stopping triggered at iteration {iteration} (no improvement for {self.early_stopper.patience} evals)")
                    break

            # ── PERIODIC CHECKPOINT ───────────────────────────────────────────
            if iteration % self.checkpoint_every == 0 and iteration > 0:
                self._save_checkpoint(iteration)

            self._global_step += 1

        logger.info("Training complete.")
        self._save_checkpoint(self.total_iterations, tag="final")

    # ── Rollout Phase ─────────────────────────────────────────────────────────

    def _collect_rollout(self, iteration: int) -> list:
        """Collect `rollout_size` trajectories from the current policy."""
        from src.modules.replay_buffer import Trajectory
        from src.models.policy import get_log_probs
        import re

        trajectories = []
        self.policy_model.eval()

        for _ in range(self.rollout_size):
            try:
                # 1. Sample task
                topic, difficulty = self.curriculum.get_next()
                task = self.task_generator.generate(topic, difficulty)

                # 2. Generate solution
                solution = self.solution_generator.generate(task.prompt)

                # 3. Evaluate
                eval_result = self.evaluator.evaluate(task, solution)

                # 4. Reference model log-probs (for KL penalty)
                try:
                    ref_lp = get_log_probs(
                        self.ref_model, self.tokenizer,
                        task.prompt, solution.text,
                        max_length=self.train_cfg.get("max_seq_length", 512),
                    ).cpu()
                except Exception:
                    ref_lp = None

                # 5. Compute reward
                reward = self.reward_model.compute_reward(
                    task, solution, eval_result, ref_lp
                )

                # 6. Store trajectory
                traj = Trajectory(
                    task_id=task.task_id,
                    query=task.prompt,
                    response=solution.text,
                    reward=reward,
                    log_probs=solution.log_probs,
                    ref_log_probs=ref_lp,
                    eval_correct=eval_result.correct,
                    difficulty=difficulty,
                    topic=topic,
                    iteration=iteration,
                    metadata={"ground_truth": task.metadata.get("answer")}
                )
                trajectories.append(traj)

            except Exception as e:
                logger.warning(f"Rollout error (skipping): {e}")
                continue

        # Push batch triggers hindsight relabeling internally
        self.replay_buffer.push_batch(trajectories)
        return trajectories

    # ── PPO Update Phase ──────────────────────────────────────────────────────

    def _perform_ppo_update(self, iteration: int) -> dict:
        """Sample from replay buffer and run one PPO update."""
        self.policy_model.train()

        batch = self.replay_buffer.sample(
            self.update_batch_size, strategy="stratified"
        )

        if hasattr(self.ppo_trainer, "step"):
            queries   = [self.tokenizer.encode(t.query,    return_tensors="pt")[0] for t in batch]
            responses = [self.tokenizer.encode(t.response, return_tensors="pt")[0] for t in batch]
            rewards   = [torch.tensor(t.reward) for t in batch]

            try:
                stats = self.ppo_trainer.step(queries, responses, rewards)
            except Exception as e:
                logger.warning(f"PPO update failed at iteration {iteration}: {e}")
                stats = {}
        else:
            if not self._warned_no_step_api:
                logger.warning(
                    "Installed TRL PPOTrainer has no step() API; using reward-weighted "
                    "policy-gradient fallback updates."
                )
                self._warned_no_step_api = True
            stats = self._manual_policy_update(batch)

        # Free CUDA cache after update
        torch.cuda.empty_cache()
        gc.collect()

        return stats

    def _manual_policy_update(self, batch: list) -> dict:
        """Fallback update for TRL versions without PPOTrainer.step()."""
        if self.optimizer is None:
            return {}

        device = next(self.policy_model.parameters()).device
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

        losses = []
        max_len = self.train_cfg.get("max_seq_length", 512)
        for traj, reward in zip(batch, rewards):
            token_log_probs = self._compute_response_log_probs(traj.query, traj.response, max_len)
            if token_log_probs is None or token_log_probs.numel() == 0:
                continue
            losses.append(-(reward * token_log_probs.mean()))

        if not losses:
            return {}

        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        clip = self.config.get("ppo", {}).get("max_grad_norm", 0.5)
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), clip)

        self.optimizer.step()
        return {
            "fallback/loss": float(loss.detach().cpu().item()),
            "fallback/batch_size": len(losses),
        }

    def _compute_response_log_probs(self, prompt: str, response: str, max_length: int):
        """Compute current-policy token log-probs for response tokens with gradients enabled."""
        device = next(self.policy_model.parameters()).device
        full_text = prompt + response

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )["input_ids"]
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len <= 0:
            return None

        outputs = self.policy_model(**inputs)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        response_ids = inputs["input_ids"][:, prompt_len:]
        if response_ids.numel() == 0:
            return None

        response_log_probs = log_probs[:, prompt_len - 1 : -1, :]
        token_log_probs = response_log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)
        return token_log_probs

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, iteration: int, trajectories: list, ppo_stats: dict, start_time: float):
        """Log training metrics to console, JSONL, and W&B."""
        elapsed = time.time() - start_time
        rewards = [t.reward for t in trajectories]
        success = sum(1 for t in trajectories if t.eval_correct) / max(len(trajectories), 1)

        buf_stats = self.replay_buffer.stats()
        curr_stats = self.curriculum.stats()
        rw_stats = self.reward_model.stats()

        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        metrics = {
            "iteration":         iteration,
            "train/mean_reward": sum(rewards) / max(len(rewards), 1),
            "train/max_reward":  max(rewards) if rewards else 0.0,
            "train/success_rate": success,
            "train/iter_time_s": elapsed,
            "buffer/size":       buf_stats.get("size", 0),
            "buffer/success":    buf_stats.get("success_rate", 0),
            "buffer/hindsight":  buf_stats.get("hindsight_relabels", 0),
            "reward/mean":       rw_stats.get("reward_mean", 0),
            "reward/std":        rw_stats.get("reward_std", 1),
            "curriculum/phase":  curr_stats.get("phase", "unknown"),
            "curriculum/global_sr": curr_stats.get("global_success", 0),
            "system/vram_gb":    float(f"{vram_gb:.2f}"),
            **{f"ppo/{k}": v for k, v in ppo_stats.items() if isinstance(v, (int, float))},
        }

        # Console Log
        logger.info(
            f"[{iteration:4d}] R={metrics['train/mean_reward']:+.2f} | "
            f"SR={success:.1%} | Buff={metrics['buffer/size']} "
            f"(SR={metrics['buffer/success']:.1%}) | "
            f"VRAM={metrics['system/vram_gb']}GB | {elapsed:.1f}s"
        )

        # JSONL Local Log
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # W&B Log
        if self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=iteration)
            except Exception:
                pass
                
    def _check_vram(self, iteration: int):
        """Warn if VRAM is approaching the 6 GB limit."""
        if not torch.cuda.is_available():
            return
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        if alloc_gb > 5.5:
            logger.warning(f"🚨 VRAM CRITICAL at iter {iteration}: {alloc_gb:.2f} GB allocated. OOM imminent.")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _run_evaluation(self, iteration: int) -> float:
        """Run held-out benchmark evaluation. Returns mean accuracy across all benchmarks."""
        from src.evaluation.benchmarks import run_benchmark_evaluation
        logger.info(f"[{iteration}] Running benchmark evaluation...")

        try:
            results = run_benchmark_evaluation(
                self.policy_model,
                self.tokenizer,
                benchmarks=self.config.get("evaluation", {}).get("benchmarks", []),
                max_seq_length=self.train_cfg.get("max_seq_length", 512),
            )
            
            accs = []
            for name, acc in results.items():
                logger.info(f"  {name}: {acc:.2%}")
                accs.append(acc)
                if self.wandb_run:
                    self.wandb_run.log({f"eval/{name}": acc}, step=iteration)
            
            mean_acc = sum(accs) / len(accs) if accs else 0.0
            if self.wandb_run:
                self.wandb_run.log({"eval/mean_accuracy": mean_acc}, step=iteration)
                
            return mean_acc
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, iteration: int, tag: str = ""):
        """Save LoRA adapter weights AND transient module states to disk."""
        base_name = "best_model" if tag == "best" else f"iter_{iteration}"
        if tag and tag != "best":
            base_name += f"_{tag}"
            
        save_path = self.checkpoint_dir / base_name
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. HuggingFace Model & Tokenizer
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # 2. Transient Engine States
            state_dir = save_path / "engine_state"
            state_dir.mkdir(exist_ok=True)
            self.curriculum.save(state_dir / "curriculum.json")
            self.reward_model.save(state_dir / "reward_stats.json")
            self.replay_buffer.save(state_dir / "replay_buffer.json")
            
            # 3. Training Loop State
            with open(state_dir / "trainer.json", "w") as f:
                json.dump({
                    "iteration": iteration,
                    "best_eval_acc": self.best_eval_acc,
                    "early_stop_counter": self.early_stopper.counter
                }, f)
            
            logger.info(f"💾 Checkpoint saved: {save_path}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

    def _load_transient_state(self, iteration: int):
        """Restore transient engine state from a checkpoint directory."""
        state_dir = self.checkpoint_dir / f"iter_{iteration}" / "engine_state"
        if not state_dir.exists():
            logger.warning(f"No engine state found at {state_dir}. Fresh start for curriculum/buffer.")
            return
            
        try:
            self.curriculum.load(state_dir / "curriculum.json")
            self.reward_model.load(state_dir / "reward_stats.json")
            self.replay_buffer.load(state_dir / "replay_buffer.json")
            
            trainer_file = state_dir / "trainer.json"
            if trainer_file.exists():
                with open(trainer_file) as f:
                    data = json.load(f)
                    self.best_eval_acc = data.get("best_eval_acc", -1.0)
                    self.early_stopper.counter = data.get("early_stop_counter", 0)
                    
            logger.info("♻️ Successfully restored engine state from checkpoint.")
        except Exception as e:
            logger.error(f"Failed to load engine state: {e}")
