"""
rl_sage/src/modules/reward_model.py

Reward Model: computes the composite reward signal used to train the policy.

R(t, s) = α₁·Rc + α₂·Rr + α₃·Rd + α₄·Rf + R_KL

Where:
    Rc  = correctness reward      (binary: +1 correct, −penalty incorrect)
    Rr  = reasoning quality       (DeBERTa NLI scorer)
    Rd  = diversity reward        (anti-repetition, Jaccard-based)
    Rf  = format reward           (bonus for well-structured CoT)
    R_KL = KL divergence penalty  (vs reference model)

Enhancements over v1:
  - Online Welford mean/variance for reward normalisation (replaces batch z-score)
  - Difficulty-scaled wrong penalty
  - Format reward: bonus for <reasoning> tags + ANSWER line
  - Diversity history window increased to 256
  - save()/load() for reward statistics across checkpoints
"""

import math
import json
import logging
from collections import deque
from pathlib import Path
from typing import List, Optional

import torch

from src.modules.task_generator import Task
from src.modules.solution_generator import Solution
from src.modules.evaluator import EvalResult

logger = logging.getLogger(__name__)


# ── Welford Online Statistics ─────────────────────────────────────────────────

class WelfordStats:
    """
    Online mean / variance using Welford's algorithm.
    Numerically stable, O(1) per update.
    """
    def __init__(self):
        self.n:    int   = 0
        self.mean: float = 0.0
        self.M2:   float = 0.0

    def update(self, x: float):
        self.n += 1
        delta  = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def normalize(self, x: float) -> float:
        """Z-score normalize a value."""
        if self.n < 2:
            return x
        return (x - self.mean) / (self.std + 1e-8)

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "M2": self.M2}

    @classmethod
    def from_dict(cls, d: dict) -> "WelfordStats":
        s = cls()
        s.n    = d.get("n",    0)
        s.mean = d.get("mean", 0.0)
        s.M2   = d.get("M2",   0.0)
        return s


# ── Reward Model ──────────────────────────────────────────────────────────────

class RewardModel:
    """
    Computes the PPO reward signal from evaluation results.

    Args:
        config: Reward section of training_config.yaml
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # ── Reward weights ────────────────────────────────────────────────────
        self.alpha_c        = self.config.get("alpha_correctness",  0.65)
        self.alpha_r        = self.config.get("alpha_reasoning",    0.15)
        self.alpha_d        = self.config.get("alpha_diversity",    0.05)
        self.alpha_f        = self.config.get("alpha_format",       0.05)
        self.diff_bonus     = self.config.get("difficulty_bonus",   0.30)
        self.max_clip       = self.config.get("max_reward_clip",    1.5)

        # Wrong penalty scales with difficulty: easy mistakes are softer
        self._base_wrong_penalty = self.config.get("wrong_answer_penalty", -0.5)

        # ── Normalisation ─────────────────────────────────────────────────────
        self.normalize  = self.config.get("normalize_rewards", True)
        self._stats     = WelfordStats()

        # ── Diversity history (256 solutions) ─────────────────────────────────
        self._solution_history: deque = deque(maxlen=256)

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        task: Task,
        solution: Solution,
        eval_result: EvalResult,
        ref_log_probs: Optional[torch.Tensor] = None,
        kl_coeff: float = 0.10,
    ) -> float:
        """
        Compute the total scalar reward for one trajectory.

        Args:
            task:         The generated task
            solution:     The policy's generated solution
            eval_result:  Output of Evaluator.evaluate()
            ref_log_probs: Per-token log-probs from frozen reference model
            kl_coeff:     KL penalty coefficient (β)

        Returns:
            Scalar reward (float)
        """
        rc  = self._correctness_reward(eval_result, task.difficulty)
        rr  = self._reasoning_reward(eval_result.reasoning_quality)
        rd  = self._diversity_reward(solution.text)
        rf  = self._format_reward(solution.text)
        rkl = self._kl_penalty(solution.log_probs, ref_log_probs, kl_coeff)

        raw = (
            self.alpha_c * rc
            + self.alpha_r * rr
            + self.alpha_d * rd
            + self.alpha_f * rf
            + rkl
        )

        # Difficulty bonus: solving harder tasks earns more
        raw *= 1.0 + self.diff_bonus * task.difficulty

        # Online normalisation
        if self.normalize:
            self._stats.update(raw)
            raw = self._stats.normalize(raw)

        # Hard clip
        total = float(max(-self.max_clip, min(self.max_clip, raw)))

        # Update diversity window
        self._solution_history.append(solution.text)

        return total

    def compute_batch_rewards(
        self,
        tasks:            List[Task],
        solutions:        List[Solution],
        eval_results:     List[EvalResult],
        ref_log_probs_list: Optional[List[Optional[torch.Tensor]]] = None,
        kl_coeff:         float = 0.10,
    ) -> List[float]:
        """Compute rewards for a batch."""
        if ref_log_probs_list is None:
            ref_log_probs_list = [None] * len(tasks)

        return [
            self.compute_reward(t, s, e, rl, kl_coeff)
            for t, s, e, rl in zip(tasks, solutions, eval_results, ref_log_probs_list)
        ]

    # ── Sub-reward Components ─────────────────────────────────────────────────

    def _correctness_reward(self, eval_result: EvalResult, difficulty: float) -> float:
        """
        Correctness reward with difficulty-scaled wrong penalty.

            +1.0   → correct
            penalty → incorrect (penalty scales: easy=-0.3, hard=-0.9)
             0.0   → format error (answer unparsed / timeout)
        """
        # Neutral on format errors
        if not eval_result.correct:
            details = eval_result.details or {}
            if details.get("reason") in ("answer_unparsed", "no_choice_found", "timeout"):
                return 0.0
            # Scale penalty: easier tasks → softer penalty; harder tasks → harsher
            penalty = self._base_wrong_penalty * (0.5 + difficulty)
            return float(max(-1.0, penalty))

        return 1.0

    def _reasoning_reward(self, quality_score: float) -> float:
        """Map DeBERTa quality score [0, 1] → reward in [-1, +1]."""
        return 2.0 * quality_score - 1.0

    def _diversity_reward(self, solution_text: str) -> float:
        """
        Reward solutions that differ from recent history.
        Higher Jaccard similarity to recent solutions → lower reward.
        """
        if not self._solution_history:
            return 0.5   # Neutral on first solution

        tokens = set(solution_text.lower().split())
        sims: List[float] = []
        for prev in self._solution_history:
            prev_tokens = set(prev.lower().split())
            union = len(tokens | prev_tokens)
            inter = len(tokens & prev_tokens)
            sims.append(float(inter) / float(union) if union > 0 else 0.0)

        mean_sim = sum(sims) / len(sims)
        return 1.0 - mean_sim    # high similarity → low diversity reward

    def _format_reward(self, solution_text: str) -> float:
        """
        Award a small reward for well-formatted chain-of-thought solutions.

        Checks for:
          +0.5 if solution contains <reasoning>...</reasoning> tags
          +0.3 if ANSWER: line is present
          +0.2 if solution has ≥ 3 distinct reasoning steps (numbered or bullet)
        """
        score: float = 0.0

        if "<reasoning>" in solution_text and "</reasoning>" in solution_text:
            score += 0.5
        if re.search(r"ANSWER\s*:", solution_text, re.IGNORECASE):
            score += 0.3

        # Count reasoning steps: lines starting with numbers or bullets
        steps = re.findall(
            r"^\s*(?:\d+[\.\):]|[-•*])\s+\S",
            solution_text, re.MULTILINE
        )
        if len(steps) >= 3:
            score += 0.2

        # Normalize to [0, 1]
        return min(1.0, score)

    def _kl_penalty(
        self,
        policy_log_probs: Optional[torch.Tensor],
        ref_log_probs:    Optional[torch.Tensor],
        beta: float,
    ) -> float:
        """
        Compute KL penalty: −β · KL(π_θ || π_ref)

        Uses the closed-form KL divergence for log-prob tensors.
        Returns 0.0 if reference log-probs are unavailable.
        """
        if policy_log_probs is None or ref_log_probs is None:
            return 0.0
        try:
            min_len = min(len(policy_log_probs), len(ref_log_probs)) # type: ignore
            p = policy_log_probs[:min_len].float() # type: ignore
            q = ref_log_probs[:min_len].float() # type: ignore
            # KL(p||q) = Σ exp(p) * (p - q)
            kl = (p.exp() * (p - q)).sum().item()
            kl = max(0.0, kl)
            return -beta * kl
        except Exception:
            return 0.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save Welford statistics for consistent normalization on resume."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "welford":  self._stats.to_dict(),
            "n_solutions_seen": len(self._solution_history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"RewardModel stats saved: {path}")

    def load(self, path: str):
        """Load saved Welford statistics."""
        try:
            with open(path) as f:
                data = json.load(f)
            self._stats = WelfordStats.from_dict(data.get("welford", {}))
            logger.info(
                f"RewardModel stats loaded: n={self._stats.n}, "
                f"mean={self._stats.mean:.3f}, std={self._stats.std:.3f}"
            )
        except FileNotFoundError:
            logger.debug(f"No saved reward stats at {path}, starting fresh.")

    def reset_history(self):
        """Clear the solution diversity history (not the Welford stats)."""
        self._solution_history.clear()

    def stats(self) -> dict:
        """Return reward model statistics for logging."""
        return {
            "n_rewards":    self._stats.n,
            "reward_mean":  float(f"{self._stats.mean:.4f}"),
            "reward_std":   float(f"{self._stats.std:.4f}"),
            "history_size": len(self._solution_history),
        }


# ── Import needed for _format_reward ────────────────────────────────────────
import re  # noqa: E402 (at bottom to avoid circular grouping)
