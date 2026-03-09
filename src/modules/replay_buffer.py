"""
rl_sage/src/modules/replay_buffer.py

Replay Buffer: CPU-resident circular buffer storing collected trajectories.

Enhancements over v1:
  - Stratified sampling (guarantees at least 1 sample per topic per batch)
  - Hindsight Relabeling (gives partial credit to wrong-but-well-reasoned solutions,
    and corrects the final answer to the ground truth to teach valid CoT paths).
  - save() / load() to persist experience across training restarts
"""

import math
import json
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import torch

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """
    One complete rollout trajectory for PPO training.
    Stores all information needed to compute the PPO loss.
    """
    task_id:         str
    query:           str            # The task prompt
    response:        str            # The generated solution text
    reward:          float          # Scalar reward
    log_probs:       Optional[torch.Tensor] = None # Policy log-probs [resp_len]
    ref_log_probs:   Optional[torch.Tensor] = None # Reference model log-probs
    eval_correct:    bool  = False
    difficulty:      float = 0.0
    topic:           str   = "unknown"
    iteration:       int   = 0
    hindsight:       bool  = False  # True if this was synthetically relabelled
    metadata:        dict  = field(default_factory=dict)

    def to_ppo_dict(self, device: str = "cuda") -> dict:
        """Convert to tensors on device for PPO update."""
        return {
            "query":     self.query,
            "response":  self.response,
            "reward":    torch.tensor(self.reward, dtype=torch.float32).to(device),
            "log_probs": self.log_probs.to(device) if self.log_probs is not None else None, # type: ignore
        }

    def to_json_dict(self) -> dict:
        """Serialize for disk persistence (excludes large tensors)."""
        return {
            "task_id":      self.task_id,
            "query":        self.query,
            "response":     self.response,
            "reward":       self.reward,
            "eval_correct": self.eval_correct,
            "difficulty":   self.difficulty,
            "topic":        self.topic,
            "iteration":    self.iteration,
            "hindsight":    self.hindsight,
            # We don't save log_probs to disk (they use too much space and are
            # only strictly valid for the specific policy weights that generated them).
            # When loaded from disk, these will be None, which means they can
            # be used as behavior cloning data (if needed) but not exact PPO updates.
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "Trajectory":
        return cls(
            task_id=d["task_id"],
            query=d["query"],
            response=d["response"],
            reward=d["reward"],
            eval_correct=d.get("eval_correct", False),
            difficulty=d.get("difficulty", 0.0),
            topic=d.get("topic", "unknown"),
            iteration=d.get("iteration", 0),
            hindsight=d.get("hindsight", False),
            log_probs=None,
            ref_log_probs=None,
            metadata={},
        )


class ReplayBuffer:
    """
    CPU-resident circular replay buffer with optional prioritized and stratified sampling.

    Args:
        capacity (int): Maximum number of trajectories to store
        alpha (float): PER exponent (0 = uniform, 1 = full prioritization)
    """

    def __init__(self, capacity: int = 1024, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        self._buffer.append(trajectory)

    def push_batch(self, trajectories: List[Trajectory]):
        """Add multiple trajectories."""
        for t in trajectories:
            self.push(t)

        # Hindsight Relabeling step
        self._hindsight_relabel(trajectories)

    def _hindsight_relabel(self, trajectories: List[Trajectory]):
        """
        Synthetically correct incorrect answers that had good reasoning.

        If a trajectory was wrong, but received a high reward (meaning the reasoning
        and format were excellent but the final calculation failed), we create a
        synthetic copy where the ANSWER: is replaced with the ground truth answer,
        assign it a partial positive reward (+0.2), and push it to the buffer.
        """
        for t in trajectories:
            if not t.eval_correct and t.reward > 0.3:
                # We need the ground truth answer to relabel.
                # It should be passed in metadata by the evaluator.
                gt_ans = t.metadata.get("ground_truth")
                if not gt_ans:
                    continue

                # Find "ANSWER: whatever" and replace it
                import re
                new_resp, count = re.subn(
                    r"ANSWER\s*:\s*.+$",
                    f"ANSWER: {gt_ans}",
                    t.response,
                    flags=re.IGNORECASE | re.MULTILINE
                )

                if count > 0:
                    synth_t = Trajectory(
                        task_id=f"{t.task_id}_hindsight",
                        query=t.query,
                        response=new_resp,
                        reward=0.2,            # Partial credit
                        log_probs=None,        # Invalidated by text change
                        ref_log_probs=None,
                        eval_correct=True,     # Synthetically correct now
                        difficulty=t.difficulty,
                        topic=t.topic,
                        iteration=t.iteration,
                        hindsight=True,
                    )
                    self.push(synth_t)


    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, strategy: str = "stratified") -> List[Trajectory]:
        """
        Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample
            strategy: "stratified" | "prioritized" | "uniform" | "recent"

        Returns:
            List of Trajectory objects
        """
        n = len(self._buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)

        if strategy == "stratified":
            return self._stratified_sample(batch_size)
        elif strategy == "uniform":
            return random.sample(list(self._buffer), batch_size)
        elif strategy == "recent":
            return self._recent_sample(batch_size)
        else:
            return self._prioritized_sample(batch_size)

    def _recent_sample(self, batch_size: int) -> List[Trajectory]:
        n = len(self._buffer)
        recent = list(self._buffer)[-min(batch_size * 4, n):] # type: ignore
        return random.sample(recent, min(batch_size, len(recent)))

    def _prioritized_sample(self, batch_size: int, candidates: Optional[List[Trajectory]] = None) -> List[Trajectory]:
        """Sample proportional to |reward|^alpha."""
        buf = candidates if candidates is not None else list(self._buffer)
        n = len(buf)
        if n == 0:
            return []
        if batch_size >= n:
            return list(buf)

        priorities = torch.tensor(
            [abs(t.reward) + 1e-6 for t in buf], dtype=torch.float32
        )
        probs = (priorities ** self.alpha)
        probs = probs / probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=False)
        return [buf[i] for i in indices.tolist()]

    def _stratified_sample(self, batch_size: int) -> List[Trajectory]:
        """
        Guarantee uniform representation of all topics present in the buffer,
        then fill the remainder using Prioritized Experience Replay (PER).
        """
        buf = list(self._buffer)

        # Group by topic
        by_topic: Dict[str, List[Trajectory]] = {}
        for t in buf:
            by_topic.setdefault(t.topic, []).append(t)

        sampled: List[Trajectory] = []

        # 1. Take 1 from each topic if possible
        available_topics = list(by_topic.keys())
        for topic in available_topics:
            if not by_topic[topic] or len(sampled) >= batch_size:
                continue
            # Pick highest reward from this topic
            best_t = max(by_topic[topic], key=lambda x: x.reward)
            sampled.append(best_t)
            by_topic[topic].remove(best_t)

        # 2. Fill remainder with PER from all remaining
        remaining_needed = batch_size - len(sampled)
        if remaining_needed > 0:
            pool = []
            for t_list in by_topic.values():
                pool.extend(t_list)
            sampled.extend(self._prioritized_sample(remaining_needed, candidates=pool))

        random.shuffle(sampled)
        return sampled


    # ── Persistence & Stats ───────────────────────────────────────────────────

    def save(self, path: str):
        """Save the buffer to disk (JSON format)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [t.to_json_dict() for t in self._buffer]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"ReplayBuffer saved {len(data)} items to {path}")

    def load(self, path: str):
        """Load the buffer from disk."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.clear()
            for d in data:
                self.push(Trajectory.from_json_dict(d))
            logger.info(f"ReplayBuffer loaded {len(self)} items from {path}")
        except FileNotFoundError:
            logger.debug(f"No buffer file at {path}, starting fresh.")

    def get_recent(self, n: int) -> List[Trajectory]:
        """Return the n most recently added trajectories."""
        buf = list(self._buffer)
        return buf[-min(n, len(buf)):] # type: ignore

    def history_texts(self) -> List[str]:
        """Return all stored solution texts (for diversity scoring)."""
        return [t.response for t in self._buffer]

    def success_rate(self, window: int = 50) -> float:
        """Compute fraction of correct solutions in the last `window` trajectories."""
        recent = self.get_recent(window)
        if not recent:
            return 0.0
        return sum(1 for t in recent if t.eval_correct) / len(recent)

    def mean_reward(self, window: int = 50) -> float:
        """Mean reward of the last `window` trajectories."""
        recent = self.get_recent(window)
        if not recent:
            return 0.0
        return sum(t.reward for t in recent) / len(recent)

    def stats(self) -> dict:
        """Return summary statistics of the buffer."""
        if not self._buffer:
            return {"size": 0}

        rewards = [t.reward for t in self._buffer]
        hindsight_count = sum(1 for t in self._buffer if t.hindsight)

        return {
            "size":         len(self._buffer),
            "capacity":     self.capacity,
            "mean_reward":  float(f"{sum(rewards) / len(rewards):.3f}"),
            "max_reward":   float(f"{max(rewards):.3f}"),
            "min_reward":   float(f"{min(rewards):.3f}"),
            "success_rate": float(f"{self.success_rate():.3f}"),
            "hindsight_relabels": hindsight_count,
        }

    def clear(self):
        """Empty the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={len(self)}/{self.capacity}, "
            f"mean_reward={self.mean_reward():.3f})"
        )
