"""
rl_sage/src/modules/curriculum.py

Curriculum Scheduler: adaptively controls task difficulty and topic distribution.

Enhancements over v1:
  - Cosine Adaptive Warmup: mathematically smooth difficulty ramp
  - Per-topic difficulty: tracks independent difficulty levels per topic
  - Mastery Gates: requires 80% success over 100 trials to unlock next difficulty band
  - Save/Load logic and Export Stats JSON for monitoring
"""

import math
import json
import random
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class DifficultyRecord:
    """Tracks recent success/failure at a given difficulty band."""
    n_correct: int = 0
    n_total:   int = 0

    def success_rate(self) -> float:
        return self.n_correct / self.n_total if self.n_total > 0 else 0.5

    def update(self, correct: bool):
        self.n_total += 1
        if correct:
            self.n_correct += 1

    def to_dict(self) -> dict:
        return {"n_correct": self.n_correct, "n_total": self.n_total}

    @classmethod
    def from_dict(cls, d: dict) -> "DifficultyRecord":
        return cls(n_correct=d.get("n_correct", 0), n_total=d.get("n_total", 0))


class CurriculumScheduler:
    """
    Adaptive difficulty curriculum for RL-SAGE.

    Algorithm:
        - Optional cosine warmup schedule for the first N iterations.
        - After warmup, maintains independent difficulty levels per topic.
        - Adjusts topic difficulty based on sliding window success rate.
        - Mastery Gate: caps difficulty at 0.5 until a topic achieves 80% success rate.
    """

    DIFFICULTY_BANDS = {
        "very_easy":   (0.00, 0.25),
        "easy":        (0.25, 0.50),
        "medium":      (0.50, 0.70),
        "hard":        (0.70, 0.85),
        "very_hard":   (0.85, 1.00),
    }

    DEFAULT_TOPICS = [
        "arithmetic",
        "algebra",
        "word_problems",
        "geometry",
        "commonsense",
        "logic",
        "science",
    ]

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # ── Core Hyperparameters ──────────────────────────────────────────────
        self.initial_diff = self.config.get("initial_difficulty", 0.0)
        self.step         = self.config.get("difficulty_step",    0.05)
        self.thresh_up    = self.config.get("success_threshold_up",   0.75)
        self.thresh_dn    = self.config.get("success_threshold_down", 0.40)
        self.window       = self.config.get("window_size", 50)
        self.topics       = self.config.get("topics", self.DEFAULT_TOPICS)

        # ── Advanced Features ─────────────────────────────────────────────────
        self.warmup_iters = self.config.get("warmup_iterations", 500)
        self.mastery_gate = self.config.get("mastery_gate_threshold", 0.80)

        # ── State Tracking ────────────────────────────────────────────────────
        self._iteration = 0
        self._global_recent: deque = deque(maxlen=self.window * 2)

        # Independent difficulty level per topic
        self._topic_diff: Dict[str, float] = {t: self.initial_diff for t in self.topics}

        # Success trackers
        self._topic_records: Dict[str, DifficultyRecord] = {t: DifficultyRecord() for t in self.topics}
        self._band_records:  Dict[str, DifficultyRecord] = {b: DifficultyRecord() for b in self.DIFFICULTY_BANDS}

        # Sliding success windows for Mastery Gates per topic
        self._topic_windows: Dict[str, deque] = {t: deque(maxlen=100) for t in self.topics}

        logger.info(f"CurriculumScheduler initialized | topics={len(self.topics)} | warmup={self.warmup_iters}")


    # ── Public API ────────────────────────────────────────────────────────────

    def get_next(self) -> Tuple[str, float]:
        """
        Return (topic, difficulty) for the next task to generate.
        """
        topic = self._sample_topic()
        base_diff = self._get_target_difficulty(topic)
        final_diff = self._jitter_difficulty(base_diff)
        return topic, final_diff

    def update(self, eval_correct: bool, topic: str = "unknown"):
        """Record evaluation outcome."""
        self._global_recent.append(eval_correct)
        self._iteration += 1

        if topic in self.topics:
            self._topic_records[topic].update(eval_correct)
            self._topic_windows[topic].append(eval_correct)

            # Adapt topic difficulty every `window` steps
            if self._iteration % self.window == 0 and self._iteration > self.warmup_iters:
                self._adapt_topic_difficulty(topic)

        # Update global band record
        if topic in self._topic_diff:
            band = self._get_band(self._topic_diff[topic])
            self._band_records[band].update(eval_correct)


    # ── Difficulty Logic ──────────────────────────────────────────────────────

    def _get_target_difficulty(self, topic: str) -> float:
        """
        Compute the current difficulty rule (Warmup vs Adaptive).
        """
        # 1. Cosine Warmup Phase
        if self._iteration < self.warmup_iters:
            # Ramps from initial_diff up to 0.50 smoothly using a half-cosine curve
            progress = self._iteration / max(1, self.warmup_iters)
            ramp = 0.5 * (1.0 - math.cos(math.pi * progress))
            target = self.initial_diff + ramp * (0.50 - self.initial_diff)
            # Sync all topics to the warmup drift
            for t in self.topics:
                self._topic_diff[t] = float(target)
            return float(target)

        # 2. Adaptive Phase (Post-warmup)
        diff = self._topic_diff.get(topic, 0.5)

        # Apply Mastery Gate: block >0.5 difficulty until 80% rolling success is achieved
        if diff >= 0.50:
            window = self._topic_windows[topic]
            if len(window) >= 50:
                mastery_rate = sum(window) / len(window)
                if mastery_rate < self.mastery_gate:
                    # Gated! Topic is struggling, cap it at 0.50 until it improves
                    diff = 0.50

        return diff

    def _adapt_topic_difficulty(self, topic: str):
        """Increase or decrease a specific topic's difficulty based on rolling success."""
        window = self._topic_windows[topic]
        if not window:
            return

        sr = sum(window) / len(window)
        old = self._topic_diff[topic]

        if sr > self.thresh_up:
            self._topic_diff[topic] = min(1.0, old + self.step)
        elif sr < self.thresh_dn:
            self._topic_diff[topic] = max(0.0, old - self.step)

        if self._topic_diff[topic] != old:
            direction = "↑" if self._topic_diff[topic] > old else "↓"
            logger.debug(
                f"Curriculum {direction} [{topic}] diff: {old:.2f} → {self._topic_diff[topic]:.2f} "
                f"(SR={sr:.0%})"
            )

    def _jitter_difficulty(self, base_diff: float) -> float:
        """Add small Gaussian noise to prevent exactly identical prompts."""
        jitter = random.gauss(0, self.step * 0.4)
        return float(max(0.0, min(1.0, base_diff + jitter)))


    # ── Topic Selection ───────────────────────────────────────────────────────

    def _sample_topic(self) -> str:
        """
        Sample a topic.
        Uses Inverse Success Rate weighting: topics the model struggles with
        are sampled more frequently.
        """
        if not self.topics:
            return "arithmetic"

        weights = []
        for t in self.topics:
            sr = self._topic_records[t].success_rate()
            # weight = max(0.1, 1 - success_rate)
            weights.append(max(0.1, 1.0 - sr))

        total = sum(weights)
        probs = [w / total for w in weights]

        r = random.random()
        cumulative = 0.0
        for topic, p in zip(self.topics, probs):
            cumulative += p
            if r < cumulative:
                return topic
        return self.topics[-1]


    # ── Utilities & Persistence ───────────────────────────────────────────────

    def _get_band(self, difficulty: float) -> str:
        for band, (lo, hi) in self.DIFFICULTY_BANDS.items():
            if lo <= difficulty < hi:
                return band
        return "very_hard"

    def recent_success_rate(self) -> float:
        """Global success rate across all topics."""
        if not self._global_recent:
            return 0.5
        return sum(self._global_recent) / len(self._global_recent)

    def stats(self) -> dict:
        """Return curriculum statistics for logging."""
        phase = "warmup" if self._iteration < self.warmup_iters else "adaptive"

        topic_stats = {}
        for t in self.topics:
            window = self._topic_windows[t]
            sr = sum(window) / len(window) if window else 0.5
            topic_stats[t] = {
                "difficulty": round(float(self._topic_diff[t]), 3),
                "success_rate": round(float(sr), 3),
            }

        return {
            "phase":          phase,
            "iteration":      self._iteration,
            "global_success": round(float(self.recent_success_rate()), 3),
            "topics":         topic_stats,
        }

    def save(self, path: str):
        """Save curriculum state to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration":  self._iteration,
            "topic_diff": self._topic_diff,
            "records":    {t: r.to_dict() for t, r in self._topic_records.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"CurriculumScheduler state saved: {path}")

    def load(self, path: str):
        """Load curriculum state from JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._iteration = data.get("iteration", 0)
            self._topic_diff.update(data.get("topic_diff", {}))

            recs = data.get("records", {})
            for t in self.topics:
                if t in recs:
                    self._topic_records[t] = DifficultyRecord.from_dict(recs[t])

            logger.info(f"CurriculumScheduler state loaded (iter={self._iteration})")
        except FileNotFoundError:
            logger.debug(f"No curriculum state at {path}, starting fresh.")

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler(iter={self._iteration}, "
            f"phase={'warmup' if self._iteration < self.warmup_iters else 'adaptive'}, "
            f"global_sr={self.recent_success_rate():.2%})"
        )
