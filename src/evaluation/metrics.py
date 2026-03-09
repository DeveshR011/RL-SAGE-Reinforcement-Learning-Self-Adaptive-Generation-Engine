"""
rl_sage/src/evaluation/metrics.py

Custom evaluation metrics for RL-SAGE:
    - Accuracy
    - Self-Improvement Rate (SIR)
    - Diversity Score
    - Generalization Gap
    - Reasoning Quality Score
"""

import math
import logging
import re
import string
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Accuracy ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact-match accuracy.

    Args:
        predictions: List of predicted answer strings
        references: List of reference answer strings

    Returns:
        Accuracy in [0, 1]
    """
    if not predictions:
        return 0.0
    n_correct: int = sum(
        1 for p, r in zip(predictions, references)
        if _normalize(p) == _normalize(r)
    )
    return n_correct / len(predictions)


# ── Self-Improvement Rate ─────────────────────────────────────────────────────

def compute_self_improvement_rate(
    baseline_accuracy: float,
    current_accuracy: float,
    n_iterations: int,
) -> float:
    """
    Compute Self-Improvement Rate (SIR).

    SIR = (accuracy_current - accuracy_baseline) / n_iterations * 1000
    Represents accuracy gain per 1000 training iterations.

    Args:
        baseline_accuracy: Accuracy before RL training
        current_accuracy:  Accuracy at current iteration
        n_iterations:      Number of training iterations elapsed

    Returns:
        SIR as a float (e.g., 1.5 means +1.5% per 1000 iterations)
    """
    if n_iterations <= 0:
        return 0.0
    delta: float = current_accuracy - baseline_accuracy
    return (delta / n_iterations) * 1000.0


# ── Diversity Score ────────────────────────────────────────────────────────────

def _compute_ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
    """Return the set of character n-gram tuples for `text`."""
    tokens: List[str] = text.lower().split()
    result: Set[Tuple[str, ...]] = set()
    upper: int = len(tokens) - n + 1
    for i in range(upper):
        gram: Tuple[str, ...] = tuple(tokens[i : i + n])
        result.add(gram)
    return result


def _jaccard_sim(
    a: Set[Tuple[str, ...]],
    b: Set[Tuple[str, ...]],
) -> float:
    """Jaccard similarity between two frozensets."""
    union_size: int = len(a | b)
    inter_size: int = len(a & b)
    if union_size == 0:
        return 0.0
    return float(inter_size) / float(union_size)


def compute_diversity_score(solutions: List[str], ngram_n: int = 3) -> float:
    """
    Compute solution diversity as 1 - mean pairwise Jaccard similarity of n-grams.

    Args:
        solutions: List of solution text strings
        ngram_n:   N-gram size for comparison

    Returns:
        Diversity in [0, 1]. Higher = more diverse.
    """
    if len(solutions) < 2:
        return 1.0

    gram_sets: List[Set[Tuple[str, ...]]] = [
        _compute_ngrams(s, ngram_n) for s in solutions
    ]

    similarities: List[float] = []
    n_sets: int = len(gram_sets)
    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            sim: float = _jaccard_sim(gram_sets[i], gram_sets[j])
            similarities.append(sim)

    if not similarities:
        return 1.0
    mean_sim: float = sum(similarities) / len(similarities)
    return 1.0 - mean_sim


# ── Generalization Gap ────────────────────────────────────────────────────────

def compute_generalization_gap(
    in_distribution_acc: float,
    out_distribution_acc: float,
) -> float:
    """
    Generalization Gap = train-distribution accuracy - holdout accuracy.
    Smaller is better (model generalises to unseen distributions).
    """
    return in_distribution_acc - out_distribution_acc


# ── Reasoning Quality Score ───────────────────────────────────────────────────

def compute_mean_reasoning_quality(quality_scores: List[float]) -> float:
    """
    Compute mean reasoning quality across a set of solutions.

    Args:
        quality_scores: List of DeBERTa reasoning quality scores in [0, 1]

    Returns:
        Mean reasoning quality
    """
    if not quality_scores:
        return 0.0
    return sum(quality_scores) / len(quality_scores)


# ── Reward Statistics ─────────────────────────────────────────────────────────

def _percentile(data: List[float], p: float) -> float:
    """
    Return the p-th percentile of a pre-sorted float list.

    Args:
        data: Pre-sorted list of floats
        p:    Percentile [0, 100]
    """
    if not data:
        return 0.0
    idx: int = int(p / 100.0 * len(data))
    idx = min(idx, len(data) - 1)
    return float(data[idx])


def compute_reward_statistics(rewards: List[float]) -> Dict[str, float]:
    """
    Compute descriptive statistics for a list of rewards.

    Returns:
        Dict with keys: mean, std, min, max, p25, p50, p75
    """
    if not rewards:
        return {}

    n: int = len(rewards)
    sorted_r: List[float] = sorted(rewards)
    mean: float = sum(rewards) / n
    variance: float = sum((r - mean) ** 2 for r in rewards) / n

    return {
        "mean": mean,
        "std":  math.sqrt(variance),
        "min":  min(rewards),
        "max":  max(rewards),
        "p25":  _percentile(sorted_r, 25.0),
        "p50":  _percentile(sorted_r, 50.0),
        "p75":  _percentile(sorted_r, 75.0),
    }


# ── Topic-level Accuracy ──────────────────────────────────────────────────────

def compute_per_topic_accuracy(
    correct_flags: List[bool],
    topics: List[str],
) -> Dict[str, float]:
    """
    Compute accuracy broken down by topic.

    Args:
        correct_flags: Boolean correct/incorrect per trajectory
        topics: Topic label per trajectory

    Returns:
        Dict mapping topic → accuracy
    """
    # Use plain dicts with .get() for full type-checker compatibility
    count_map: Dict[str, int]   = {}
    correct_map: Dict[str, int] = {}

    for flag, topic in zip(correct_flags, topics):
        count_map[topic]   = count_map.get(topic, 0) + 1
        correct_map[topic] = correct_map.get(topic, 0) + (1 if flag else 0)

    result: Dict[str, float] = {}
    for topic_key, total in count_map.items():
        if total > 0:
            n_correct_t: int = correct_map.get(topic_key, 0)
            result[topic_key] = float(n_correct_t) / float(total)
    return result


# ── Full Metrics Report ───────────────────────────────────────────────────────

def compute_full_metrics(
    rewards:          List[float],
    correct_flags:    List[bool],
    solutions:        List[str],
    topics:           List[str],
    reasoning_scores: Optional[List[float]] = None,
    baseline_acc:     Optional[float] = None,
    current_acc:      Optional[float] = None,
    n_iterations:     Optional[int] = None,
    holdout_acc:      Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all RL-SAGE evaluation metrics in one call.
    """
    n_flags: int = len(correct_flags)
    n_correct_total: int = sum(1 for f in correct_flags if f)
    accuracy: float = float(n_correct_total) / float(n_flags) if n_flags > 0 else 0.0

    metrics: Dict[str, float] = {
        "accuracy":  accuracy,
        "diversity": compute_diversity_score(solutions),
    }

    for k, v in compute_reward_statistics(rewards).items():
        metrics[f"reward_{k}"] = v

    for t, a in compute_per_topic_accuracy(correct_flags, topics).items():
        metrics[f"topic_acc/{t}"] = a

    if reasoning_scores:
        metrics["reasoning_quality"] = compute_mean_reasoning_quality(reasoning_scores)

    if baseline_acc is not None and current_acc is not None and n_iterations:
        metrics["sir"] = compute_self_improvement_rate(baseline_acc, current_acc, n_iterations)

    if holdout_acc is not None:
        metrics["generalization_gap"] = compute_generalization_gap(accuracy, holdout_acc)

    return metrics
