"""RL-SAGE evaluation package: benchmark runners and metrics."""
from src.evaluation.benchmarks import run_benchmark_evaluation
from src.evaluation.metrics import (
    compute_accuracy,
    compute_self_improvement_rate,
    compute_diversity_score,
    compute_full_metrics,
)

__all__ = [
    "run_benchmark_evaluation",
    "compute_accuracy",
    "compute_self_improvement_rate",
    "compute_diversity_score",
    "compute_full_metrics",
]
