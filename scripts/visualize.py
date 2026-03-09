"""
rl_sage/scripts/visualize.py

Visualization script: plots training curves from W&B logs or local JSON logs.

Usage:
    python scripts/visualize.py --log-dir logs/
    python scripts/visualize.py --wandb-run <run-id>
"""

import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rl_sage.visualize")


def parse_args():
    parser = argparse.ArgumentParser(description="RL-SAGE Visualization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--log-dir",   type=str, help="Directory containing local metric JSON files")
    group.add_argument("--wandb-run", type=str, help="W&B run ID to download and plot")
    parser.add_argument("--output-dir", type=str, default="plots/", help="Where to save plot images")
    return parser.parse_args()


def plot_from_local(log_dir: str, output_dir: str):
    """Plot training curves from local JSON log files."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as mstyle
        mstyle.use("seaborn-v0_8-darkgrid")
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return

    log_path = Path(log_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prefer the training loop's JSONL file; fall back to legacy metrics_*.json.
    all_logs = []
    metrics_jsonl = log_path / "metrics.jsonl"
    if metrics_jsonl.exists():
        with open(metrics_jsonl, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        for f in sorted(log_path.glob("metrics_*.json")):
            with open(f, "r", encoding="utf-8") as fp:
                all_logs.append(json.load(fp))

    if not all_logs:
        logger.error(f"No metric logs found in {log_dir}")
        return

    iterations  = [d.get("iteration", d.get("train/iteration", i)) for i, d in enumerate(all_logs)]
    rewards     = [d.get("train/mean_reward", 0) for d in all_logs]
    success     = [d.get("train/success_rate", 0) for d in all_logs]
    curriculum_sr = [d.get("curriculum/global_sr", 0) for d in all_logs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("RL-SAGE Training Curves", fontsize=16, fontweight="bold")

    # Reward
    axes[0, 0].plot(iterations, rewards, color="#4A90D9", linewidth=1.5)
    axes[0, 0].set_title("Mean Reward per Iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Success Rate
    axes[0, 1].plot(iterations, [s * 100 for s in success], color="#E84B3C", linewidth=1.5)
    axes[0, 1].set_title("Task Success Rate (%)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].axhline(75, color="green", linestyle="--", linewidth=0.8, label="Threshold ↑")
    axes[0, 1].axhline(40, color="orange", linestyle="--", linewidth=0.8, label="Threshold ↓")
    axes[0, 1].legend(fontsize=8)

    # Curriculum global success
    axes[1, 0].plot(iterations, curriculum_sr, color="#7B68EE", linewidth=1.5)
    axes[1, 0].set_title("Curriculum Global Success")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Success Rate [0-1]")
    axes[1, 0].set_ylim(0, 1)

    # Eval accuracy (if present)
    eval_iters = [d.get("iteration", d.get("train/iteration")) for d in all_logs if "eval/gsm8k" in d]
    gsm8k_acc  = [d.get("eval/gsm8k", 0) * 100 for d in all_logs if "eval/gsm8k" in d]
    arc_acc    = [d.get("eval/arc_easy", 0) * 100 for d in all_logs if "eval/arc_easy" in d]

    if eval_iters:
        axes[1, 1].plot(eval_iters, gsm8k_acc, "o-", label="GSM8K", color="#F5A623")
        axes[1, 1].plot(eval_iters, arc_acc,   "s-", label="ARC-Easy", color="#7ED321")
        axes[1, 1].set_title("Benchmark Accuracy (%)")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No eval data yet", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=12, color="gray")
        axes[1, 1].set_title("Benchmark Accuracy")

    plt.tight_layout()
    out_file = output_path / "training_curves.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved: {out_file}")
    plt.show()


def plot_from_wandb(run_id: str, output_dir: str):
    """Download and plot a W&B run."""
    try:
        import wandb
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("wandb / matplotlib not installed.")
        return

    api = wandb.Api()
    run = api.run(run_id)

    history = run.history(samples=1000)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if "train/mean_reward" in history.columns:
        history[["_step", "train/mean_reward"]].dropna().plot(
            x="_step", y="train/mean_reward", title="Mean Reward (W&B)"
        )
        plt.savefig(output_path / "reward_wandb.png", dpi=150)
        logger.info(f"W&B plot saved: {output_path / 'reward_wandb.png'}")
        plt.show()


def main():
    args = parse_args()
    if args.log_dir:
        plot_from_local(args.log_dir, args.output_dir)
    else:
        plot_from_wandb(args.wandb_run, args.output_dir)


if __name__ == "__main__":
    main()
