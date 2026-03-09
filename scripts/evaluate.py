"""
rl_sage/scripts/evaluate.py

Standalone evaluation script: loads a checkpoint and runs benchmark evaluation.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/iter_500
    python scripts/evaluate.py --checkpoint checkpoints/iter_500 --benchmarks gsm8k arc_easy arc_challenge
    python scripts/evaluate.py --checkpoint checkpoints/final --all-benchmarks
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmarks import run_benchmark_evaluation
from src.evaluation.metrics import compute_full_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_sage.evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="RL-SAGE Evaluation")
    parser.add_argument("--checkpoint",     type=str, required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--config",         type=str, default="config/training_config.yaml")
    parser.add_argument("--benchmarks",     nargs="+", default=["gsm8k", "arc_easy"])
    parser.add_argument("--all-benchmarks", action="store_true", help="Run all benchmarks")
    parser.add_argument("--n-samples",      type=int, default=200, help="Samples per benchmark")
    parser.add_argument("--output",         type=str, default=None, help="Save results to JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_id    = config["model"]["base_model"]
    max_seq_len = config["training"].get("max_seq_length", 512)
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load Base + LoRA ──────────────────────────────────────────────────────
    logger.info(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading LoRA checkpoint: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    # ── Configure Benchmarks ──────────────────────────────────────────────────
    all_benchmarks = [
        {"name": "gsm8k",         "n_samples": args.n_samples, "decode": "greedy"},
        {"name": "arc_easy",      "n_samples": args.n_samples, "decode": "greedy"},
        {"name": "arc_challenge", "n_samples": args.n_samples, "decode": "greedy"},
    ]

    if args.all_benchmarks:
        benchmarks = all_benchmarks
    else:
        names = set(args.benchmarks)
        benchmarks = [b for b in all_benchmarks if b["name"] in names]

    # ── Run Evaluation ────────────────────────────────────────────────────────
    logger.info("Running evaluation...")
    results = run_benchmark_evaluation(model, tokenizer, benchmarks, max_seq_len, device)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Evaluation Results — Checkpoint: {args.checkpoint}")
    print("=" * 50)
    for name, acc in results.items():
        print(f"  {name:<20} {acc:.2%}")
    print("=" * 50 + "\n")

    # ── Save Results ──────────────────────────────────────────────────────────
    if args.output:
        import json
        output_data = {
            "checkpoint": args.checkpoint,
            "model": model_id,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
