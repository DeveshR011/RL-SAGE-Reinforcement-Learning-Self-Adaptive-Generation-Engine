"""
rl_sage/src/evaluation/benchmarks.py

Benchmark evaluation: runs the policy on standard benchmarks (GSM8K, ARC)
and computes accuracy.
"""

import re
import logging
from typing import Any, Callable, List, Dict, Optional

# Third-party runtime dependencies — guarded with try/except so static analysers
# do not flag them as hard errors when packages are not yet installed.
# Run `pip install -r requirements.txt` to resolve all imports.
try:
    import torch as _torch                          # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]

try:
    from datasets import load_dataset as _load_ds   # type: ignore[import-untyped]
    _has_datasets = True
except ImportError:  # pragma: no cover
    _has_datasets = False

try:
    from tqdm import tqdm as _tqdm                  # type: ignore[import-untyped]
    _has_tqdm = True
except ImportError:  # pragma: no cover
    _has_tqdm = False


def _get_torch() -> Any:
    if _torch is None:
        raise RuntimeError("torch not installed. Run: pip install torch")
    return _torch


def load_dataset(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
    """Thin wrapper — forwards to HuggingFace datasets.load_dataset."""
    if not _has_datasets:
        raise RuntimeError("datasets not installed. Run: pip install datasets")
    from datasets import load_dataset as _ld        # type: ignore[import-untyped]
    return _ld(*args, **kwargs)


def tqdm(iterable: Any, **kwargs: Any) -> Any:     # type: ignore[misc]
    """Thin wrapper — forwards to tqdm.tqdm, falls back to plain iteration."""
    if _has_tqdm:
        from tqdm import tqdm as _t                 # type: ignore[import-untyped]
        return _t(iterable, **kwargs)
    return iterable


logger = logging.getLogger(__name__)



SOLVE_PROMPT = """\
You are a precise reasoning assistant. Solve the following problem step by step.
Show ALL your work. Put your final answer on the last line as: ANSWER: [value]

Problem: {problem}

Solution:"""


def run_benchmark_evaluation(
    model,
    tokenizer,
    benchmarks: List[dict],
    max_seq_length: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate the model on all configured benchmarks.

    Args:
        model: Policy model (PEFT + LM)
        tokenizer: Tokenizer
        benchmarks: List of benchmark config dicts from training_config.yaml
        max_seq_length: Token limit
        device: Compute device

    Returns:
        Dict mapping benchmark_name → accuracy (float in [0, 1])
    """
    results = {}
    model.eval()

    for bench_cfg in benchmarks:
        name = bench_cfg["name"]
        n_samples = bench_cfg.get("n_samples", 100)

        try:
            if name == "gsm8k":
                acc = _eval_gsm8k(model, tokenizer, n_samples, max_seq_length, device)
            elif name == "arc_easy":
                acc = _eval_arc(model, tokenizer, "ARC-Easy",      n_samples, max_seq_length, device)
            elif name == "arc_challenge":
                acc = _eval_arc(model, tokenizer, "ARC-Challenge",  n_samples, max_seq_length, device)
            else:
                logger.warning(f"Unknown benchmark: {name}, skipping.")
                continue

            results[name] = acc
            logger.info(f"  [{name}] accuracy = {acc:.2%}")

        except Exception as e:
            logger.error(f"Benchmark {name} failed: {e}")
            results[name] = 0.0

    return results


# ── GSM8K ─────────────────────────────────────────────────────────────────────

def _eval_gsm8k(
    model, tokenizer, n_samples: int, max_seq_length: int, device: str
) -> float:
    """Evaluate on GSM8K grade school math (test split)."""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    correct_flags: List[bool] = []
    for row in tqdm(list(dataset), desc="GSM8K", leave=False):
        problem: str  = str(row["question"])
        ref_ans: str  = _extract_gsm8k_answer(str(row["answer"]))
        pred_ans: str = _generate_answer(model, tokenizer, problem, max_seq_length, device)
        correct_flags.append(_numeric_equal(pred_ans, ref_ans))

    n_total: int   = len(correct_flags)
    n_correct: int = sum(1 for f in correct_flags if f)
    return float(n_correct) / float(max(n_total, 1))


def _extract_gsm8k_answer(answer_text: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    match = re.search(r"####\s*([\d,\.\-]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


# ── ARC ───────────────────────────────────────────────────────────────────────

def _eval_arc(
    model, tokenizer, config_name: str, n_samples: int, max_seq_length: int, device: str
) -> float:
    """Evaluate on ARC-Easy or ARC-Challenge (test split)."""
    dataset = load_dataset("allenai/ai2_arc", config_name, split="test")
    dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    correct_flags: List[bool] = []
    for row in tqdm(list(dataset), desc=config_name, leave=False):
        problem: str   = _format_arc_problem(row)
        ref_key: str   = str(row["answerKey"]).upper()
        pred_text: str = _generate_answer(model, tokenizer, problem, max_seq_length, device)
        pred_key: str  = _extract_choice(pred_text)
        correct_flags.append(bool(pred_key) and pred_key.upper() == ref_key)

    n_total: int   = len(correct_flags)
    n_correct: int = sum(1 for f in correct_flags if f)
    return float(n_correct) / float(max(n_total, 1))


def _format_arc_problem(row: dict) -> str:
    """Format an ARC row as a multiple-choice question."""
    choices = row["choices"]
    options = "\n".join(
        f"({label}) {text}"
        for label, text in zip(choices["label"], choices["text"])
    )
    return f"{row['question']}\n\nOptions:\n{options}"


# ── Generation Helper ─────────────────────────────────────────────────────────

def _generate_answer(
    model, tokenizer, problem: str, max_seq_length: int, device: str
) -> str:
    """Generate a greedy answer for a single problem."""
    prompt = SOLVE_PROMPT.format(problem=problem)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    ).to(device)

    with _get_torch().no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,    # Greedy for eval
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Answer Extraction ─────────────────────────────────────────────────────────

def _extract_answer(text: str) -> str:
    """Extract final answer from solution text."""
    match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def _extract_choice(text: str) -> str:
    """Extract A/B/C/D choice from text."""
    patterns = [
        r"ANSWER\s*:\s*\(?([ABCD])\)?",
        r"answer is\s*\(?([ABCD])\)?",
        r"\(([ABCD])\)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def _numeric_equal(pred: str, ref: str, tol: float = 1e-6) -> bool:
    """Check numeric equality with tolerance."""
    # Try exact string match first
    if pred.strip() == ref.strip():
        return True
    # Try numeric match
    try:
        return abs(float(pred) - float(ref)) < tol
    except ValueError:
        return False
