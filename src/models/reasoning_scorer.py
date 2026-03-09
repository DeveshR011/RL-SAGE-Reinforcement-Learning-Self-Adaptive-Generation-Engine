"""
rl_sage/src/models/reasoning_scorer.py

Reasoning Quality Scorer: a lightweight DeBERTa-v3-small model that runs on CPU
and scores the logical coherence of a chain-of-thought solution on [0, 1].

The scorer uses NLI (Natural Language Inference) to measure whether each reasoning
step is entailed by the problem statement and prior steps.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Labels from NLI classifiers: [contradiction, neutral, entailment]
ENTAILMENT_IDX = 2
CONTRADICTION_IDX = 0


class ReasoningScorer:
    """
    Scores the reasoning quality of a chain-of-thought solution.

    Uses a cross-encoder (DeBERTa-v3-small fine-tuned on NLI) to measure
    logical coherence between the problem and the reasoning chain.

    Score interpretation:
        0.0 → Completely incoherent / contradictory reasoning
        0.5 → Neutral / unsupported reasoning
        1.0 → Logically entailed, well-structured reasoning
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        batch_size: int = 8,
    ):
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading reasoning scorer: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info("Reasoning scorer loaded.")

    @torch.no_grad()
    def score(self, problem: str, reasoning: str) -> float:
        """
        Score how well the reasoning supports the problem solution.

        Args:
            problem: The original task/question string
            reasoning: The chain-of-thought reasoning text

        Returns:
            float in [0, 1]
        """
        if not reasoning or len(reasoning.strip()) < 5:
            return 0.1   # Minimal or empty reasoning

        # Truncate to model limits
        premise = problem[:512]
        hypothesis = reasoning[:512]

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits           # [1, 3]
        probs = torch.softmax(logits, dim=-1)[0]       # [3]

        # Score = entailment_prob - 0.5 * contradiction_prob, clipped to [0,1]
        entailment = probs[ENTAILMENT_IDX].item()
        contradiction = probs[CONTRADICTION_IDX].item()
        raw_score = entailment - 0.5 * contradiction
        return float(max(0.0, min(1.0, raw_score + 0.5)))  # shift to [0,1]

    @torch.no_grad()
    def score_batch(
        self,
        problems: List[str],
        reasonings: List[str],
    ) -> List[float]:
        """
        Score a batch of (problem, reasoning) pairs efficiently.

        Args:
            problems: List of problem strings
            reasonings: List of reasoning strings

        Returns:
            List of float scores in [0, 1]
        """
        scores = []
        for i in range(0, len(problems), self.batch_size):
            batch_probs = problems[i : i + self.batch_size]
            batch_reas = reasonings[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_probs,
                batch_reas,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            logits = self.model(**inputs).logits           # [B, 3]
            probs = torch.softmax(logits, dim=-1)          # [B, 3]

            entailment = probs[:, ENTAILMENT_IDX]
            contradiction = probs[:, CONTRADICTION_IDX]
            raw = entailment - 0.5 * contradiction
            clipped = (raw + 0.5).clamp(0.0, 1.0)
            scores.extend(clipped.tolist())

        return scores


def extract_reasoning(solution_text: str) -> str:
    """
    Extract the reasoning chain from a solution string.

    Supports two formats:
        1) Tagged: <reasoning>...</reasoning>
        2) Untagged: everything except the last "ANSWER: ..." line

    Args:
        solution_text: Full solution string

    Returns:
        Extracted reasoning text
    """
    import re

    # Format 1: tagged reasoning
    match = re.search(r"<reasoning>(.*?)</reasoning>", solution_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Format 2: everything before the final answer line
    lines = solution_text.strip().split("\n")
    reasoning_lines = [l for l in lines if not l.upper().startswith("ANSWER:")]
    return "\n".join(reasoning_lines).strip()
