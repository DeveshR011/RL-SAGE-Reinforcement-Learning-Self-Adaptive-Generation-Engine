"""
rl_sage/src/modules/evaluator.py

Evaluator: determines whether a generated solution is correct.
Supports multiple evaluation modes based on task type.
"""

import re
import string
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional

from src.modules.task_generator import Task
from src.modules.solution_generator import Solution

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one solution against a task."""
    correct:           bool
    correctness_score: float        # 1.0 = correct, 0.0 = wrong, -1.0 = unparsed
    reasoning_quality: float        # [0, 1] from reasoning scorer
    eval_mode:         str          # "exact", "normalized", "choice", "code"
    details:           dict = None  # Extra info (e.g., diff, error message)


class Evaluator:
    """
    Multi-mode evaluator for RL-SAGE tasks.

    Modes:
        - exact_match:   Numeric or string equality after normalization
        - choice:        Multiple-choice letter extraction (ARC style)
        - code:          Python execution with unit tests (HumanEval style)
        - llm_judge:     Fallback LM-based judge (optional)
    """

    def __init__(
        self,
        reasoning_scorer=None,
        code_timeout: int = 5,
    ):
        """
        Args:
            reasoning_scorer: ReasoningScorer instance (optional, can be None)
            code_timeout: Seconds allowed for code execution
        """
        self.reasoning_scorer = reasoning_scorer
        self.code_timeout = code_timeout

    def evaluate(
        self,
        task: Task,
        solution: Solution,
    ) -> EvalResult:
        """
        Evaluate a solution against its task.

        Args:
            task: Task dataclass
            solution: Solution dataclass

        Returns:
            EvalResult
        """
        # Choose evaluation strategy
        mode = self._detect_mode(task)

        if mode == "choice":
            result = self._eval_choice(task, solution)
        elif mode == "code":
            result = self._eval_code(task, solution)
        else:
            result = self._eval_exact(task, solution)

        # Reasoning quality
        rq = 0.5  # default
        if self.reasoning_scorer is not None and solution.reasoning:
            try:
                rq = self.reasoning_scorer.score(task.problem, solution.reasoning)
            except Exception as e:
                logger.warning(f"Reasoning scorer failed: {e}")

        result.reasoning_quality = rq
        return result

    # ── Mode Detection ─────────────────────────────────────────────────────────

    def _detect_mode(self, task: Task) -> str:
        topic = task.topic.lower()
        if "code" in topic or "python" in topic or task.source == "humaneval":
            return "code"
        if self._is_choice_task(task.problem):
            return "choice"
        return "exact"

    @staticmethod
    def _is_choice_task(problem: str) -> bool:
        """Detect multiple-choice format: options labeled (A) (B) (C) (D)."""
        return bool(re.search(r"\([ABCD]\)", problem))

    # ── Exact / Numeric Match ──────────────────────────────────────────────────

    def _eval_exact(self, task: Task, solution: Solution) -> EvalResult:
        predicted = self._normalize(solution.answer)
        reference = self._normalize(task.reference_answer)

        if predicted == "unparsed" or predicted == "":
            return EvalResult(
                correct=False,
                correctness_score=0.0,
                reasoning_quality=0.0,
                eval_mode="exact",
                details={"reason": "answer_unparsed"},
            )

        correct = (predicted == reference) or self._numeric_match(predicted, reference)

        return EvalResult(
            correct=correct,
            correctness_score=1.0 if correct else -0.0,
            reasoning_quality=0.0,
            eval_mode="exact",
            details={"predicted": predicted, "reference": reference},
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip punctuation, remove extra whitespace."""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _numeric_match(pred: str, ref: str, tol: float = 1e-6) -> bool:
        """Check if both strings represent the same number."""
        try:
            return abs(float(pred) - float(ref)) < tol
        except ValueError:
            return False

    # ── Multiple Choice ────────────────────────────────────────────────────────

    def _eval_choice(self, task: Task, solution: Solution) -> EvalResult:
        predicted_letter = self._extract_choice_letter(solution.text)
        reference_letter = self._extract_choice_letter(task.reference_answer)

        if not predicted_letter:
            return EvalResult(
                correct=False,
                correctness_score=0.0,
                reasoning_quality=0.0,
                eval_mode="choice",
                details={"reason": "no_choice_found"},
            )

        correct = (predicted_letter.upper() == reference_letter.upper())
        return EvalResult(
            correct=correct,
            correctness_score=1.0 if correct else 0.0,
            reasoning_quality=0.0,
            eval_mode="choice",
            details={"predicted": predicted_letter, "reference": reference_letter},
        )

    @staticmethod
    def _extract_choice_letter(text: str) -> str:
        """Extract A/B/C/D choice from text."""
        # Formats: "(A)", "A)", "Answer: A", "ANSWER: A", "the answer is A"
        patterns = [
            r"ANSWER\s*:\s*\(?([ABCD])\)?",
            r"answer is\s*\(?([ABCD])\)?",
            r"^\s*\(?([ABCD])\)?\s*$",
            r"\(([ABCD])\)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()
        return ""

    # ── Code Execution ─────────────────────────────────────────────────────────

    def _eval_code(self, task: Task, solution: Solution) -> EvalResult:
        """
        Execute generated Python code against unit tests.
        Runs in a subprocess with a timeout for safety.
        """
        # Extract code block from solution
        code_match = re.search(r"```python\n(.*?)```", solution.text, re.DOTALL)
        code = code_match.group(1) if code_match else solution.text

        test_code = task.metadata.get("test_code", "")
        full_code = code + "\n\n" + test_code

        try:
            result = subprocess.run(
                ["python", "-c", full_code],
                capture_output=True,
                text=True,
                timeout=self.code_timeout,
            )
            correct = result.returncode == 0
            return EvalResult(
                correct=correct,
                correctness_score=1.0 if correct else 0.0,
                reasoning_quality=0.0,
                eval_mode="code",
                details={
                    "returncode": result.returncode,
                    "stderr": result.stderr[:200],
                },
            )
        except subprocess.TimeoutExpired:
            return EvalResult(
                correct=False,
                correctness_score=0.0,
                reasoning_quality=0.0,
                eval_mode="code",
                details={"reason": "timeout"},
            )
        except Exception as e:
            return EvalResult(
                correct=False,
                correctness_score=0.0,
                reasoning_quality=0.0,
                eval_mode="code",
                details={"reason": str(e)},
            )
