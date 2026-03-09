"""
rl_sage/src/modules/solution_generator.py

Solution Generator: runs the policy (trainable LM) to generate chain-of-thought
solutions and captures per-token log-probabilities for PPO training.

Enhancements over v1:
  - Structured CoT prompt wrapper (<reasoning>...</reasoning> + ANSWER:)
  - max_retries=2 loop: retries if ANSWER is "UNPARSED"
  - generate_with_best_of_n(): samples N candidates, returns highest-reward one
  - Shared answer/reasoning parsers (robust multi-pattern extraction)
"""

import re
import gc
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Callable

import torch

logger = logging.getLogger(__name__)


# ── CoT Prompt Template ───────────────────────────────────────────────────────

_COT_WRAPPER = """\
{prompt}

Think step by step. Show your full reasoning.
<reasoning>
[write your reasoning here]
</reasoning>
ANSWER: [write only the final answer here]"""


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Solution:
    """A generated solution to a task."""
    text:      str            # Full generated text (CoT + answer)
    reasoning: str            # Extracted chain-of-thought
    answer:    str            # Extracted final answer
    log_probs: torch.Tensor   # Per-token log-probs [seq_len] on CPU
    n_tokens:  int            # Number of generated tokens
    n_retries: int = 0        # Number of generation retries needed
    metadata:  dict = field(default_factory=dict)

    @property
    def is_parseable(self) -> bool:
        """True if the answer was successfully extracted."""
        return self.answer != "UNPARSED"


# ── Solution Generator ────────────────────────────────────────────────────────

class SolutionGenerator:
    """
    Wraps the trainable policy model for solution generation.

    Captures per-token log-probabilities during generation, required for PPO.

    Args:
        model: Trainable PEFT LoRA model (policy)
        tokenizer: Tokenizer
        config: Dict with generation hyperparameters
        wrap_cot: Whether to wrap prompts with the structured CoT template
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[dict] = None,
        wrap_cot: bool = True,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config or {}
        self.wrap_cot  = wrap_cot

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_retries: int = 2) -> Solution:
        """
        Generate a solution with log-probability capture.

        Retries up to `max_retries` times if the answer is unparseable.

        Args:
            prompt: Full task prompt string
            max_retries: Number of additional attempts on parse failure

        Returns:
            Solution dataclass (answer may still be "UNPARSED" after all retries)
        """
        wrapped = self._make_prompt(prompt)
        solution = self._generate_once(wrapped)

        for retry in range(max_retries):
            if solution.is_parseable:
                break
            logger.debug(f"Answer unparsed, retry {retry + 1}/{max_retries}")
            solution = self._generate_once(wrapped, temperature_boost=0.2 * (retry + 1))
            solution.n_retries = retry + 1

        return solution

    def generate_with_best_of_n(
        self,
        prompt: str,
        n: int = 4,
        score_fn: Optional[Callable[[Solution], float]] = None,
    ) -> Solution:
        """
        Sample N candidate solutions and return the best-scoring one.

        Args:
            prompt: Task prompt
            n: Number of candidates to generate
            score_fn: Optional callable (Solution → float) for scoring.
                      Defaults to: parseable=1.0, not parseable=-1.0

        Returns:
            Best solution among the N candidates
        """
        if score_fn is None:
            score_fn = lambda s: 1.0 if s.is_parseable else -1.0

        candidates = [self.generate(prompt, max_retries=1) for _ in range(n)]
        best = max(candidates, key=score_fn)
        best.metadata["candidates_generated"] = n
        return best

    def generate_batch(self, prompts: List[str], max_retries: int = 2) -> List[Solution]:
        """Generate solutions for a batch of prompts (sequential, VRAM-safe)."""
        solutions = []
        for p in prompts:
            try:
                solutions.append(self.generate(p, max_retries=max_retries))
            except Exception as e:
                logger.warning(f"Generation failed for prompt (len={len(p)}): {e}")
                solutions.append(self._empty_solution())
        return solutions

    # ── Internal Generation ───────────────────────────────────────────────────

    def _make_prompt(self, prompt: str) -> str:
        """Optionally wrap in structured CoT template."""
        if self.wrap_cot:
            return _COT_WRAPPER.format(prompt=prompt)
        return prompt

    def _generate_once(
        self, prompt: str, temperature_boost: float = 0.0
    ) -> Solution:
        """
        Internal: single generation pass with log-prob capture.

        Args:
            prompt: Fully prepared prompt string
            temperature_boost: Added to base temperature (for retries)
        """
        max_new_tokens = self.config.get("max_new_tokens", 256)
        temperature    = self.config.get("temperature", 0.7) + temperature_boost
        top_p          = self.config.get("top_p", 0.95)
        max_length     = self.config.get("max_seq_length", 512)
        rep_penalty    = self.config.get("repetition_penalty", 1.2)

        # Enforce temperature bounds
        temperature = float(max(0.1, min(2.0, temperature)))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(self._device())

        prompt_len = inputs["input_ids"].shape[1]

        # ── Generate ──────────────────────────────────────────────────────────
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # ── Extract generated token ids ───────────────────────────────────────
        generated_ids = output.sequences[:, prompt_len:]   # [1, resp_len]
        resp_len = generated_ids.shape[1]

        # ── Compute per-token log-probs ───────────────────────────────────────
        if output.scores and len(output.scores) > 0:
            logits = torch.stack(output.scores, dim=1)         # [1, resp_len, V]
            log_probs_full = torch.log_softmax(logits, dim=-1) # [1, resp_len, V]
            token_log_probs = log_probs_full.gather(
                2, generated_ids.unsqueeze(-1)
            ).squeeze(-1).squeeze(0).cpu()                     # [resp_len]
        else:
            token_log_probs = torch.zeros(resp_len)

        # ── Decode text ───────────────────────────────────────────────────────
        solution_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        reasoning = extract_reasoning(solution_text)
        answer    = extract_answer(solution_text)

        # Free GPU memory from the generation output
        del output, inputs
        torch.cuda.empty_cache()

        return Solution(
            text=solution_text,
            reasoning=reasoning,
            answer=answer,
            log_probs=token_log_probs,
            n_tokens=resp_len,
        )

    def _device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def _empty_solution() -> Solution:
        """Return a blank solution placeholder for failed generations."""
        return Solution(
            text="[GENERATION FAILED]",
            reasoning="",
            answer="UNPARSED",
            log_probs=torch.zeros(1),
            n_tokens=0,
            metadata={"error": True},
        )


# ── Shared Parsing Utilities ──────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """
    Extract the final answer from generated text.

    Priority order:
        1. Explicit ANSWER: label
        2. Structured </reasoning>\\nANSWER: label
        3. GSM8K #### format
        4. "The answer is X" / "The result is X"
        5. Last number in text (fallback)
    """
    # Priority 1 & 2: ANSWER: label (catches both structured and unstructured)
    match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Priority 3: GSM8K format
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()

    # Priority 4: natural language
    match = re.search(
        r"(?:the answer is|answer:|result:|therefore[,:]?\s+the answer is)\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Priority 5: last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return "UNPARSED"


def extract_reasoning(text: str) -> str:
    """
    Extract the chain-of-thought reasoning from a solution.

    Supports both:
        - Tagged format: <reasoning>...</reasoning>
        - Untagged: everything before the ANSWER line
    """
    # Tagged format (structured CoT template)
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Untagged: everything before ANSWER / ####
    lines = text.strip().split("\n")
    reasoning_lines = [
        line for line in lines
        if not re.match(r"^\s*(ANSWER|####)\s*[:\s]", line, re.IGNORECASE)
    ]
    return "\n".join(reasoning_lines).strip()
