"""
rl_sage/src/modules/task_generator.py

Task Generator: uses the base LM in frozen inference mode to generate
novel training tasks conditioned on a topic and difficulty level.
"""

import re
import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A single generated task."""
    task_id:          str
    prompt:           str          # Full prompt shown to the policy
    problem:          str          # Just the problem statement
    reference_answer: str          # Ground-truth answer string
    difficulty:       float        # [0.0, 1.0]
    topic:            str
    source:           str          # "seed" | "self_generated" | "dataset"
    metadata:         dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id":          self.task_id,
            "prompt":           self.prompt,
            "problem":          self.problem,
            "reference_answer": self.reference_answer,
            "difficulty":       self.difficulty,
            "topic":            self.topic,
            "source":           self.source,
            "metadata":         self.metadata,
        }


TASK_GENERATION_PROMPT_TEMPLATE = """\
You are an educational AI that creates math and reasoning problems.

Topic: {topic}
Difficulty: {difficulty:.2f}  (0.0 = easiest, 1.0 = hardest)
Reference examples for variety (DO NOT COPY THESE):
{seed_examples}

Create ONE new, original problem. Format your response EXACTLY as:
PROBLEM: [problem statement]
SOLUTION: [step-by-step solution]
ANSWER: [final answer only — a number or short phrase]

Problem:"""


SOLVE_PROMPT_TEMPLATE = """\
You are a precise reasoning assistant. Solve the following problem step by step.
Show ALL your work. Put your final answer on the last line as: ANSWER: [value]

Problem: {problem}

Solution:"""


class TaskGenerator:
    """
    Generates novel training tasks using the frozen base LM.

    Args:
        model: Frozen Hugging Face CausalLM
        tokenizer: Corresponding tokenizer
        seed_tasks: List of Task objects used as few-shot seeds
        config: Dict with generation parameters
    """

    def __init__(
        self,
        model,
        tokenizer,
        seed_tasks: List[Task],
        config: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.seed_tasks = seed_tasks
        self.config = config or {}
        self._task_counter = 0

    def generate(self, topic: str, difficulty: float) -> Task:
        """
        Generate a single new task.

        Args:
            topic: Subject area (e.g., "arithmetic", "logic")
            difficulty: Float in [0.0, 1.0]

        Returns:
            Task dataclass
        """
        # Pick 2–3 random seed examples as context
        seeds = random.sample(self.seed_tasks, min(3, len(self.seed_tasks)))
        seed_text = "\n".join(
            f"- Q: {s.problem}  A: {s.reference_answer}" for s in seeds
        )

        prompt = TASK_GENERATION_PROMPT_TEMPLATE.format(
            topic=topic,
            difficulty=difficulty,
            seed_examples=seed_text,
        )

        raw_output = self._generate_text(
            prompt,
            max_new_tokens=self.config.get("max_new_tokens", 200),
            temperature=self.config.get("temperature", 0.9),
            top_p=self.config.get("top_p", 0.95),
        )

        task = self._parse_generated_task(raw_output, topic, difficulty)
        return task

    def generate_batch(self, topics: List[str], difficulties: List[float]) -> List[Task]:
        """Generate multiple tasks sequentially."""
        return [self.generate(t, d) for t, d in zip(topics, difficulties)]

    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> str:
        """Run inference on the frozen model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _parse_generated_task(
        self, raw: str, topic: str, difficulty: float
    ) -> Task:
        """Parse the PROBLEM / SOLUTION / ANSWER from generated text."""
        problem = self._extract_field(raw, "PROBLEM")
        answer = self._extract_field(raw, "ANSWER")

        # Fallback: use the entire output as problem with unknown answer
        if not problem:
            problem = raw.strip()[:300]
            answer = "UNKNOWN"

        self._task_counter += 1
        task_id = f"gen_{self._task_counter:06d}"

        solve_prompt = SOLVE_PROMPT_TEMPLATE.format(problem=problem)

        return Task(
            task_id=task_id,
            prompt=solve_prompt,
            problem=problem,
            reference_answer=answer,
            difficulty=difficulty,
            topic=topic,
            source="self_generated",
        )

    @staticmethod
    def _extract_field(text: str, field: str) -> str:
        """Extract a labeled field from generated text."""
        pattern = rf"{field}:\s*(.*?)(?:\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @classmethod
    def from_dataset(cls, model, tokenizer, dataset, config=None) -> "TaskGenerator":
        """
        Build a TaskGenerator seeded from a Hugging Face dataset.

        Expects dataset rows with keys: 'question' and 'answer'.
        """
        seed_tasks = []
        for i, row in enumerate(dataset):
            if i >= 100:
                break
            answer = row.get("answer", "")
            # GSM8K: answer is "...#### <number>"
            if "####" in answer:
                answer = answer.split("####")[-1].strip()

            seed_tasks.append(Task(
                task_id=f"seed_{i:04d}",
                prompt=SOLVE_PROMPT_TEMPLATE.format(problem=row["question"]),
                problem=row["question"],
                reference_answer=answer,
                difficulty=0.3,
                topic="arithmetic",
                source="seed",
            ))

        logger.info(f"Created TaskGenerator with {len(seed_tasks)} seed tasks.")
        return cls(model, tokenizer, seed_tasks, config)
