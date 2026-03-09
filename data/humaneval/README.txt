# HumanEval Dataset directory
# HumanEval is used for the optional code generation training track.
# Disabled by default (datasets.humaneval.enabled: false in config).
#
# Manual download:
#   from datasets import load_dataset
#   ds = load_dataset("openai_humaneval")
#   ds.save_to_disk("data/humaneval/")
#
# Format: JSON with fields: "task_id", "prompt" (docstring + signature), "test", "canonical_solution"
# Evaluation: unit test execution via sandbox subprocess
