# GSM8K dataset directory
# GSM8K will be downloaded automatically from Hugging Face Hub on first run.
# Downloaded to: ~/.cache/huggingface/datasets/openai__gsm8k/
#
# Manual download:
#   from datasets import load_dataset
#   ds = load_dataset("openai/gsm8k", "main")
#   ds.save_to_disk("data/gsm8k/")
#
# Format: JSONL with fields: "question" (str), "answer" (str, ends with #### <number>)
