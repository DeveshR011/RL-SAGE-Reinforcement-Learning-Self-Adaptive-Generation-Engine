# ARC Dataset directory
# ARC-Easy and ARC-Challenge are downloaded automatically from Hugging Face Hub.
# Downloaded to: ~/.cache/huggingface/datasets/allenai__ai2_arc/
#
# Manual download:
#   from datasets import load_dataset
#   easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
#   challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")
#   easy.save_to_disk("data/arc/easy/")
#   challenge.save_to_disk("data/arc/challenge/")
#
# Format: JSON with fields: "question", "choices" ({"label": [...], "text": [...]}), "answerKey"
