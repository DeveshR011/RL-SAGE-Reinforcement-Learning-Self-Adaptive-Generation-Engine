# RL-SAGE checkpoints directory
# LoRA adapter weights are saved here every N iterations.
# Each checkpoint dir contains:
#   adapter_config.json
#   adapter_model.safetensors
#   tokenizer files
#
# Resume training from a checkpoint:
#   python scripts/train.py --config config/training_config.yaml --resume checkpoints/iter_500
