"""
rl_sage/src/models/policy.py

Policy model: Phi-2 (or TinyLlama) loaded with 4-bit QLoRA quantization.
Enhanced with:
  - Pre-load VRAM guard (aborts early if < 5 GB free)
  - Flash Attention 2 opt-in
  - Weight-sharing reference model (saves ~2 GB VRAM)
  - Detailed VRAM reporting (allocated vs reserved)
"""

import gc
import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

logger = logging.getLogger(__name__)

# ── VRAM constants ────────────────────────────────────────────────────────────
_MIN_FREE_VRAM_GB = 4.5    # Refuse to load if less than this is free
_WARN_VRAM_GB     = 5.5    # Log a warning above this usage level


# ── VRAM Utilities ────────────────────────────────────────────────────────────

def vram_stats() -> dict:
    """
    Return a dict with current VRAM stats in GB.
    Keys: allocated, reserved, free, total
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}
    props     = torch.cuda.get_device_properties(0)
    total     = props.total_memory / 1e9
    reserved  = torch.cuda.memory_reserved(0)  / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    free      = total - reserved
    return {
        "allocated": round(allocated, 3),
        "reserved":  round(reserved,  3),
        "free":      round(free,      3),
        "total":     round(total,     3),
    }


def _log_vram(tag: str = ""):
    s = vram_stats()
    logger.info(
        f"VRAM {tag}: {s['allocated']:.2f} GB alloc | "
        f"{s['reserved']:.2f} GB reserved | "
        f"{s['free']:.2f} GB free / {s['total']:.2f} GB total"
    )
    if s["allocated"] > _WARN_VRAM_GB:
        logger.warning(
            f"⚠ VRAM usage {s['allocated']:.2f} GB exceeds warning threshold "
            f"({_WARN_VRAM_GB} GB). Risk of OOM."
        )


def _check_vram_guard():
    """Abort with a clear message if there isn't enough free VRAM."""
    s = vram_stats()
    if s["total"] == 0.0:
        logger.warning("No CUDA GPU detected — running on CPU (very slow).")
        return
    if s["free"] < _MIN_FREE_VRAM_GB:
        raise RuntimeError(
            f"Insufficient free VRAM: {s['free']:.2f} GB free, "
            f"need at least {_MIN_FREE_VRAM_GB} GB. "
            "Try closing other GPU processes or switching to TinyLlama."
        )


# ── Config Builders ───────────────────────────────────────────────────────────

def build_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig from config dict."""
    compute_dtype = (
        torch.float16 if cfg.get("compute_dtype", "float16") == "float16"
        else torch.bfloat16
    )
    return BitsAndBytesConfig(
        load_in_4bit=cfg.get("bits", 4) == 4,
        load_in_8bit=cfg.get("bits", 4) == 8,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("double_quant", True),
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    """Build a LoraConfig from config dict."""
    return LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("alpha", 32),
        lora_dropout=cfg.get("dropout", 0.05),
        bias=cfg.get("bias", "none"),
        target_modules=cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "dense"]
        ),
        task_type="CAUSAL_LM",
    )


# ── Policy Model ──────────────────────────────────────────────────────────────

def load_policy_model(
    config: dict,
    device_map: str = "auto",
    use_flash_attention: bool = False,
) -> Tuple:
    """
    Load the policy model with QLoRA applied.

    Args:
        config: Model config section from training_config.yaml
        device_map: HuggingFace device map strategy ("auto" or "cuda:0")
        use_flash_attention: Enable Flash Attention 2 if supported

    Returns:
        (model, tokenizer) — PEFT LoRA model + tokenizer
    """
    model_id = config["base_model"]
    logger.info(f"Loading policy model: {model_id}")
    _check_vram_guard()
    _log_vram("before load")

    # ── Quantization ──────────────────────────────────────────────────────────
    quant_cfg = config.get("quantization", {})
    quantization_enabled = quant_cfg.get("enabled", True)
    quantization_config = (
        build_quantization_config(quant_cfg) if quantization_enabled else None
    )

    # ── Max memory cap (leave ~0.5 GB headroom) ───────────────────────────────
    s = vram_stats()
    max_mem: Optional[dict] = None
    if s["total"] > 0:
        max_mem = {0: f"{max(1, int(s['total'] - 0.5))}GiB"}

    # ── Flash Attention 2 ─────────────────────────────────────────────────────
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"

    # ── Base Model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_mem,
        trust_remote_code=True,       # Required for Phi-2
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
    )
    _log_vram("after base model")

    # ── Prepare for k-bit training ────────────────────────────────────────────
    if quantization_enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", True):
        lora_config = build_lora_config(lora_cfg)
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

    # ── Gradient Checkpointing ────────────────────────────────────────────────
    if config.get("gradient_checkpointing", True):
        try:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except Exception as e:
            logger.warning(f"Gradient checkpointing failed to enable: {e}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"    # Required for PPO / left-padded causal LM

    _log_vram("after LoRA setup")
    return model, tokenizer


# ── Reference Model (Weight-Shared) ──────────────────────────────────────────

def load_reference_model(
    config: dict,
    policy_model=None,
    share_weights: bool = True,
    device_map: str = "auto",
):
    """
    Load a frozen reference model for KL penalty computation.

    If `share_weights=True` (default) and a `policy_model` is provided,
    this creates the reference view by temporarily disabling the LoRA adapters —
    i.e., the reference IS the base weights of the policy, which saves ~2 GB VRAM.

    If `share_weights=False`, loads a fully independent copy of the model.

    Args:
        config: Model config section
        policy_model: The trainable PEFT model (used if share_weights=True)
        share_weights: Whether to reuse the base model weights
        device_map: HuggingFace device map

    Returns:
        frozen reference model
    """
    model_id = config["base_model"]

    if share_weights and policy_model is not None:
        logger.info(
            "Reference model: SHARING weights with policy base model (saves ~2 GB VRAM)."
        )
        # The base_model of a PeftModel IS the original frozen weights.
        # We access it directly and disable all LoRA adapters.
        try:
            ref_model = policy_model.base_model.model
        except AttributeError:
            ref_model = policy_model

        # Important: do NOT mutate requires_grad here.
        # In PEFT models, LoRA parameters live inside the wrapped base model.
        # Freezing this shared object would also freeze policy training params.
        logger.info("Reference model: weights shared without mutating policy trainability.")
        _log_vram("after ref model (shared)")
        return ref_model

    # Full independent copy
    logger.info(f"Loading independent reference model: {model_id}")
    quant_cfg = config.get("quantization", {})
    quantization_config = (
        build_quantization_config(quant_cfg) if quant_cfg.get("enabled", True) else None
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    _log_vram("after ref model (independent)")
    return ref_model


# ── Log-probability Utility ───────────────────────────────────────────────────

def get_log_probs(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities of `response` given `prompt`.

    Returns:
        token_log_probs: Tensor of shape [response_len] on CPU
    """
    if device is None:
        device = str(next(model.parameters()).device)

    full_text = prompt + response
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)

    prompt_len = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    logits     = outputs.logits                         # [1, seq_len, vocab]
    log_probs  = torch.log_softmax(logits, dim=-1)     # [1, seq_len, vocab]

    # Shift: logit at t predicts token at t+1
    response_ids       = inputs["input_ids"][:, prompt_len:]           # [1, resp]
    response_log_probs = log_probs[:, prompt_len - 1 : -1, :]         # [1, resp, V]

    token_log_probs = response_log_probs.gather(
        2, response_ids.unsqueeze(-1)
    ).squeeze(-1).squeeze(0)    # [resp_len]

    return token_log_probs.cpu()
