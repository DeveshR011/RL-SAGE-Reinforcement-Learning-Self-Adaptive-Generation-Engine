"""
rl_sage/src/models/value_head.py

A lightweight scalar value head added on top of the language model's
last hidden state. Used by PPO to estimate V(s) for advantage computation.
"""

import torch
import torch.nn as nn
from typing import Optional, cast


class ValueHead(nn.Module):
    """
    Scalar value head for PPO.

    Takes the last hidden state of the language model and projects it
    to a single scalar (the state value estimate).

    Architecture:
        hidden_state [batch, seq_len, hidden_dim]
            → mean pool across seq dim                [batch, hidden_dim]
            → LayerNorm
            → Linear(hidden_dim, 256)
            → GELU
            → Dropout(0.1)
            → Linear(256, 1)
            → scalar value [batch]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights to near-zero for training stability."""
        for raw_module in self.modules():
            if isinstance(raw_module, nn.Linear):
                linear: nn.Linear = cast(nn.Linear, raw_module)
                nn.init.normal_(linear.weight, mean=0.0, std=0.01)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len], 1 for real tokens, 0 for pad

        Returns:
            values: [batch] — scalar state value estimate
        """
        # Mean pool over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()   # [batch, seq_len, 1]
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)            # [batch, hidden_dim]

        x = self.norm(pooled)
        value = self.net(x).squeeze(-1)                   # [batch]
        return value


class PolicyWithValueHead(nn.Module):
    """
    Wraps a Hugging Face CausalLM model with an attached ValueHead.
    Used by PPOTrainer as the full actor-critic model.
    """

    def __init__(self, lm_model, value_head: ValueHead):
        super().__init__()
        self.lm = lm_model
        self.value_head = value_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass: returns both LM logits and value estimate.

        Returns:
            logits: [batch, seq_len, vocab_size]
            values: [batch]
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        logits = outputs.logits                           # [batch, seq, vocab]
        last_hidden = outputs.hidden_states[-1]           # [batch, seq, hidden]
        values = self.value_head(last_hidden, attention_mask)
        return logits, values

    def generate(self, *args, **kwargs):
        """Delegate generation to the underlying LM."""
        return self.lm.generate(*args, **kwargs)


def build_value_head(model, dropout: float = 0.1) -> ValueHead:
    """
    Infer hidden_dim from the LM config and build a ValueHead.

    Args:
        model: Hugging Face CausalLM model (with PEFT adapters applied)
        dropout: Dropout rate for value head

    Returns:
        ValueHead instance
    """
    # Try common config attribute names
    cfg = model.config
    hidden_dim = getattr(cfg, "hidden_size",
                 getattr(cfg, "n_embd",
                 getattr(cfg, "d_model", 2560)))  # 2560 = Phi-2 default

    return ValueHead(hidden_dim=hidden_dim, dropout=dropout)
