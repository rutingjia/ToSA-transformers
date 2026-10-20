import logging
import os

import torch
import torch.nn as nn

from .config import (
    get_mix_nsa_backend,
    get_mix_nsa_config,
    get_visual_token_mask_from_position_ids,
    use_mix_nsa,
)
from .official_backend import compute_visual_mix_nsa_output_official
from .qwen3_backend import compute_visual_mix_nsa_output_qwen3

logger = logging.getLogger(__name__)


def _should_log_mix_debug(module: nn.Module) -> bool:
    enabled = os.environ.get("QWEN3VL_MIX_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return False
    max_steps = int(os.environ.get("QWEN3VL_MIX_DEBUG_STEPS", "5"))
    current = int(getattr(module, "_mix_nsa_debug_step", 0))
    if current >= max_steps:
        return False
    setattr(module, "_mix_nsa_debug_step", current + 1)
    return True


def init_mix_nsa_projections(module: nn.Module, hidden_size: int, num_attention_heads: int) -> None:
    if not use_mix_nsa(module.config):
        return
    module.g_proj_1 = nn.Linear(hidden_size, hidden_size, bias=True)
    module.g_proj_2 = nn.Linear(hidden_size, num_attention_heads * 3, bias=True)
    _reset_gate_initialization(module, num_attention_heads)



def run_attention_with_mix_nsa(
    module: nn.Module,
    attention_interface,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    multimodal_position_ids: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if os.environ.get("QWEN3VL_MIX_DEBUG") == "1":
        logger.info("Entering MixNSA Routing: mode=%s, has_pos_ids=%s", getattr(module.config, 'attention_mode', 'N/A'), multimodal_position_ids is not None)

    debug_this_step = _should_log_mix_debug(module)
    if debug_this_step:
        logger.info(
            "enter run_attention_with_mix_nsa: module=%s, attention_mode=%s, backend=%s, hidden=%s, query=%s, key=%s, value=%s",
            module.__class__.__name__,
            getattr(module.config, 'attention_mode', None),
            getattr(module.config, 'mix_nsa_backend', None),
            tuple(hidden_states.shape),
            tuple(query_states.shape),
            tuple(key_states.shape),
            tuple(value_states.shape),
        )
    #计算mask
    visual_mask = get_visual_token_mask_from_position_ids(multimodal_position_ids)
  
    if debug_this_step:
        logger.info(
            "token split: visual_tokens=%d, text_tokens=%d, total_tokens=%d, has_visual=%s",
            int(visual_mask.sum().item()) if visual_mask is not None else 0,
            int((~visual_mask).sum().item()) if visual_mask is not None else query_states.shape[0] * query_states.shape[2],
            query_states.shape[0] * query_states.shape[2],
            bool(visual_mask is not None and visual_mask.any()),
        )

    if not use_mix_nsa(module.config) or visual_mask is None or not visual_mask.any():
        return attention_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    if not hasattr(module, "g_proj_1") or not hasattr(module, "g_proj_2"):
        raise ValueError("`attention_mode='mix_nsa'` requires MixNSA projections to be initialized at model build time.")

    text_mask = ~visual_mask
   

    #计算text部分的attention
    text_query_states = query_states * text_mask.unsqueeze(1).unsqueeze(-1).to(query_states.dtype)
    text_attn_output, attn_weights = attention_interface(
        module,
        text_query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )
    text_output = text_attn_output * text_mask.unsqueeze(-1).unsqueeze(-1).to(text_attn_output.dtype)
 


    if debug_this_step:
        logger.info(
            "text path output: shape=%s, abs_mean=%.6f, abs_max=%.6f",
            tuple(text_output.shape),
            float(text_output.abs().mean().item()),
            float(text_output.abs().max().item()),
        )

    mix_config = get_mix_nsa_config(module.config)
    backend = get_mix_nsa_backend(module.config)

    #开始分支，进入mixnsa
    if backend == "videonsa_official":
        visual_output = compute_visual_mix_nsa_output_official(
            module=module,
            hidden_states=hidden_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            visual_mask=visual_mask,
            mix_config=mix_config,
        )
    else:
        visual_output = compute_visual_mix_nsa_output_qwen3(
            module=module,
            hidden_states=hidden_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            visual_mask=visual_mask,
            mix_config=mix_config,
            debug=debug_this_step,
        )
    return text_output + visual_output.transpose(1, 2), attn_weights


def _reset_gate_initialization(module: nn.Module, num_attention_heads: int) -> None:
    if module.g_proj_2.bias.device.type == 'meta':
        return
    nn.init.normal_(module.g_proj_1.weight, mean=0.0, std=0.02)
    nn.init.constant_(module.g_proj_1.bias, 0.0)

    nn.init.normal_(module.g_proj_2.weight, mean=0.0, std=0.02)

    target_prob = 0.5
    logit_val = torch.log(torch.tensor(target_prob) / (1 - target_prob))  # 这里改回原样即可

    output_bias = torch.full(
        (num_attention_heads * 3,),
        logit_val,
        device=module.g_proj_2.bias.device,
        dtype=module.g_proj_2.bias.dtype,
    )
    module.g_proj_2.bias.copy_(output_bias)

