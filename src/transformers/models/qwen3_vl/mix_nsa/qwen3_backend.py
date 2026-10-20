import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
_flex_attention = torch.compiler.disable(flex_attention)
from flash_attn import flash_attn_func

from .config import MixNSAConfig
from .ops import build_block_mask, masked_softmax, mean_pool_blocks, repeat_kv
from .selection_kernel import selection_attention
from fla.ops.nsa.parallel import parallel_nsa_topk
from fla.ops.utils.pooling import mean_pooling



logger = logging.getLogger(__name__)


def compute_visual_mix_nsa_output_qwen3(
    module: nn.Module,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    visual_mask: torch.Tensor,
    mix_config: MixNSAConfig,
    debug: bool = False,
) -> torch.Tensor:
    visual_mask_seq = visual_mask.unsqueeze(-1)
    visual_mask_attn = visual_mask.unsqueeze(1).unsqueeze(-1)

    masked_hidden_states = hidden_states * visual_mask_seq.to(hidden_states.dtype)
    masked_queries = query_states * visual_mask_attn.to(query_states.dtype)
    masked_keys = key_states * visual_mask_attn.to(query_states.dtype)
    masked_values = value_states * visual_mask_attn.to(query_states.dtype)

    gate_logits = module.g_proj_2(F.gelu(module.g_proj_1(hidden_states)))
    gate_logits = gate_logits.view(
        query_states.shape[0], query_states.shape[2], module.config.num_attention_heads, 3
    ).transpose(1, 2)
    gate_logits = gate_logits / getattr(module.config, "mix_nsa_gate_temperature", 0.3)

    g_cmp, g_slc, g_swa = torch.sigmoid(gate_logits).unbind(-1)

    # 文本位置的 gate 清零
    g_cmp = g_cmp * visual_mask.unsqueeze(1).to(g_cmp.dtype)
    g_slc = g_slc * visual_mask.unsqueeze(1).to(g_slc.dtype)
    g_swa = g_swa * visual_mask.unsqueeze(1).to(g_swa.dtype)

    # Pad Q to satisfy HEADS_PER_GROUP >= 16 for FLA ParallelNSAFunction.
    # H=16, G=8 → HEADS_PER_GROUP=2 (too small for Triton tl.dot, which requires >= 16).
    # Use `repeat_interleave` so that each original Q head is replicated in-place,
    # producing layout [Q0×N, Q1×N, ..., Q_{H-1}×N]. This preserves GQA pairing:
    # kernel head range [g*16 : (g+1)*16] still maps to the same original heads
    # that should pair with kv_head g. Using `cat([Q]*N)` would scramble this
    # mapping (every kv_head would see all original Q heads), breaking GQA.
    B, H, M, D = masked_queries.shape
    num_kv_groups = key_states.shape[1]
    target_heads = num_kv_groups * 16
    copies_per_head = target_heads // H
    assert target_heads % H == 0, f"Cannot pad {H} heads to {target_heads} evenly"

    masked_queries_padded = masked_queries.repeat_interleave(copies_per_head, dim=1)  # (B, target_heads, M, D)

    # 1. 压缩分支
    compression_output, lse_cmp = compression_branch(
        module.scaling, masked_queries_padded, masked_keys, masked_values, mix_config
    )

    # 2. selection 挑选分支
    selection_output = selection_branch(
        module.scaling, masked_queries_padded, masked_keys, masked_values, mix_config, lse_cmp
    )

    # 3. sliding 常规分支
    sliding_output = sliding_window_branch(
        module.scaling, masked_queries_padded, masked_keys, masked_values, mix_config
    )

    # Reduce padded outputs back to original head count.
    # Because of `repeat_interleave`, kernel head [h*N : (h+1)*N] holds N identical
    # copies of original head h's output. We view as (H, copies_per_head) and take
    # the mean over the N copies — they should be numerically identical, so mean is
    # mathematically equivalent to picking any single copy, while providing slight
    # numerical stability against floating-point reduction-order noise.
    compression_output = compression_output.view(B, H, copies_per_head, M, D).mean(dim=2)
    # selection_output / sliding_output layout is (B, M, target_heads, D)
    selection_output = selection_output.view(B, M, H, copies_per_head, D).mean(dim=3)
    sliding_output = sliding_output.view(B, M, H, copies_per_head, D).mean(dim=3)

    # Transpose selection and sliding to (B, H, M, D) to match compression and gates.
    selection_output = selection_output.permute(0, 2, 1, 3).contiguous()
    sliding_output = sliding_output.permute(0, 2, 1, 3).contiguous()

    # 4. 汇总三个分支
    mixed_output = (
        compression_output * g_cmp.unsqueeze(-1)
        + selection_output * g_slc.unsqueeze(-1)
        + sliding_output * g_swa.unsqueeze(-1)
    )

    return mixed_output * visual_mask.unsqueeze(1).unsqueeze(-1).to(mixed_output.dtype)



def compression_branch(
    scaling: float,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    mix_config: MixNSAConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, M, D = query_states.shape
    _, G, N, _ = key_states.shape

    # Mean pool on raw GQA K/V using FLA's mean_pooling kernel (aligned with VideoNSA).
    # Input layout: (B, G, N, D) with head_first=True → kernel will transpose internally.
    # Output: (B, G, num_blocks, D), with the last block correctly averaged over real
    # tokens only (boundary-aware), avoiding zero-padding dilution.
    k_cmp = mean_pooling(key_states, mix_config.block_size, head_first=True)
    v_cmp = mean_pooling(value_states, mix_config.block_size, head_first=True)
    num_blocks = k_cmp.shape[2]

    # NSA-style bidirectional block mask: query can attend to its own block's compressed representation
    def cmp_mask(b, h, q_idx, kv_idx):
        return q_idx <= (kv_idx + 1) * mix_config.block_size - 1



    block_mask = create_block_mask(cmp_mask, B, H, M, num_blocks, device=query_states.device)

    # flex_attention with GQA (aligned with NSA's compression_attention)
    o_cmp, lse_cmp = _flex_attention(
        query_states,
        k_cmp,
        v_cmp,
        block_mask=block_mask,
        enable_gqa=True,
        scale=scaling,
        return_lse=True,
    )
    
    return o_cmp, lse_cmp


def selection_branch(
    scaling: float,
    query_states: torch.Tensor,   # (B, H, M, D)
    key_states: torch.Tensor,     # (B, G, N, D)
    value_states: torch.Tensor,   # (B, G, N, D)
    mix_config: MixNSAConfig,
    lse_cmp: torch.Tensor = None, # (B, H, M)
) -> torch.Tensor:
    B, H, M, D = query_states.shape
    _, G, N, _ = key_states.shape

    # Mean pool K to compressed blocks using FLA's mean_pooling (aligned with VideoNSA).
    # (B, G, N, D) -> (B, G, num_blocks, D), boundary-aware over the last block.
    k_cmp = mean_pooling(key_states, mix_config.block_size, head_first=True)

    # Permute to FLA layout: q (B,H,M,D)->(B,M,H,D), k_cmp (B,G,blocks,D)->(B,blocks,G,D)
    q_permuted = query_states.permute(0, 2, 1, 3).contiguous()
    k_cmp_permuted = k_cmp.permute(0, 2, 1, 3).contiguous()

    # Use FLA's parallel_nsa_topk (aligned with VideoNSA)
    block_indices = parallel_nsa_topk(
        q=q_permuted,           # (B, M, H, D)
        k=k_cmp_permuted,       # (B, num_blocks, G, D)
        lse=lse_cmp.permute(0, 2, 1).contiguous(),  # (B, M, H)
        block_counts=mix_config.block_count,
        block_size=mix_config.block_size,
        scale=scaling,
    )  # -> (B, M, G, T)

    # Permute K/V to FLA layout: (B, G, N, D) -> (B, N, G, D)
    k_permuted = key_states.permute(0, 2, 1, 3).contiguous()
    v_permuted = value_states.permute(0, 2, 1, 3).contiguous()

    out_permuted = selection_attention(
        q_permuted, k_permuted, v_permuted,
        block_indices, mix_config.block_count, mix_config.block_size, scaling,
    )  # -> (B, M, H, D)

    return out_permuted




def sliding_window_branch(
    scaling: float,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    mix_config: MixNSAConfig,
) -> torch.Tensor:
    if mix_config.window_size <= 0:
        return torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[1],
            query_states.shape[3],
            device=query_states.device,
            dtype=query_states.dtype,
        )

    # flash_attn expects (B, M, H, D) layout
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    o = flash_attn_func(
        q, k, v,
        causal=True,
        window_size=(mix_config.window_size - 1, 0),
        softmax_scale=scaling,
    )
    return o  # (B, M, H, D)
