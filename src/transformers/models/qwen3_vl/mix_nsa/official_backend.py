import torch
import torch.nn.functional as F


def _load_official_nsa_func():
    try:
        from ..vendors.videonsa_nsa import nsa_func
    except Exception as exc:
        raise ImportError(
            "The `videonsa_official` MixNSA backend requires the vendored VideoNSA package and its runtime "
            "dependencies (`flash_attn`, `triton`, `fla`, `loguru`)."
        ) from exc
    return nsa_func


def _maybe_pad_heads_for_official_backend(
    query_states: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, callable]:
    num_heads = query_states.shape[1]

    if num_heads == 28:
        zeros_q = torch.zeros(
            query_states.shape[0],
            8,
            query_states.shape[2],
            query_states.shape[3],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        padded_query = torch.cat([query_states, query_states, zeros_q], dim=1)

        zeros_g = torch.zeros(g_cmp.shape[0], g_cmp.shape[1], 8, device=g_cmp.device, dtype=g_cmp.dtype)
        g_cmp = torch.cat([g_cmp, g_cmp, zeros_g], dim=-1)
        g_slc = torch.cat([g_slc, g_slc, zeros_g], dim=-1)
        g_swa = torch.cat([g_swa, g_swa, zeros_g], dim=-1)

        def restore(nsa_output: torch.Tensor) -> torch.Tensor:
            front = nsa_output[:, :, :28, :]
            copied = nsa_output[:, :, 28:56, :]
            return (front + copied) * 0.5

        return padded_query, g_cmp, g_slc, g_swa, restore

    def restore(nsa_output: torch.Tensor) -> torch.Tensor:
        return nsa_output

    return query_states, g_cmp, g_slc, g_swa, restore


def compute_visual_mix_nsa_output_official(
    module,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    visual_mask: torch.Tensor,
    mix_config,
) -> torch.Tensor:
    nsa_func = _load_official_nsa_func()

    bsz, _, q_len, _ = query_states.shape
    zeros = torch.zeros_like(query_states)
    vision_mask_expanded = visual_mask.unsqueeze(1).unsqueeze(3).expand_as(query_states)
    kv_mask_expanded = visual_mask.unsqueeze(1).unsqueeze(3).expand_as(key_states)

    vision_only_q = torch.where(vision_mask_expanded, query_states, zeros)
    vision_only_k = torch.where(kv_mask_expanded, key_states, torch.zeros_like(key_states))
    vision_only_v = torch.where(kv_mask_expanded, value_states, torch.zeros_like(value_states))

    gate_hidden = F.gelu(module.g_proj_1(hidden_states))
    gates = module.g_proj_2(gate_hidden).view(bsz, q_len, module.config.num_attention_heads, 3)
    gate_temperature = getattr(module.config, "mix_nsa_gate_temperature", 1.0)
    gates = gates / gate_temperature

    vision_gate_mask = visual_mask.unsqueeze(2).unsqueeze(3).expand_as(gates)
    vision_only_gates = torch.where(vision_gate_mask, gates, torch.zeros_like(gates))
    g_cmp, g_slc, g_swa = vision_only_gates.sigmoid().unbind(-1)

    query_for_nsa, g_cmp, g_slc, g_swa, restore_fn = _maybe_pad_heads_for_official_backend(
        vision_only_q, g_cmp, g_slc, g_swa
    )

    nsa_output = nsa_func(
        query_for_nsa.transpose(1, 2),
        vision_only_k.transpose(1, 2),
        vision_only_v.transpose(1, 2),
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_count=mix_config.block_count,
        block_size=mix_config.block_size,
        window_size=mix_config.window_size,
        scale=None,
        layer_idx=getattr(module, "layer_idx", None),
    )
    nsa_output = restore_fn(nsa_output).transpose(1, 2)
    return nsa_output * visual_mask.unsqueeze(1).unsqueeze(-1).to(nsa_output.dtype)
