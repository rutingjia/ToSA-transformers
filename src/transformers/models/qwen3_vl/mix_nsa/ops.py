import torch
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_block_mask(visual_seq_len: int, num_blocks: int, block_size: int, device: torch.device) -> torch.Tensor:
    query_block_ids = torch.div(torch.arange(visual_seq_len, device=device), block_size, rounding_mode="floor")
    allowed_blocks = torch.arange(num_blocks, device=device).unsqueeze(0) <= query_block_ids.unsqueeze(-1)
    return allowed_blocks.unsqueeze(0).unsqueeze(0)


def pad_to_multiple(hidden_states: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    seq_len = hidden_states.shape[2]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return hidden_states, 0
    return F.pad(hidden_states, (0, 0, 0, pad_len)), pad_len


def mean_pool_blocks(hidden_states: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    hidden_states, pad_len = pad_to_multiple(hidden_states, block_size)
    batch_size, num_heads, seq_len, head_dim = hidden_states.shape
    pooled = hidden_states.view(batch_size, num_heads, seq_len // block_size, block_size, head_dim).mean(dim=3)
    return pooled, pad_len


def gather_blocks(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_heads, _, head_dim = key_states.shape
    token_offsets = torch.arange(block_size, device=key_states.device)
    token_indices = selected_blocks.unsqueeze(-1) * block_size + token_offsets
    token_indices = token_indices.reshape(batch_size, num_heads, -1)
    gather_index = token_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    gathered_keys = torch.gather(key_states, 2, gather_index)
    gathered_values = torch.gather(value_states, 2, gather_index)
    return gathered_keys, gathered_values, token_indices


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(masked_scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
    return probs / denom
