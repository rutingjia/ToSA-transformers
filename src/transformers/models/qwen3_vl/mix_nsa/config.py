from dataclasses import dataclass


@dataclass(frozen=True)
class MixNSAConfig:
    block_size: int
    block_count: int
    window_size: int


def use_mix_nsa(config) -> bool:
    return getattr(config, "attention_mode", "default") == "mix_nsa"


def get_mix_nsa_backend(config) -> str:
    return getattr(config, "mix_nsa_backend", "qwen3_native")


def get_mix_nsa_config(config) -> MixNSAConfig:
    return MixNSAConfig(
        block_size=getattr(config, "mix_nsa_block_size", 64),
        block_count=getattr(config, "mix_nsa_block_count", 16),
        window_size=getattr(config, "mix_nsa_window_size", 256),
    )


def get_visual_token_mask_from_position_ids(multimodal_position_ids):
    if multimodal_position_ids is None:
        return None
    if multimodal_position_ids.ndim != 3 or multimodal_position_ids.shape[0] != 3:
        raise ValueError(
            "`multimodal_position_ids` must have shape `(3, batch_size, sequence_length)` for MixNSA routing."
        )
    temporal_ids, height_ids, width_ids = multimodal_position_ids
    is_text = (temporal_ids == height_ids) & (height_ids == width_ids)
    return ~is_text
