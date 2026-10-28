"""GQA/MQA use the same student TODOs, with fewer stored K/V heads."""

from labs.attention.config import AttentionConfig
from labs.attention.reference.mha import TorchAttention


class TorchGQA(TorchAttention):
    """Fill the shared TODO methods in mha.py; this constructor is complete."""

    def __init__(self, config: AttentionConfig):
        if config.num_kv_heads >= config.num_heads:
            raise ValueError("GQA requires H_kv < H_q (H_kv=1 is MQA)")
        super().__init__(config)
