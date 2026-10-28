from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    hidden_size: int = 128
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 32
    max_position: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    qkv_bias: bool = False
    page_size: int = 256  # FlashAttention paged KV requires multiples of 256.

    def __post_init__(self):
        if min(self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError("Require H_q % H_kv == 0 and an even head_dim")
        if self.page_size <= 0 or self.page_size % 256:
            raise ValueError("page_size must be a positive multiple of 256")

    @property
    def mode(self):
        return "MHA" if self.num_heads == self.num_kv_heads else (
            "MQA" if self.num_kv_heads == 1 else "GQA"
        )

    @property
    def q_size(self):
        return self.num_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_kv_heads * self.head_dim
