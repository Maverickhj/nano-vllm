"""Pure PyTorch learning skeleton with Qwen3-compatible parameter names/layouts.

No FlashAttention, Triton, distributed state, or lab Oracle is used here.
Implement the TODO methods yourself, then run the harness with --backend student.
"""

import torch
from torch import nn

from labs.attention.config import AttentionConfig


class TorchAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.qkv_proj = nn.Linear(
            config.hidden_size, config.q_size + 2 * config.kv_size,
            bias=config.qkv_bias,
        )
        self.o_proj = nn.Linear(config.q_size, config.hidden_size, bias=False)
        if not config.qkv_bias:
            # Match Qwen3's per-head, learned Q/K RMSNorm weights.
            self.q_norm = nn.ParameterDict({"weight": nn.Parameter(torch.ones(config.head_dim))})
            self.k_norm = nn.ParameterDict({"weight": nn.Parameter(torch.ones(config.head_dim))})

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """[T, hidden_size] -> Q[T,H_q,D], K/V[T,H_kv,D].

        TODO: packed QKV linear projection, split, and head layout.
        Weight layout is [out_features, in_features], as in nn.Linear.
        """
        # TODO: Apply qkv_proj, split Q/K/V, and reshape their head dimensions.

        assert hidden_states.shape(-1) == hidden_size

        qkv_proj = torch.matmul(hidden_states, self.qkv_proj)
        q_proj, k_proj, v_proj = qkv_proj.split([config.q_size, config.kv_size, config.kv_size], dim=-1)
        q_proj = q_proj.view(-1, config.num_heads, config.head_size)
        k_proj = k_proj.view(-1, config.num_kv_heads, config.head_size)
        v_proj = v_proj.view(-1, config.num_kv_heads, config.head_size)

        return q_proj, k_proj, v_proj

        # raise NotImplementedError("TODO: implement QKV projection and head reshaping")

    def normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """TODO: per-head Q/K RMSNorm; bypass when config.qkv_bias is True.

        Match layers/layernorm.py: accumulate variance in FP32, cast normalized
        values back to the input dtype, then multiply by the learned weight.
        """
        # TODO: Implement per-head Q/K RMSNorm and the qkv_bias bypass.

        if self.q_norm:
            q = torch.matmul(q, self.q_norm)
            k = torch.matmul(k, self.k_norm)
        return q, k

        # raise NotImplementedError("TODO: implement Q/K RMSNorm")

    def apply_rope(self, positions: torch.Tensor, q: torch.Tensor,
                   k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """TODO: full-head, split-half (not adjacent-pair) RoPE.

        positions[T] are absolute token positions, including any prefix offset.
        Match layers/rotary_embedding.py's FP32 trigonometry/cast boundaries.
        """
        # TODO: Build position-dependent cos/sin and apply split-half RoPE to Q/K.



        # raise NotImplementedError("TODO: implement split-half RoPE")

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  query_positions: torch.Tensor, key_positions: torch.Tensor) -> torch.Tensor:
        """Q[Q,H_q,D], K/V[L,H_kv,D] -> [Q,H_q,D].

        TODO: map query heads to their KV groups, scaled scores, absolute-position
        causal mask, stable softmax, and weighted V aggregation. The scale is
        D**-0.5. Q may be shorter than L; a top-left tril(Q,L) is incorrect.
        History K/V are already normalized/rotated: do not transform them again.
        Keep KV stored with H_kv heads. This is a correctness exercise, not a
        production kernel or evidence of reduced memory traffic.
        """
        # TODO: Map each query head to its KV head for MHA/GQA/MQA.
        # TODO: Compute QK scores and apply the head_dim**-0.5 scale.
        # TODO: Mask invisible keys using absolute query/key positions.
        # TODO: Compute numerically stable softmax over visible keys.
        # TODO: Aggregate V and return [Q, H_q, D] head outputs.
        raise NotImplementedError("TODO: implement attention scores/mask/softmax/aggregation")

    def output_projection(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """TODO: [T,H_q,D] -> [T,hidden_size] using o_proj."""
        # TODO: Merge the query heads and apply o_proj.
        raise NotImplementedError("TODO: implement output projection")

    def prepare_qkv(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """Wiring only. The harness observes each stage after TODOs are filled."""
        q, k, v = self.project_qkv(hidden_states)
        stages = {name: tensor.detach().clone() for name, tensor in
                  (("q_proj", q), ("k_proj", k), ("v_proj", v))}
        q, k = self.normalize_qk(q, k)
        stages.update(q_norm=q.detach().clone(), k_norm=k.detach().clone())
        q, k = self.apply_rope(positions, q, k)
        stages.update(q_rope=q.detach().clone(), k_rope=k.detach().clone())
        return q, k, v, stages

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """Single-sequence full causal prefill. Incremental wiring lives in the adapter."""
        q, k, v, _ = self.prepare_qkv(positions, hidden_states)
        heads = self.attention(q, k, v, positions, positions)
        return self.output_projection(heads)


class TorchMHA(TorchAttention):
    def __init__(self, config: AttentionConfig):
        if config.num_kv_heads != config.num_heads:
            raise ValueError("MHA requires H_kv == H_q")
        super().__init__(config)
