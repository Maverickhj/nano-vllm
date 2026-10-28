"""Independent, fully implemented test Oracle. Not the student exercise.

FP64 by default. Physical KV repetition is allowed ONLY in this small Oracle.
The production FlashAttention implementation is never called here.
"""

import torch
import torch.nn.functional as F


def attention(q, k, v, query_positions, key_positions):
    """[Q,Hq,D], [L,Hkv,D]; causal alignment uses absolute positions."""
    dtype = q.dtype
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    q, k, v = [t.transpose(0, 1) for t in (q, k, v)]
    mask = key_positions[None, :] <= query_positions[:, None]
    if not mask.any(dim=-1).all():
        raise ValueError("Every query must have at least one visible key")
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = scores.softmax(dim=-1)
    return {
        "attention": (probabilities @ v).transpose(0, 1).to(dtype),
        "scores": scores, "probabilities": probabilities,
        "lse": scores.logsumexp(dim=-1),
    }


def _rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def _rope(x, positions, theta):
    inv_freq = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device,
                                       dtype=x.dtype) / x.shape[-1])
    angle = positions.to(x.dtype)[:, None] * inv_freq[None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


def layer(config, values, hidden, positions, dtype=torch.float64):
    """High precision end-to-end expectation, including observable intermediates."""
    values = {name: tensor.to(dtype) for name, tensor in values.items()}
    packed = F.linear(hidden.to(dtype), values["qkv_proj.weight"], values.get("qkv_proj.bias"))
    q, k, v = packed.split([config.q_size, config.kv_size, config.kv_size], dim=-1)
    q = q.reshape(-1, config.num_heads, config.head_dim)
    k, v = [t.reshape(-1, config.num_kv_heads, config.head_dim) for t in (k, v)]
    stages = {"q_proj": q, "k_proj": k, "v_proj": v}
    if not config.qkv_bias:
        q = _rms_norm(q, values["q_norm.weight"], config.rms_norm_eps)
        k = _rms_norm(k, values["k_norm.weight"], config.rms_norm_eps)
    stages.update(q_norm=q, k_norm=k)
    q, k = [_rope(t, positions, config.rope_theta) for t in (q, k)]
    stages.update(q_rope=q, k_rope=k)
    stages.update(attention(q, k, v, positions, positions))
    stages["output"] = F.linear(stages["attention"].flatten(1), values["o_proj.weight"])
    return stages
