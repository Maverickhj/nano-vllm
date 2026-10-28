import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from labs.attention import oracle
from labs.attention.config import AttentionConfig
from labs.attention.fixtures import inputs, load_weights, weights
from labs.attention.metrics import compare
from labs.attention.reference.gqa import TorchGQA
from labs.attention.reference.mha import TorchMHA


@pytest.mark.parametrize("kv", [8, 2, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_oracle_matches_sdpa_math_and_fp64(kv, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(142)
    q, k, v = [torch.randn(shape, generator=generator).to(device) for shape in
               [(3, 8, 32), (9, kv, 32), (9, kv, 32)]]
    query_pos, key_pos = torch.arange(26, 29, device=device), torch.arange(20, 29, device=device)
    actual = oracle.attention(q, k, v, query_pos, key_pos)
    high = oracle.attention(q.double(), k.double(), v.double(), query_pos, key_pos)
    mask = key_pos[None, :] <= query_pos[:, None]
    with sdpa_kernel(SDPBackend.MATH):
        sdpa = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.repeat_interleave(8 // kv, 1).transpose(0, 1),
            v.repeat_interleave(8 // kv, 1).transpose(0, 1), attn_mask=mask,
        ).transpose(0, 1)
    torch.testing.assert_close(actual["attention"], high["attention"].float(), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual["attention"], sdpa, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual["probabilities"].sum(-1), torch.ones(8, 3, device=device))
    assert torch.isfinite(high["lse"]).all()


def test_extreme_scores_are_stable_and_no_future_leaks():
    q = torch.tensor([[[1000., 0.]], [[-1000., 0.]]], dtype=torch.float64)
    k = q.clone()
    v = torch.tensor([[[1., 2.]], [[100., 200.]]], dtype=torch.float64)
    pos = torch.arange(2)
    result = oracle.attention(q, k, v, pos, pos)
    assert torch.isfinite(result["attention"]).all()
    assert torch.isfinite(result["lse"]).all()
    torch.testing.assert_close(result["attention"], v, atol=0, rtol=0)
    assert result["probabilities"][0, 0, 1] == 0


@pytest.mark.parametrize("mutation", ["position", "mask", "kv_head_mapping"])
def test_harness_detects_deliberate_errors(mutation):
    config = AttentionConfig(num_kv_heads=2)
    values = weights(config, 7, "cpu", torch.float64)
    hidden, pos = inputs(config, [7], [17], 8, "cpu", torch.float64)
    reference = oracle.layer(config, values, hidden[0], pos[0])
    q, k, v = [reference[name] for name in ("q_rope", "k_rope", "v_proj")]
    qp, kp = pos[0][-2:].clone(), pos[0].clone()
    if mutation == "position":
        wrong_pos = pos[0].clone()
        wrong_pos[-1] += 13
        wrong = oracle.layer(config, values, hidden[0], wrong_pos)["attention"][-2:]
    else:
        if mutation == "mask":
            qp = kp[:2]  # Wrong top-left alignment for Q_len < L.
        else:
            k, v = k.flip(1), v.flip(1)
        wrong = oracle.attention(q[-2:], k, v, qp, kp)["attention"]
    assert not compare(wrong, reference["attention"][-2:], "attention", "bfloat16")["passed"]


@pytest.mark.parametrize("kv", [8, 2, 1])
def test_student_skeleton_is_explicitly_unfinished(kv):
    config = AttentionConfig(num_kv_heads=kv)
    module = (TorchMHA if kv == 8 else TorchGQA)(config)
    load_weights(module, weights(config, 7, "cpu", torch.float32))
    with pytest.raises(NotImplementedError, match="QKV"):
        module(torch.arange(2), torch.randn(2, config.hidden_size))


def test_metrics_reject_nonfinite_shape_and_corruption():
    expected = torch.ones(7, 8, 32)
    assert not compare(expected * 2, expected, "output", "bfloat16")["passed"]
    with pytest.raises(AssertionError, match="shape"):
        compare(expected[:1], expected, "output", "bfloat16")
    with pytest.raises(AssertionError, match="non-finite"):
        compare(expected * torch.nan, expected, "output", "bfloat16")
