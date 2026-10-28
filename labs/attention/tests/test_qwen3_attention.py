import pytest

from labs.attention.checks import core_case, layer_case


@pytest.mark.parametrize("kv", [8, 2, 1])
def test_qwen3_prefill_and_incremental_decode(tp1, kv):
    result = layer_case("nanovllm", kv, [7, 5])
    assert result["passed"], result


@pytest.mark.parametrize("kv", [8, 2, 1])
def test_core_extreme_scores_and_causal_alignment(tp1, kv):
    result = core_case(kv)
    assert result["passed"], result
