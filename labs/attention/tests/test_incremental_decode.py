import pytest

from labs.attention.checks import layer_case, skipped_slot_case


@pytest.mark.parametrize("kv", [8, 2, 1])
def test_unequal_context_and_page_boundary(tp1, kv):
    result = layer_case("nanovllm", kv, [258, 259], starts=[5, 71], prefix_lengths=[255, 258])
    assert result["passed"], result


def test_cache_skips_negative_slot(tp1):
    assert skipped_slot_case()["passed"]
