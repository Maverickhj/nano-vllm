from pathlib import Path

import pytest
import torch

from labs.attention.fixtures import single_rank_group


@pytest.fixture(scope="session")
def tp1():
    if not torch.cuda.is_available():
        pytest.skip("nano-vllm baseline needs CUDA and FlashAttention")
    with single_rank_group(Path(__file__).resolve().parents[1] / "results" / ".scratch"):
        yield
