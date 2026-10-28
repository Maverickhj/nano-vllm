"""Reproducible inputs and explicit weights; no checkpoint or tokenizer."""

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from labs.attention.config import AttentionConfig


def weights(config: AttentionConfig, seed: int, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def normal(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)
    result = {
        "qkv_proj.weight": normal((config.q_size + 2 * config.kv_size, config.hidden_size),
                                  config.hidden_size ** -0.5),
        "o_proj.weight": normal((config.hidden_size, config.q_size), config.q_size ** -0.5),
    }
    if config.qkv_bias:
        result["qkv_proj.bias"] = normal((config.q_size + 2 * config.kv_size,), 0.05)
    else:
        result["q_norm.weight"] = 1 + normal((config.head_dim,), 0.1)
        result["k_norm.weight"] = 1 + normal((config.head_dim,), 0.1)
    return result


def inputs(config, lengths, starts, seed, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = [(0.35 * torch.randn(n, config.hidden_size, generator=generator)).to(device, dtype)
              for n in lengths]
    positions = [torch.arange(start, start + n, device=device) for n, start in zip(lengths, starts)]
    return hidden, positions


def load_weights(module, values):
    params = dict(module.named_parameters())
    if params.keys() != values.keys():
        raise AssertionError(f"Parameter mismatch: {params.keys()} vs {values.keys()}")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(values[name])


@contextmanager
def single_rank_group(scratch: Path):
    """Own only the process group we create. All rendezvous files live on NAS."""
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError("The lab requires TP=1")
        yield
        return
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tp1-", dir=scratch) as directory:
        dist.init_process_group(
            "nccl", init_method=Path(directory, "rendezvous").resolve().as_uri(),
            rank=0, world_size=1, timeout=timedelta(seconds=120),
            device_id=torch.device("cuda:0"),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()
