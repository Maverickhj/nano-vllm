"""Shared assertions for the CLI and automated tests."""

from dataclasses import asdict

import torch

from labs.attention import oracle
from labs.attention.adapters import NanoAdapter, StudentAdapter, runtime_context
from labs.attention.config import AttentionConfig
from labs.attention.fixtures import inputs, weights
from labs.attention.metrics import compare


STAGES = ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "q_rope", "k_rope",
          "attention", "output")


def _comparisons(actual, references, seq_ids, spans, dtype):
    return {
        name: compare(actual[name], torch.cat([
            references[seq][name][start:stop]
            for seq, (start, stop) in zip(seq_ids, spans)
        ]), name, dtype)
        for name in STAGES
    }


def _merge(destination, source):
    for stage, metrics in source.items():
        if stage not in destination:
            destination[stage] = dict(metrics)
        else:
            for name in ("max_abs_error", "p99_abs_error", "relative_L2"):
                destination[stage][name] = max(destination[stage][name], metrics[name])
            destination[stage]["passed"] &= metrics["passed"]


@torch.inference_mode()
def layer_case(backend, num_kv_heads, lengths, starts=None, seed=20261028,
               dtype_name="bfloat16", prefix_lengths=None, qkv_bias=False):
    """Full prefill vs FP64, followed by incremental decode or cached chunked prefill."""
    dtype, device = getattr(torch, dtype_name), torch.device("cuda:0")
    config = AttentionConfig(num_kv_heads=num_kv_heads, qkv_bias=qkv_bias)
    starts = starts or [17 + 29 * i for i in range(len(lengths))]
    values = weights(config, seed, device, dtype)
    hidden, positions = inputs(config, lengths, starts, seed + 1, device, dtype)
    references = [oracle.layer(config, values, x, p) for x, p in zip(hidden, positions)]
    cls = NanoAdapter if backend == "nanovllm" else StudentAdapter
    adapter = cls(config, values, lengths, device, dtype)
    result = {"kind": "layer" if prefix_lengths is None else "cached_chunk",
              "backend": backend, "config": asdict(config), "mode": config.mode,
              "lengths": lengths, "starts": starts, "seed": seed, "dtype": dtype_name,
              "prefix_lengths": prefix_lengths}
    ids = list(range(len(lengths)))
    try:
        full = adapter.execute(hidden, positions, ids, [0] * len(ids), "prefill")
        result["prefill"] = _comparisons(full, references, ids, [(0, n) for n in lengths], dtype_name)
        full_output = full["output"].clone()
        result["qkv_head_layout"] = {name: list(full[name].shape)
                                     for name in ("q_proj", "k_proj", "v_proj")}
        adapter.reset_cache()
        if prefix_lengths is None:
            incremental = [torch.empty_like(x) for x in hidden]
            result["decode"] = {}
            for token in range(max(lengths)):
                active = [seq for seq, n in enumerate(lengths) if token < n]
                chunks = [x[token:token + 1] for seq, x in enumerate(hidden) if seq in active]
                pos = [positions[seq][token:token + 1] for seq in active]
                stages = adapter.execute(chunks, pos, active, [token] * len(active), "decode")
                _merge(result["decode"], _comparisons(
                    stages, references, active, [(token, token + 1)] * len(active), dtype_name))
                for row, seq in enumerate(active):
                    incremental[seq][token] = stages["output"][row]
            result["decode_vs_prefill"] = {
                "output": compare(torch.cat(incremental), full_output, "output", dtype_name)
            }
        else:
            prefix = adapter.execute(
                [x[:n] for x, n in zip(hidden, prefix_lengths)],
                [p[:n] for p, n in zip(positions, prefix_lengths)], ids,
                [0] * len(ids), "prefill")
            result["prefix"] = _comparisons(prefix, references, ids,
                                              [(0, n) for n in prefix_lengths], dtype_name)
            chunks = adapter.execute(
                [x[n:] for x, n in zip(hidden, prefix_lengths)],
                [p[n:] for p, n in zip(positions, prefix_lengths)],
                ids, prefix_lengths, "chunk")
            spans = list(zip(prefix_lengths, lengths))
            result["chunk"] = _comparisons(chunks, references, ids, spans, dtype_name)
            # Exercise different context_lens in one paged decode call as well.
            adapter.reset_cache()
            histories = [n - 1 for n in lengths]
            adapter.execute([x[:-1] for x in hidden], [p[:-1] for p in positions],
                            ids, [0] * len(ids), "prefill")
            decoded = adapter.execute([x[-1:] for x in hidden], [p[-1:] for p in positions],
                                      ids, histories, "decode")
            result["unequal_decode"] = _comparisons(decoded, references, ids,
                                                     [(n - 1, n) for n in lengths], dtype_name)
        result["cache"] = adapter.cache_report()
    finally:
        adapter.close()
    groups = ("prefill", "decode", "decode_vs_prefill", "prefix", "chunk", "unequal_decode")
    result["passed"] = all(metric["passed"] for group in groups
                           for metric in result.get(group, {}).values())
    return result


@torch.inference_mode()
def core_case(num_kv_heads, dtype_name="bfloat16", seed=20261028):
    """Real Attention.forward: extreme scores, bottom-aligned mask, future leakage."""
    from nanovllm.layers.attention import Attention
    device, dtype = "cuda:0", getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(seed)
    q, k, v = [(torch.randn(shape, generator=generator) * scale).to(device, dtype)
               for shape, scale in [((7, 8, 32), 32), ((7, num_kv_heads, 32), 32),
                                    ((7, num_kv_heads, 32), 1)]]
    module = Attention(8, 32, 32 ** -0.5, num_kv_heads)
    def run(query, key, value):
        nq, nk = len(query), len(key)
        with runtime_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, nq], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, nk], device=device, dtype=torch.int32),
            max_seqlen_q=nq, max_seqlen_k=nk,
        ):
            return module(query, key, value).clone()
    output = run(q, k, v)
    pos = torch.arange(7, device=device)
    expected = oracle.attention(q.double(), k.double(), v.double(), pos, pos)["attention"]
    extreme = compare(output, expected, "attention", dtype_name)
    chunk = run(q[-3:].contiguous(), k, v)
    expected_chunk = oracle.attention(q[-3:].double(), k.double(), v.double(), pos[-3:], pos)["attention"]
    aligned = compare(chunk, expected_chunk, "attention", dtype_name)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[4:] *= -10
    changed_v[4:] += 100
    changed = run(q, changed_k, changed_v)
    torch.testing.assert_close(output[:4], changed[:4], atol=0, rtol=0)
    return {"kind": "core_extreme_mask", "num_kv_heads": num_kv_heads,
            "dtype": dtype_name, "seed": seed, "extreme_scores": extreme,
            "bottom_aligned_chunk": aligned, "no_future_leakage": True,
            "passed": extreme["passed"] and aligned["passed"]}


@torch.inference_mode()
def skipped_slot_case():
    """Check the actual Triton writer, including -1 and untouched cache cells."""
    from nanovllm.layers.attention import store_kvcache
    key = torch.arange(3 * 2 * 32, device="cuda", dtype=torch.float32).reshape(3, 2, 32).bfloat16()
    value = -key
    k_cache = torch.full((2, 256, 2, 32), -7, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.full_like(k_cache, 9)
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    expected_k[1, 1], expected_v[1, 1] = key[0], value[0]
    expected_k[0, 255], expected_v[0, 255] = key[2], value[2]
    store_kvcache(key, value, k_cache, v_cache,
                  torch.tensor([257, -1, 255], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(k_cache, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, expected_v, atol=0, rtol=0)
    return {"kind": "slot_mapping_minus_one", "passed": True}
