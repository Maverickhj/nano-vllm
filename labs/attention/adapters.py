"""Test-only adapters; production source and global Context are preserved."""

from contextlib import contextmanager

import torch

from labs.attention.fixtures import load_weights


@contextmanager
def runtime_context(**kwargs):
    from nanovllm.utils.context import get_context, set_context
    previous = get_context()
    # Keep the original tensors: dataclasses.asdict would deep-copy CUDA tensors.
    saved = {name: getattr(previous, name) for name in previous.__dataclass_fields__}
    set_context(**kwargs)
    try:
        yield
    finally:
        set_context(**saved)


class NanoAdapter:
    def __init__(self, config, values, lengths, device, dtype):
        from nanovllm.models.qwen3 import Qwen3Attention
        self.config, self.lengths = config, lengths
        self.device, self.dtype = device, dtype
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            # Match ModelRunner: model weights BF16, explicit RoPE cache FP32.
            with torch.device(device):
                self.model = Qwen3Attention(
                    hidden_size=config.hidden_size, num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
                    max_position=config.max_position, rms_norm_eps=config.rms_norm_eps,
                    qkv_bias=config.qkv_bias, rope_theta=config.rope_theta,
                ).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        load_weights(self.model, values)
        pages = [(n + config.page_size - 1) // config.page_size for n in lengths]
        # Reverse physical order and reserve page 0 as an untouched guard page.
        table = torch.full((len(lengths), max(pages)), -1, dtype=torch.int32, device=device)
        physical = sum(pages)
        for batch, count in enumerate(pages):
            table[batch, :count] = torch.arange(physical, physical - count, -1, device=device)
            physical -= count
        self.block_tables = table
        shape = (sum(pages) + 1, config.page_size, config.num_kv_heads, config.head_dim)
        self.model.attn.k_cache = torch.full(shape, 0.125, device=device, dtype=dtype)
        self.model.attn.v_cache = torch.full(shape, -0.25, device=device, dtype=dtype)
        self.stages = {}
        self.hooks = []
        self._install_hooks()
        self.reset_cache()

    def _install_hooks(self):
        def packed_hook(module, args, output):
            c = self.config
            q, k, v = output.split([c.q_size, c.kv_size, c.kv_size], -1)
            for name, tensor, heads in (("q", q, c.num_heads), ("k", k, c.num_kv_heads),
                                         ("v", v, c.num_kv_heads)):
                self.stages[name + "_proj"] = tensor.reshape(-1, heads, c.head_dim).detach().clone()
            if c.qkv_bias:
                self.stages.update(q_norm=self.stages["q_proj"], k_norm=self.stages["k_proj"])
        self.hooks.append(self.model.qkv_proj.register_forward_hook(packed_hook))
        def capture(name):
            def hook(module, args, output):
                self.stages[name] = output.detach().clone()
            return hook
        if not self.config.qkv_bias:
            for name in ("q_norm", "k_norm"):
                self.hooks.append(getattr(self.model, name).register_forward_hook(capture(name)))
        def rope_hook(module, args, output):
            self.stages.update(q_rope=output[0].detach().clone(), k_rope=output[1].detach().clone())
        self.hooks.append(self.model.rotary_emb.register_forward_hook(rope_hook))
        self.hooks.append(self.model.attn.register_forward_hook(capture("attention")))
        self.hooks.append(self.model.o_proj.register_forward_hook(capture("output")))

    def close(self):
        # RoPE is lru_cached in production; remove hooks so cases cannot leak into each other.
        for hook in self.hooks:
            hook.remove()

    def reset_cache(self):
        self.model.attn.k_cache.fill_(0.125)
        self.model.attn.v_cache.fill_(-0.25)
        self.expected_k = self.model.attn.k_cache.clone()
        self.expected_v = self.model.attn.v_cache.clone()

    @torch.inference_mode()
    def execute(self, hidden, positions, seq_ids, offsets, mode):
        c = self.config
        sizes = [len(x) for x in hidden]
        tables = self.block_tables[seq_ids]
        slots = []
        for row, offset, n in zip(tables, offsets, sizes):
            logical = torch.arange(offset, offset + n, device=self.device)
            slots.append(row[logical // c.page_size].long() * c.page_size + logical % c.page_size)
        def cumulative(lengths):
            return torch.tensor([0] + lengths, dtype=torch.int32, device=self.device).cumsum(0).int()
        visible = [offset + n for offset, n in zip(offsets, sizes)]
        kwargs = dict(is_prefill=mode != "decode", slot_mapping=torch.cat(slots))
        if mode == "decode":
            if any(n != 1 for n in sizes):
                raise ValueError("Standard decode must have one query per active sequence")
            kwargs.update(context_lens=torch.tensor(visible, dtype=torch.int32, device=self.device),
                          block_tables=tables)
        else:
            kwargs.update(cu_seqlens_q=cumulative(sizes), cu_seqlens_k=cumulative(visible),
                          max_seqlen_q=max(sizes), max_seqlen_k=max(visible),
                          block_tables=tables if mode == "chunk" else None)
        self.stages = {}
        with runtime_context(**kwargs):
            self.model(torch.cat(positions), torch.cat(hidden))
        self.stages["attention"] = self.stages["attention"].reshape(-1, c.num_heads, c.head_dim)
        # Independently address every expected cache cell, including untouched tails/guard page.
        token = 0
        for seq, offset, n in zip(seq_ids, offsets, sizes):
            for index in range(n):
                logical = offset + index
                page = self.block_tables[seq, logical // c.page_size].item()
                within = logical % c.page_size
                self.expected_k[page, within] = self.stages["k_rope"][token]
                self.expected_v[page, within] = self.stages["v_proj"][token]
                token += 1
        torch.testing.assert_close(self.model.attn.k_cache, self.expected_k, atol=0, rtol=0)
        torch.testing.assert_close(self.model.attn.v_cache, self.expected_v, atol=0, rtol=0)
        return self.stages

    def cache_report(self):
        c = self.config
        per_token = 2 * c.num_kv_heads * c.head_dim * self.model.attn.k_cache.element_size()
        return {"layout": "paged", "page_size": c.page_size, "bytes_per_token": per_token,
                "logical_bytes": sum(self.lengths) * per_token,
                "allocated_bytes": 2 * self.model.attn.k_cache.numel() *
                                   self.model.attn.k_cache.element_size(),
                "guard_and_padding_unchanged": True}


class StudentAdapter:
    """Contiguous test cache only. Student code computes all attention mathematics."""
    def __init__(self, config, values, lengths, device, dtype):
        from labs.attention.reference.mha import TorchMHA
        from labs.attention.reference.gqa import TorchGQA
        self.config, self.lengths, self.dtype = config, lengths, dtype
        cls = TorchMHA if config.num_heads == config.num_kv_heads else TorchGQA
        self.model = cls(config).to(device=device, dtype=dtype).eval()
        load_weights(self.model, values)
        self.reset_cache()

    def reset_cache(self):
        self.history = [None] * len(self.lengths)

    def close(self):
        """No hooks or external resources to release in the contiguous adapter."""

    @torch.inference_mode()
    def execute(self, hidden, positions, seq_ids, offsets, mode):
        collected = {}
        for x, pos, seq, offset in zip(hidden, positions, seq_ids, offsets):
            q, k, v, stages = self.model.prepare_qkv(pos, x)
            previous = self.history[seq]
            if offset != (0 if previous is None else len(previous[0])):
                raise ValueError("Student contiguous cache requires append-only offsets")
            if previous is not None:
                k, v, key_positions = [torch.cat(pair) for pair in zip(previous, (k, v, pos))]
            else:
                key_positions = pos
            self.history[seq] = (k, v, key_positions)
            heads = self.model.attention(q, k, v, pos, key_positions)
            stages.update(attention=heads, output=self.model.output_projection(heads))
            for name, tensor in stages.items():
                collected.setdefault(name, []).append(tensor.detach().clone())
        return {name: torch.cat(tensors) for name, tensors in collected.items()}

    def cache_report(self):
        per_token = 2 * self.config.num_kv_heads * self.config.head_dim * torch.empty((), dtype=self.dtype).element_size()
        return {"layout": "contiguous_student_adapter", "bytes_per_token": per_token,
                "logical_bytes": sum(self.lengths) * per_token}
