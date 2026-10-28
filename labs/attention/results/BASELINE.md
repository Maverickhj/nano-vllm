# 原版 nano-vllm 正确性基线（2026-10-28）

本报告记录初始空骨架建立时的验证结果。后续 `reference/` 中的手写进度尚未重新验收，`student-unimplemented.json` 也是初始骨架的历史快照；下一次运行验证将使用 Megatron-Bridge 镜像。

结果：**28/28 组 GPU baseline 通过，14/14 项 Oracle/骨架/故障注入测试通过**。学生骨架尚未实现数学计算，单独的 student smoke 明确返回 `not_implemented` 和退出码 1。

被测代码为原版 `Qwen3Attention` / `Attention.forward` / `store_kvcache`。`nanovllm/` 无修改。对照是独立 FP64 数学 Oracle；FP32 Oracle 另外与 SDPA math backend 对照。数据全部为固定 seed 的合成张量，不加载任何 checkpoint。

## 环境与范围

- Git HEAD：`8ca2df7a3b4ed6da5a1907b86b7d9b2e12779a3b`；工作树 dirty，实验代码尚未提交。
- minimind 容器，Python 3.12.3，NVIDIA L20Y，SM 9.0，单卡 TP=1。
- Torch `2.12.0a0+0291f960b6.nv26.4.48445190`，CUDA 13.2，内核 driver 535.129.03。
- Triton `3.6.0+git5d72932fc5.nv26.4`，FlashAttention `2.7.4.post1+git5231d95fe1.45164347`，Transformers 5.8.1。
- BF16，seed `20261028`，`hidden_size=128,H_q=8,head_dim=32`，page size 256。
- 本次设置 `TORCHDYNAMO_DISABLE=1`，关闭 Norm/RoPE 的 `torch.compile` 包装；原版数学代码、Triton cache writer、FlashAttention GPU kernels 照常执行。编译路径未纳入此次验收。
- 测量范围仅为正确性，没有计时、benchmark 或实际 HBM 流量结论。

## 最终输出误差

每列分别取该模式全部 full prefill、prefix、chunk、incremental decode 对 FP64 Oracle 的最坏值。p99 列是各次比较的 p99 最大值，不是跨所有用例重新计算的全局分位数。

| 模式 | H_kv | max_abs_error | max p99_abs_error | max relative_L2 | BF16 KV bytes/token/layer |
|---|---:|---:|---:|---:|---:|
| MHA | 8 | 0.00409320 | 0.00358583 | 0.00651986 | 1024 |
| GQA | 2 | 0.00423726 | 0.00296925 | 0.00695985 | 256 |
| MQA | 1 | 0.00418764 | 0.00315301 | 0.00657358 | 128 |

本次常规及 bias 用例中，逐 token decode 与同一实例 full causal prefill 的最终输出最大绝对差均为 **0**；这只是此次形状与执行路径的观测，不将 bitwise 相等作为通用验收条件。

分层检查同时覆盖 Q/K/V projection、Q/K Norm、Q/K RoPE、head output、output projection。固定容差在运行前写入 `metrics.py`，未根据结果调整。完整原始误差、各 stage shape 和每项容差见 [`baseline-bf16.json`](baseline-bf16.json)。

## 覆盖证据

- 18 组常规层比较：`B=1/2,S=2/7/33,H_kv=8/2/1`，每组同时执行 full prefill 和从空缓存开始的逐 token decode。
- 3 组不等长分页 prefix/chunk：总长度 `[258,259]`，prefix `[255,258]`，非零起始位置 `[5,71]`，新 chunk `[3,1]`；另检查一个 batch 中 `context_lens=[258,259]` 的 decode。
- 3 组 QKV bias 路径：长度 `[7,5]`，按原版规则跳过 Q/K Norm。
- 3 组原版 attention core：极端 scores、`Q_len=3<L=7` 的 bottom-aligned causal mask，以及扰动未来 K/V 后早期输出不变。
- 1 组实际 Triton cache writer：`slot_mapping=[257,-1,255]`，检查跳过写入、跨页槽位和全部未写缓存区域。
- 14 项 pytest：CPU/GPU FP32 vs FP64/SDPA、极端分数、故意错误的 Position/Mask/KV head mapping、学生 TODO 状态、误差检查的 shape/NaN/数值破坏检测。

所有原版层测试都检查非顺序物理页的缓存内容、未写尾部及 guard page。容量表为逻辑 KV payload；JSON 另列分页分配容量，包含 padding 与 guard page，不把理论容量当作实际内存读流量。

## 复现

在容器内 `/opt/nano-vllm` 执行：

```bash
TORCHDYNAMO_DISABLE=1 \
TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
TRITON_CACHE_DIR=/data/cache/triton \
/opt/venv/bin/python -m labs.attention.run \
  --backend nanovllm --suite full --output labs/attention/results/baseline-bf16.json

TORCHDYNAMO_DISABLE=1 /opt/venv/bin/python -m pytest labs/attention/tests/test_oracle.py -q
```

学生入口的实际状态记录见 [`student-unimplemented.json`](student-unimplemented.json)。共享前缀生命周期、MLA、性能 sweep 和编译路径均不属于此次通过结论。
