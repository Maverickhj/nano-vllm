# MLA Learning Lab：基于 nano-vllm 的 Attention 演进与性能实验方案

> **2026-10-27 / 设计稿**。仅规划，尚未实现 MLA，也尚未运行新的正确性或性能实验。
>
> 进度与验收：[`ROADMAP.md`](../../ROADMAP.md)。理论来源：[AI-Infra / MLA 计算强度分析基线](https://github.com/Maverickhj/AI-Infra/blob/main/01_Inbox/MLA%20%E8%AE%A1%E7%AE%97%E5%BC%BA%E5%BA%A6%E5%88%86%E6%9E%90%E5%9F%BA%E7%BA%BF.md)。

## 0. 数学符号表

按 **row-vector** 记法，与 M a v/AI-Infra MLA 基线统一；代码里按实际张量布局显式注明 batch/heads 轴。

| 符号 | 说明 |
|---|---|
| `B` | 活跃序列的 batch size |
| `S` | 无前缀 Prefill 的序列长度 |
| `Q_len` | 本次执行的 query token 数；标准 Decode 每条序列为 1 |
| `L` | 可见 KV token 数，包括当前正在写入的 token |
| `d` | Hidden Size |
| `H_q` | Query Head 数 |
| `H_kv` | 传统 MHA/GQA/MQA 的 KV Head 数 |
| `d_h` | 传统 Attention 的 Q/K/V 单头宽度（选相同 head_dim 的教学配置） |
| `d_c` | MLA 压缩 KV latent 维度 |
| `d_c'` | MLA 压缩 Q 维度 |
| `d_k` | MLA 内容 Q/K Head 维度 |
| `d_r` | MLA 解耦 RoPE 分支宽度（所有 K Heads 共享位置分支） |
| `d_v` | MLA 每个 Head 的展开 V 维度 |
| `p=d_k+d_r` | MLA 展开 Q/K 的 Head 维度 |
| `p_a=d_c+d_r` | MLA 吸收 Q/K 宽度（不参与重算 Softmax scale） |
| `b` | 一个 Cache 元素的字节数 |
| `c_i` | 第 `i` 个 token 经过降维与 RMSNorm 的 KV latent，宽度 `d_c` |
| `c'_t` | 第 `t` 个 token 经过降维与 RMSNorm 的 Q latent，宽度 `d_c'` |
| `k_{i,R}` | token `i` 的共享 RoPE K 位置分支，宽度 `d_r` |
| `W` | 各投影矩阵；不同分支与 Head 使用不同下标 |
| `α_{t,i}^{(s)}` | 第 `s` 个 Head 的 query `t` 对 KV `i` 的 Softmax 权重 |

公式只用于明确定义/正确性；计算强度与访存分析直接以 AI-Infra 基线文档为主，避免维护两套相互漂移的公式。

## 1. 定位：不造第二个 nano-vllm

### 核心目标

1. **理解与复用 MHA/GQA/MQA：** nano-vllm 现有代码已经通过 `Qwen3Attention(num_heads, num_kv_heads)` 实现它们的核心路径。学习阶段只新增测试、Oracle、Benchmark，不重新写生产级 Attention。
2. **自己实现 MLA 中当前仓库没有的部分：** Expanded Reference → Absorbed Reference → 压缩 KV Cache → Triton kernels → nano-vllm Backend。
3. **验证正确性优先：** 无真实 prompt、tokenizer、模型权重、生成文本；采用固定 seed 的合成 Hidden States、Weights、Positions、Cache Metadata，要求逐层可对照。
4. **用性能回答理论问题：** 区分 MHA/GQA 的 KV Head 数下降、MLA 的 latent 压缩/吸收、Head 间跨 Cache 读取复用，以及 Prefill/Decode 的不同瓶颈。
5. **利用真实 Runtime，但不让 Runtime 主导早期学习：** 只有核心数学稳定后才改 Cache 分配、Page Metadata 和调度。

### 明确不做（本期）

- 不重写 MHA/GQA/MQA 主算子、不重新开发一套独立推理引擎。
- 不要求 Text In/Text Out、Tokenizer、预训练 Checkpoint、Sampling 或端到端质量指标。
- 不先做 TP/EP/CP、多模态模型、反向传播、RL 或 FP8 量化；它们是后续独立扩展。
- 不将不同模型结构随机初始化后的最终输出要求数值相等，也不把数学等价误判为逐 bit 相等。

## 2. 基于现有 Fork 的代码审计与修改边界

| 当前源码 | 已有能力 | 学习/扩展策略 |
|---|---|---|
| `nanovllm/models/qwen3.py` 的 `Qwen3Attention` | 组合 QKV 投影、Q/K Norm、RoPE、Attention、输出投影；`num_kv_heads` 可调 | **直接复用**：`H_q=8, H_kv=8/2/1` 对应 MHA/GQA/MQA |
| `nanovllm/layers/attention.py` | `store_kvcache`、FlashAttention varlen Prefill、`flash_attn_with_kvcache` Decode | 先黑盒测试；MLA 走独立 Backend，不破坏原路径 |
| `nanovllm/utils/context.py` | `cu_seqlens`、`slot_mapping`、`context_lens`、`block_tables` | 将其作为后期 Adapter 的 Metadata Contract |
| `nanovllm/engine/model_runner.py` | 初始化模型、分配传统二维 K/V Cache、Prefill/Decode、CUDA Graph | M5 再引入支持压缩 Cache 的按层分配策略 |
| `nanovllm/engine/block_manager.py` | KV Page 与 Prefix Cache 生命周期 | 初期复用块管理语义；后期适配 MLA 的实际缓存布局 |
| `nanovllm/layers/linear.py` / `layernorm.py` | Linear、RMSNorm | MLA 投影优先复用 |
| `nanovllm/layers/rotary_embedding.py` | Q/K RoPE（当前实现要求 `rotary_dim == head_size`） | MLA 只对 `d_r` 分支旋转，需小范围适配并测试布局，不强行传入不兼容的 Head 宽度 |

**当前的结构性缺口：** `ModelRunner.allocate_kv_cache` 使用每层传统 K/V 两份 Cache（逻辑大小 `2 × blocks × block_size × H_kv × d_h`）。MLA 压缩缓存应保存 `[c_i, k_{i,R}]`，不能仅替换 FlashAttention 调用而沿用旧内存预算和写入接口。

**测试环境警告：** `Qwen3Attention` / `LinearBase` 在初始化时调用 `torch.distributed.get_world_size/get_rank`，因此 GPU 组件测试先建立 TP=1 的进程组；并显式初始化这些模块中 `torch.empty` 创建的权重。不要直接启动需要真实 checkpoint 的 `ModelRunner` 来完成 M0～M3。

当前代码要求 `H_q` 与 `H_kv` 都能被 TP size 整除，因此 `H_kv=1` 的 MQA 首先限定 TP=1；任何 TP>1 支持需要单独设计 Head 复制/映射方案。

## 3. 分层实现：Reuse → Oracle → MLA → Runtime

```text
nano-vllm 已有 Qwen3Attention + Attention
       │
       ├── MHA / GQA / MQA，配置 H_kv 改变（不重写）
       │       ├── PyTorch 数学 Oracle（仅用于测试）
       │       └── synthetic-input correctness + benchmark
       │
       └── MLA 实验路径（自行开发）
               ├── Expanded MLA（独立 PyTorch Reference）
               ├── Absorbed MLA（独立 PyTorch Reference）
               ├── 压缩 KV Cache / token-by-token Decode
               ├── Triton Decode (naive → tiled → optimized)
               ├── Triton Prefill（独立研究分支）
               └── nano-vllm MLA Backend + Paged KV
                       └── FlashInfer / 兼容的 Dense FlashMLA 对照
```

### 3.1 MHA/GQA/MQA：只做 Harness，不写第二套实现

- 生产被测体：真实的 `Qwen3Attention` 和 `Attention.forward`；使用合成 Hidden States/Positions 与固定初始化权重；确保 Q/K Norm、Rotary、输出投影与现有代码一致。
- 数学 Oracle：PyTorch `scaled_dot_product_attention` 的 math backend，或显式 `repeat_interleave` 的小规模测试路径。**物理重复 KV 只允许在 Oracle 里**；不能把它作为证明生产路径省访存的依据。
- 每种配置用独立正确性测试；若需测试“GQA/MQA 与 MHA 等价”，必须构造 KV 权重共享/重复的特殊 MHA 对照，不能拿不同随机权重强行比较输出。
- Cache 理论容量（每 token、每 layer，忽略 metadata）：

$$
D_{\mathrm{KV,traditional}}=2bH_{\mathrm{kv}}d_h.
$$

- 对外可观测：每层 Q/K/V Head Layout、最终 Attention Output、Cache 占用和 Core 延迟；可选 `scores/LSE` 通过数学 Oracle 或调试接口获得，不要求 FlashAttention 暴露所有中间矩阵。

**M0 验收：** TP=1，固定 seed，`H_q=8,H_kv=8/2/1` 三组测试全部有数值报告；完全不依赖真实模型文本输入或 checkpoint。

### 3.2 MLA Expanded Reference：先规定数学

按 AI-Infra 基线的行向量记法：

$$
\begin{aligned}
c'_t &= \operatorname{RMSNorm}(x_tW_{DQ}),\\
c_i &= \operatorname{RMSNorm}(x_iW_{DKV}),\\
q_{t,C}^{(s)} &= c'_tW_{QC}^{(s)}, \qquad
q_{t,R}^{(s)} = \operatorname{RoPE}_t(c'_tW_{QR}^{(s)}),\\
k_{i,C}^{(s)} &= c_iW_{KC}^{(s)}, \qquad
k_{i,R} = \operatorname{RoPE}_i(x_iW_{KR}),\\
v_i^{(s)} &= c_iW_V^{(s)}.
\end{aligned}
$$

每个 Head 将内容与位置分数相加，再按原始 `p=d_k+d_r` 缩放；`Causal Mask → Stable Softmax → Weighted V → Output Projection` 的算子边界在测试中分别记录。该路径先使用普通 PyTorch 算子，**不追求快**。

### 3.3 MLA Absorbed Reference：验证数学等价

内容 K 吸收到 Query：

$$
(q_{t,C}^{(s)}W_{KC}^{(s)\top})c_i^\top
=q_{t,C}^{(s)}(c_iW_{KC}^{(s)})^\top.
$$

V 投影移到聚合之后：

$$
\left(\sum_i\alpha_{t,i}^{(s)}c_i\right)W_V^{(s)}
=\sum_i\alpha_{t,i}^{(s)}(c_iW_V^{(s)}).
$$

压缩 Cache **仅**存 `[c_i, k_{i,R}]`；它的每 token 容量为：

$$
D_{\mathrm{KV,MLA}}=b(d_c+d_r).
$$

吸收后的 Q/K 维度是 `p_a=d_c+d_r`，但分数缩放仍为 `1/\sqrt{d_k+d_r}`，**不能**使用 `1/\sqrt{p_a}`。位置分支独立 RoPE，不允许把可旋转内容 K 的操作错误地吸收到 Query 侧。

先按原权重分别执行 `Q_C W_{KC}^\top`、后置 `u W_V`；预先合并复合权重属于后续可选优化，必须重算 FLOPs 和权重带宽模型。

**M3 验收：** 两路径在相同输入、权重、Position、Mask 下逐项对齐 scores、Softmax/LSE、Head Outputs、最终输出；缓存占用可根据 Tensor 实际元素数核对。

### 3.4 Decode / Prefill 与 Cache Contract

- Prefill（无历史前缀）`Q_len=S`，对角线在内的因果 Attention；Decode（已有前缀）`Q_len=1, L=history+1`，历史 KV 不得重复 Projection/RoPE。
- 验证 `full_causal_prefill(x)` 的末位置与同一模型从空 Cache 逐 token Decode 的末位置输出一致（考虑浮点舍入误差）。
- 对 `Q_len < L` 的 Chunked Prefill / 验证场景，使用**绝对位置或正确的 right/bottom-aligned causal mask**；不能直接使用误对齐的 `tril(Q_len,L)`。
- 先连续压缩 Cache，再支持 `slot_mapping=-1`、Page 非对齐尾块、`block_tables` 和不同 `context_lens`。Prefix-sharing 的生命周期需与既有 BlockManager 协议匹配。
- 将「Cache 已含当前 token 的写入」与「Attention 对多少个可见 token 读取」区别计数，防止重复写入或者 off-by-one。

## 4. Correctness Harness 设计

建议独立目录 `labs/attention/`，但 **不要复制 nano-vllm 的生产 MHA/GQA 代码**：

```text
labs/attention/
├── README.md                       # 实验启动与边界（后续创建）
├── fixtures.py                     # seed / synthetic tensor / random weights
├── oracle.py                       # 仅供测试的 SDPA/naive 计算
├── reference/
│   ├── mla_expanded.py             # M2
│   └── mla_absorbed.py             # M3
├── cache/
│   ├── contiguous_mla.py           # M3
│   └── paged_mla.py                # M5
├── kernels/
│   ├── decode_naive.py             # M4-V0
│   ├── decode_tiled.py             # M4-V1
│   └── decode_optimized.py         # M4-V2
├── tests/
│   ├── test_qwen3_attention.py      # M0：已有 MHA/GQA/MQA
│   ├── test_incremental_decode.py   # M1
│   ├── test_mla_equivalence.py      # M2/M3
│   └── test_mla_kernel.py           # M4
├── benchmarks/
│   ├── bench_attention_core.py      # 一致输入/范围/计时
│   ├── bench_mla_layer.py
│   └── bench_decode_step.py
└── results/                        # 配置与报告，勿提交海量原始 trace
```

测试矩阵（先小后大）：

| 类型 | 必选用例 |
|---|---|
| 常规正确性 | `B=1/2`、`S=2/7/33`、`H_q=8`、`H_kv=8/2/1`；至少一个非 2 的幂的长度 |
| 位置/Mask | 长前缀 + 新 query、非零起始 position、不同 batch context length、无未来泄漏 |
| Cache | 空缓存→追加→读取、Page 边界前后、共享/不共享 Prefix（后期） |
| Softmax 稳定性 | 含大正/负 scores，测试稳定归一化，避免全零数据造成伪通过 |
| MLA 等价 | `d=1024,d_c'=d_c=256,H_q=8,d_k=128,d_r=64,d_v=128`，按基线示例 |
| 性能 Sweep | `B=1/4/16`，`L=128/512/2048/8192`，`Q_len=1`；Prefill 单独测试；维度/显存按设备裁剪 |
| 精度 | 小规模 FP64 或高精度数学 Oracle；GPU FP32 与 BF16 分别校验 |

数据 `torch.manual_seed` + 显式权重初始化；为避免 Softmax 饱和，主用例对输入/投影的方差做合理控制，另设专门极端分数用例。绝不使用未经初始化的 `torch.empty` 权重。

默认记录：`max_abs_error`、`p99_abs_error`、`relative_L2`、可选 `max_rel_error`（须为近零值定义分母下限）。FP32/FP64 和 BF16 容差按阶段、shape 与累加方式分别制定；禁用将 FP64 的 bitwise 相等强加给 BF16 Kernel。建议先在小 FP64 数据上做等式断言，再校准 FP32/BF16 阈值并写入测试，不临时放宽阈值掩盖缺陷。

## 5. 代码集成边界

- **Phase 1（M0～M3）：** 只新增 `labs/attention/` 与测试；无需修改 `nanovllm/engine`、Scheduler、Tokenizer 或采样器。若现有 API 不易独立调用，可添加最小的测试适配层，不在生产 Kernel 中插入研究特例。
- **Phase 2（M4）：** 每个 Triton Kernel 可以单独接收合成 Q / 压缩 KV / Metadata，先脱离 Runtime 跑通。
- **Phase 3（M5）：** 新 `MLAAttention`/Backend + 缓存 Layout Contract。`ModelRunner` 按 Layer/Backend 分配 K/V 或 latent Cache，控制 Prefix Cache、Page 地址和占用统计；旧 Qwen3 Attention 的输出与 Cache 不回归。
- **Phase 4（M6）：** 添加 third-party Adapter，不改实验核心的正确性定义；适配后产生可能的 dtype/layout 转换必须显式统计。

不建议把 MLA 计算塞入原 `Qwen3Attention` 的某个布尔分支。已有普通 QKV Head 与 MLA latent 表示是不同的 Cache Contract，应在高一层选择 Backend。

## 6. Performance 与对照的实验协议

### 统一统计边界

| 边界 | 输入 → 输出 | 不应混入 |
|---|---|---|
| Attention Core | 已生成 Q、Cache → 注意力聚合结果 | 投影权重准备/模型加载 |
| MLA Layer | Hidden States → Q/KV 压缩、Attention、后置 V/Output | Tokenizer、Sampling |
| Decode Step | 本步 KV 生成/写入、Cache 读取 → MLA Output | Scheduler 等未列入的 CPU 操作 |

必须明确：扩展形式 Core 输出是展开 V 聚合，吸收形式 Core 输出是 latent 聚合；**它们的向量维度可能不同**，最终数值等价应在后置投影之后判断，Core 性能则应分别报操作范围。

### 测量与记录

- 固定 seed、shape、dtype、position、有效可见范围与 Cache Layout；相同原始 MLA 权重才能对照 Expanded/Absorbed 的最终输出。
- 预热、`torch.cuda.synchronize`、CUDA Events；对 Decode 小 Kernel 使用 CUDA Graph replay 消除 host overhead 并单独列出 Eager 启动结果，统一 repeat 次数。
- 保存 median/p50、p95 和可复现参数；性能差异没有环境元数据视为不可复现。
- 将 AI-Infra 理论 `FLOPs`、理想 `KV-read Bytes` 与 Nsight Compute 实测 HBM bytes、带宽、SM occupancy、寄存器/Spill、Warp stall 分栏；**理论字节 ≠ 真实 HBM 流量**。
- 对跨 Head 复用，增加 `H_q` 与 `L` Sweep，并比较每次 KV Read 服务的 Head 数；缓存「只存一次」本身不能证明「只读一次」。

### 外部实现优先级与公平性

1. **FlashInfer MLA：** 可作为独立 Kernel Adapter 候选，固定版本、API、Plan/Run 边界。
2. **vLLM / SGLang：** 主要研究 Backend 选择、Batching 与数据布局，谨慎比较 Runner 级开销。
3. **FlashMLA：** 查清目标版本的硬件/精度/Head 维度约束，固定兼容 Dense MLA 的 Commit；当前主线可能以 sparse/DSA 为主，不能直接把稀疏算法与 Dense MLA 当作同一工作负载。

硬件门禁：开发机 RTX 4070 Ti SUPER 可以用于 PyTorch/Triton 与兼容的 FlashAttention 路线；要求 Hopper/Blackwell 的实现只在支持的 GPU 上执行，报告中标记 `unsupported / skipped`，不能把编译失败当成性能 0。对比跨 GPU 的绝对延迟之前需要明确架构差异与吞吐/带宽上限。

## 7. 里程碑与交付物

以 [Roadmap](../../ROADMAP.md) 为唯一任务状态源。每个 M 的交付物固定三部分：

1. **Correctness：** 最少一个自动化测试入口、seed/config、Oracle 与失败时的分层差异。
2. **Engineering：** 新增最小实现/Adapter，标明入口与输入输出、Cache Contract；尽量保持 nano-vllm 旧路径可用。
3. **Evidence：** 关键输出、误差表、Benchmark 结果（如果本阶段涉及性能），以及环境/Git SHA。

版本控制建议：一个 Milestone 多个小 Commit，先 tests/reference，后功能，再优化与 profiling。新增数学符号时同步更新本文第 0 节及其他技术文档中的数学符号表。

## 8. 第一轮可以立即执行的任务（M0）

1. 为现有 `Qwen3Attention` 准备 TP=1 GPU fixture：初始化单 rank 进程组，显式初始化投影权重，构造 positions 和合法 Runtime Context。
2. 固定 `H_q=8`，依次测试 `H_kv=8/2/1`；使用同一现有 Attention 路径，验证 shape、有限值及与 Oracle 的分层数值差异。
3. 捕获并保存 QKV 输出 Head Layout、Cache bytes 公式、Prefill/Decode 一次最小样例。
4. 将测试配置、CUDA/GPU/Torch/FlashAttention 版本、执行命令以及错误统计纳入一次可审查的实验记录。

**M0 完成后不急着写 Triton。** 下一轮先建立可靠的增量 Decode/Mask/Cache Harness，随后直接进入 MLA Expanded → Absorbed 的数学实现。
