# Attention → MLA Learning Lab Roadmap

> 状态：**Planned**（2026-10-27）；目标分支：`feature/mla-learning-lab`；基线：从 `main` 创建。本文件列出待完成的任务，不代表已实现或已测量。
>
> 详细方案：[`docs/mla-learning-lab/PLAN.md`](docs/mla-learning-lab/PLAN.md)。理论背景：[AI-Infra / MLA 计算强度分析基线](https://github.com/Maverickhj/AI-Infra/blob/main/01_Inbox/MLA%20%E8%AE%A1%E7%AE%97%E5%BC%BA%E5%BA%A6%E5%88%86%E6%9E%90%E5%9F%BA%E7%BA%BF.md)。

## 北极星目标

以 **nano-vllm 为唯一 Runtime 底座**：直接复用已有 MHA/GQA（以及 `num_kv_heads=1` 下的 MQA）实现；用合成输入/随机初始化权重建立可验证的 Attention/MLA 计算基线；逐步开发 Expanded → Absorbed MLA、压缩 KV Cache、Triton Kernel，最后与业界实现做公平性能对照。

约束：**不要求真实 prompt、tokenizer、checkpoint、生成文本；必须能验证数值正确性。** 当前阶段只提交规划，不更改原有 Runtime 的计算逻辑。

## 当前焦点

**M0：已有 MHA/GQA/MQA 的代码与测试基线。** 从 nano-vllm 现有 `Qwen3Attention` / `Attention` 开始，而不是重写三个 Attention。M0 完成后才启动 MLA 开发。

## Milestones（顺序执行，门禁制）

### M0. 复用与环境基线（Planned）

- [ ] 固定上游 Fork 起点、Python/Torch/Triton/FlashAttention/CUDA/Driver/GPU 信息，记录复现命令。
- [ ] 梳理 `Qwen3Attention → Attention.forward → Context → ModelRunner → KV Cache` 数据流，记录输入输出形状、Prefill/Decode 路由、TP=1 的限制。
- [ ] 使用同一现有实现的 `num_heads=8, num_kv_heads=8/2/1` 覆盖 MHA/GQA/MQA，不添加生产级重写。
- [ ] 用固定随机种子初始化 `torch.empty` 权重，构造合成 hidden_states/positions，跑通至少一个真实 nano-vllm Attention GPU 前向。
- [ ] 对齐独立的 PyTorch 数学 Oracle（优先 SDPA math backend；必要时显式 KV repeat 仅限 Oracle），验证输出形状、数值、Causal Mask 和 Q/K Norm + RoPE。
- [ ] 记录每种模式的 KV Cache 理论元素数及一次可重复的 GPU baseline（不宣称三者输出在随机独立权重下相等）。

**DoD：** 单卡 TP=1 的 MHA/GQA/MQA 全部通过参数化正确性测试；测试不依赖预训练 checkpoint；`README/报告` 中给出版本、执行命令和误差统计。未通过不得进入 M1。

### M1. Attention Correctness Harness & 增量缓存（Planned）

- [ ] 将合成输入配置、权重初始化、误差统计与可选中间结果收集提取为可复用测试工具。
- [ ] 验证不等长序列、非平凡因果 Mask、位置偏移；明确多 query decode 的 mask 对齐方式。
- [ ] 以现有普通 KV Cache 测试从空缓存开始的 Prefill 与逐 token Decode 一致性。
- [ ] 覆盖不同 batch 的 context length、Page 边界、slot_mapping/prefix cache 基本语义。
- [ ] 区分 FP64/FP32 小规模参考与 BF16 Kernel 的容差；保存 max_abs/p99_abs/relative_L2。

**DoD：** 给定 seed/config 可重复复现 Prefill/Decode 差异；由自动化测试发现故意的错误 Position、Mask 或 KV Head 映射。

### M2. Expanded MLA（PyTorch 正确性基线，Planned）

- [ ] 依据 AI-Infra 基线明确符号、维度、RMSNorm、解耦 RoPE、Softmax Scale、权重布局、Mask。
- [ ] 复用现有 Linear / RMSNorm；为 MLA 不同 Q/K RoPE 分支提供最小适配（不强行复用不兼容的接口）。
- [ ] 完成纯 PyTorch Expanded MLA Reference：Q/K/V 展开、Score、Softmax、Aggregation、Output Projection。
- [ ] 同一组 seed/weights 下，支持从位置/Head/Score 到最终输出的调试与测试。
- [ ] 在小规模 FP64/FP32 数值检查之后，补充 BF16 检查。

**DoD：** Expanded MLA Prefill 与独立数学/形状断言一致，所有核心中间结果可观测；不需要 ModelRunner、checkpoint 或 tokenizer。

### M3. Absorbed MLA + 压缩 KV Cache（Planned）

- [ ] 实现吸收 Q 侧内容 K 展开投影及后置 V 投影，保留原始 scale `1/sqrt(d_k+d_r)`。
- [ ] 实现共享压缩 Cache：每位置只存 `[c_KV, k_R]`，而不是为每个 Head 单独存展开 K/V。
- [ ] 验证 Expanded 与 Absorbed 的 attention score、概率/LSE、Head 输出和最终输出的数值等价。
- [ ] 实现增量 Decode，不重新投影已有历史 K/V；验证完整因果 Prefill vs token-by-token Decode。
- [ ] 通过参数化测试覆盖多 Head、不同 context length、非对齐长度、空前缀/共享前缀。

**DoD：** 两条 MLA 计算链路共享同一组权重与输入、达到声明的数值容差；Cache 实际容量与理论维度公式相符。

### M4. MLA Kernel（分阶段，Planned）

- [ ] V0：Naive Triton Decode，正确性优先，允许中间 Tensor 和重复 KV 读取。
- [ ] V1：Tiled + Online Softmax Decode，验证分块合并/归一化、非整块长度。
- [ ] V2：跨 Head 共享读取、Split-K/并行优化，记录每项优化前后结果。
- [ ] Prefill 路线独立推进：Q/K Tiling、Causal Mask，与 PyTorch Oracle 对齐。
- [ ] 每个 Kernel 变体通过相同的 Correctness Harness；不允许只测性能不测误差。

**DoD：** 三代 Decode Kernel 均能独立运行、复现误差与延迟；优化差异能归因于明确的计算/访存机制。

### M5. nano-vllm Runtime 适配（Planned）

- [ ] 定义 MLA 专用 Attention/Cache Backend 接口；保留现有 MHA/GQA 路径不回归。
- [ ] 扩展 `ModelRunner.allocate_kv_cache` 以支持按层选择传统 K/V 或 MLA 压缩格式；明确 block_bytes 公式。
- [ ] 适配 `slot_mapping`、`block_tables`、`context_lens`、Prefill/Decode、可选 CUDA Graph；TP>1 延后。
- [ ] 使用合成 hidden_states 或合成批次测试 Runtime；真实文本推理不是验收门槛。
- [ ] 回归已有 nano-vllm Qwen3 MHA/GQA 计算路径。

**DoD：** 合成请求能穿过 Runner + MLA Backend，输出对齐独立 MLA Reference；旧 Attention 路径保持正确。

### M6. Performance & 业界实现对照（Planned）

- [ ] 统一 `core-only`、`MLA-layer`、`decode-step` 三种计时边界，分开报告。
- [ ] Sweep `B/Q_len/L/dtype/head-layout/block-size`；使用 CUDA Events/Graph Replay 并记录测量方式。
- [ ] 对比理论 FLOPs、KV Cache Bytes、理想读取次数与 Nsight 实测 HBM 流量/Kernel 耗时。
- [ ] 接入可用的 FlashInfer MLA API；固定适配版本和必要的 Layout 转换。
- [ ] 仅在硬件、dtype、形状受支持时比较 FlashMLA（固定 Dense MLA 兼容 Commit）；DSA/sparse 路径不能当 Dense MLA 同条件基线。
- [ ] 保存完整配置、硬件与 Git SHA、warmup/repeat/统计量、数值误差，并撰写性能归因报告。

**DoD：** 至少包含 Reference、个人 Triton 和一个兼容的第三方 Backend；每项延迟结果能追溯环境与输入，性能比较具备一致边界与正确性证明。

## 工作规则

1. **小步提交。** 每个 Milestone 至少拆为「测试/Oracle → 实现 → benchmark/文档」的小 PR 或 Commit，禁止一次改动模型、缓存、调度和 Kernel 导致难以定位。
2. **先证正确，再谈性能。** Kernel 不通过相同输入与权重的正确性测试，不进入 Benchmark 报告。
3. **先隔离后集成。** M2/M3 可脱离完整 nano-vllm Runner；直到 M5 才变更 Runner 分配与调度。
4. **保留主线。** 只在本 feature 分支开发；合并回 main 之前要求现有 Qwen3 路径无回归。
5. **报告写清基准边界。** 理论 HBM 字节数不是实测流量；不同 Attention 结构若权重不同，不能要求输出相等。
6. **数学符号表同步维护。** 正式技术文档在开头保留符号表，引入新符号时更新对应定义。

## 实验结果最小元数据

每组记录：`git_sha`、`baseline_sha`、`gpu_name/sm`、`cuda/driver`、`torch/triton/flash-attn`、`dtype`、`B/Q_len/L/H_q/H_kv/dims`、`seed`、`cache_layout`、`kernel_variant`、`warmup/repeats`、`timing_method`、`p50/p95_latency_us`、`max_abs_error/relative_L2`；Profiling 额外记录指标来源与测量边界。

## 进度更新原则

只在 DoD 实际满足、测试有证据时勾选 `[x]`；初始全部为 `[ ]`。参考方案见 [实施方案](docs/mla-learning-lab/PLAN.md)。
