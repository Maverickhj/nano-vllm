# Attention learning lab

本轮实现 MHA/GQA 教学骨架和 PLAN §4 的传统 attention 正确性基线。生产代码保持原样；MLA 等价、MLA kernels、共享前缀生命周期和性能 sweep 留待相应里程碑。

当前 `reference/` 已开始手写实现，仍有 TODO，尚未通过 student 正确性验收。`results/` 保留初始骨架阶段的历史报告；后续运行验证使用 Megatron-Bridge 镜像，本次提交不调整环境依赖。

## 先手写的部分

[`reference/mha.py`](reference/mha.py) 的 `TorchAttention` 提供普通 `nn.Linear`、Q/K Norm 参数、forward 串接和以下 TODO：

1. `project_qkv`：投影、拆分、head 布局。
2. `normalize_qk`：每 head RMSNorm；QKV bias 模式与原版一致地跳过 Norm。
3. `apply_rope`：绝对位置、split-half RoPE。
4. `attention`：MHA/GQA head 映射、scale、因果 mask、稳定 Softmax、V 聚合。
5. `output_projection`：合并 heads、输出投影。

所有待实现步骤均有显式的 `# TODO:` 注释，可以用编辑器全局搜索或 `rg -n '# TODO:' labs/attention/reference` 定位。GQA/MQA 共享 `mha.py` 中的待实现方法，`gqa.py` 的构造器和 Harness 接线已经完成。

`TorchMHA` 要求 `H_kv=H_q`；[`reference/gqa.py`](reference/gqa.py) 的 `TorchGQA` 使用同一骨架，`H_kv=1` 即 MQA。待实现方法标有 TODO，学生计算路径不调用 Oracle。学习骨架不依赖 FlashAttention/Triton/分布式进程组。

| 教学接口 | 原版阅读入口 |
|---|---|
| `project_qkv` | `Qwen3Attention.forward`、`layers/linear.py::QKVParallelLinear` |
| `normalize_qk` | `layers/layernorm.py::RMSNorm` |
| `apply_rope` | `layers/rotary_embedding.py::apply_rotary_emb` |
| `attention` | `layers/attention.py::Attention.forward` |
| `output_projection` | `layers/linear.py::RowParallelLinear`，本实验固定 TP=1 |

张量约定：Hidden `[T,d]`、Q `[T,H_q,d_h]`、K/V `[T,H_kv,d_h]`、attention 输出 `[T,H_q,d_h]`。`attention` 接收完整历史 K/V，以及独立的 query/key 绝对位置；允许 `Q_len < L`。学生 adapter 只负责连续 cache 的 append 与测试接线，不代写数学计算。

原版数据流是 `Qwen3Attention` 的投影/Norm/RoPE → `Attention.forward` 根据 `Context` 写 KV 并选择 prefill/decode → 输出投影。生产 `ModelRunner` 负责分配缓存和准备 metadata；本实验由 adapter 提供这些输入，不启动 Runner。`slot_mapping` 表示物理 token 槽位，`block_tables` 把逻辑 page 映射到物理 page，decode 的 `context_lens` **包括本次已经写入的 token**。测试初始化 TP=1 进程组，结束时恢复 Context 并移除 hooks。

## 已实现的检查

- 固定 seed，合成输入，显式初始化所有权重；不加载模型、tokenizer 或 checkpoint。
- 独立 FP64 Oracle，显式 KV repeat 仅用于 Oracle；CPU/GPU FP32 对照 SDPA math backend。
- 原版 `Qwen3Attention` 的 Q/K/V 投影、Q/K Norm、RoPE、head output、最终输出逐层比较。原版 FlashAttention 的 scores/LSE 不可见；这些中间量仅在 Oracle 中检查。
- `H_q=8,H_kv=8/2/1`，完整矩阵 `B=1/2,S=2/7/33`；额外覆盖 QKV bias 路径。
- 空缓存开始逐 token decode，对照每个位置的完整因果 prefill 和 Oracle。
- 非零起始位置、不等长 batch、长前缀与新 chunk、bottom-aligned mask、不等长 `context_lens`。
- 非顺序物理页、256-token page 边界、`slot_mapping=-1`；逐元素检查写入内容、未写区域和 guard page。
- 大正/负 scores、未来 token 扰动；故意错误的位置、mask、KV head 映射必须被误差检查发现。
- 报告固定容差及 `max_abs_error/p99_abs_error/relative_L2`，记录形状、seed、Git SHA、软件和 GPU 版本、缓存容量。Decode 汇总中的 p99 是各 step p99 的最大值，不是拼接所有误差后的分位数。

`metrics.py` 的容差按投影、Norm/RoPE、输出和 dtype 预先声明。FlashAttention 2 只接收 FP16/BF16；原版 GPU baseline 不伪装成 FP32 路径。FP32/FP64 由数学 Oracle 测试单独覆盖。

## 在 minimind 容器运行

使用容器预装的 `/opt/venv/bin/python`，工作目录 `/opt/nano-vllm`；不需要同步项目 `.venv` 或重装 torch。

```bash
cd /opt/nano-vllm
export TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor
export TRITON_CACHE_DIR=/data/cache/triton

# 默认基线只关闭 torch.compile 包装，原版算子、Triton cache writer 和 FA 均照常执行。
# JSON 会记录 torch_compile_disabled；移除该环境变量即可检查编译路径。
TORCHDYNAMO_DISABLE=1 /opt/venv/bin/python -m labs.attention.run \
  --backend nanovllm --suite full --output labs/attention/results/baseline-bf16.json

# CPU + GPU 单测；pytest 是实验工具，不是生产运行依赖。
TORCHDYNAMO_DISABLE=1 /opt/venv/bin/python -m pytest labs/attention/tests -q

# 补完 TODO 后，使用同一批输入、权重和 FP64 Oracle 检查学生实现。
TORCHDYNAMO_DISABLE=1 /opt/venv/bin/python -m labs.attention.run \
  --backend student --suite full --output labs/attention/results/student-bf16.json
```

骨架抛出 `NotImplementedError` 时，student 命令返回非零状态，并记录 `not_implemented`；其他异常记录为 `error`，不会算成通过。两个 backend 使用相同 fixture/Oracle：原版 baseline 与 student 报告可逐 stage 对照，但不同 `H_kv` 的独立随机模型不应强行比较输出相等。

快速检查可用 `--suite smoke`；FP16 用 `--dtype float16`。无 GPU 时可运行 `python -m pytest labs/attention/tests/test_oracle.py -q`，其中 GPU 子用例会明确 skip。所有源码、JSON、rendezvous、编译缓存均位于 NAS 挂载目录。

测试入口不会修改 `nanovllm/`、启动 ModelRunner 或测试采样；也不报告性能数字。

本机实测记录见 [`results/BASELINE.md`](results/BASELINE.md)，完整逐阶段数据见 [`results/baseline-bf16.json`](results/baseline-bf16.json)。
