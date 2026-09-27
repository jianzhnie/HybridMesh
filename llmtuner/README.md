# llmtuner

> **DeviceMesh 之上的混合并行训练框架** —— 一个从零搭建、用于学习的分布式训练包
> （FSDP / TP / PP / CP / EP），按文件与上游
> [torchtitan](https://github.com/pytorch/torchtitan) 逐块对齐。

GitHub 仓库名是 **TorchLLMTuner**，Python 包名是 **`llmtuner`**（`import llmtuner`）。

## 设计立场

- **只有两个抽象。** 分组配置 `LLMTunerConfig`（`llmtuner/config/`）与模型包装
  `HFTransformerModel`（`llmtuner/models/hf_wrapper.py`）。模型本体用 `transformers`
  的 `AutoModelForCausalLM`，分布式复杂度全部集中在 `parallel/`，训练循环保持端到端可读。
- **`apply_*` 在 degree == 1 时是 no-op。** 所以同一份 trainer 代码从单设备一直跑到
  全混合并行，调用点没有一处 `if degree > 1`；`parallelize_hf_transformers` 因此可以
  无条件调用每个维度。
- **装配顺序就是契约。** TP/CP/EP → activation checkpointing → `torch.compile` →
  FSDP（最外层），顺序落在 `parallel/stages.py` 的数据表里，未切分路径与 PP 逐 chunk
  路径共用同一张表，两条路径无法漂移成不同顺序。
- **不支持的组合 loud-raise。** 组合裁决的单一来源是 `parallel/matrix.py` 加各配置组的
  `__post_init__`；未验证或语义无意义的组合抛
  `UnsupportedCombinationError`，环境缺依赖抛 `EnvironmentUnsupportedError`，
  绝不静默退化成一个"能跑但算错"的模型（异常约定见 `llmtuner/errors.py`）。

## 安装

```bash
git clone https://github.com/jianzhnie/TorchLLMTuner.git
cd TorchLLMTuner
pip install -e .          # 需要 torch>=2.12，见下
```

| 依赖                                                                                         | 版本                | 说明                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python                                                                                       | `>=3.10`            |                                                                                                                                                                                                                                                         |
| torch                                                                                        | `>=2.12`            | pyproject 声明的下限。trainer/CLI 在模块级导入 `torch.distributed.tensor.DTensor`、`torch.distributed.pipelining` 等 API；**实测** torch 2.2.2 下 `import llmtuner` 正常，而 `llmtuner.Trainer` 与 CLI 直接 `ImportError: cannot import name 'DTensor'` |
| transformers                                                                                 | `>=5.9,<5.10`       | 只固定一个 minor：MoE 专家布局（`gate_up_proj` / `down_proj`）是 5.x 形状                                                                                                                                                                               |
| grain                                                                                        | `==0.2.18`          | 数据层建的 Grain 图；`grain.experimental` 的六个节点 API 仍在变                                                                                                                                                                                         |
| spmd_types                                                                                   | `==0.2.5`           | SPMD 类型检查与 mesh 助手；按上游同样精确固定                                                                                                                                                                                                           |
| datasets / tokenizers / jinja2 / pillow / einops / torchvision / requests / numpy / colorama | 见 `pyproject.toml` | `torchvision` 只被多模态路径惰性用到                                                                                                                                                                                                                    |

`pip install -e . --no-deps` 可跳过依赖求解，仅安装包本体（会装出 `llmtuner-train`
控制台脚本）。开发依赖：`pip install -e ".[dev]"`（pytest / ruff / pre-commit）。

## 快速开始

```bash
# 单进程 —— 走完全部装配路径，各维度 degree=1 全是 no-op
python -m llmtuner --steps 20
llmtuner-train --steps 20                      # 等价入口

# 数据并行 FSDP，2 个进程（-1 = 用剩余 rank 推导 dp_shard）
torchrun --nproc_per_node=2 -m llmtuner --data_parallel_shard_size -1

# 检查点（默认关闭，--enable 才开）
python -m llmtuner --steps 20 --enable --interval 10 --dump_folder ./outputs

# 真实语料（默认是合成随机 token）
python -m llmtuner --steps 20 --dataset local_jsonl \
    --dataset_path ./corpus.jsonl --tokenizer_path ./tokenizer

# 指标与 profiling（stdout 恒开；TensorBoard / WandB / profiler 均 opt-in）
python -m llmtuner --steps 20 --enable_tensorboard
python -m llmtuner --steps 20 --enable_wandb --tag baseline
python -m llmtuner --steps 20 --enable_profiling --profile_freq 4
```

CLI 用 `HfArgumentParser` 把九个配置组摊平成扁平旗标（`--tensor_parallel_size`、
`--learning_rate`、`--steps`……），也支持位置参数给一个 YAML/JSON 配置文件；
嵌套组的字段（`--enable`、`--interval`、`--dataset`）同样是裸旗标。

## 代码地图（每个文件对应一个概念）

| 路径                                                                           | 职责                                                                                                                                                                                    |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py` / `__main__.py`                                                  | 公开面：`LLMTunerConfig`、惰性 `Trainer`；`python -m llmtuner`                                                                                                                          |
| `errors.py`                                                                    | 三类失败语义（Config / UnsupportedCombination / EnvironmentUnsupported），零依赖                                                                                                        |
| `config/`                                                                      | 配置唯一来源：顶层组 Model / Parallel / Optimizer / Training，嵌套组 Checkpoint / Dataloader / Metrics / Profiler / LRScheduler；校验在各组 `__post_init__`；**只描述不构建**           |
| `accelerator/`                                                                 | 设备与后端发现（NPU/CUDA/MLU/MUSA）、PG 超时与 EP-aware `clip_grad_norm_`、显存监控与 peak FLOPS、SPMD ambient 上下文；`dist.py` / `dist_utils.py` 是从 mmengine 裁剪 vendored 的工具箱 |
| `parallel/parallelize.py`                                                      | 唯一装配入口 `parallelize_hf_transformers`（pp>1 时返回 `PipelineParallelSetup` 而非模型）                                                                                              |
| `parallel/stages.py`                                                           | 装配顺序契约，以数据表形式存在（不导入引擎，任何层都可读）                                                                                                                              |
| `parallel/matrix.py`                                                           | 跨层组合裁决的唯一来源（装配期 / probe 期），config 期校验留在配置里                                                                                                                    |
| `parallel/parallel_dims.py`                                                    | 进程拓扑：`ParallelDims` + `build_mesh`                                                                                                                                                 |
| `parallel/head_sharding.py`                                                    | attention 头数整除守卫（`heads % tp` / `heads % (tp*cp)`），对应上游 `config/validation.py` 的 `head_shard_degree`                                                                      |
| `parallel/tensor_parallel/`                                                    | 声明式 TP plan + 融合原语（`AllGatherLinear` / `LinearReduceScatter`）                                                                                                                  |
| `parallel/fully_shard/`                                                        | FSDP2 `fully_shard`（HSDP；纯 `dp_replicate` 走 DDP 兜底）                                                                                                                              |
| `parallel/context_parallel/`                                                   | CP：kv_allgather 与 Ulysses 两种策略、flex kernel、输入分片与 load balancer                                                                                                             |
| `parallel/expert_parallel/`                                                    | EP：HF MoE 块替换（搬权重而非重新初始化）+ all-to-all dispatcher + 布局 probe                                                                                                           |
| `parallel/pipeline_parallel/`                                                  | PP：stage 切分（`pipeline.py`）与 schedule 驱动（`apply.py`）                                                                                                                           |
| `models/hf_wrapper.py`                                                         | 模型唯一抽象 `HFTransformerModel`（HF `ForCausalLM` + 并行化方式）                                                                                                                      |
| `models/hf_factory.py`                                                         | 建模前助手：HF config 构造、meta 模型 materialize、`num_flops_per_token`                                                                                                                |
| `models/hf_state_dict_adapter.py`                                              | HF safetensors 命名映射（wrapper 多一层 `model.` 前缀，无张量变换）                                                                                                                     |
| `models/common/`                                                               | vendored 模型组件词汇表：注意力片段、MoE、FFN、norm/激活、RoPE、多模态胶水                                                                                                              |
| `trainer/train.py`                                                             | CLI 入口（`llmtuner.trainer.train:main`，即 `llmtuner-train`）                                                                                                                          |
| `trainer/builder.py`                                                           | 装配顺序（＝契约）：PG/rank → 度解析 → 组合守卫 → 播种 → mesh → 模型 → 并行化 → 优化器/调度/EMA/aux hooks → 数据 / checkpoint / 指标                                                    |
| `trainer/trainer.py`                                                           | 训练循环：`train` → `train_step` → `forward_backward_step` → `_forward_backward_body`                                                                                                   |
| `trainer/batch.py`                                                             | 批处理：loader 构建与取数契约、token 计数（分母）、device move 与 preprocess 缝                                                                                                         |
| `trainer/pp_steps.py`                                                          | PP 微批次切行与 schedule 驱动的前后向体（并在驱动前发布 loss 分母）                                                                                                                     |
| `trainer/validate.py`                                                          | 验证 pass 与可行性检查（含 loud-raise 的两条拒绝）                                                                                                                                      |
| `trainer/seed.py`                                                              | mesh-aware 播种：只在 PP 轴偏移，同组 rank 共用基种子                                                                                                                                   |
| `components/checkpointer/`                                                     | `base` 契约与保留/发现策略、`dcp`（实跑后端）、`torch_checkpointing`（第三方包未装则 ImportError 带安装提示）                                                                           |
| `components/loss.py`                                                           | 交叉熵（含 vocab-parallel 形式与 chunked CE）+ next-token 目标构造                                                                                                                      |
| `components/metrics.py` / `components/profiler.py` / `components/tokenizer.py` | 指标聚合与上报（stdout / TensorBoard / WandB）、profiler 与显存快照、tokenizer                                                                                                          |
| `components/optimizer/`                                                        | 优化器容器（正则分组、per-group lr/wd）、WSD 学习率调度、EMA、FQN 键状态序列化                                                                                                          |
| `datasets/`                                                                    | Grain 数据层：`build_dataloader` 是唯一装配入口，其余为 dataset / packing / sources / collators / loader；`text/` 渲染 chat 模板，`multimodal/` 惰性依赖 torchvision                    |
| `utils/`                                                                       | logger、训练步边界 GC                                                                                                                                                                   |

## 并行维度支持情况

| 维度       | CLI 旗标                         | 入口                          | 状态                                                                           |
| ---------- | -------------------------------- | ----------------------------- | ------------------------------------------------------------------------------ |
| FSDP2      | `--data_parallel_shard_size`     | `parallel/fully_shard/`       | 已落地；mesh 按 torchtitan 轴语义重建                                          |
| HSDP / DDP | `--data_parallel_replicate_size` | 同上                          | 已落地（纯 replicate 走 DDP 兜底）                                             |
| TP         | `--tensor_parallel_size`         | `parallel/tensor_parallel/`   | 已落地；TP 天然 sequence-parallel（`enable_sequence_parallel=false` 直接拒绝）；头数须整除 `tp`，装配期即拒绝 |
| CP         | `--context_parallel_size`        | `parallel/context_parallel/`  | 已落地：kv_allgather / ulysses，varlen 与 packed 支持，headtail 负载均衡       |
| EP         | `--expert_parallel_size`         | `parallel/expert_parallel/`   | 已落地：Qwen3Moe / OLMoE / Mixtral / DeepSeek-V2/V3 / GLM4 的可表示布局        |
| PP         | `--pipeline_parallel_size`       | `parallel/pipeline_parallel/` | 1F1B / Interleaved1F1B 闭环 + DCP 续训；数值等价性有未闭合项，见下             |

被**明确拒绝**的组合（全部 loud-raise，逐行对应 `parallel/matrix.py` 的表）：

| 组合                                                            | 结论                                                                                                               |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `tp > 1` 且 `ep > 1` 且 `cp > 1`                                | 拒绝（未验证）；拆成 `tp×ep`（cp=1）或 `ep×cp`（tp=1）                                                             |
| `pp > 1` × (`cp > 1` 或 `ep > 1`)                               | 拒绝；CP 切的是 schedule 消费的 batch，EP 按 chunk 换 MoE 块，两条路径都没穿过 PP                                  |
| `pp > 1` × 真实语料                                             | 拒绝；打包语料的逐 token positions 没有穿过 schedule                                                               |
| `pp > 1` × 权重绑定                                             | 拒绝；embedding 在第一 stage、head 在最后 stage，各自深拷贝会训练出两份共享权重                                    |
| `pp > 1` × chunked loss / activation checkpointing / validation | 拒绝；三者的接缝都在 PP 训练驱动器之外                                                                             |
| shared-expert 块 × `tp > 1`                                     | 拒绝；dense colwise/rowwise realizer 与 MoE 序列边界 collective 的组合未验证（`ep > 1` 时由 EP swap 负责共享专家） |
| `ulysses` × 负载均衡器                                          | 拒绝；每个 rank 按 all-to-all 到达顺序注意力，重排会让语料变成置换后的序列（静默训错，因此必须拒绝）               |
| `ptrr` 负载均衡器                                               | 未实现（依赖 BlockMask，CP kernel 不消费）                                                                         |
| `ep > 1` × checkpoint                                           | 拒绝；专家权重是 rank 异构普通张量，现有后端会当复制处理                                                           |
| `deepep` / `hybridep` dispatcher                                | 未 vendored（CUDA-only）→ `EnvironmentUnsupportedError`，解锁条件写在消息里                                        |

## 一次训练的数据流

1. `train.py` 用 `HfArgumentParser` 解析九个分组，`LLMTunerConfig.from_groups` 把嵌套组
   嫁接到顶层组，每组自己的 `__post_init__` 先校验。
2. `Trainer.__init__` 是薄壳，装配全在 `trainer/builder.py`：初始化 PG、解析 rank 与
   各维度 degree、跑组合守卫、播种。
3. `build_mesh` 建 DeviceMesh；`HFTransformerModel` 载入 HF 模型（meta 构建后只
   materialize 本地分片）。
4. `parallelize_hf_transformers` 按 `stages.py` 的顺序施加 TP/CP/EP/AC/compile/FSDP；
   `pp > 1` 时先切 stage，逐 chunk 走同一张表的子序列，最后建 schedule。
5. 优化器 / lr 调度 / EMA / MoE aux 与负载均衡 hook、dataloader、checkpointer、
   指标处理器依次建好。
6. 循环：`train` → `train_step` → `forward_backward_step`。loss 是 **sum / 全批 token
   数**（分母在切 CP、切 PP 之前算好，只在 DP 轴归约），因此换切法不改上报数字；
   `gradient_accumulation_steps` 在 backward 内做除法；梯度裁剪跨 PP stage 归约范数；
   非有限值检测归约成一个全局标志，遇到 NaN 立即停。
7. 统计量经 `components/metrics` 上报；`step` / `ntokens_seen` / 模型 / 优化器 / 调度 /
   dataloader 状态一起进 checkpoint，可续训。

## 公开 API 与入口

- `llmtuner.LLMTunerConfig` —— 稳定面；`llmtuner.config` 可导入全部 `*Config`。
- `llmtuner.Trainer` —— 惰性导入（首次访问才拉起 torch 分布式重栈），
  所以 `import llmtuner` 在缺新 API 的环境下也轻量可用。
- CLI：`python -m llmtuner`，或安装后的 `llmtuner-train`。
- 其余名字（`llmtuner.parallel`、`llmtuner.models.common` 等）是内部面，稳定性策略见
  [`../docs/torchllmtuner_design.md`](../docs/torchllmtuner_design.md) §3.4。

## 测试与验证

```bash
# CPU 单测（本机 macOS 需要 -p no:capture 规避 pytest 的 readline workaround）
python -m pytest -p no:capture tests/unit_tests -q

# 集成等价性测试（torchrun + gloo/nccl，不是 pytest；--list 列出全部脚本与启动命令）
python tests/integration_tests/run_all.py --list
PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/cp_wiring_equivalence.py
```

- **本机基线**（2026-09-27，Python 3.11.5 / torch 2.2.2 / CPU）：`9 failed, 143 passed,
  59 skipped`。9 个失败全部是环境性，与代码无关：7 个 profiler 用例（本机 torch 没有
  `torch.OutOfMemoryError`）与 2 个配置解析用例（本机没有 `torch.distributed.pipelining`）。
- **环境门禁是能力标记，不是 ignore 清单。** import 级硬依赖（DTensor、spmd_types、
  grain、flex_attention、pipelining、DCP 私有面等）由 `tests/caps.py::require_env` 在
  模块顶部声明，缺失即模块级 skip，`-rs` 就是环境覆盖报告。
- 多卡数值验证需要 torch≥2.12 + 多卡（或目标设备），命令清单见
  [`../docs/torchllmtuner_design.md`](../docs/torchllmtuner_design.md) §8 末尾。

## 已知限制

- **PP 的数值等价性尚未闭合**：目标容器（torch 2.10）复核中 PP 从 step 2 起偏离单卡
  参照（4 步最大约 `8.5e-3`），当前状态记为"未通过"，不能以"完全对齐"描述——见
  [`../docs/torchllmtuner_design.md`](../docs/torchllmtuner_design.md) §8 第 10 条。
- **routed experts 上的纯 TP 是有意保留的差异**：上游已弃用该路径并加了守卫，llmtuner
  的结构化实现（沿 F 维原地切分 + 块边界 AG/RS 对偶）不需要复制 token，因此不照搬守卫，
  见 §8 第 11 条。
- **vocab 分片的 `lm_head` + 端到端 vocab-parallel loss 完成第一步**：loss 侧已接线
  （四个调用点收 `tp_group`/`global_vocab_size`，按形状分派，复制 head 下逐位不变），
  缺的是模型侧第二步——head 真分片 + "head 已分片但 loss 未被告知" 时的 loud-raise，
  需多卡环境复跑（登记为 D 类两步走）。
- 本机（Intel macOS / torch 2.2.2）只能验证 `import llmtuner` 与 CPU 单测；多卡、
  数值等价性与端到端 smoke 都需要 torch≥2.12 + 多卡或目标设备。

## 文档

| 文档                                                                                                     | 回答什么                                                     |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [`../docs/README.md`](../docs/README.md)                                                                 | 文档索引与阅读顺序                                           |
| [`../docs/torchllmtuner_design.md`](../docs/torchllmtuner_design.md)                                     | 为什么这样设计、运行时契约、哪些组合被拒绝                   |
| [`../docs/llmtuner_upstream_map.md`](../docs/llmtuner_upstream_map.md)                                   | 每个文件来自上游哪里、按哪种策略维护（A/B/C/D 分类唯一权威） |
| [`../docs/llmtuner_torchtitan_symbol_guide.md`](../docs/llmtuner_torchtitan_symbol_guide.md)             | 符号级对应关系与正确性结论                                   |
| [`../docs/llmtuner_trainer_walkthrough.md`](../docs/llmtuner_trainer_walkthrough.md)                     | 训练主路径逐步对照与差异清单                                 |
| [`../docs/llmtuner_torchtitan_alignment_workflow.md`](../docs/llmtuner_torchtitan_alignment_workflow.md) | 下一轮对齐怎么执行、什么算完成                               |
