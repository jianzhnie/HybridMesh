# hpmesh

> **Hybrid Parallel training over a Torch DeviceMesh** — a minimal, learning-oriented
> distributed-training framework (FSDP / TP / PP / CP / EP).

`hpmesh` 是一个**从零搭建、用于学习**的分布式训练框架。目标不是交付一个产品，
而是让你**通过亲手实现**搞懂分布式训练的核心模块。模型本体用 `transformers`
的 `AutoModelForCausalLM`（离线可用 `AutoConfig.for_model` 构造小模型，无需联网），
分布式的硬活集中在 `mesh` + `parallel/`，训练循环保持端到端可读。

> 说明：GitHub 仓库名为 `HybridMesh`，Python 包名为 `hpmesh`。

## 设计：只有两个抽象

- **分组 `HybridMeshConfig`**（`hpmesh/trainer/config.py`）—— 按关注点分组
  （Model / Parallel / Optimizer / Training）再**组合**成单一配置；每组在自己的
  `__post_init__` 里校验。CLI 用 `HfArgumentParser` 暴露成扁平旗标
  （`--steps`、`--data_parallel_shard_degree`、`--learning_rate`），也支持 YAML/JSON
  配置文件。
- **`HFTransformerModel`**（`hpmesh/models/hf_wrapper.py`）—— 模型唯一抽象：一个 HF
  模型 + 并行化它的方式。

## 安装

```bash
git clone https://github.com/jianzhnie/HybridMesh.git
cd HybridMesh
pip install -e .
```

## 快速开始

```bash
# 第 0 步：单设备（无需 GPU/torchrun）
python -m hpmesh --steps 20

# 第 1 步：数据并行 FSDP，2 进程（需 CUDA/NCCL）
torchrun --nproc_per_node=2 -m hpmesh --data_parallel_shard_degree 2
```

## 代码地图（每个文件对应一个核心概念）

| 文件 | 核心概念 | 状态 |
|---|---|---|
| `hpmesh/trainer/config.py` | 分组组合配置（Model / Parallel / Optimizer / Training）+ 校验 | 可运行 |
| `hpmesh/mesh.py` | **DeviceMesh / 进程拓扑** + torchrun 初始化 | 可运行 |
| `hpmesh/models/hf_wrapper.py` | HF 模型包装成统一的 decoder forward（返回 logits，loss 在 trainer 里算） | 可运行 |
| `hpmesh/trainer/trainer.py` | 训练循环：`train` -> `train_step` -> `forward_backward_step`，token 归一化 loss + 梯度裁剪 + 非有限值检测 | 可运行 |
| `hpmesh/datasets/random_data.py` | `Batch` + 无限微批次迭代器（源耗尽即中止整步，不训练半个 batch） | 可运行 |
| `hpmesh/datasets/{loader,sources,packing,hf/text}.py` | Grain 数据层：语料 -> 打包 -> 每 DP rank 分片；`DATALOADER` 状态进 checkpoint | 可运行 |
| `hpmesh/datasets/hf/multimodal/` | 多模态语料（图/视频/文本处理器 + collator）——**尚未接线**：只有单测在调 | 已实现 |
| `hpmesh/components/checkpointer/{base,dcp,torch_checkpointing}.py` | 每 rank 一份检查点，`step` / `ntokens_seen` / 模型 / 优化器，可续训；`base.py` 是共用骨架，两种后端各一个 manager | 可运行 |
| `hpmesh/components/loss.py` | 交叉熵（含 vocab-parallel 形式）+ next-token 目标构造 | 已实现 |
| `hpmesh/components/{metrics,profiler}.py` | 训练指标 + profiler | 可运行 |
| `hpmesh/components/optimizer/{optimizer,lr_scheduler,utils}.py` | 优化器容器（正则分组 + per-group lr/wd）、WSD 学习率调度、FQN-keyed checkout 状态序列化 | 可运行 |
| `hpmesh/parallel/collectives.py` | mesh 感知的 `dist_sum` / `dist_max` / `clip_grad_norm_`（跨 PP stage 归约范数） | 可运行 |
| `hpmesh/parallel/fully_shard/fsdp.py` | 数据并行（FSDP2 `fully_shard`） | 已实现 |
| `hpmesh/parallel/tensor_parallel/linear.py` | async-TP 融合原语（`AllGatherLinear` / `LinearReduceScatter`） | 已实现（CUDA） |
| `hpmesh/parallel/tensor_parallel/tp.py` | 张量并行（声明式 sharding -> 融合原语） | 已实现（CUDA） |
| `hpmesh/parallel/pipeline_parallel/{pipeline,pp}.py` | PP：stage 切分 + `apply_pp` / schedule 驱动（1F1B 闭环，pp+cp/ep 未接线） | 已实现 |
| `hpmesh/parallel/context_parallel/` + `expert_parallel/` | 上下文并行（KV all-gather 接线）/ 专家并行（Qwen3Moe MoE 替换 + all-to-all） | 已实现 |
| `hpmesh/trainer/train.py` | 入口：`HfArgumentParser` 解析 config -> `Trainer(cfg).train()` | 可运行 |

结构审计见 `docs/hpmesh_structure.md`；设计见 `docs/hybridmesh_design.md`
（`docs/FRAMEWORK_DESIGN.md` 是**立项前的评估稿，已归档**，其中的 `hftrain/`
目录骨架未落地，读之前先看它的抬头）。
优化器 checkpoint 的磁盘格式在 `0fd6cbe` 变更过（改为扁平 FQN keying），
旧 checkpoint 不再能加载，见 `docs/optimizer_checkpoint_format.md`。

## 学习路径

每一步都能跑、都能跟官方实现对拍。**先跑通，再读官方实现，再动手改/写。**

```text
第 0 步  单设备纯训练      已实现   python -m hpmesh --steps 20
第 1 步  +FSDP 数据并行    已实现   torchrun --nproc_per_node=2 -m hpmesh --data_parallel_shard_degree 2
第 2 步  +TP 张量并行      已实现   parallel/tensor_parallel/ (声明式 -> 融合 GEMM)
第 3 步  +PP 流水线并行    已实现   parallel/pipeline_parallel/ (1F1B 闭环, pp_equivalence 对拍)
第 4 步  +CP 或 EP         已实现   parallel/context_parallel/ + expert_parallel/ (KV all-gather / all-to-all)
```

## 验证方法

- **确定性**：`config.deterministic=True`（默认）+ 固定 `seed`，所有 rank 构建相同
  初始权重、产生相同合成数据 —— 这是 bit-exact 对拍的前提。学习期间绝不要关闭。
- **对拍**：每加一维，验证"开 vs 关该维度"在同一 seed 下 loss 一致（数据并行下应与
  单卡逐位一致或仅差浮点累加顺序）。重点盯浮点累加的精度与顺序。

## 已知限制

- 依赖：`torch`、`transformers`、`spmd_types`（TP 用到），以及数据层的一组
  `grain` / `datasets` / `tokenizers` / `jinja2` / `pillow` / `einops` /
  `torchvision` / `requests`（见 `pyproject.toml` 的注释；后五个只在多模态路径上
  才被 import，但都声明成了硬依赖）。
- 第 1 步起的多进程（torchrun + FSDP）需要 **CUDA/NCCL**。在无 CUDA 的机器
  （如 Apple Silicon / MPS）上：第 0 步可正常运行，但 FSDP2 `fully_shard` 面向
  NCCL 设计，在 CPU+gloo 上不可用 —— 请在 GPU 机器上做第 1 步及以后。
- **第 2 步 TP 同样是 CUDA-only**：`parallel/tensor_parallel/linear.py` 的融合算子走
  `torch.ops.symm_mem.fused_*`（对称内存），本机 `symm_mem.is_available()==False`。
  CPU 上只能验证声明层与权重切分（见 `tests/test_tp.py`），完整的 all-gather /
  reduce-scatter 前反向要在 GPU 上跑。
- **PP 只支持 `--dataset random`**：打包语料的 `positions` 没有穿过 schedule 的通道
  （`pp.py` 里显式 raise）。见 `docs/hybridmesh_design.md` §5.5。

## License

Apache-2.0（见 `LICENSE`）。
