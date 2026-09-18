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
  （`--steps`、`--dp`、`--learning_rate`），也支持 YAML/JSON 配置文件。
- **`ModelBundle`**（`hpmesh/bundle.py`）—— 模型唯一抽象：一个 HF 模型 + 并行化它的方式。

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
torchrun --nproc_per_node=2 -m hpmesh --dp 2
```

## 代码地图（每个文件对应一个核心概念）

| 文件 | 核心概念 | 状态 |
|---|---|---|
| `hpmesh/trainer/config.py` | 分组组合配置 + `derive_dp`（`world_size = dp*cp*tp*pp`） | 可运行 |
| `hpmesh/mesh.py` | **DeviceMesh / 进程拓扑** + torchrun 初始化 | 可运行 |
| `hpmesh/bundle.py` | HF 模型包装成统一 `(input_ids, labels) -> loss` | 可运行 |
| `hpmesh/trainer/trainer.py` | 训练循环 + 确定性 seeding + DP 数据切分 | 可运行 |
| `hpmesh/parallel/fsdp.py` | 数据并行（FSDP2 `fully_shard`） | 已实现 |
| `hpmesh/parallel/linear.py` | async-TP 融合原语（`AllGatherLinear` / `LinearReduceScatter`） | 已实现（CUDA） |
| `hpmesh/parallel/tp.py` | 张量并行（声明式 sharding -> 融合原语） | 已实现（CUDA） |
| `hpmesh/parallel/pp.py` | 流水线并行（1F1B 调度） | 学习练习 |
| `hpmesh/parallel/cp_ep.py` | 上下文并行 / 专家并行 | 学习练习 |
| `hpmesh/trainer/train.py` | 入口：`HfArgumentParser` 解析 config -> `Trainer(cfg).train()` | 可运行 |

## 学习路径

每一步都能跑、都能跟官方实现对拍。**先跑通，再读官方实现，再动手改/写。**

```text
第 0 步  单设备纯训练      已实现   python -m hpmesh --steps 20
第 1 步  +FSDP 数据并行    已实现   torchrun --nproc_per_node=2 -m hpmesh --dp 2
第 2 步  +TP 张量并行      已实现   parallel/linear.py + tp.py (声明式 -> 融合 GEMM)
第 3 步  +PP 流水线并行    练习     parallel/pp.py (pipelining 1F1B)
第 4 步  +CP 或 EP         练习     parallel/cp_ep.py (KV all-gather / all-to-all)
```

## 验证方法

- **确定性**：`config.deterministic=True`（默认）+ 固定 `seed`，所有 rank 构建相同
  初始权重、产生相同合成数据 —— 这是 bit-exact 对拍的前提。学习期间绝不要关闭。
- **对拍**：每加一维，验证"开 vs 关该维度"在同一 seed 下 loss 一致（数据并行下应与
  单卡逐位一致或仅差浮点累加顺序）。重点盯浮点累加的精度与顺序。

## 已知限制

- 依赖：`torch`、`transformers`、`tyro`。TP 那一步才需要独立的 `spmd_types` 包。
- 第 1 步起的多进程（torchrun + FSDP）需要 **CUDA/NCCL**。在无 CUDA 的机器
  （如 Apple Silicon / MPS）上：第 0 步可正常运行，但 FSDP2 `fully_shard` 面向
  NCCL 设计，在 CPU+gloo 上不可用 —— 请在 GPU 机器上做第 1 步及以后。
- **第 2 步 TP 同样是 CUDA-only**：`parallel/linear.py` 的融合算子走
  `torch.ops.symm_mem.fused_*`（对称内存），本机 `symm_mem.is_available()==False`。
  CPU 上只能验证声明层与权重切分（见 `tests/test_tp.py`），完整的 all-gather /
  reduce-scatter 前反向要在 GPU 上跑。

## License

Apache-2.0（见 `LICENSE`）。
