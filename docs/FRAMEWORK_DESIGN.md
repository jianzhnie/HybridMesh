# 基于 transformers_modeling_backend 接口构建新框架：剥离分析与设计

> 目标：评估"拿掉 TorchTitan 抽象、以 `experiments/transformers_modeling_backend`
> 的接口为主体设计新框架"的可行性，并给出两条落地路线。
>
> 本文四部分：
> **§1 依赖剥离图** —— 这个模块到底依赖 TorchTitan 的哪些符号、耦合深浅如何；
> **§2 最小框架骨架** —— 自研轻量框架的接口形状（只定签名，不实现）；
> **§3 精简 Titan 路线** —— 留哪些删哪些、两条路线工作量/代码量对比与建议；
> **§4 精度对齐方法论** —— 两条路线各自如何验证与官方仓库的数值一致性；
> **§5 学习路径** —— 若目标是学懂分布式训练核心模块，该选哪条路、怎么学。
>
> 所有 import 与行数均取自当前代码实测。

---

## 0. 一个决定成败的前提

`spmd_types` **不是 TorchTitan 的抽象**。它是一个独立的 Meta pip 包
（`spmd_types==0.2.1`，github.com/meta-pytorch/spmd_types，"A type system for
distributed (SPMD) tensor computations"）。TorchTitan 里的
`distributed/spmd_types.py`（529 行）只是一层 glue shim（`MeshAxisName` 映射、
TLS mesh 栈、redistribute 校验）。

**结论先行**：这个模块的声明式并行能力来自 `spmd_types` + `torch.distributed`，
两者都独立于 TorchTitan。所以"拿掉 TorchTitan 抽象"**不等于**重写 SPMD 类型系统——
真正要重写的是 Titan 在上面的**编排层**（mesh 构建、sharding 应用引擎、FSDP/PP/AC
编排、训练循环、配置系统）。

---

## §1. 依赖剥离图

### 1.1 模块对外接口（干净、可保留）

模块通过 `__init__.py` 的 `model_registry()` 暴露一个 ModelSpec 形状的包——
这是它**最有价值、最该保留**的部分：

```python
ModelSpec(
    name="transformers_modeling_backend",
    model=HFTransformerModel.Config(...),     # 模型 + HF 配置
    parallelize_fn=parallelize_hf_transformers,  # 函数式接入点
    pipelining_fn=pipeline_hf_transformers,
    post_optimizer_build_fn=register_moe_load_balancing_hook,
    state_dict_adapter=HFTransformerStateDictAdapter,
)
```

### 1.2 逐文件依赖清单（实测）

下面按文件列出它从 `torchtitan.*` 引入的**确切符号**，并标注耦合性质。

#### `__init__.py`（接口组装）
| 依赖 | 符号 | 耦合 |
|---|---|---|
| `protocols.model_spec` | `ModelSpec` | 薄 —— 一个 8 字段 dataclass，可自研 |
| `components.optimizer` | `register_moe_load_balancing_hook` | 中 —— MoE 负载均衡钩子 |

#### `model.py`（HF 模型包装，1412 行核心）
| 依赖 | 符号 | 耦合 |
|---|---|---|
| `protocols.model` | `BaseModel` | 薄（128 行协议） |
| `protocols.module` | `Module`, `ModuleDict` | **厚** —— `model.parallelize()` 引擎所在（589 行） |
| `distributed.parallel_dims` | `ParallelDims` | **厚** —— mesh 构建（530 行） |
| `distributed.spmd_types` | `annotate_input_spmd_types` | 中 —— glue shim |
| `distributed.context_parallel.api` | `prepare_context_parallel_input` | 中（255 行） |
| `distributed.utils` | `is_in_batch_invariant_mode` | 薄 |
| `models.common.attention` | `create_attention_mask`, mask_mods | 中（951 行，flex mask 工具） |
| `models.common.decoder_sharding` | `decoder_input_sharding` | 中（371 行 placement helper） |
| `models.utils` | `quadratic_attention_flops_per_token` | 薄 —— 一个 FLOPs 公式 |

#### `hf_sharding.py`（声明式 TP/EP，506 行）
| 依赖 | 符号 | 耦合 |
|---|---|---|
| `protocols.sharding` | `ShardingConfig` | **厚（语义核心）** —— 但本身小（126 行） |
| `distributed.parallel_dims` | `MeshAxisName` | 薄 —— 一个枚举 |
| `models.common.decoder_sharding` | `dense_*_placement` ×3 | 中 —— placement 工厂 |
| **外部** `spmd_types` | `spmd.S/R/P/V/I`, `SpmdType` | **独立包，不算 Titan 依赖** |

#### `parallelize.py`（并行编排，410 行）
| 依赖 | 符号 | 耦合 |
|---|---|---|
| `distributed.fsdp` | `resolve_fsdp_mesh`, `resolve_sparse_fsdp_mesh`, `enable_fsdp_symm_mem`, `disable_fsdp_gradient_division`, `get_fsdp_reshard_after_forward_policy` | **厚** —— FSDP2 编排（431 行） |
| `distributed.compile` | `apply_compile` | 薄（154 行） |
| `distributed.activation_checkpoint` | `ActivationCheckpointingConfig` | 中（428 行策略体系） |
| `config` | `ParallelismConfig`, `CompileConfig`, `TrainingConfig`, `TORCH_DTYPE_MAP` | 薄 —— 纯 dataclass |

#### `pipeline.py`（PP 切分，422 行）
| 依赖 | 符号 | 耦合 |
|---|---|---|
| `distributed.pipeline_parallel` | `_build_get_mesh_callback`, `_build_pipeline_schedule` | **厚** —— PP schedule 编排（643 行） |
| `protocols.module` | `ModuleDict`, `ModuleList` | 厚（同上） |
| `components.loss` | `LossFunction` | 薄 —— 类型别名 |
| `models.common.nn_modules` | `Identity` | 薄 —— `nn.Identity` 别名 |

#### `moe_replacement.py`（HF MoE → Titan MoE，603 行）
| 依赖 | 符号 | 耦合 |
|---|---|---|
| `models.common.moe` | `MoE`, `GroupedExperts` | **厚** —— Titan MoE 实现（566 行） |
| `models.common.moe_sharding` | `set_moe_sharding_config` | 厚（357 行） |
| `models.common.config_utils` | `make_moe_config` 等 4 个 | 中（448 行配置工厂） |
| `models.deepseek_v3` | `make_deepseek_v3_router_config` | 中 —— 复用了一个**具体模型**的 router |

#### 其余薄依赖
`module_conversion.py`（`Embedding`）、`state_dict_adapter.py`（`StateDictAdapter`，
140 行协议）、`tokenizer.py`（`HuggingFaceTokenizer`）。

#### `config_registry.py`（preset，299 行）
继承 `Trainer.Config`，**直接用了约 15 个一级配置字段**（loss/optimizer/
lr_scheduler/dataloader/checkpoint/metrics/profiler/parallelism/training/
activation_checkpoint/tokenizer/hf_assets_path/debug + 自加的 `hf_model`）。
这是与 Titan 训练栈**最深的耦合点**。

### 1.3 耦合分级汇总

| 级别 | 内容 | 处理建议 |
|---|---|---|
| **可整体搬走（薄）** | `ModelSpec`、`BaseModel`、`ShardingConfig`(126 行)、`StateDictAdapter`、各 `*Config` dataclass、`MeshAxisName` 枚举、mask_mods、FLOPs 公式 | 抽进新框架的 `protocols/`，几乎零改写 |
| **需精简后复用（中）** | placement 工厂（`decoder_sharding`）、`spmd_types` glue、`prepare_context_parallel_input`、`apply_compile`、AC 策略、MoE config 工厂 | 抽核心路径，砍冷门分支 |
| **耦合死结（厚，重写成本最高）** | `protocols/module.py` 的 `parallelize()`/`init_states()` 引擎（589 行）、`ParallelDims`（530 行）、`distributed/fsdp.py`（431 行）、`pipeline_parallel.py`（643 行）、Titan MoE（566+357 行）、`Trainer.Config` 训练栈 | **自研 vs 精简的分水岭就在这里** |
| **不该重写的独立件** | `spmd_types`（pip 包）、`torch.distributed`/`fully_shard`/`pipelining`/`torch.compile`、`transformers` | 直接依赖 |

**死结估算**：上表"厚"档合计约 **3.5k 行** Titan 引擎代码（module 协议 +
ParallelDims + fsdp + pipeline + MoE）。这是"拿掉抽象"的真实代价。

---

## §2. 最小框架骨架

设计原则：**对外只暴露两个抽象（ModelBundle + 扁平 TrainConfig），引擎复用
`spmd_types` + PyTorch 原生件，砍掉 Titan 的 Configurable 全套。**

### 2.1 目录结构

```text
hftrain/
├── __init__.py
├── bundle.py          # ModelBundle：模型 + 并行化接入点（对应 ModelSpec 形状）
├── config.py          # 扁平 TrainConfig + preset 函数（对应 config_registry，无 Configurable）
├── protocols.py       # 极简协议：ShardingConfig / StateDictAdapter / BaseModel
├── parallelism/
│   ├── mesh.py        # build_mesh()：从度数建 DeviceMesh（精简 ParallelDims）
│   ├── sharding.py    # apply_sharding()：ShardingConfig -> spmd_types redistribute
│   ├── fsdp.py        # apply_fsdp()：fully_shard 编排
│   ├── tensor.py      # apply_tp()/apply_ep()：声明式 TP/EP
│   ├── pipeline.py    # apply_pp()：pipelining schedule 切分
│   ├── context.py     # apply_cp()：flex all-gather KV
│   └── activation.py  # apply_ac()：激活检查点
├── models/
│   └── hf_wrapper.py  # HFTransformerModel 等价物（动态 import + init patch）
├── moe.py             # HF MoE -> 分组专家 MoE 替换
├── trainer.py         # Trainer：训练循环
└── checkpoint.py      # DCP 保存/加载 + HF 适配
```

### 2.2 核心接口签名（只定形状）

#### `bundle.py` —— 模型包（对外抽象 #1）
```python
@dataclass
class ModelBundle:
    model: nn.Module
    parallelize: ParallelizeFn            # (model, mesh, cfg) -> None
    pipeline: PipelineFn | None = None    # (model, mesh, cfg, loss) -> (schedule, parts, ...)
    state_dict_adapter: type[StateDictAdapter] | None = None

def hf_bundle(hf_model: str, *, moe: bool = False, **arch_overrides) -> ModelBundle:
    """构建一个 HF 模型的 bundle（动态 import + 包装 + 可选 MoE 替换）。"""
```

#### `config.py` —— 扁平配置（对外抽象 #2）
```python
@dataclass
class TrainConfig:
    # 并行度数（扁平，-1 = 从 world_size 推导）
    dp: int = -1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    # 训练
    lr: float = 3e-4
    steps: int = 1000
    tokens_per_microbatch: int = 8192
    max_seq_len: int = 2048
    # 模型
    hf_model: str = ""
    # 开关
    compile: bool = False
    activation_checkpoint: Literal["selective", "full", "none"] = "selective"
    ckpt_interval: int = 0
    # CLI 用 tyro 直接暴露本 dataclass；preset = 返回 TrainConfig 的普通函数

def debugmodel() -> TrainConfig: ...
def full_moe() -> TrainConfig: ...
```

#### `protocols.py` —— 极简 sharding（保留声明式灵魂，砍表面）
```python
@dataclass
class ShardSpec:
    """每模块可选地声明 sharding；比 Titan ShardingConfig 的 6 槽位收敛到 2-3 键。"""
    weight: Placement | None = None   # 参数如何切（spmd_types placement）
    out: Placement | None = None      # 输出如何放

def apply_sharding(model: nn.Module, mesh: DeviceMesh) -> None:
    """遍历带 ShardSpec 的模块, 用 spmd_types 在 forward 前后自动 redistribute。"""
```

#### `parallelism/mesh.py`
```python
def build_mesh(cfg: TrainConfig, world_size: int) -> DeviceMesh:
    """校验 world_size = dp*cp*tp*pp, 构建多维 mesh 及子网格 (dp/tp/pp/cp/ep)。"""
```

#### `parallelism/fsdp.py` / `pipeline.py` / `context.py`
```python
def apply_fsdp(model, mesh, *, mp_policy, reshard_after_forward=True) -> None: ...
def apply_pp(bundle, mesh, cfg, loss_fn) -> tuple[Schedule, list[nn.Module]]: ...
def apply_cp(model, mesh) -> None: ...   # flex all-gather KV
def apply_ac(model, mode) -> None: ...
```

#### `trainer.py`
```python
class Trainer:
    def __init__(self, bundle: ModelBundle, cfg: TrainConfig): ...
    def train(self) -> None: ...
    # 内部: build_mesh -> bundle.parallelize -> apply_ac -> compile -> apply_fsdp
    #       -> (可选) apply_pp -> 训练循环 -> checkpoint
```

### 2.3 明确砍掉（相对 Titan 的价值主张）

- `Trainer.Config` 嵌套树 + `Configurable`(`__init_subclass__`/`build`/`traverse`) 全套 → 扁平 dataclass + tyro
- `@override` / `config/transform` / `ModelConfigTransform` → preset 函数
- Module protocol 的 `init_states`/`verify_module_protocol` 递归 → 显式 `apply_sharding()`
- 12 原生模型注册表、`ModelSpec.traverse` → 只做 HF 后端
- PP 5 种 schedule / EP 6 种 dispatcher / AC 4 种策略 → **先各保留 1 个**（1F1B、all-to-all、selective），用到再加

---

## §3. 精简 Titan 路线（HF-only 发行版）

思路：**不拿掉 B 层（薄壳），而是删掉 Titan 里用不到的部分**，把 `spmd_types` +
这个实验模块的接口做成一个精简发行版。

### 3.1 保留（核心引擎 + 本模块）
```text
torchtitan/protocols/            (model/module/model_spec/sharding/state_dict_adapter)
torchtitan/distributed/          parallel_dims, fsdp, pipeline_parallel, spmd_types,
                                 compile, activation_checkpoint, context_parallel, utils
torchtitan/models/common/        (attention, moe, feed_forward, rope, embedding, ...)
torchtitan/components/           (data, loss, optimizer, checkpoint, tokenizer)
torchtitan/observability/
torchtitan/config/ (基础)         configs/manager —— 但砍掉 override/transform
torchtitan/trainer.py, train.py
torchtitan/experiments/transformers_modeling_backend/   <- 主体
```

### 3.2 删除（与本场景无关）
```text
torchtitan/models/{llama3,qwen3*,deepseek_v3/v4,gpt_oss,kimi*,flux,muse_glimmer}/
        -> 但保留 deepseek_v3 的 make_deepseek_v3_router_config (MoE router 复用),
           抽到 models/common/
torchtitan/experiments/ (除 transformers_modeling_backend 外)
torchtitan/config/override.py, config/transform/ (CP/quant/lora transform)
torchtitan/quantization/  (若不需要)
torchtitan/overrides/     (fused_mla/helion_rope 等实验覆盖, 按需)
torchtitan_recipes/, torchtitan/experiments/rl/
```

> 唯一需要小重构的：`moe_replacement.py` 依赖 `models.deepseek_v3` 的一个 router
> config 工厂。把它上移到 `models/common/` 即可删掉整个 deepseek_v3 模型目录
> （符合 TorchTitan "通用组件放 common/" 的既有原则）。

### 3.3 两条路线对比

| 维度 | 路线 A：彻底自研（§2） | 路线 B：精简 Titan（§3） |
|---|---|---|
| **要重写的引擎** | ~3.5k 行（module 协议 + ParallelDims + fsdp + pipeline + MoE）+ 训练循环/ckpt | ≈ 0（复用现有，仅删 + 1 处 router 上移） |
| **对外接口简洁度** | 最简（两个抽象 + 扁平 config） | 中（仍带 Trainer.Config，但砍了 override/transform） |
| **声明式 sharding** | 保留（重写 apply 引擎） | 保留（原样） |
| **跟上游 Titan 同步** | 断（自己维护） | 可定期 rebase/挑 cherry-pick |
| **风险** | 高 —— SPMD/PP/EP 引擎重写易引入静默数值错误 | 低 —— 用的是已收敛验证过的代码路径 |
| **适合** | 长期独立维护的框架、要彻底不同的设计哲学 | 想快速得到"HF-only、更小的 Titan" |
| **工作量量级** | 大（周级 + 数值验证） | 小（天级，主要是删除 + 测试） |
| **最终代码量** | ≈ 14,000 行（其中 ~4.5–6k 行要新写） | ≈ 31,000 行（新增 ≈ 0） |

> 代码量口径：Python 行数（含注释/空行），`wc -l` 实测当前仓库；两条路线都保留
> transformers_modeling_backend 本体（≈ 6,400 行）。明细见 §3.4。

### 3.4 代码量估算（实测）

**路线 B（精简 Titan）≈ 31,000 行。** 要保留的子模块实测累加：

| 保留部分 | 行数 |
|---|---|
| transformers_modeling_backend（主体） | 6,398 |
| models/common/（attention/moe/rope/embedding…） | 6,465 |
| components/（data/loss/optimizer/checkpoint/tokenizer） | 5,938 |
| distributed/（用到的子集） | 3,975 |
| observability/ | 3,058 |
| hf_datasets/ | 1,732 |
| trainer.py + train.py + models/utils | 1,763 |
| protocols/ | 1,081 |
| config/（仅 configs/manager/configurable，砍 override/transform） | 895 |
| **合计** | **≈ 31,300** |

相对全仓库核心（`torchtitan/` 去掉 experiments ≈ **64,000 行**），路线 B 砍掉约
**一半**——主要删：其余 11 个原生模型（~24.8k）、quantization/overrides/recipes
（~5.7k）、config override/transform（~1.5k）。**新增/改写代码 ≈ 0**（唯一重构：
把 `deepseek_v3` 的 router config 工厂上移 ~几十行）。

**路线 A（彻底自研）≈ 14,000 行，其中约 1/3 要新写。** 构成不同——保留的薄件更少，
但要新写一个引擎：

- **新写/重写（真实工作量）≈ 4,400–6,100 行**：
  - 并行引擎（module 协议 `parallelize()` + ParallelDims + fsdp + pipeline + MoE
    系列）：Titan 现状 3,935 行，砍冷门分支后自研 ≈ 2,500–3,500 行
  - 训练循环 + checkpoint（去掉 Configurable/嵌套 config）：Titan 现状
    1,145 + 1,909 行，自研精简 ≈ 1,500–2,000 行
  - 扁平 config + CLI + bundle 骨架（§2 新增）：≈ 400–600 行
- **可直接复用的薄件 ≈ 9,300 行**：模块本体 6,398 + spmd_types glue/compile/AC/
  CP-api/attention mask 2,317 + ShardingConfig/StateDictAdapter/BaseModel/各
  Config dataclass ~600。

**对比一览：**

| | 路线 B 精简 Titan | 路线 A 彻底自研 |
|---|---|---|
| 最终代码量 | ≈ 31,000 行 | ≈ 14,000 行 |
| 其中要新写的 | ≈ 0–100 行 | ≈ 4,500–6,000 行 |
| 相对全仓库 core（64k） | 砍 ~51% | 砍 ~78% |
| 高风险新代码 | 几乎无 | 全部集中在并行/训练引擎 |

**结论**：路线 A 最终代码只有路线 B 的 ~45%（1.4 万 vs 3.1 万），但代价是新写
4–6k 行最难的分布式引擎并重做数值验证；路线 B 代码量翻倍，但几乎不写新代码、
全程站在已验证路径上。折中路径（先 B 再逐个替换抽象）的最终代码量落在两者之间，
随替换进度逐步逼近 A。

### 3.5 建议

- **默认推荐路线 B**。它用 ~10% 的成本拿到 ~90% 的简洁，且站在已验证的数值
  正确性之上——这对分布式训练框架是决定性的（参考 Titan 自己的规则：保护
  已收敛代码路径、改分布式的静默正确性风险极高）。
- **选路线 A 的正当理由是**：你本就想设计一套不同的框架哲学（扁平 config、
  无 Configurable、极小协议），并接受为此重写引擎 + 做数值对拍。若是这样，
  务必用 Titan 的数值验证方法（`--debug.seed=42 --debug.deterministic`，
  loss/grad_norm bitwise 对拍）逐个并行维度验证新引擎。
- **折中**：先走 B 得到一个能跑的 HF-only 精简版，再把其中你想换掉的抽象
  （如 Trainer.Config -> 扁平 config）**逐个**替换成 §2 的形状——每次替换都
  能做数值对拍，比一次性重写安全得多。

---

## §4. 精度对齐方法论

两条路线的"对齐"含义根本不同：**路线 B 几乎不用对齐**（它复用的就是官方代码，
只需回归确认）；**路线 A 才是要对齐的那个**，而它面对的是分布式训练最难的问题——
逐位一致（bit-exact）。下面所有方法都基于仓库里**已存在**的工具。

### 4.0 先统一精度基线（任何对拍的前提）

1. **固定随机性**：`--debug.seed=42 --debug.deterministic`
   （`config/configs.py:361-382`）。deterministic 用确定性算法（更慢但可复现）。
2. **固定初始权重**：用 **seed checkpoint**（`loss_compare.py` 默认开启），
   两边从同一份初始权重出发，差异才来自引擎而非初始化。
3. **固定数据顺序**：同一数据集、同一 DP rank 分片、同一 batch 顺序。

> 铁律（项目规则）：**绝不**用 `--debug.deterministic_warn_only`——它只警告不报错，
> 会让不确定性悄悄混入。

### 4.1 现有工具（直接可用，不要重造）

| 工具 | 位置 | 作用 |
|---|---|---|
| `scripts/loss_compare.py`（1345 行） | 仓库根 | 主力。比较两个 commit / 两套配置的 loss，**自动开 deterministic + seed checkpoint**；`--assert-equal` 用于 CI 断言逐位一致；`--export-result/--import-result` 导出/导入基线。 |
| TensorBoard 全精度 | — | stdout 只打 5 位有效数字，**不够**；按项目规则用 `loss_compare.py` 的方式从 TensorBoard 取 loss/grad_norm 全精度。 |
| `numerical_equivalence.py` | 模块 `.claude/skills/add_moe_model/scripts/` | **模块级**对拍：单层 HF MoE vs Titan-replaced MoE，同一输入，比 **KL 散度 / cosine 相似度 / max abs diff**。 |
| `HF_BACKEND_LOGIT_DUMP` 钩子 | `model.py:1315` | 设环境变量后逐 forward dump 各 rank logits（含 CP 坐标），用于 CP-only vs CP+PP 对拍。 |
| `tests/` 数值测试 | 模块 `tests/` | `cp_pp_numerical.py`、`test_flex_cp_numerical.py`、`test_moe_parallelism.py` 等现成并行对拍。 |

### 4.2 路线 B：回归确认（而非"对齐"）

路线 B 复用官方引擎，loss 应当逐位等于官方。要做的就是验证**删除动作没引入意外**：

```bash
# 1. 在未改动的官方 commit 上导出期望 loss
python scripts/loss_compare.py . . \
  --baseline-module=transformers_modeling_backend \
  --baseline-config=transformers_modeling_backend_debugmodel \
  --export-result=expected_losses.txt

# 2. 在精简后的仓库上对拍 (不一致即非零退出, 作 CI 门禁)
python scripts/loss_compare.py . . --assert-equal \
  --baseline-module=transformers_modeling_backend \
  --baseline-config=transformers_modeling_backend_debugmodel \
  --import-result=expected_losses.txt
```

- 唯一重构点（`deepseek_v3` router config 上移）是纯搬移，属**非计算性改动**，
  必须 `seed=42 + deterministic` 下 loss/grad_norm **逐位一致**。
- 覆盖每个保留的并行维度组合各跑一次（dense 的 FSDP/TP/PP/CP + MoE 的 FSDP/EP）。

> 已知例外（README 明确）：HF 建模下 `FSDP=2` vs `FSDP=2+PP=2` 的 loss/grad_norm
> **不逐位一致**（但收敛），疑似 `register_buffer` 问题。对拍时**避开此组合**或标记
> 为已知 xfail，别当成自己的回归。

### 4.3 路线 A：分层对拍（真正的对齐战场）

路线 A 重写了引擎，必须证明**新引擎 == 官方引擎**。关键是**分层对拍**——不要一上来
比端到端 loss（差一点无法定位），而是自底向上逐层锁死：

```text
        ┌─────────────────────────────┐
   L4   │ 端到端: N 步 loss/grad_norm  │  最终判据 (bitwise 或收敛)
        ├─────────────────────────────┤
   L3   │ 单步: 前向 logits / 反向梯度  │  HF_BACKEND_LOGIT_DUMP 式对拍
        ├─────────────────────────────┤
   L2   │ 模块级: MoE / attention 块    │  numerical_equivalence.py 式
        ├─────────────────────────────┤
   L1   │ 原语级: 单个并行算子           │  all-gather/reduce-scatter/a2a
        └─────────────────────────────┘
```

原则：L1 逐位一致了再验 L2，依此类推；哪层开始出现差异，问题就在哪层。
**绝不要**跳层直接比 L4。

**各层方法：**

- **L1 原语级**：对单个集合通信（all-gather / reduce-scatter / all-to-all /
  redistribute）喂同一固定张量，断言 `torch.equal`。新 `apply_sharding`/`fsdp`/
  `pipeline` 引擎逐个过。
- **L2 模块级**：复用 `numerical_equivalence.py` 模式——单层、生产尺寸、随机权重、
  同一输入，比 **max abs diff（目标 0）** + cosine + KL。**优先这层**，它能隔离 MoE
  替换、attention 等最易错单元。
- **L3 单步级**：同一 seed checkpoint、同一 batch，比前向 logits 和反向梯度（逐位）。
  借 `HF_BACKEND_LOGIT_DUMP` 思路 dump 中间张量。
- **L4 端到端**：用 `loss_compare.py` 跑 N 步。**非计算性改动**（纯搬移/重构）必须
  **逐位一致**；**计算性改动**（重写了累加顺序）允许不逐位，但必须**收敛行为一致**
  （loss 曲线重合、grad_norm 量级一致）。

### 4.4 做不到逐位时怎么对齐（三类已知浮点来源）

`numerical_equivalence.py` 的注释列出三类导致**非逐位但语义等价**的来源，正是路线 A
最常踩的坑，逐条对齐即可消除：

1. **累加精度**：bf16 里 `scatter_add` vs f32 累加最后再 cast
   -> 对齐：**f32 累加，最后才降精度**。
2. **累加顺序**：expert 排序序 vs token 序（`scatter_add` vs `reshape(N,K,D).sum(1)`）
   -> 对齐：**unsort 回 token 序再 sum**。
3. **topk 排序**：`topk(sorted=False)` vs `sorted=True`（影响 f32 求和顺序）
   -> 对齐：**统一 `sorted=True`**。

该脚本证实：修掉这三类后，Mixtral/Qwen3-30B/DeepSeek-V2-Lite/OLMoE 全部
`max_diff = 0.00`。即"逐位对齐官方"**可达到**，关键在于把浮点累加的**精度与顺序**
与官方逐点对齐。

### 4.5 浮点噪音判据

- **bitwise（首选）**：同配置、同 seed、deterministic 下 `torch.equal`。
- **容差（次选，仅当语义等价但顺序不同）**：`max_abs_diff == 0` 理想；确有顺序差异时
  要求 f32 参考下相对误差 < 1e-6，且 **L4 收敛曲线重合**。
- **接受不逐位的唯一正当理由**：能**逐条解释**差异来自哪个累加顺序变化（如上面三类），
  且收敛一致。**解释不了 why 的差异，按 bug 处理，不要放过。**

### 4.6 落地为 CI 门禁（两条路线通用）

```bash
# 一次性: 官方仓库导出黄金基线 (基线文件随仓库提交)
python scripts/loss_compare.py . . \
  --baseline-module=transformers_modeling_backend \
  --baseline-config=transformers_modeling_backend_debugmodel \
  --export-result=golden/losses_debug.txt

# 每次改动后: 对拍 (失败即 CI 红)
python scripts/loss_compare.py . . --assert-equal \
  --baseline-module=transformers_modeling_backend \
  --baseline-config=transformers_modeling_backend_debugmodel \
  --import-result=golden/losses_debug.txt
```

再配一层模块级对拍（`numerical_equivalence.py` 风格）守护 MoE/attention 单元。
官方仓库升级时：先重新生成基线，再对拍。

### 4.7 一句话总结

- **路线 B**：不用"对齐"，用 `loss_compare.py --assert-equal` 做**回归确认**，避开已知
  的 `FSDP2 vs FSDP2+PP2` 不逐位坑。
- **路线 A**：**分层对拍**（原语->模块->单步->端到端），用 `numerical_equivalence.py`
  的方法把浮点**累加精度 + 顺序**逐条对齐到官方，目标 `max_diff=0`；解释不了的差异
  一律当 bug。端到端用 `loss_compare.py` + seed checkpoint 收口。

---

## §5. 若目标是深入学习分布式训练

如果真正的意图是**学懂分布式训练的核心模块**，那么结论和"做框架"完全相反——
**选路线 A（自研），但要换一种学法**。原因：学习的本质是把难的部分亲手做一遍，
而路线 B"太好了"——引擎现成，你只删不用碰，`ParallelDims`、ShardingConfig→
redistribute、FSDP 编排、PP schedule、EP dispatcher 这些核心一行都不用写，
学完只会"调用"，不会"造"。

### 5.1 核心模块 = 学习清单

分布式训练真正的硬核知识，全在 §1.3 标记为"厚/死结"的那 ~3.9k 行里。
这些只有重写一遍才会真正懂（看代码懂 30%，调通懂 60%，自己写出来并对拍到
bit-exact 才懂 90%+）：

| 核心模块 | 能学到什么 |
|---|---|
| `ParallelDims` / DeviceMesh | 多维进程拓扑、子网格切分、`world_size = dp*cp*tp*pp` 推导 |
| ShardingConfig -> redistribute | **声明式并行的灵魂**：placement 如何驱动 forward 前后的集合通信 |
| FSDP2 `fully_shard` 编排 | all-gather/reduce-scatter 时机、混合精度、reshard 策略 |
| PP schedule | 微批次流水、气泡、1F1B、stage 间 P2P |
| EP token dispatcher | MoE 的 all-to-all、token 路由与负载均衡 |
| CP | 序列切分 + 注意力的 KV all-gather |
| 数值对齐（§4.4） | 浮点累加顺序/精度为何决定 bit-exact -- 最隐性也最深的功夫 |

### 5.2 学法：逐模块重建 + 逐层对拍（而非一次性交付）

不要按 §2 那种"先定完整骨架再实现"的产品思路。学习导向应该：

1. **从最小可跑开始**：单 GPU、单模型、纯 PyTorch 训练循环（不用任何并行）打地基。
2. **一次只加一个并行维度**，每个都走"读懂官方实现 -> 自己写简化版 -> 用对拍证明等价"：
   FSDP（最简单，all-gather/reduce-scatter 两原语）-> TP（声明式 sharding）->
   PP（微批次 + P2P）-> CP / EP（最难，序列切分 / MoE all-to-all）。
3. **每加一维，就用 `loss_compare.py --assert-equal` 跟官方对拍**。对不上就去查——
   **"查为什么对不上"的过程就是学习本身**。§4.4 那三类坑（累加精度/顺序/topk 排序）
   大概率会亲自踩一遍，踩过就刻进脑子里。

> §4.3 的分层对拍金字塔在这里不只是"验证手段"，更是**学习脚手架**：
> L1 逼你懂集合通信，L2 逼你懂 MoE/attention，L4 逼你懂端到端数值。

### 5.3 用官方代码当"参考答案"，而不是"要删的包袱"

学习场景下路线 A 和 B 不是二选一，而是**用 B 当 A 的参照系**：

- 保留官方仓库在旁边（不动它），当作**标准答案 + 对拍基线**。
- 在旁边另起一个最小框架，**逐个模块重写**，每写完一个就跟官方比。
- 卡住了就去看官方怎么写（如 `distributed/fsdp.py` 431 行），看懂再回来写自己的。

这样既有"亲手造"的深度，又有"随时对答案"的安全网——避免纯自研最常见的死法：
引擎写错了却不知道，在错误的基础上越学越偏。

### 5.4 起步顺序（每步都能跑、能对拍）

```text
第 0 步: 单卡纯 PyTorch 训练一个 HF 小模型 (Llama-3.2-1B 或 debugmodel)
         -> 只搞懂 train loop / loss / optimizer / backward
第 1 步: 加 FSDP (fully_shard)         -> 搞懂数据并行 + 混合精度
第 2 步: 加 TP (spmd_types 声明式)     -> 搞懂 placement 驱动通信
第 3 步: 加 PP (pipelining 1F1B)      -> 搞懂微批次流水
第 4 步: 加 CP 或 EP (选一个深入)      -> 搞懂序列并行 或 MoE
每一步: 用 loss_compare.py 跟官方对拍到一致为止
```

全程只需 `spmd_types` + `torch.distributed` + `transformers` 三个外部依赖，其余自己写——
正好逼你把每个核心模块吃透。

> **一句话**：学习分布式训练，选 A，但以"逐模块重建 + 每层对拍 + 官方当参考答案"
> 的方式学，而不是当产品一次性交付。路线 B 留给"将来真要用这个框架"的那天。

