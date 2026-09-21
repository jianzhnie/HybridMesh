# HybridMesh 设计文档

一个移除 TorchTitan `Configurable` 与 `Module` 两层抽象、直接接入 HuggingFace
`transformers` 的大模型并行训练框架。

## 0. 结论

hpmesh 拿掉了 TorchTitan 的 `Configurable` 与 `Module` 两个抽象层，换来一个明显更短
的框架：约 91 个模块、20.6k 行，覆盖 TP / FSDP2 / CP / EP / PP 五条并行路径的装配、
训练循环、checkpoint 与等价性测试。

拿掉抽象不等于拿掉复杂度，只是把复杂度换成另一种形式。hpmesh 选择的形式是：

- **配置只沿一个方向流动**：`CLI -> Config -> 顶层显式传参`，没有 config 树遍历、
  没有 `build()` 递归物化。
- **契约是函数签名，不是 protocol 类**：每个 `apply_*` 是一个普通函数，依赖写在
  参数列表里。
- **模型契约是 nn.Module 的现有形状**：`named_children` 暴露固定五个部件名，
  不需要模型继承任何框架基类。

## 1. 背景：拿掉的是什么

### 1.1 `Configurable`（torchtitan/config/configurable.py，180 行）

三段式协议：每个组件继承 `Configurable`，定义嵌套
`class Config(Configurable.Config)` dataclass，`__init__` 只收一个 config；
`__init_subclass__` 自动把 `Config._owner` 接线回组件类，于是任何组件都能用
`config.build()` 物化。Trainer 本身也是一个 `Configurable`，整个训练任务是一棵
Config 树：`Trainer.Config` 组合 `model_spec / optimizer / lr_scheduler /
dataloader / tokenizer / checkpoint / loss / metrics ...`，训练入口就是
`config_manager.parse_args().build().train()`。

代价（torchtitan fork 实测规模）：

| 部分 | 规模 |
|---|---|
| Configurable 本体 | 180 行 |
| 配置机器（CLI / override / 组合 / Function） | ~1278 行 |
| 全仓库 Configurable/Module 子类 | ~177 个，嵌套 Config 268 个，波及 58 个文件 |
| HF 适配后端（transformers_modeling_backend） | ~3700 行 |

每个组件要写两个类（Config + 本体）；`slots=True` dataclass 与 HF
`PretrainedConfig` 不兼容，HF 后端被迫双继承并覆写 `__init__` / `build` /
`_replace` 三处——这是适配成本最集中的地方。

### 1.2 `Module` protocol（torchtitan/protocols/module.py，589 行）

在 `nn.Module` 上叠加三件横切能力：递归权重初始化（`init_states` 按
`param_init` 查表）、声明式 SPMD 并行化（`ShardingConfig` 挂在 config 上，
`parallelize()` 递归分片并把 forward 包成「输入 redistribute -> forward -> 输出
redistribute」）、remat 区域命名。配套还有 `BaseModel`（`preprocess_inputs`、
`verify_module_protocol`）、协议兼容容器 `ModuleList/ModuleDict/Sequential`
（+`protocols/` 其余文件合计 ~1081 行）。

代价：模型要么继承这套基类，要么写转换层。HF 模型走的是后者——wrapper 里猴子
补丁 HF 的权重初始化、给 HF 子模块逐个挂 `ShardingConfig`（hf_sharding.py 506
行）、再做模块结构转换。

### 1.3 边界：哪些是 TorchTitan，哪些不是

"拿掉抽象"不等于重写整个并行层。hpmesh 依赖的底层里有三样东西与 TorchTitan
无关，照用即可：

| 依赖 | 归属 | 提供什么 |
|---|---|---|
| `torch.distributed` | PyTorch | `DeviceMesh` / `ProcessGroup` / FSDP2 / `pipelining` |
| `spmd_types` | 独立 Meta pip 包（`spmd_types==0.2.5`，~11.5k 行，零依赖） | SPMD 类型系统：`SpmdType` / `TensorSharding` / `MeshAxis` / `assert_type` / `local_map` / `redistribute`，以及 mesh 作用域 `set_current_mesh` |
| `transformers` | HuggingFace | `AutoModelForCausalLM`、模型自带的 `_tp_plan` |

hpmesh 的 SPMD glue shim **不是** TorchTitan 的抽象，而是套在
`spmd_types` 包外面的一层 glue（`MeshAxisName` 映射、TLS mesh 栈、state-dict
转换、redistribute 校验），现按依赖方向拆成两层：底层 `utils/spmd_context.py`
（mesh 作用域 + 轴查询，只依赖 torch + PyPI `spmd_types`），上层
`parallel/spmd_shims.py`（state-dict 转换与 redistribute 校验，仅 parallel 层用）。

真正被移除后需要补回的接口面其实很窄，全部用普通 Python 手段补回：

| TorchTitan 机制 | hpmesh 替代物 |
|---|---|
| `Config.build()` 构造协议 | 构造函数显式传参（`Trainer(cfg)`、`HFTransformerModel(hf_config)`） |
| `init_states` 递归初始化 | HF 模型自己的 `_init_weights`（`model_cls(config)` 内部完成） |
| 声明式 `ShardingConfig.parallelize()` | 顶层函数 `apply_tp / apply_cp / apply_ep / apply_fsdp`，顺序写在 `parallelize_hf.py` |
| `preprocess_inputs` | trainer 自己切 batch（`_dp_slice`），模型 forward 收 flat token |
| `state_dict_adapter` | 暂不需要：checkpoint 每 rank 各存各的（见 §5.6） |
| `ModelSpec` / registry | 不需要：TP plan 直接用 HF 模型自带的 `_tp_plan`（见 §4.2） |

## 2. 设计目标

| | 目标 |
|---|---|
| G1 | 任意 HF `AutoModelForCausalLM` 不改模型代码即可并行 |
| G2 | 配置只沿一个方向流动（`CLI -> Config -> 顶层显式传参`） |
| G3 | 每个 `apply_*` 只依赖它的契约（函数签名），不 import trainer |
| G4 | 任何非计算改动必须逐位一致，且有等价性测试兜底 |

G1 决定模型层只能依赖 HF 公共约定（`config.architectures`、常见 embed/norm 命名、
`_tp_plan`），不能要求模型作者配合。G4 是允许大胆删抽象的前提：CP/EP/TP 都有
"多卡分片 == 单卡全量"的数值等价测试（见 §7）。

## 3. 总体架构

```
+---------------------------------------------------------------+
|  CLI (HfArgumentParser)              trainer/config.py        |
|  Model/Parallel/Optimizer/TrainingArguments -> HybridMeshConfig|
+---------------------------------------------------------------+
                          |  组装层读取; 不向下传
                          v
+---------------------------------------------------------------+
|  trainer/trainer.py                                           |
|    init_distributed -> build_mesh -> HFTransformerModel       |
|    -> parallelize_hf_transformers -> AdamW -> train loop      |
+---------------------------------------------------------------+
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
+----------------+ +----------------+ +----------------+
| apply_tp       | | apply_cp/ep    | | apply_fsdp     |
| (m, mesh, cfg) | | (m, mesh, cfg) | | (m, mesh, cfg) |
+----------------+ +----------------+ +----------------+
        |                 |                 |
        +-----------------+-----------------+
        顺序即契约, 写在 parallel/parallelize_hf.py 一个文件里
                          |
             [SEAM 1]  HF wrapper 契约 (§4.2)
                          |
+---------------------------------------------------------------+
|  models/hf_wrapper.py    唯一的 HF wrapper                    |
|    forward(input_ids, *, positions, attention_masks) -> logits|
|    named_children() -> tok_embeddings/layers/norm/lm_head/... |
|    tp_plan property    <- 重写 HF 模型自带的 _tp_plan          |
+---------------------------------------------------------------+
                          |
             [SEAM 2]  分布式运行时上下文 (线程局部, §4.3)
                          |
+---------------------------------------------------------------+
|  utils/spmd_context.py                                    |
|    spmd_context(parallel_dims)  # contextmanager              |
|    spmd_mesh_group(axis) / spmd_mesh_size(axis)               |
+---------------------------------------------------------------+
                          |
+---------------------------------------------------------------+
|  models/common/*  (rope, masks, qkv, aux_loss, moe, dist_gemm)|
|    通过上下文取 group, 不接收 cfg, 不 import trainer            |
+---------------------------------------------------------------+
```

分层方向：`trainer -> parallel -> models` 单向，反向零依赖。

目录结构（91 模块，约 20.6k 行）：

```
hpmesh/
  __main__.py / __init__.py     入口: python -m hpmesh
  mesh.py                       init_distributed / build_parallel_dims / build_mesh
  trainer/      4 模块          config.py / trainer.py / train.py
  models/      20 模块          hf_wrapper.py + common/{rope,masks,qkv,moe,...}
  parallel/    23 模块          config.py (ParallelConfig) + tensor_parallel/
                                fully_shard/ pipeline_parallel/ context_parallel/
                                expert_parallel/ spmd_shims.py (state-dict/校验)
                                parallel_dims.py parallelize_hf.py sharding.py
  components/  14 模块          loss / checkpointer(DCP) / metrics / profiler /
                                optimizer(lr_scheduler)
  datasets/    18 模块          Grain 数据图 + random_data + hf/{text,multimodal}
  utils/        9 模块          spmd_context.py (SPMD mesh 作用域 + 轴查询, 最底层)
                                checkpoint_keys.py (checkpoint 键常量)
```

## 4. 核心契约（三条缝）

### 4.1 SEAM 0：唯一配置入口

`HybridMeshConfig`（trainer/config.py）由四组 dataclass **组合**（不是多继承）：
`ModelArguments / ParallelArguments / OptimizerArguments / TrainingArguments`。
CLI 用 `HfArgumentParser` 平铺解析四组 flag，组合后经
`cfg.auto_fill_model()`（hub id 时从 HF 拉架构补齐）得到唯一配置对象。

配置流动遵守 G2：只有 trainer 组装层读 `HybridMeshConfig`；往下传递时拆成显式
参数——`ParallelDims.from_config(cfg.parallel, world_size)` 读度数，
`apply_*` 收 `cfg.parallel`（`ParallelConfig`，与其余配置组一起住在
`trainer/config.py`），只读自己的字段（`cfg.tp` / `cfg.cp` 等短别名 property）；少数
训练侧标量（`compile` / `global_batch_size` / `dataset`）由调用方显式传入。
没有 config 树 `traverse`，没有运行时 override 机制——要改配置就改 CLI flag 或
改 dataclass 默认值。

### 4.2 SEAM 1：HF wrapper 契约

`HFTransformerModel(nn.Module)`（models/hf_wrapper.py）是唯一 wrapper，
`__init__(config: PretrainedConfig)` 内按 `config.architectures` 解析
`ForCausalLM` 类并直接 `model_cls(config=config)`——用 HF 自己的初始化，
无 monkey-patch。对并行层暴露的契约只有三条：

1. **forward 签名**：`forward(input_ids: (T,) flat, *, positions=None,
   attention_masks=None) -> logits`。token 是一维平坦流（packing 是一等公民），
   wrapper 内部加/去 batch 维；RoPE 由显式 `positions` 驱动；flex 路径用
   `attention_masks` 构造 BlockMask。
2. **部件命名**：`named_children()` 固定 yield `tok_embeddings / layers / norm /
   lm_head / rotary_emb` 五个部件（embed 名按 `embed_tokens/wte/...` 探测一次，
   norm 同理），并行层 walk children 时看到的是部件而不是单个 `model` blob；
   state_dict key 因此不带 `model.` 前缀，与 HF checkpoint 对齐。
3. **TP plan**：`tp_plan` property 把 HF 模型自带的 `_tp_plan` 统一重写为本
   wrapper 的模块路径。声明是纯数据，且数据源在 HF 侧——这就是不需要
   TorchTitan 式 model registry 的原因。

### 4.3 SEAM 2：apply_* 函数契约与分布式上下文

每个并行维度一个顶层函数，签名统一（`cfg` 是 `ParallelConfig`——parallel 层
不接触 `HybridMeshConfig`，其它关注点如 `compile` 走显式参数）：

```python
def apply_tp(model, mesh, cfg, plan=None) -> nn.Module      # tensor_parallel/tp.py
def apply_cp(model, mesh, cfg) -> nn.Module                 # context_parallel/apply.py
def apply_ep(model, cfg, *, ep_group=None) -> nn.Module     # expert_parallel/apply.py
def apply_fsdp(model, mesh, cfg, parallel_dims) -> nn.Module # fully_shard/
```

公共语义：`mesh is None` 或对应度数 `<= 1` 时 no-op 原样返回；否则返回就地改造
后的模型。**顺序即契约**，整个框架的编排知识集中在
`parallel/parallelize_hf.py`（81 行）一个文件里：

```
pp>1 时转入 pipeline_parallel.apply_pp（切 stage -> 每 part 过 tp/compile/fsdp
-> 建 schedule），返回 PipelineParallelSetup；pp=1 时保持：
apply_tp -> apply_ep -> apply_cp -> torch.compile(可选) -> apply_fsdp   # FSDP 最后, outer wraps inner
```

模型内部组件（`models/common/*`）不接收 cfg、不 import trainer，需要的分布式
状态全部走线程局部上下文：`spmd_context(parallel_dims)` 是唯一的 ambient 状态
入口（trainer 在 fwd/bwd 时进入），`spmd_mesh_group("cp")` /
`spmd_mesh_size("tp")` 是查询口。singleton 轴返回 `None`/1，组件代码无须
分支判断"是否启用某并行"。

## 5. 模块设计

### 5.1 trainer

`train.py` 的 `main()` 一行说完全部：`Trainer(parse_config()).train()`。
`Trainer.__init__` 顺序固定：`init_distributed`（无 torchrun 则单进程
`(0,0,1)`）-> 定种子 -> `build_parallel_dims` / `build_mesh`（含
`covered != world_size` backstop 校验）-> 建模型 -> `parallelize_hf_transformers`
-> `AdamW`。

`train_step`：` _dp_slice` 切本地数据 -> `forward_backward_step`（在
`spmd_context` 内执行）-> `clip_grad_norm_`（跨 PP stage 归约）->
`_check_finite`（`torch._assert_async`）-> `optimizer.step` -> token 归一化
loss。

### 5.2 mesh 与 ParallelDims

`mesh.py` 刻意不自己建多维 mesh：`MESH_AXES = ("dp", "cp", "tp")`，只提供
`init_distributed` / `build_parallel_dims` / `build_mesh` 三个函数，所有视图由
`ParallelDims`（parallel/parallel_dims.py）统一 unflatten。PP 不在 mesh 轴上
（PP 是 stage 切分而非张量分片），`build_mesh` 的覆盖率校验天然成为 pp>1 的
backstop。

### 5.3 TP（tensor_parallel/tp.py）

声明层是纯数据：`ShardingConfig(kind, implementation)` frozen dataclass +
`colwise()/rowwise()` 工厂。实现层两个 fused collective+GEMM 模块：
`ColwiseLinear`（存 `[in, out/tp]`，配 all-gather）与 `RowwiseLinear`
（`[out, in/tp]` 切 dim1，配 reduce-scatter），均为 sequence-parallel 形态。
plan 为 None 时读 `model.tp_plan`（即 HF `_tp_plan` 的重写版），按路径深度
倒序替换 `nn.Linear`；遇 bias 直接 raise。可选注册对称内存
（`enable_fsdp_symm_mem`）。不做：meta-init、fused QKV、FP8。

### 5.4 CP / EP（context_parallel/ + expert_parallel/）

**CP 已接线**。拦截点是 `hf_wrapper._flex_attention_hf` 读取的
`_titan_flex_kernel`：`apply_cp`（cp>1）walk 每层 attention module 并 attach
`CPFlexKernel`（`context_parallel/cp_kernel.py`，默认 KV all-gather 策略：k/v 经
torch 的 `flex_cp_allgather` 收成全长，q 保持 token 分片；Ulysses 留
`strategy` 参数位，首版 raise）。输入分片在 trainer 侧：
`context_parallel/input_shard.py` 的 `shard_batch_for_cp`（封装 torch 私有
`_context_parallel_shard`，支持 headtail load balancer）把
input_ids/labels/positions 同步切片；BlockMask 只沿 Q 维分片
（`shard_attention_mask_for_cp`）。loss/token 归约走含 cp 轴的 `loss` mesh。

**EP 已接线**。`parallel/expert_parallel/ep.py` 的 `swap_hf_moe_blocks` 把 HF MoE block（首版支持
Qwen3Moe 形态：`gate` + `experts` ModuleList，duck-typed 探测，识别不了就带模型
类名 raise）替换为 `models/common` 的 `MoE`：router gate 与 experts 权重逐元素
直拷进 `TokenChoiceTopKRouter` / `GroupedExperts`；ep>1 时每 rank 切本地
experts 片并接 `AllToAllTokenDispatcher`（`wire_meshes(ep_group=...)`），ep==1 用
`LocalTokenDispatcher`。负载均衡 loss 走 router 上的
`MicrobatchWiseLoadBalanceLoss`（coeff 取 HF config 的 `router_aux_loss_coef`），
trainer 以梯度注入 hook 接线，不改主 loss 值。

已知边界：CP 要求 flex backend 与 `seq_len` 的整除约束（纯 CP 的梯度归约已
随 `fsdp_enabled` 入口修复解决，见 §8）。

### 5.5 PP（pipeline_parallel/）

`pipeline.py` 提供两件纯函数构件：`generate_llm_fqn_per_model_part`（纯算术，
决定哪层去哪个 stage）与 `split_model_into_stages`（每 stage deep-copy、删掉不
属于自己的部分——保留原始层索引以避免跨 rank state_dict 撞名——包成
`PipelineStage`）。vendored 自 torchtitan，改动全是删除 protocol 层。

**闭环已落地**：`pipeline_parallel/pp.py` 的 `apply_pp` 按 schedule 类推导
stage 数（looped schedule 默认每 rank 2 个），切分后对每个 model_part 依次跑
`apply_tp` → `apply_fsdp`（与单卡路径同序）；`build_pipeline_schedule` 建
schedule（`scale_grads=False`，loss 是 sum 由 trainer 归一）。trainer 侧：
`_pp_forward_backward_body` 驱动 `schedule.step`——首 stage 收 `input_ids`、
末 stage 收 labels 并返回 detach 求和的 loss 与 token 数、其余 stage 返回哨兵
-1.0；optimizer 是 `components/optimizer/` 的 `OptimizersContainer`，每个
model_part 一个内层 optimizer；checkpoint 的 optimizer state 一律按参数 FQN
扁平存取（`state.<fqn>.exp_avg` 形式），positional 索引跨 stage 撞键的问题因此
不复存在——注意非 PP 也不再是 positional 格式，见
`docs/optimizer_checkpoint_format.md`。

已知边界：pp+cp / pp+ep 组合显式 raise；tied embeddings 拒绝（deepcopy 会拆断
共享权重）；只支持 `dataset="random"`（packed 语料的 positions 没有穿过
schedule 的通道）；looped schedule 代码就绪但只有 1F1B 被等价测试覆盖。

### 5.6 components / datasets

- `loss.py`：vendored 自 torchtitan 的 `cross_entropy_loss`（sum 归约）、
  `next_token_targets`、`vocab_shard_bounds`。
- Checkpointer：DCP 格式，`ModelWrapper` 支持多 model_part；optimizer state 由
  `OptimizersContainer` 序列化为扁平 FQN 字典（不再有 `OptimizerWrapper`，
  也无论 PP 与否都是同一格式）。
- `random_data.py`：`RandomTokenSource` 确定性合成语料——`(seed, step)`
  唯一决定 batch，等价性测试和 smoke run 不需要真实数据集。

## 6. 一次训练步骤的数据流

```
RandomTokenSource --(seed,step)--> Batch(input_ids, labels)   # 全局一致
trainer._dp_slice                -> 本 dp rank 的批切片（PP 时按 "batch" 轴）
shard_batch_for_cp（CP 时）      -> input_ids/labels/positions 沿序列维切片
spmd_context(parallel_dims)      -> TLS 压入 dense/sparse mesh
HFTransformerModel.forward       -> tok_embeddings -> layers -> norm -> lm_head
    每层内: TP 的 Colwise/RowwiseLinear 就地做 collective
            CP 的 K/V all-gather 在 attention 前收成全长（Ulysses 预留）
            EP 的 all-to-all dispatcher 在 MoE 前后换位
PP 时: schedule.step(arg_mbs / target_mbs) 驱动各 stage，末 stage 出 loss
loss (sum 归约, loss mesh) -> backward -> clip_grad_norm_ (跨 PP 归约) -> AdamW.step
```

## 7. 正确性验证策略

目录沿用 torchtitan 的分法：`tests/unit_tests/cpu/` 是 pytest 套件（把子系统
建目录），`tests/integration_tests/` 放 torchrun 起的等价性脚本 —— 后者不是
pytest，`testpaths` 不收集它们。

G4 的兜底是 `integration_tests/` 里那套"分片 == 全量"的等价性测试，全部
torchrun 2 ranks gloo 起，自建 mesh、不依赖 trainer 装配（pp_equivalence 除外，
它驱动真实 Trainer）：

| 测试 | 验证什么 |
|---|---|
| `tests/integration_tests/cp_equivalence.py` | CP 原语：分片注意力逐位 == 单卡全序列；Ulysses 往返；non-vacuity 反证 |
| `tests/integration_tests/cp_wiring_equivalence.py` | CP 接线：`apply_cp` 后分片 logits/loss == 单卡全长；含 causal/headtail/packed 三场景与 gather backward 微测 |
| `tests/integration_tests/ep_equivalence.py` | EP 原语：all-to-all MoE == 单卡全专家 MoE；fp64 对照区分归约噪声与路由错误 |
| `tests/integration_tests/ep_wiring_equivalence.py` | EP 接线：EP=2 替换后输出 == EP=1 == 原 HF；每 rank 只持本地 expert 切片 |
| `tests/integration_tests/ep_fsdp_equivalence.py` | EP×FSDP：ep=2+dp_shard=4 经 `parallelize_hf_transformers` 的 loss/梯度 == 单卡全批参照；专家参数必须落在 efsdp mesh（`moe_enabled` 回归钉） |
| `tests/integration_tests/moe_aux_loss_grad_equivalence.py` | MoE aux loss：cp=2 归约的 forward all-reduce / backward identity 语义；router 梯度 == 单卡拼接流参照 |
| `tests/integration_tests/pp_equivalence.py` | PP 闭环：pp=2 经真实 Trainer 跑 4 步，loss 轨迹逐位 == 同 chunking 单卡参照 |
| `tests/unit_tests/cpu/distributed/test_ep_swap.py` | MoE 替换单测：权重逐位直拷、logits 等价、aux loss 注入 |
| `tests/unit_tests/cpu/distributed/test_tp.py` | TP 声明层 / 权重布局 / plan 解析（CPU 单测） |
| `tests/unit_tests/cpu/distributed/test_pipeline.py` | PP 切分算术 + `split_model_into_stages` 部件归属 / 级联 forward 等价 |
| `tests/unit_tests/cpu/test_trainer.py` | loss / 数据迭代器 / checkpoint / collectives |
| `tests/unit_tests/cpu/utils/test_spmd_context.py` | ambient 上下文的 no-op 与恢复语义 |

规则：任何"只改结构不改语义"的改动（换容器、改装配顺序、拆函数）必须保持
这套测试逐位通过。

## 8. 现状与验证清单

已落地：TP（SP 前提接线：输入沿 TP 组切序列；CPU/gloo 回退路径；复制参数梯度
跨 TP 组归约）、FSDP2（mesh 按 torchtitan 轴语义重建；纯 dp_replicate 的 DDP
兜底）、CP（KV all-gather + Ulysses 接线）、EP（Qwen3Moe 形态替换 +
all-to-all dispatcher + FSDP moe_enabled 接线）、PP（1F1B/Interleaved1F1B
闭环 + DCP 续训）、训练循环、上述等价性测试。

对齐审计（对照 torchtitan 逐项核对）后修掉的主要 BUG：TP 权重布局双重转置、
3D 激活喂 2D 原语、TP 序列不切分导致梯度放大 tp 倍、FSDP 丢弃
`DataParallelMeshDims` 导致的多轴错读、纯 dp_replicate 梯度不归约、EP swap
缺 `moe_enabled` 接线、aux loss 归约 backward 语义错误（梯度放大
group_size 倍）、CP+packed 单文档批次必崩、random 数据源 resume 数据流断裂、
loss 上报 collective 的局部门槛挂死风险。

剩余边界（均为 loud-raise，不静默错）：

1. pp+cp / pp+ep 组合未接线；tied embeddings 的 PP 拒绝。
2. ptrr load balancer 未实现；Ulysses 不支持 packed（BlockMask Q 分片与全长
   attention 冲突，attach 时拒绝）且不与 load balancer 组合。
3. looped PP schedule 已覆盖 Interleaved1F1B；V 风格（DualPipeV/ZBV）未测，
   `pipeline_parallel_schedule_csv` 拒绝。
4. EP 支持 Qwen3Moe 形态为主（transformers 5.x fused experts 布局）；带
   per-expert bias 的 GPT-OSS 等显式拒绝。
5. ep>1 且 `max_norm > 0` 拒绝训练：EP 感知的 grad norm 分组归约未移植
   （专家参数是 plain tensor，torchtitan 的 DTensor 分组路径前提不成立）；
   `max_norm <= 0` 可跑但上报的 grad_norm 缺 EP 组归约（仅指标偏差）。
6. EP>1 的 checkpoint 不对称：专家权重是各 rank 内容不同的 plain tensor，
   DCP checkpointer 按 replicated 存取——save 只留单 rank 切片，resume 会把
   同一切片广播到所有 rank（塌缩）。修复依赖专家参数 DTensor 化，未做；
   `apply_ep` 在 ep>1 时打 warning 提示不要 resume。
7. TP+FSDP（dp_shard>1）组合与 4-rank 以上混合拓扑只有 mesh 形状 smoke，
   无完整训练等价测试；CUDA symm-mem fused 路径本机（macOS）未实测。
8. loss 上报的归约 mesh 未含 tp 轴：TP 开启时上报 loss 偏小的显示问题
   （不影响梯度）；`efsdp*ep > num_experts` 的 placement 选择比上游严格
   ep 倍（数值正确、placement 次优）。

多卡 GPU 验证清单（开发机是 macOS，无 nccl；以下在 GPU 环境执行）：

```bash
# 等价性（nccl）
PYTHONPATH=. torchrun --nproc_per_node=4 tests/integration_tests/cp_wiring_equivalence.py
PYTHONPATH=. torchrun --nproc_per_node=4 tests/integration_tests/cp_ulysses_equivalence.py
PYTHONPATH=. torchrun --nproc_per_node=4 tests/integration_tests/ep_wiring_equivalence.py
PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/pp_equivalence.py            # 1F1B
PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/pp_equivalence.py Interleaved1F1B
PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py full  $W
PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume $W

# 端到端 smoke：各维度单独与组合，loss 对拍单卡基线
torchrun --nproc_per_node=2 -m hpmesh --tensor_parallel_size 2 --steps 20 ...
torchrun --nproc_per_node=2 -m hpmesh --context_parallel_size 2 --steps 20 ...
torchrun --nproc_per_node=2 -m hpmesh --pipeline_parallel_size 2 --steps 20 ...
torchrun --nproc_per_node=4 -m hpmesh --tensor_parallel_size 2 --context_parallel_size 2 ...
torchrun --nproc_per_node=4 -m hpmesh --pipeline_parallel_size 2 --tensor_parallel_size 2 ...
```
