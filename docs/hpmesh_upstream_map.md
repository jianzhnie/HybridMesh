# hpmesh → torchtitan 对应关系表

[hpmesh](../hpmesh) 拿掉了 TorchTitan 的 `Configurable` 与 `Module` 两个抽象层，换来一个
明显更短的框架：94 个 Python 模块、约 27.4k 行，覆盖 TP / FSDP2 / CP / EP / PP 五条
并行路径的装配、训练循环、checkpoint 与等价性测试。本文是这些模块与 torchtitan 之间
对应关系的**唯一权威**。

## 怎么用这张表

**先查表，再动手。** 不要在 hpmesh 源码里加 `# upstream: <path> @ <sha>` 之类的来源
标注——那种标注试过又被退了，因为 sha 会腐烂而表不会。

**分类不是装饰，是操作指令。** 把 A 类的高保真同步规则套到 B 类文件上会毁掉设计；套到
C 类上会把项目**故意删掉**的抽象又拽回来。

**比结构用 AST，不要比 diff 行数。** `utils/filesystem.py` 的 diff 有几十行，代码差异
是 **0**——全是改写措辞。做法是剥掉 docstring、`ast.unparse`、再
`difflib.SequenceMatcher`。下表 `ratio` 列就是这么来的。

目录级速查（细节以下方分类表为准）：

| hpmesh | torchtitan |
| --- | --- |
| `parallel/**` | `distributed/` |
| `components/metrics.py` | `observability/metrics.py` |
| `models/common/scatter_add.py` | `ops/scatter_add.py` |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` |
| `datasets/text/text.py` | `hf_datasets/text_datasets.py` |
| `utils/filesystem.py` | `tools/filesystem.py` |

## 图例

- **A 移植（vendored）**——从 torchtitan 移植。上游改了，hpmesh **应该**逐项核对。
- **B 适配（adapted）**——同一个想法、不同的形状。上游改了，**读意图、不要抄形状**。
- **C hpmesh 独有**——上游没有对应物（或同名不同源）。不要"对齐"它。
- **D 缺口**——上游有，hpmesh **真的没有**。补还是不补是决策，不是疏漏。

分类按文件的**主要维护策略**划分；一个文件只能有一个主分类。文件内部若混合了移植与
适配逻辑，在"改写点"列中另行说明，避免同一文件同时收到互相冲突的操作指令。

`ratio` 是 AST 相似度的历史快照，**历史行是"全量最佳匹配"**（重现脚本的第 1 列），
仅对 A/B 类有意义。**1.000 不代表逐字相同**——剥掉 docstring 后 `ast.unparse` 归一化了
空白和引号；要判断"真逐字"，看 ratio 为 1.000 且人工确认过的那两个。少数行的"改写点"
列会额外给出**同名比较**值——两者差得远时，最佳匹配多半是噪音，以同名值为准
（见"重现这张表"末尾的教训）。

2026-09-21 复核后从 A1 移入 A2 的十行使用当前工作树的**指定对应文件直接比较**值，不是
全量最佳匹配；这些值用于解释为何维护策略已经变化，不与历史排名混用。

## A1 —— 高保真移植（改动需逐位验证）

只有 `components/checkpointer/utils.py` 和 `utils/filesystem.py` 在该快照中经人工确认
属于"去 docstring 后结构等价"；其余行即使 ratio 很高也不是逐字复制。

| hpmesh | torchtitan | ratio |
| --- | --- | --- |
| `components/checkpointer/utils.py` | `components/checkpointer/utils.py` | 1.000 |
| `utils/filesystem.py` | `tools/filesystem.py` | 1.000 |
| `components/optimizer/utils.py` | `components/optimizer/utils.py` | 0.996 |
| `datasets/multimodal/mm_image.py` | `hf_datasets/multimodal/utils/image.py` | 0.977 |
| `datasets/multimodal/mm_text_utils.py` | `hf_datasets/multimodal/utils/text.py` | 0.971 |
| `datasets/multimodal/mm_video.py` | `hf_datasets/multimodal/utils/video.py` | 0.967 |
| `components/tokenizer.py` | `components/tokenizer.py` | 0.907 |

## A2 —— 移植但已改写（比例中等，需逐处核对）

| hpmesh | torchtitan | ratio | 改写点 |
| --- | --- | --- | --- |
| `components/checkpointer/__init__.py` | `components/checkpointer/__init__.py` | 0.757 | |
| `components/checkpointer/base.py` | `components/checkpointer/base.py` | 0.717 | |
| `components/checkpointer/dcp.py` | `components/checkpointer/dcp.py` | 0.735 | 本地/remote storage、HF export 与生命周期已重塑 |
| `components/checkpointer/torch_checkpointing.py` | `components/checkpointer/torch_checkpointing.py` | 0.265 | |
| `components/loss.py` | `components/loss.py` | 0.297 | 去 loss 类层次，保留自由函数与 vocab-parallel 数学 |
| `components/metrics.py` | `observability/metrics.py` | 0.381 | |
| `components/optimizer/lr_scheduler.py` | `components/optimizer/lr_scheduler.py` | 0.373 | 去 `Configurable` |
| `components/optimizer/optimizer.py` | `components/optimizer/optimizer.py` | 0.240 | 容器化改写；`OptimizerWrapper` 已删 |
| `components/profiler.py` | `observability/profiler.py` | 0.413 | |
| `datasets/collators.py` | `components/data/collators.py` | 0.145 | |
| `datasets/dataset.py` | `components/data/dataset.py` | 0.267 | 去 `Configurable`；三个节点类去 `Config` 后缀，构建走自由函数 `build_dataset` |
| `datasets/loader.py` | `components/data/loader.py` | 0.590 | 去 `Configurable`；`GrainDataLoader` 直接收参数，无 config 类 |
| `datasets/multimodal/mm_collator.py` | `hf_datasets/multimodal/mm_collator.py` | 0.777 | 增加 MRoPE grid/run/长度校验，当前已是契约适配 |
| `datasets/multimodal/mm_datasets.py` | `hf_datasets/multimodal/mm_datasets.py` | 0.310 | 去 `Configurable`；packing 改自由函数 `build_mm_sample_packing` |
| `datasets/packing.py` | `components/data/packing.py` | 0.065 | 自由函数外还增加文档容量、padding mask、长文档切分和可恢复 remainder，按语义维护 |
| `datasets/sources.py` | `components/data/sources.py` | 0.700 | |
| `datasets/text/text.py` | `hf_datasets/text_datasets.py` | 0.767 | 路径与 processor 构造契约已适配 |
| `datasets/types.py` | `components/data/types.py` | 0.506 | 去 Configurable 后重塑 build context 与 iteration policy |
| `models/common/aux_loss.py` | `models/common/aux_loss.py` | 0.682 | |
| `models/common/dist_gemm.py` | `models/common/dist_gemm.py` | 0.527 | |
| `models/common/feed_forward.py` | `models/common/feed_forward.py` | 0.560 | **曾写完又被退**，不要在没有明确指令时重新引入 |
| `models/common/linear.py` | `models/common/linear.py` | 0.620 | |
| `models/common/masks.py` | `models/common/attention.py` | 0.380 | 拆出了 mask 部分 |
| `models/common/moe.py` | `models/common/moe.py` | 0.155 | |
| `models/common/multimodal.py` | `models/common/multimodal.py` | 0.888 | 保留算法来源，但加入同步规避与更严格的 span/run 校验 |
| ~~`models/common/param_init.py`~~ | — | — | **2026-09-25 移除**：torchtitan parity 的 vendored 死代码（hpmesh 走 HF 模型自带 `_init_weights`，全仓零引用） |
| `models/common/qkv.py` | `models/common/attention.py` | 0.242 | |
| `models/common/rope.py` | `models/common/rope.py` | 0.616 | 上游持续重构后结构已分叉；同步公式与边界修复，不同步 Module/缓存形状 |
| `models/common/scatter_add.py` | `ops/scatter_add.py` | 0.711 | |
| `models/common/token_dispatcher.py` | `models/common/token_dispatcher.py` | 0.441 | 2026-09-25 起含 `TorchAOTokenDispatcher` 可选导入适配层（torchao `permute_and_pad` 委托，未装 loud-raise）；DeepEP/HybridEP 保持登记缺口，见 D 表 |
| `parallel/activation_checkpoint.py` | `distributed/activation_checkpoint.py` | 0.374 | **FullAC + SelectiveAC + MemoryBudgetAC 已移植**（后者按上游语义设 `torch._functorch.config.activation_memory_budget`，需 compile，torch 无该 knob 时 loud-raise）；RegionAC 未移植（需 `torch_remat` + `Module.configure_remat_regions`，配置即 NotImplementedError），理由见文件 docstring |
| `parallel/fully_shard/fsdp.py` | `distributed/fsdp.py` | 0.815 | 多轴 mesh 重建、HF decoder 与 MoE placement 是 hpmesh 适配 |
| `parallel/parallel_dims.py` | `distributed/parallel_dims.py` | 0.772 | hpmesh 扩展 world/loss/sparse mesh 视图，不能按旧 A1 结构覆盖 |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` | 0.686 | `None` → `nn.Identity`；每 stage 追加 `rotary_emb`；stage 内 layer 保留原始索引（不重新编号），避免多 stage state-dict FQN 冲突 |
| `parallel/tensor_parallel/linear.py` | `models/common/dist_gemm.py`（原 `distributed/linear.py`，上游 e72fd863d 搬迁并改名 `Async*`，数学不变） | 0.511 | 保留 fused/fallback 数学意图，但运行时上下文和 autograd 形状已适配 hpmesh |
| `accelerator/collectives.py` | `distributed/utils.py`（vendored `set_pg_timeouts` 与 EP 感知 `clip_grad_norm_` 两个符号） | 部分 | 2026-09-24 从 `parallel/` 迁入 `accelerator/`；EP 裁剪按物理本地 expert 参数适配（免 DTensor "ep" 轴断言）；同日复核后由 C 改标 A2 |

## B —— 适配层（读意图，不要抄形状）

这几个是 **torchtitan 每个模型一个文件** 的那种东西的**替代品**。照搬它们的形状会破坏
分片契约。

| hpmesh | 替代掉的上游 | ratio |
| --- | --- | --- |
| `models/hf_wrapper.py` | `experiments/transformers_modeling_backend/model.py` 的包装层；上游另有 `models/*/model.py` 各一份 | 0.059 |
| `models/hf_state_dict_adapter.py` | `experiments/transformers_modeling_backend/state_dict_adapter.py`；hpmesh 更强：读 safetensors index 做 missing/unexpected 严格校验；上游的 `hf_to_titan_moe_state_dict` 转换对因 hpmesh EP swap 直接搬运 HF 权重（无第二 key 布局）而不需要 | — |
| `parallel/parallelize_hf.py` | `experiments/transformers_modeling_backend/parallelize.py` + 各 `models/*/parallelize.py` | 0.089 |
| `parallel/tensor_parallel/tp.py` | 各模型 TP plan；上游 `distributed/tensor_parallel.py` 已随 DTensor 后端删除、无后继文件。hpmesh 是**手写 plan realizer**，不是声明式 `_sharding_config` | 0.056 |
| `parallel/expert_parallel/apply.py` + `swap.py` | `experiments/.../moe_replacement.py` + 各模型 EP parallelize；hpmesh 搬运 HF 权重而非重新初始化 | 0.036–0.146 |
| `parallel/fully_shard/apply.py` | 各 `models/*/parallelize.py` 的 FSDP driver；HF 五部件适配 | 0.155 |
| `parallel/pipeline_parallel/apply.py` | `distributed/pipeline_parallel.py`；hpmesh 直接消费 HF stage 部件 | 0.130 |
| `trainer/trainer.py` | `trainer.py`，基本重写 | 0.065 |
| `trainer/config.py` | `config/configs.py` | 0.189 |
| `trainer/train.py` | `train.py` | 0.186 |
| `accelerator/mesh.py` | `distributed/parallel_dims.py` + `trainer.py` 中分散的 mesh 逻辑 | 0.111 |
| `models/common/grouped_experts.py` | `models/common/grouped_experts.py` + `models/gpt_oss/moe.py` | 0.119 |
| `utils/gc.py` | `tools/utils.py` 的 GC helper，去 structured logger | 0.211 |

**注意 `accelerator/mesh.py`**：上游没有单一对应物——mesh 逻辑散在 `distributed/parallel_dims.py`
和 `trainer.py` 里，不是某个文件的移植。

## C —— hpmesh 独有（不要对齐上游）

| hpmesh | 说明 |
| --- | --- |
| `accelerator/spmd_context.py` | `spmd_types` pip 包的**独立活跃适配层**，由 trainer 和 `models/common/*` 使用；2026-09-24 从 `utils/` 迁入 |
| `parallel/context_parallel/apply.py` | 0.058；CP 的编排层，上游无对应文件 |
| `parallel/context_parallel/cp_kernel.py` | 0.051；hpmesh 独有的 CP flex kernel |
| `parallel/context_parallel/input_shard.py` | 0.078 |
| `utils/logger_utils.py` | 0.070，上游无对应；2026-09-24 起全仓模块 logger 统一经 `get_logger`（handler 挂模块 logger，rank 过滤在发射时判定，修掉了"import 时 rank 未知"的旧缺陷） |
| `accelerator/monitoring.py` | 与 `tools/utils.py` 0.107，独立实现（含 `get_peak_flops`）；2026-09-24 从 `utils/` 迁入 |
| `utils/checkpoint_keys.py` | 上游无 |
| `accelerator/device.py` | 上游无（0.382 是噪音，命中实验目录）；2026-09-24 从 `utils/` 迁入 `accelerator/` |
| `utils/batch_invariant.py` | 上游无；上游把 batch-invariant 开关放在 `trainer.py`/`config/configs.py` 里，没有独立模块 |
| `models/common/activation.py` | 与上游同名但不同源；公式由 hpmesh 自持，不能按 A 类覆盖 |
| `models/common/embedding.py` | 与上游同名但不同源；包含 hpmesh 的 vocab-shard 契约 |
| `datasets/random_data.py` | 合成语料，上游无 |
| `datasets/build.py` | 工厂；上游把 `build()` 放在 config 上 |
| `accelerator/dist.py` + `accelerator/dist_utils.py` | 2026-09-24 加入：vendored 自 OpenMMLab `mmengine.dist`（**不是 torchtitan 来源**），已去 mmengine 化，设备谓词与后端表统一由同包的 `accelerator/device.py` 提供；不进 trainer 装配路径 |
| `utils/seed.py` | 2026-09-24 加入：上游 `distributed/utils.py::set_determinism` 的 distinct-seed 派生公式的纯函数提取（仅该项，非全文件移植）；DTensor RNG tracker 不移植 |

**已清理悬空链**：`parallel/sharding.py` 与 `parallel/spmd_shims.py` 没有运行时消费者，
已在 2026-09-21 一并删除。`accelerator/spmd_context.py` 是独立活代码，不在删除组内。TP 的
活跃实现继续是 `parallel/tensor_parallel/tp.py` 的 plan 引擎。

**2026-09-23 死代码清理**（第二轮排查后删除，均为全仓零调用者，含 tests/examples）：
`parallel/context_parallel/primitives.py`（曾列为 B 类，等价实现早已并入
`cp_kernel.py`；上游 `models/common/cp_attention.py` 的语义对照因此直接落到
`cp_kernel.py`）、`models/common/flex_kernel.py`（`HFFlexKernel` 从未实例化，其
`_sharding_config` 兼容字段随之消失）、`models/common/nn_modules.py`（仅被
`__init__.py` 再导出的 nn 别名）、`parallel_dims.py` 的 7 个未用 API
（`unfold_dp_axis*`、`get_dense_tp_mesh`、`resolve_mesh`、`get_activated_mesh`、
`world_mesh`、`fsdp_enabled`、`seq_len_divisor`）、`apply_fsdp_to_vision_encoder`、
`ParallelConfig.backend` 字段（backend 由设备类型推导，不再有环境变量覆盖）
及若干零散项。明细见审计记录。

## D —— 真正缺失

| 上游 | 影响 |
| --- | --- |
| `distributed/compile.py` | **已移植**（2026-09-24，批 8）：逐 block compile、async TP `_micro_pipeline_tp`、`regional_inductor`、`capture_scalar_outputs` 四件全部落 `hpmesh/parallel/compile.py` + `CompileConfig`，见下"已从 D 移除" |
| `models/common/moe_sharding.py` | **部分移除**（2026-09-25）。其载荷 MoE-under-TP 已在 `parallel/tensor_parallel/tp.py` 落 B 类适配：HF plan 的 `packed_colwise`/`packed_rowwise`/`moe_tp_experts` 规格不再 raise，专家权重沿 F 维原地切分、router Replicate、块边界 AG/RS 对偶 collective；**tp×ep 同日起按上游语义放行**（TP 只切 dense、EP 独占 routed 专家沿 E 切、router Replicate，`apply_tp` 在 ep>1 时把块留给 swap，专家梯度排除由 `_tp_sharded_param_ids` 统一判定；tp×ep×cp 与 shared-expert×tp 保持 loud-raise）。声明层+装配层就位并有 CPU 单测，但真多卡前后向等价性**环境未覆盖**（本机 torch 2.2.2 无分布式执行栈），待 torch≥2.12 多卡复跑后方可视为完整移除。见下"已从 D 移除（部分）" |
| `components/optimizer/ema.py`（2026-09 新增，515 行） | **已移植**（2026-09-24，`hpmesh/components/optimizer/ema.py`）：在线 EMA 模型平均，config/trainer/checkpointer 三侧接线完成，见下"已从 D 移除" |
| Ulysses CP × varlen/packed（baff3c681） | **已移植**（2026-09-25，批 5）：`apply_cp` 不再 fail-fast，wrapper 全长透传文档 mask、kernel 按 mask Q 长度分派，见下"已从 D 移除" |
| 多轮对话 SFT 的 renderer 路径（4a0d8dab3） | **已适配为可选路径**（2026-09-25，§9.1 第 12 项）：不引入硬依赖、不复制 Configurable 外形。`components/renderer.py` 为可选导入适配层（`build_chat_renderer` + `RendererTokenizerWrapper`），`ChatProcessor(renderer=...)` 走多-turn renderer 分支，`--chat_renderer`/`--messages_field` 接线 `local_jsonl_sft`；未装 `renderers` 时启用 loud-raise（ImportError 带安装指引），默认关闭逐位不变。真实库数值**未验证**（本机无 renderers，单测以 fake 模块覆盖接口与 mask 移位语义）；解锁条件：pyproject 加 optional extra `renderers==0.1.11` 后装包复跑 |
| `models/common/token_dispatcher.py` 的 DeepEP/HybridEP 两个 dispatcher | 登记缺口（2026-09-25，§9.1 第 13 项）：CUDA-only（`deep_ep`/`hybridep` 内核 + GB200/NVLink72 假设）且 dispatch/combine 经上游 `distributed/deepep/` wrappers（1155 行）驱动，可选导入无法忠实表达契约，故不 vendor；`ParallelConfig.ep_token_dispatcher="deepep"/"hybridep"` 配置期 NotImplementedError（含解锁条件），swap 入口防御性同语义。解锁条件：vendor 上游 wrappers + pyproject 加 CUDA-only optional extra + CUDA 目标设备复跑数值。`AllToAllTokenDispatcher` 满足同一 dispatch/combine 契约 |
| `models/common/token_dispatcher.py` 的 `TorchAOTokenDispatcher` | **已适配为可选导入适配层**（2026-09-25，§9.1 第 13 项）：torchao 不进 pyproject、不复制上游 Config 嵌套。`TorchAOTokenDispatcher(num_experts, top_k, pad_multiple)` 继承 `AllToAllTokenDispatcher`，仅 `_permute`/`_unpermute` 改委托 torchao `permute_and_pad`（expert-major 重排 + 每组 pad 到 `pad_multiple`，EP=1 本地 padded permute 路径一并移植），构造期 lazy import，未装 torchao loud-raise ImportError（带 `pip install torchao` 指引）；`ParallelConfig.ep_token_dispatcher="torchao"` + `ep_torchao_pad_multiple`（默认 16=FP8）接线 `apply_ep` → swap，默认 `alltoall` 逐位不变。数值**环境未覆盖**（本机无 torchao/CUDA，单测以 sys.modules fake 覆盖 sentinel-row padding 契约与 EP=1 combine 等价性）；解锁条件：CUDA 目标设备装 torchao 复跑 |
| DSA（DeepSeek sparse attention）的稠密 additive mask 路径 | 上游 `model.py` 的 `_build_dense_attention_mask` + indexer 支持；**2026-09-24 起 hpmesh wrapper 构造期对 `index_topk` fail-fast**（静默走 flex BlockMask 的错误语义已消除），稠密 mask 执行路径本身仍未移植，无消费者 |

**已从 D 移除（部分）**（2026-09-25）：`models/common/moe_sharding.py`——上游该文件是
声明层：`ShardingConfig` 声明 router 参数 TP Replicate、routed 专家权重仅在 EP 开时
沿专家维 E 取 placement（DP_REPLICATE/EFSDP 为 R,EP 为 S(0)），由上游 Module 协议
的 parallelize 引擎消费。hpmesh 按 B 类语义适配，不复制其 Config 协议：MoE-under-TP
的声明改由 HF tp_plan 的 `packed_colwise`/`packed_rowwise`/`moe_tp_experts` 规格承载
（`_resolve_plan` 解析为 None），执行落在 `parallel/tensor_parallel/tp.py` 的结构路
径——`_shard_experts_for_tp`（`down_proj (E,D,F)` 切 dim 2，`gate_up_proj (E,2F,D)`
gate/up 两半各自切 dim 1,router 不动）+ `_TPMoeSequenceBoundary`（`__class__` swap
安装块边界 sequence all-gather / reduce-scatter，与 dense TP 同一对偶契约，序列维
-2)。梯度语义：边界 collective 的注册反向互为对偶；router 权重 Replicate，梯度由
`_allreduce_replicated_tp_grads` 求和；被切专家参数经块上 `_tp_sharded_param_ids`
从该归约排除。state_dict FQN 不变、tp=1 逐位不变。**tp×ep（同日第二段）**：按上游
语义放行——TP 只切 dense,routed 专家由 EP 独占沿专家维 E 切，router Replicate;
`apply_tp` 在 `cfg.ep > 1` 时跳过 MoE 块扫描/分片/边界安装（块留给 `apply_ep`
swap,swap 后的原生 MoE 直接消费/产出 T/tp 序列分片，即上游 ep+sp 的
sequence-parallel 布局，无边界 collective);trainer 的排除判定抽为模块级
`_tp_sharded_param_ids`（三类：dense TP realizer、MoE-under-TP 的 F 分片、EP 的
`GroupedExperts` E 切片；EP 专家梯度按 rank 完备，跨 TP 求和会混不同专家的梯度）。
组合矩阵终态：tp>1×ep>1（cp=1）放行；tp>1×ep>1×cp>1 在
`ParallelConfig.__post_init__` fail-fast（未验证）;shared-expert 块 ×tp 两条路径均
loud-raise(ep=1 边界处、tp×ep 的 swap `_convert_block` 处）;plan 声明 MoE 规格但
探针找不到块（ep=1）loud-raise;GPT-OSS 布局 loud-raise。aux loss / padding-mask
LB / quantile hook 的归约轴此前已按 ep_enabled 含 tp 书写，放行后不重复计数、无需
改动。测试
`tests/unit_tests/cpu/distributed/test_tp_moe.py`：规格解析、分片重建、单进程
partial-sum 等价（reduce-scatter 求和的算术内容，无进程组）、FQN 稳定、幂等、
ep>1 时 apply_tp 原样放行 MoE 块、shared-expert×tp×ep 拒绝、梯度排除规则、组合
矩阵各格。**未覆盖**：真多卡 forward/backward 等价（本机 torch 2.2.2 无
DTensor/spmd 执行栈，gloo 下功能 collective 未验证）——待 torch≥2.12 多卡复跑。

**已从 D 移除**（2026-09-25 批 5 移植）：Ulysses CP × varlen/packed（上游
baff3c681）——上游形态是把 Ulysses 的 token↔head resharding 提为
`UlyssesCPInnerAttention` 共享层，`UlyssesCPVarlenInnerAttention` 借 MRO 把
`super().forward` 派发到 `VarlenInnerAttention`；varlen 元数据（cu_seqlens）不随输入
分片（`cp_shard` 把 `attention_masks` 摘出再原样放回），因为 all-to-all 后每个 rank
都持有全长 token 流。hpmesh 按 B 类语义适配、不复制类层次：packed 语料的"varlen
元数据"在 HF/flex 集成里是烘进 BlockMask 的文档结构，因此
`hf_wrapper.preprocess_inputs` 在 `ulysses` 策略下把全长文档 mask **不 Q 分片**透传
（`set_cp_mesh` 新增 `strategy` 闩锁），`CPFlexKernel._forward_ulysses` 按 mask 的 Q
长度 == 全长序列 分派：全长即用传入 mask，否则照旧重建全长 causal mask。决策全部
config/shape 驱动、rank 对称。`apply_cp` 对 ulysses×`block_causal` 的 fail-fast 移除，
ulysses×load-balancer 拒绝与 heads÷(tp×cp) 校验不变；不启用 varlen 的稠密路径逐位不
变（kernel 重建的 causal mask 与 wrapper 同源同参）。测试：
`tests/integration_tests/cp_ulysses_varlen_equivalence.py`（2-rank gloo：seam 全长
mask 契约、分片 logits/loss == 单卡稠密 block-causal 参考、SDPA 内层下前后向
collective 对偶与 varlen mask 梯度等价、non-vacuity 反证）——本机 torch 2.2.2 无
flex 模块，**环境未覆盖，待 torch≥2.12 + 目标设备复跑**；单测补
`test_ulysses_packed_is_accepted_and_the_strategy_is_latched`。上游 GPT-OSS 的
Ulysses 拒绝（per-head sinks 只走 TP 分片）不适用：hpmesh 尚无 GPT-OSS 支持。

**已从 D 移除**（2026-09-24 批 8 移植）：`distributed/compile.py`——四件互相独立的
能力全部落 `hpmesh/parallel/compile.py::apply_compile`，由
`trainer/config.py::CompileConfig`（`training.compile_config`，默认全关）驱动，
装配顺序不变（AC 之后、FSDP 之前；PP 下每 chunk 经 `apply_pp` 同一函数）：

* **逐 block compile**（`per_block=True`）：每个 decoder layer `block.compile(
  backend=..., fullgraph=True)`（`Module.compile` 就地，state-dict 键与 FSDP 包装
  不变）；默认 False 保持整体 `torch.compile(model, backend="inductor")`，与旧
  路径逐位一致。
* **async TP**（`enable_async_tensor_parallel=True`）：设
  `torch._inductor.config._micro_pipeline_tp` 并为 TP group 注册 symmetric
  memory（按 group 名去重，PP 每 chunk 重入安全）。配置校验期（
  `HybridMeshConfig.__post_init__`）拒绝 无 compile / tp=1 两种组合；装配期对
  无 TP mesh、torch 无 `_micro_pipeline_tp`、无 `enable_symm_mem_for_group` 三种
  情形 loud-raise，不静默跳过。
* **regional_inductor**：flex 只有 inductor lowering，故非 inductor backend 下
  flex 模型必须 scoop。`backend="aot_eager"` 且模型走 flex（wrapper 新 property
  `uses_flex_attention`）时用 `torch.fx.passes.regional_inductor` 包
  `aot_autograd`；annotation 落 `hf_wrapper._flex_attention_hf` 的
  `maybe_regional_inductor({})`（默认 nullcontext，inductor/eager 路径零开销）。
  flex 模型配其他非 inductor backend → `ValueError`；torch 无 regional_inductor
  → `NotImplementedError`；sdpa 模型 backend 原样透传。inductor_configs 传空
  （hpmesh 走 HF 的 flex 集成，不带上游 FlexInnerAttention 的 autotune 配置）。
* **capture_scalar_outputs**：按上游条件在编译的 model part 含 token-choice MoE
  block（`_iter_moe_layers` 非空，即 EP swap 后的 hpmesh MoE 栈）时设
  `torch._dynamo.config.capture_scalar_outputs=True`；dense 模型不动该全局量
  （逐位不变）；torch 无此 knob 时 loud-raise。

未移植（登记）：上游同文件的 `skip_fwd_side_effects_in_bwd_under_checkpoint`
（AC+compile 的 side-effect 重放分歧开关，hpmesh 未遇到其失败场景，需要时按上游
注释补）与 `FakeTensorMode.__init__` 的 `torch.compiler.disable` monkeypatch
（修上游 pytorch#178887，等上游修复即废弃的临时措施）。`CompileConfig.components`
未移植（hpmesh 只编译 model，loss 无 compile 通路）。

**已从 D 移除**（2026-09-24 批 4 移植）：`pipeline_with_first_stage_modules`——
多模态 first-stage 模块并入 stage 0，落为 `apply_pp` 的可选关键字参数
`first_stage_module_fqns: Sequence[str] | None`（默认 None，默认时切分与
state-dict 键逐位不变）+ `parallel/pipeline_parallel/apply.py::
_prepend_first_stage_modules`（仅作用于自动生成的切分，把存在的 FQN 按序前插
stage 0；已被切分占有的 FQN 与重复 FQN loud-raise，缺失模块跳过；显式
`module_fqns_per_model_part` 给定时忽略并告警，与上游委托语义一致）。配套改动
`split_model_into_stages`：wrapper `named_children()` 不呈现的额外顶层模块
（注册在 decoder 旁的多模态编码器等）在非属主 stage 上一律置 `nn.Identity`
——上游"pruned on other stages"语义；装五部件的容器（wrapper 内层 HF 模型）
通过"包含已呈现部件"判定跳过，绝不置空。不变量：stage FQN 稳定（并入模块保持
原名顶层子模块，optimizer/checkpoint 键不跨 stage 冲突）、五部件契约与
`named_children()` 语义不变。当前无真实消费者（多模态 vision encoder 路径未
接线），属"能力就位 + 契约测试"；多 stage 真跑待目标设备（torch≥2.12）复跑。

**已从 D 移除**（2026-09-24 批 4 移植）：validation 循环——上游
`components/validate.py::Validator` 落 `trainer/trainer.py` 的
`Trainer.validate`/`should_validate`/`_check_validation_feasibility` +
`trainer/config.py::ValidationConfig`（`training.validation_config`，默认
None 关闭，关闭时训练循环逐位不变；programmatic-only，同 `ema_config`）。
语义对齐：eval 模式 + `no_grad`、结束恢复 train；loss 按全局有效 token 数
归一化，token 计数走 dp mesh、loss 和走 dp×cp×tp loss mesh（与训练 loss 同一
对 mesh、同一归一化）；每次 pass 新建并关闭临时 dataloader（`repeat=False`
对应 `steps=-1`），不进 checkpoint、不动 `ntokens_seen`；训练循环内调用点在
checkpoint save 之后、profiler.step 之前，与上游同序。两条上游 bug fix 一并
移植：零 batch / 零有效 token 报 `ValueError`（上游 6c2dadbb3），dp>1 拒绝
`steps=-1`（上游 90b25912f，在 trainer 构造期、真实 dp degree 已知后检查）；
另对 random 无限语料的 `steps=-1` 同样 fail-fast。PP × validation 未支持：
hpmesh 的 PP loss 计算内嵌在 schedule 的训练步里，无 `pp_schedule.eval`
对应的 eval 通路，构造期 `NotImplementedError`（loud-raise，不静默跳过）。

**已从 D 移除**（2026-09-24 批 2 移植）：quantile-balanced MoE routing——
`QuantileBalancedTopKRouter` + `QuantileBalancer` + `register_moe_quantile_balancing_hook`
（biased top-(K+1) cutoff、1000-bin 直方图、分位数 mean-centred 覆写 bias），与
sign-based bias 互斥（同层 raise、跨层 hook raise），经 `ParallelConfig.
moe_quantile_balancing` 启用；MoE padding-mask 负载均衡——`MoE.set_padding_mask`
一次性暂存通道（HF layer 签名穿不了 mask），mask 只过滤负载均衡统计
（tokens_per_expert、aux loss f/p、quantile 直方图），不动 routing 执行，无 mask
逐位不变；CP/TP 由 `shard_padding_mask_for_cp/tp` 与 token 流同序切分。

**已从 D 移除**（2026-09-24 批 3a 移植）：在线 EMA——`hpmesh/components/optimizer/ema.py`
（515 行上游 `components/optimizer/ema.py` 的语义移植）：`EMA` 复用
`OptimizersContainer` 的 flat FQN state-dict 契约，`decay = 2**(-1/(half_life_fraction*num_updates))`
动态计划或固定 decay，firing count 由 trainer step 推导（resume 不重置 decay），
`step_bias` 支持阶段重编号，`start_step`/`update_every_n_steps` 门控，可选
`buffer_patterns` 浮点 buffer 跟踪（整型 buffer 拒绝）；checkpointer 增加 `ema`
state 键与 `_find_load_step(max_step=)`，trainer/config 完成三侧接线。上游
DTensor unwrap/rewrap 与 CUDA `torch._foreach_lerp_` 专项未移植（hpmesh 的 FSDP2
张量本身就是 DTensor，容器 state dict 直接交给 DCP）。

**已从 D 移除**（2026-09-24 批 1 移植）：`CastLinear`——lm_head compute-dtype
变换，落 `models/common/cast_linear.py`（`nn.Linear` 子类，state-dict FQN 不变），
经 `ModelConfig.compute_dtype` 启用，默认关闭；router `_debug_force_load_balance`
——落 `TokenChoiceTopKRouter` 同名构造参数，round-robin 语义与上游逐字一致；
PP per-stage seed——`utils/seed.py` 的 `derive_distinct_seed`（上游
`distinct_seed_mesh_dims=["pp"]` 同公式），trainer 在 `pp_enabled` 时按 stage rank
偏移，pp=1 逐位不变；DTensor RNG tracker 不移植（初始化走 materialize 路径）。

**故意删除，不是缺口**（不要"补回来"）：`components/quantization/`、
`structured_logger/`、`protocols/`、`configurable.py`。

**已从 D 移除**：`distributed/activation_checkpoint.py` 的 `SelectiveAC`——于
2026-09-21 移植（见 A2）；`MemoryBudgetAC`——于 2026-09-24 移植：它没有策略代码，
只是设一个 `torch._functorch.config.activation_memory_budget` 全局量让 compile
partitioner 做取舍，落为 `training.activation_checkpoint_mode='memory_budget'` +
`MemoryBudgetACConfig`（budget ∈ [0,1]，同上游校验），按上游 trainer 校验在
compile 关闭时 fail-fast；torch 无该 knob（本机 2.2.2 即如此）时 loud-raise 而非
静默设一个没人读的全局量；上游的 `visualize_memory_budget_pareto`（往 dump folder
倒 SVG）未移植，hpmesh 的 AC 路径没有 dump folder 概念。该文件仍有两处未移植，都是
环境依赖而非删减：`RegionAC` 需要 `torch_remat`（hpmesh 不依赖，且它的"模型声明
region"建立在 hpmesh 没有的 `Module.configure_remat_regions` 协议上）——配置
`mode='region'` 在 config 与 `apply_ac` 两处都是显式 `NotImplementedError`，解锁
条件写在报错与文件 docstring 里；`_disable_dynamo_lru_cache`（修的是 SAC+PP 的
重编译交互，而 hpmesh 在 `pp > 1` 上直接拒绝 AC，够不到那个场景）。

**已从 D 移除**：`tools/validate.py`——上一版既写了它、又写"上游也没有这个路径，已从
表里移除"，自相矛盾。核实：上游 `torchtitan/tools/validate.py` **确实不存在**，这一行
没有意义，删掉。

## E —— 包面（`__init__.py` 与入口）

重组出口，不是移植内容：这些文件定义 hpmesh 的**公开 API 面**，上游对应物是同名
`__init__.py`（若存在）。改上游的导出列表时才需要看这里。

| hpmesh | 行数 | torchtitan |
| --- | --- | --- |
| `__init__.py` | 30 | `__init__.py` |
| `__main__.py` | 6 | `train.py` 的入口对等物 |
| `models/common/__init__.py` | 61 | `models/common/__init__.py` |
| `datasets/__init__.py` | 64 | `components/data/__init__.py` |
| `parallel/__init__.py` | 35 | `distributed/__init__.py` |
| `parallel/context_parallel/__init__.py` | 23 | `distributed/context_parallel/__init__.py` |
| `parallel/expert_parallel/__init__.py` | 20 | 上游无对应（见 C 类） |
| `parallel/pipeline_parallel/__init__.py` | 12 | 上游无对应 |
| `trainer/__init__.py` | 26 | 上游无对应 |
| `accelerator/__init__.py` | 73 | 上游无对应（PEP 562 懒加载包面） |

`__init__.py` 的 ratio 平均偏低（0.3 上下）是正常的——它们导出的是各自的公开面，不是
从上游抄结构。表里的数值是"文件行数"，不是 ratio。

**空文件**（0 行，不必查表）：`components/__init__.py`、`models/__init__.py`、
`parallel/fully_shard/__init__.py`、`parallel/tensor_parallel/__init__.py`、
`utils/__init__.py`。

**两个子包**：`datasets/` 下按语料分 `text/` 和 `multimodal/`，其余 8 个模块平铺在
`datasets/` 根下。上游的 `hf_datasets/` 是 `components/data/` 的兄弟目录，hpmesh 曾用
`datasets/hf/` 镜像它（4 层，三个 0 字节的 `__init__.py`），先被溶解成单层，再于重组
时按语料分成两个子包。

划分依据是**内容**而非上游路径：`datasets/` 根下的模块与上游 `components/data/` 一一
对应，且都被两边共用（`loader.py` 默认 `TextCollator`、`multimodal/mm_collator.py`
复用 `collators.py` 的 `Collator`/`TrainerBatch`），所以不进任何一边；只有 `text.py`
和 5 个 `mm_*.py` 是语料专属。两个子包的 `__init__.py` 都**刻意不导入子模块**——惰性
导入契约靠这一点维持。

**上游路径对应关系因此不再一一成立**（`hf_datasets/multimodal/utils/image.py` 在
hpmesh 侧是 `datasets/multimodal/mm_image.py`），本表的 hpmesh 列是唯一权威。

## 版本与漂移

- 本文最近一次人工审计工作树：hpmesh `5749d19`（+本轮改动），TorchTitan `b64103072`；
  详细验证记录见
  `hpmesh_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区）。
  上一轮审计（hpmesh `8a2f269` × TorchTitan `c6e416bbd`）引用的
  `hpmesh_torchtitan_alignment_audit_2026-09-21.md` 不在当前工作区。
- 检查后续漂移：`git -C <torchtitan> log b64103072..HEAD -- torchtitan/`。
- 2026-09-23 映射修订：上游 `distributed/linear.py` 已删除、内容迁入
  `models/common/dist_gemm.py`（改名 `AsyncAllGatherLinear`/`AsyncLinearReduceScatter`，
  数学不变），此后上游 dist_gemm.py 同时对应 hpmesh 的 `parallel/tensor_parallel/linear.py`
  （autograd 原语）与 `models/common/dist_gemm.py`（模块层），一对二；
  `distributed/tensor_parallel.py` 已随 DTensor 后端整体删除、无后继。
- 早前基线：hpmesh `8a2f269`，TorchTitan `c6e416bbd`。
- 表中的 ratio 除 A2 中明确标为 2026-09-21 复核的十行外，来自早期结构快照
  （hpmesh `58eb279` 附近），只用于解释来源，**不是当前工作树的实时相似度**。源码
  变化后应运行下方脚本重算，不能据旧 ratio 判定漂移。
- 2026-09-22 设备验证补充：Qwen3-8B 已按 TorchTitan 的 meta 构建 → FSDP → `to_empty`
  → checkpoint load 顺序完成 8 卡 HCCL、4096 序列的真实训练，并完成完整 DCP
  save→resume。训练示例显式使用 `last_save_model_only=False`；上游默认的 model-only
  最终 checkpoint 只适合作为导出物，不能作为续训状态。
- 同日并行复核修正了 EP 不应计入 world-size 乘积的 config helper，以及 Torch 2.10
  functional-collective 的 TP fallback API。2-rank FSDP/TP/CP/EP-grad-norm 等价性
  通过；PP 1F1B 的多步轨迹仍有约 `8.5e-3` 最大偏差，保持未通过状态。

## 重现这张表

保存为 `recompute_map.py`，把两个根目录指向你自己的 checkout，然后
`python recompute_map.py | sort -t$'\t' -k1 -rn`。它是产出本表的**实际脚本**（不是
伪码），两分钟内跑完。

```python
"""hpmesh -> torchtitan 结构相似度。输出: ratio \t ratio2 \t hpmesh路径 \t 上游路径 \t 代码长度"""
import ast, difflib
from pathlib import Path

HP = Path("<你的>/HybridMesh/hpmesh")
TT = Path("<你的>/torchtitan/torchtitan")   # 必须限定在 torchtitan/ 下!

def strip_src(p):
    t = ast.parse(p.read_text())
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef, ast.Module)):
            if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                n.body.pop(0)
            if not n.body:
                n.body.append(ast.Pass())
    return ast.unparse(ast.fix_missing_locations(t))

up = {p.relative_to(TT).as_posix(): strip_src(p) for p in TT.rglob("*.py")}

for p in sorted(HP.rglob("*.py")):
    a = strip_src(p)
    if len(a) < 40:            # 空 __init__.py 等,跳过(见下方教训)
        continue
    best = second = 0.0
    best_name = second_name = ""
    for name, b in up.items():
        # 上界剪枝: ratio <= 2*min/(sum)。比不过当前最优就不必构造 matcher。
        if 2 * min(len(a), len(b)) / (len(a) + len(b)) <= best:
            continue
        r = difflib.SequenceMatcher(None, a, b).ratio()   # autojunk 保持默认 True
        if r > best:
            second, second_name = best, best_name
            best, best_name = r, name
        elif r > second:
            second, second_name = r, name
    print(f"{best:.3f}\t{second:.3f}\t{p.relative_to(HP).as_posix()}\t{best_name}\t{len(a)}")
```

使用要点（均为实测教训）：

- **必须先把 torchtitan 的候选集限定在 `torchtitan/torchtitan/` 下**，否则会匹配到
  `experiments/rl/` 之类的噪音。
- **性能**：不要用 `autojunk=False`，91 × 444 会跑十几分钟。默认 `autojunk=True` 加
  那条上界剪枝就能在两分钟内跑完。
- **不要用 `diff` 行数判断改动量**；`quick_ratio()` 也不能用来筛候选——它是上界不是
  估计，会漏掉真正的对应物（实测漏过 `components/loss.py`）。
- **第 2 列是次优匹配**，它比最优值还重要：两者接近说明这个文件的归属有歧义，需要
  人工判；两者差距大，最优才可信。
