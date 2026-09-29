# llmtuner → torchtitan 对应关系表

[llmtuner](../llmtuner) 拿掉了 TorchTitan 的 `Configurable` 与 `Module` 两个抽象层，换来一个
明显更短的框架：123 个 Python 模块（99 个实现模块）、约 29.7k 行，覆盖 TP / FSDP2 /
CP / EP / PP 五条并行路径的装配、训练循环、checkpoint 与等价性测试。本文是这些模块与
torchtitan 之间对应关系的**唯一权威**。

## 怎么用这张表

**先查表，再动手。** 不要在 llmtuner 源码里加 `# upstream: <path> @ <sha>` 之类的来源
标注——那种标注试过又被退了，因为 sha 会腐烂而表不会。

**分类不是装饰，是操作指令。** 把 A 类的高保真同步规则套到 B 类文件上会毁掉设计；套到
C 类上会把项目**故意删掉**的抽象又拽回来。

**比结构用 AST，不要比 diff 行数。** `components/checkpointer/filesystem.py` 的 diff 有几十行，代码差异
是 **0**——全是改写措辞。做法是剥掉 docstring、`ast.unparse`、再
`difflib.SequenceMatcher`。下表 `ratio` 列就是这么来的。

目录级速查（细节以下方分类表为准）：

| llmtuner | torchtitan |
| --- | --- |
| `parallel/**` | `distributed/` |
| `components/metrics.py` | `observability/metrics.py` |
| `models/common/scatter_add.py` | `ops/scatter_add.py` |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` |
| `datasets/text/processors.py` | `hf_datasets/text_datasets.py` |
| `components/checkpointer/filesystem.py` | `tools/filesystem.py` |

## 图例

- **A 移植（vendored）**——从 torchtitan 移植。上游改了，llmtuner **应该**逐项核对。
- **B 适配（adapted）**——同一个想法、不同的形状。上游改了，**读意图、不要抄形状**。
- **C llmtuner 独有**——上游没有对应物（或同名不同源）。不要"对齐"它。
- **D 缺口**——上游有，llmtuner **真的没有**。补还是不补是决策，不是疏漏。

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

只有 `components/checkpointer/utils.py` 和 `components/checkpointer/filesystem.py` 在该快照中经人工确认
属于"去 docstring 后结构等价"；其余行即使 ratio 很高也不是逐字复制。

| llmtuner | torchtitan | ratio |
| --- | --- | --- |
| `components/checkpointer/utils.py` | `components/checkpointer/utils.py` | 1.000 |
| `components/checkpointer/filesystem.py` | `tools/filesystem.py` | 1.000 |
| `components/optimizer/utils.py` | `components/optimizer/utils.py` | 0.996 |
| `datasets/multimodal/image.py` | `hf_datasets/multimodal/utils/image.py` | 0.977 |
| `datasets/multimodal/text_utils.py` | `hf_datasets/multimodal/utils/text.py` | 0.971 |
| `datasets/multimodal/video.py` | `hf_datasets/multimodal/utils/video.py` | 0.967 |
| `components/tokenizer.py` | `components/tokenizer.py` | 0.907 |

## A2 —— 移植但已改写（比例中等，需逐处核对）

| llmtuner | torchtitan | ratio | 改写点 |
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
| `datasets/multimodal/collator.py` | `hf_datasets/multimodal/mm_collator.py` | 0.777 | 增加 MRoPE grid/run/长度校验，当前已是契约适配 |
| `datasets/multimodal/datasets.py` | `hf_datasets/multimodal/mm_datasets.py` | 0.310 | 去 `Configurable`；packing 改自由函数 `build_mm_sample_packing` |
| `datasets/packing.py` | `components/data/packing.py` | 0.065 | 自由函数外还增加文档容量、padding mask、长文档切分和可恢复 remainder，按语义维护 |
| `datasets/sources.py` | `components/data/sources.py` | 0.700 | |
| `datasets/text/processors.py` | `hf_datasets/text_datasets.py` | 0.767 | 路径与 processor 构造契约已适配 |
| `datasets/types.py` | `components/data/types.py` | 0.506 | 去 Configurable 后重塑 build context 与 iteration policy |
| `models/common/aux_loss.py` | `models/common/aux_loss.py` | 0.682 | |
| `models/common/async_linear.py`（2026-09-26 文件名对齐上游，原 dist_gemm.py） | `models/common/async_linear.py` | 0.527 | |
| `models/common/feed_forward.py` | `models/common/feed_forward.py` | 0.560 | **曾写完又被退**，不要在没有明确指令时重新引入 |
| `models/common/linear.py` | `models/common/linear.py` | 0.620 | |
| `models/common/attention/masks.py` | `models/common/attention.py` | 0.380 | 拆出了 mask 部分 |
| `models/common/moe/`（`block`/`router`/`experts`/`dispatcher`/`load_balance`/`balancing` 六文件） | `models/common/moe.py` + `models/common/token_dispatcher.py` | 0.155 | 2026-09-28 十七次增量拆成子包 |
| `models/common/multimodal.py` | `models/common/multimodal.py` | 0.888 | 保留算法来源，但加入同步规避与更严格的 span/run 校验 |
| ~~`models/common/param_init.py`~~ | — | — | **2026-09-25 移除**：torchtitan parity 的 vendored 死代码（llmtuner 走 HF 模型自带 `_init_weights`，全仓零引用） |
| `models/common/attention/qkv.py` | `models/common/attention.py` | 0.242 | |
| `models/common/rope.py` | `models/common/rope.py` | 0.616 | 上游持续重构后结构已分叉；同步公式与边界修复，不同步 Module/缓存形状 |
| `models/common/scatter_add.py` | `ops/scatter_add.py` | 0.711 | |
| `models/common/moe/dispatcher.py` | `models/common/token_dispatcher.py` | 0.441 | 2026-09-25 起含 `TorchAOTokenDispatcher` 可选导入适配层（torchao `permute_and_pad` 委托，未装 loud-raise）；DeepEP/HybridEP 保持登记缺口，见 D 表 |
| `parallel/activation_checkpoint.py` | `distributed/activation_checkpoint.py` | 0.374 | **FullAC + SelectiveAC + MemoryBudgetAC 已移植**（后者按上游语义设 `torch._functorch.config.activation_memory_budget`，需 compile，torch 无该 knob 时 loud-raise）；RegionAC 未移植（需 `torch_remat` + `Module.configure_remat_regions`，配置即 NotImplementedError），理由见文件 docstring |
| `parallel/fully_shard/fsdp.py` | `distributed/fsdp.py` | 0.815 | 多轴 mesh 重建、HF decoder 与 MoE placement 是 llmtuner 适配（2026-09-28 逐项复核，见审计“八次增量”） |
| `parallel/parallel_dims.py` | `distributed/parallel_dims.py` | 0.772 | llmtuner 扩展 world/loss/sparse mesh 视图，不能按旧 A1 结构覆盖；`build_parallel_dims` / `build_mesh` 自 `accelerator/mesh.py` 并入，上游无单一对应物（mesh 逻辑散在 `distributed/parallel_dims.py` 与 `trainer.py`） |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` | 0.686 | `None` → `nn.Identity`；每 stage 追加 `rotary_emb`；stage 内 layer 保留原始索引（不重新编号），避免多 stage state-dict FQN 冲突 |
| `parallel/tensor_parallel/linear.py` | `models/common/async_linear.py`（原 `distributed/linear.py` → e72fd863d 搬入 dist_gemm.py → 9e159aed7 改名，数学不变） | 0.511 | 保留 fused/fallback 数学意图，但运行时上下文和 autograd 形状已适配 llmtuner |
| `accelerator/collectives.py` | `distributed/utils.py`（vendored `set_pg_timeouts` 与 EP 感知 `clip_grad_norm_` 两个符号） | 部分 | 2026-09-24 从 `parallel/` 迁入 `accelerator/`；EP 裁剪按物理本地 expert 参数适配（免 DTensor "ep" 轴断言）；同日复核后由 C 改标 A2 |

## B —— 适配层（读意图，不要抄形状）

这几个是 **torchtitan 每个模型一个文件** 的那种东西的**替代品**。照搬它们的形状会破坏
分片契约。

| llmtuner | 替代掉的上游 | ratio |
| --- | --- | --- |
| `models/hf/model.py`（2026-09-28 十八次增量随上游命名；+ `models/hf/factory.py` 构建侧、`models/hf/flops.py` 算术侧） | `experiments/transformers_modeling_backend/model.py` 的包装层；上游另有 `models/*/model.py` 各一份 | 0.059 |
| `models/hf/state_dict_adapter.py` | `experiments/transformers_modeling_backend/state_dict_adapter.py`；llmtuner 更强：读 safetensors index 做 missing/unexpected 严格校验；上游的 `hf_to_titan_moe_state_dict` 转换对因 llmtuner EP swap 直接搬运 HF 权重（无第二 key 布局）而不需要 | — |
| `parallel/parallelize.py`（2026-09-26 文件名对齐上游，原 parallelize_hf.py） | `experiments/transformers_modeling_backend/parallelize.py` + 各 `models/*/parallelize.py` | 0.089 |
| `parallel/tensor_parallel/tp.py`（+ `apply.py` 入口） | 各模型 TP plan；上游的 TP 声明层已随 DTensor 后端迁到 `protocols/sharding.py` + 各模型 `*_sharding.py`，旧的 `distributed/tensor_parallel.py` 于 `7e7f271e0` 删除。llmtuner 是**手写 plan realizer**，对应上游的声明式 `_sharding_config` 面（逐项对应见本文「TP/SP 对齐结论」） | 0.056 |
| `parallel/expert_parallel/apply.py` + `swap.py` | `experiments/.../moe_replacement.py` + 各模型 EP parallelize；llmtuner 搬运 HF 权重而非重新初始化 | 0.036–0.146 |
| `parallel/fully_shard/apply.py` | 各 `models/*/parallelize.py` 的 FSDP driver；HF 五部件适配 | 0.155 |
| `parallel/pipeline_parallel/apply.py` | `distributed/pipeline_parallel.py`；llmtuner 直接消费 HF stage 部件 | 0.130 |
| `trainer/trainer.py` | `trainer.py`，基本重写 | 0.065 |
| `config/`（顶层配置包） | `config/configs.py` | 0.189 |
| `trainer/train.py` | `train.py` | 0.186 |
| `models/common/moe/experts.py` | `models/common/grouped_experts.py` + `models/gpt_oss/moe.py` | 0.119 |
| `utils/gc.py` | `tools/utils.py` 的 GC helper，去 structured logger | 0.211 |

**注意 mesh 构建**：原 `accelerator/mesh.py` 已并入 `parallel/parallel_dims.py`，上游没有
单一对应物——mesh 逻辑散在 `distributed/parallel_dims.py` 和 `trainer.py` 里。整文件按
A2 分类（见上表），mesh 构建这一段记在该行的"改写点"里，不另占一个分类行。

## C —— llmtuner 独有（不要对齐上游）

| llmtuner | 说明 |
| --- | --- |
| `accelerator/spmd_context.py` | `spmd_types` pip 包的**独立活跃适配层**，由 trainer 和 `models/common/*` 使用；2026-09-24 从 `utils/` 迁入 |
| `parallel/context_parallel/apply.py` | 0.058；CP 的编排层，上游无对应文件 |
| `parallel/context_parallel/cp_kernel.py` | 0.051；llmtuner 独有的 CP flex kernel |
| `parallel/context_parallel/input_shard.py` | 0.078 |
| `utils/logger_utils.py` | 0.070，上游无对应；2026-09-24 起全仓模块 logger 统一经 `get_logger`（handler 挂模块 logger，rank 过滤在发射时判定，修掉了"import 时 rank 未知"的旧缺陷） |
| `accelerator/monitoring.py` | 与 `tools/utils.py` 0.107，独立实现（含 `get_peak_flops`）；2026-09-24 从 `utils/` 迁入 |
| `components/checkpointer/checkpoint_keys.py` | 上游无 |
| `accelerator/device.py` | 上游无（0.382 是噪音，命中实验目录）；2026-09-24 从 `utils/` 迁入 `accelerator/` |
| `models/common/activation.py` | 与上游同名但不同源；公式由 llmtuner 自持，不能按 A 类覆盖 |
| `models/common/embedding.py` | 与上游同名但不同源；包含 llmtuner 的 vocab-shard 契约 |
| `datasets/random_data.py` | 合成语料，上游无 |
| `datasets/build.py` | 工厂；上游把 `build()` 放在 config 上 |
| `accelerator/dist.py` + `accelerator/dist_utils.py` | 2026-09-24 加入：vendored 自 OpenMMLab `mmengine.dist`（**不是 torchtitan 来源**），已去 mmengine 化，设备谓词与后端表统一由同包的 `accelerator/device.py` 提供；不进 trainer 装配路径。2026-09-28 复核（十次增量）：只按「能力缺口」对照上游 `distributed/utils.py`，结论是**无缺口**——上游的 `dist_sum`/`dist_max`/`dist_mean`/`dist_sum_tensor` 在 llmtuner 侧是 `all_reduce`（调用点 clone + in-place），`set_pg_timeouts`/`clip_grad_norm_` 已迁入 `accelerator/collectives.py`，仅有的 `init_distributed`/fake 后端差异已单列于 D 表 |
| `trainer/seed.py` | 2026-09-24 加入：上游 `distributed/utils.py::set_determinism` 的 distinct-seed 派生公式的纯函数提取（仅该项，非全文件移植）；DTensor RNG tracker 不移植。2026-09-28（十次增量）补齐该函数剩余的两个可移植件：`PYTHONHASHSEED = str(seed % 2**32)`（为后续 spawn 的 dataloader worker 而设）与 `detect_anomaly`（`torch.autograd.set_detect_anomaly(True, check_nan=False)` + 上游同文告警），落点为 `Trainer._seed_everything` 与 `TrainingConfig.detect_anomaly`；不移植的仍是只服务上游自有栈的两件（DTensor mesh-aware RNG tracker、flex-attention 确定性内核调优） |

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
`world_mesh`、`fsdp_enabled`、`seq_len_divisor`）、`apply_fsdp_to_multimodal_encoder`
（上游 2026-09 由 `apply_fsdp_to_vision_encoder` 改名；服务于 `models/common/multimodal.py`
的 vision tower，llmtuner 的 `models/common/multimodal.py` 只有 span/gather 融合算子、
没有编码器模块，故无消费者）、`ParallelConfig.backend` 字段（backend 由设备类型推导，不再有环境变量覆盖）
及若干零散项。明细见审计记录。

## D —— 真正缺失

| 上游 | 影响 |
| --- | --- |
| `distributed/compile.py` | **已移植**（2026-09-24，批 8）：逐 block compile、async TP `_micro_pipeline_tp`、`regional_inductor`、`capture_scalar_outputs` 四件全部落 `llmtuner/parallel/compile.py` + `CompileConfig`，见下"已从 D 移除" |
| `models/common/moe_sharding.py` | **部分移除**（2026-09-25）。其载荷 MoE-under-TP 已在 `parallel/tensor_parallel/tp.py` 落 B 类适配：HF plan 的 `packed_colwise`/`packed_rowwise`/`moe_tp_experts` 规格不再 raise，专家权重沿 F 维原地切分、router Replicate、块边界 AG/RS 对偶 collective；**tp×ep 同日起按上游语义放行**（TP 只切 dense、EP 独占 routed 专家沿 E 切、router Replicate，`apply_tp` 在 ep>1 时把块留给 swap，专家梯度排除由 `tp_sharded_param_ids` 统一判定；tp×ep×cp 与 shared-expert×tp 保持 loud-raise）。声明层+装配层就位并有 CPU 单测，但真多卡前后向等价性**环境未覆盖**（本机 torch 2.2.2 的 gloo 可用，缺的是模型/并行层所需的新 API，见"版本与漂移"），待 torch≥2.12 多卡复跑后方可视为完整移除。见下"已从 D 移除（部分）" |
| `components/optimizer/ema.py`（2026-09 新增，515 行） | **已移植**（2026-09-24，`llmtuner/components/optimizer/ema.py`）：在线 EMA 模型平均，config/trainer/checkpointer 三侧接线完成，见下"已从 D 移除" |
| `components/optimizer/optimizer.py` 的 `implementation="fused_opt_states_bf16"`（+ `_register_bf16_optimizer_state_hook`） | **登记缺口**（2026-09-28，components 批次 optimizer 走查）：上游第四种实现模式用 Adam 的 step pre-hook 预建 bf16 `exp_avg`/`exp_avg_sq`（fused CUDA 核据此走 fp32 参数+bf16 状态的混合精度路径，省一半优化器状态显存），再用 `register_load_state_dict_post_hook` 在 DCP 载入后把被 torch 转回参数 dtype 的状态重新降为 bf16。llmtuner 的 `implementation` 只声明 `fused`/`foreach`/`for-loop`（配置期即 Literal 拒绝，装配期 `_build_impl_kwargs` 再兜一道 `ValueError`）。不移植的理由是**不可验证**：它的全部价值来自那个 CUDA fused 核，本机无 CUDA 也无从复现上游的显存收益；同时它改写 checkpoint 里 Adam 状态的 dtype，属续训兼容敏感面，盲写风险高于收益。解锁条件：CUDA 目标设备 + 确有优化器状态显存诉求；届时实现要点即上面两条 hook（上游 `components/optimizer/optimizer.py:339` 起） |
| `components/optimizer/optimizer.py` 的 `optimizer_factory_kwargs_by_name` | **登记缺口（无消费者）**（2026-09-28，同上）：上游用它把「实例级对象」——per-parameter compute metadata、通信 bucket 规格——按 optimizer 名传进工厂，而它现在的两个消费者（`DistMuon`、Float8 系优化器）都在 llmtuner 裁剪面内（前者属已在 C 类登记为范围外的 `distributed/flex_shard/`）。按本仓"有调用者再补"的口径不预置字段；补 `DistMuon` 时一并加 |

| Ulysses CP × varlen/packed（baff3c681） | **已移植**（2026-09-25，批 5）：`apply_cp` 不再 fail-fast，wrapper 全长透传文档 mask、kernel 按 mask Q 长度分派，见下"已从 D 移除" |
| 多轮对话 SFT 的 renderer 路径（4a0d8dab3） | **已适配为可选路径**（2026-09-25，§9.1 第 12 项）：不引入硬依赖、不复制 Configurable 外形。`datasets/text/renderer.py` 为可选导入适配层（`build_chat_renderer` + `RendererTokenizerWrapper`），`ChatProcessor(renderer=...)` 走多-turn renderer 分支，`--chat_renderer`/`--messages_field` 接线 `local_jsonl_sft`；未装 `renderers` 时启用 loud-raise（ImportError 带安装指引），默认关闭逐位不变。真实库数值**未验证**（本机无 renderers，单测以 fake 模块覆盖接口与 mask 移位语义）；解锁条件：pyproject 加 optional extra `renderers==0.1.11` 后装包复跑 |
| `models/common/moe/dispatcher.py` 的 DeepEP/HybridEP 两个 dispatcher | 登记缺口（2026-09-25，§9.1 第 13 项）：CUDA-only（`deep_ep`/`hybridep` 内核 + GB200/NVLink72 假设）且 dispatch/combine 经上游 `distributed/deepep/` wrappers（1155 行）驱动，可选导入无法忠实表达契约，故不 vendor；`ParallelConfig.ep_token_dispatcher="deepep"/"hybridep"` 配置期 NotImplementedError（含解锁条件），swap 入口防御性同语义。解锁条件：vendor 上游 wrappers + pyproject 加 CUDA-only optional extra + CUDA 目标设备复跑数值。`AllToAllTokenDispatcher` 满足同一 dispatch/combine 契约 |
| `models/common/moe/dispatcher.py` 的 `TorchAOTokenDispatcher` | **已适配为可选导入适配层**（2026-09-25，§9.1 第 13 项）：torchao 不进 pyproject、不复制上游 Config 嵌套。`TorchAOTokenDispatcher(num_experts, top_k, pad_multiple)` 继承 `AllToAllTokenDispatcher`，仅 `_permute`/`_unpermute` 改委托 torchao `permute_and_pad`（expert-major 重排 + 每组 pad 到 `pad_multiple`，EP=1 本地 padded permute 路径一并移植），构造期 lazy import，未装 torchao loud-raise ImportError（带 `pip install torchao` 指引）；`ParallelConfig.ep_token_dispatcher="torchao"` + `ep_torchao_pad_multiple`（默认 16=FP8）接线 `apply_ep` → swap，默认 `alltoall` 逐位不变。数值**环境未覆盖**（本机无 torchao/CUDA，单测以 sys.modules fake 覆盖 sentinel-row padding 契约与 EP=1 combine 等价性）；解锁条件：CUDA 目标设备装 torchao 复跑 |
| DSA（DeepSeek sparse attention）的稠密 additive mask 路径 | **已移植（2026-09-28，十九次增量）**：`models/common/attention/masks.py::build_dense_attention_mask` + `models/hf/model.py::get_attention_masks` 的 DSA 分支，与上游 `_build_dense_attention_mask` 逐行等价（causal / block_causal 两种 `attn_mask_type`）；flex 照旧运行并按 mask 类型当 `score_mask`（HF 集成分支）。**未覆盖**：真 DSA 模型端到端（transformers 的 DSA 家族需 torch≥2.4 才能建模型）与 CP×DSA（后者显式拒绝） |
| 单进程模拟多卡的 debug 后端（`comm.backend` 的 `fake` / `real_pp_fake_spmd`，2026-09 新增的 `DistributedTopology`） | 登记缺口（2026-09-28，parallel_dims 走查）：上游用 torch 的 `backend="fake"` 建一个"逻辑世界"，可在单进程内模拟任意 world_size 的 mesh（`real_pp_fake_spmd` 再叠一个真实 PP 组，供 PP 边通信）；llmtuner 只有 `world_size == 1 → parallel_dims is None` 与真多卡两条路，单机并行验证走 gloo + torchrun 集成测试。解锁条件：torch 提供 `backend="fake"`（本机 2.2.2 无此 backend）+ 决定给 `accelerator/dist_utils.py` 加一条 debug 后端；届时 mesh 构造无需改动（`build_mesh` 已是 `world_size` 驱动）。2026-09-28 十九次增量曾按该方向实现（`init_distributed` + `NGPU`/`FAKE_PP_RANK`），**按用户要求整条撤下**：fake 后端与 `DistributedTopology` 都不引入，保留本条为登记缺口；同轮的另一项（DSA 稠密 mask）不受影响 |
| vocab-sharded `lm_head` + 端到端 vocab-parallel loss | **D 类，两步走，第一步已完成（2026-09-27）**。第二步（模型侧）未实现：上游 HF 路径把 `lm_head` 的 weight/bias 沿 vocab 维 `S(0)` 切、输入从 sequence-parallel gather 回全长、输出 `S(-1)`（vocab 分片），core `cross_entropy_loss` 检测到分片后走 vocab-parallel CE（`hf_sharding.py` 的 `lm_head` 段）。llmtuner 仍把 HF plan 的 `colwise_gather_output` 解析为 None、`lm_head` 保持复制（`tensor_parallel/tp.py::resolve_plan`）。**第一步（loss 侧接线，已完成）**：`Trainer._loss_vocab_kwargs()` + `HFTransformerModel.vocab_size` 把 `tp_group`/`global_vocab_size` 送到四个调用点（`Trainer._loss_sum`、`chunked_lm_head_cross_entropy`、PP `_scalar_loss_fn`、Validator），`components/loss.py` 按形状分派，因此复制 head 下逐位不变；第二步（vocab-shard realizer + head 已分片但 loss 未被告知时 loud-raise）在多卡环境复跑后再做。

**已从 D 移除（部分）**（2026-09-25）：`models/common/moe_sharding.py`——上游该文件是
声明层：`ShardingConfig` 声明 router 参数 TP Replicate、routed 专家权重仅在 EP 开时
沿专家维 E 取 placement（DP_REPLICATE/EFSDP 为 R,EP 为 S(0)），由上游 Module 协议
的 parallelize 引擎消费。llmtuner 按 B 类语义适配，不复制其 Config 协议：MoE-under-TP
的声明改由 HF tp_plan 的 `packed_colwise`/`packed_rowwise`/`moe_tp_experts` 规格承载
（`resolve_plan` 解析为 None），执行落在 `parallel/tensor_parallel/tp.py` 的结构路
径——`shard_experts_for_tp`（`down_proj (E,D,F)` 切 dim 2，`gate_up_proj (E,2F,D)`
gate/up 两半各自切 dim 1,router 不动）+ `TPMoeSequenceBoundary`（`__class__` swap
安装块边界 sequence all-gather / reduce-scatter，与 dense TP 同一对偶契约，序列维
-2)。梯度语义：边界 collective 的注册反向互为对偶；router 权重 Replicate，梯度由
`_allreduce_replicated_tp_grads` 求和；被切专家参数经块上 `tp_sharded_param_ids`
从该归约排除。state_dict FQN 不变、tp=1 逐位不变。**tp×ep（同日第二段）**：按上游
语义放行——TP 只切 dense,routed 专家由 EP 独占沿专家维 E 切，router Replicate;
`apply_tp` 在 `cfg.ep > 1` 时跳过 MoE 块扫描/分片/边界安装（块留给 `apply_ep`
swap,swap 后的原生 MoE 直接消费/产出 T/tp 序列分片，即上游 ep+sp 的
sequence-parallel 布局，无边界 collective);trainer 的排除判定抽为模块级
`tp_sharded_param_ids`（三类：dense TP realizer、MoE-under-TP 的 F 分片、EP 的
`GroupedExperts` E 切片；EP 专家梯度按 rank 完备，跨 TP 求和会混不同专家的梯度）。
组合矩阵终态（config 期校验在各 config `__post_init__`，跨层裁决单一来源 `parallel/matrix.py`；2026-09-26 收窄）：tp>1×ep>1（cp=1）放行；tp>1×ep>1×cp>1 在
`ParallelConfig.__post_init__` fail-fast（未验证）;shared-expert 块 ×tp 两条路径均
loud-raise(ep=1 边界处、tp×ep 的 swap `convert_block` 处）;plan 声明 MoE 规格但
探针找不到块（ep=1）loud-raise;GPT-OSS 布局 loud-raise。aux loss / padding-mask
LB / quantile hook 的归约轴此前已按 ep_enabled 含 tp 书写，放行后不重复计数、无需
改动。测试
`tests/unit_tests/cpu/parallel/test_tp_moe.py`：规格解析、分片重建、单进程
partial-sum 等价（reduce-scatter 求和的算术内容，无进程组）、FQN 稳定、幂等、
ep>1 时 apply_tp 原样放行 MoE 块、shared-expert×tp×ep 拒绝、梯度排除规则、组合
矩阵各格。**未覆盖**：真多卡 forward/backward 等价（本机 torch 2.2.2 无
DTensor/spmd 执行栈，gloo 下功能 collective 未验证）——待 torch≥2.12 多卡复跑。

**已从 D 移除**（2026-09-25 批 5 移植）：Ulysses CP × varlen/packed（上游
baff3c681）——上游形态是把 Ulysses 的 token↔head resharding 提为
`UlyssesCPInnerAttention` 共享层，`UlyssesCPVarlenInnerAttention` 借 MRO 把
`super().forward` 派发到 `VarlenInnerAttention`；varlen 元数据（cu_seqlens）不随输入
分片（`cp_shard` 把 `attention_masks` 摘出再原样放回），因为 all-to-all 后每个 rank
都持有全长 token 流。llmtuner 按 B 类语义适配、不复制类层次：packed 语料的"varlen
元数据"在 HF/flex 集成里是烘进 BlockMask 的文档结构，因此
`hf/model.py` 的 `preprocess_inputs` 在 `ulysses` 策略下把全长文档 mask **不 Q 分片**透传
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
Ulysses 拒绝（per-head sinks 只走 TP 分片）不适用：llmtuner 尚无 GPT-OSS 支持。

**已从 D 移除**（2026-09-24 批 8 移植）：`distributed/compile.py`——四件互相独立的
能力全部落 `llmtuner/parallel/compile.py::apply_compile`，由
`config/training.py::CompileConfig`（`training.compile_config`，默认全关）驱动，
装配顺序不变（AC 之后、FSDP 之前；PP 下每 chunk 由
`parallel/parallelize.py::parallelize_hf_transformers` 过同一函数）：

* **逐 block compile**（`per_block=True`）：每个 decoder layer `block.compile(
  backend=..., fullgraph=True)`（`Module.compile` 就地，state-dict 键与 FSDP 包装
  不变）；默认 False 保持整体 `torch.compile(model, backend="inductor")`，与旧
  路径逐位一致。
* **async TP**（`enable_async_tensor_parallel=True`）：设
  `torch._inductor.config._micro_pipeline_tp` 并为 TP group 注册 symmetric
  memory（按 group 名去重，PP 每 chunk 重入安全）。配置校验期（
  `LLMTunerConfig.__post_init__`）拒绝 无 compile / tp=1 两种组合；装配期对
  无 TP mesh、torch 无 `_micro_pipeline_tp`、无 `enable_symm_mem_for_group` 三种
  情形 loud-raise，不静默跳过。
* **regional_inductor**：flex 只有 inductor lowering，故非 inductor backend 下
  flex 模型必须 scoop。`backend="aot_eager"` 且模型走 flex（wrapper 新 property
  `uses_flex_attention`）时用 `torch.fx.passes.regional_inductor` 包
  `aot_autograd`；annotation 落 `hf/model.py` 的 `flex_attention_hf` 的
  `maybe_regional_inductor({})`（默认 nullcontext，inductor/eager 路径零开销）。
  flex 模型配其他非 inductor backend → `ValueError`；torch 无 regional_inductor
  → `NotImplementedError`；sdpa 模型 backend 原样透传。inductor_configs 传空
  （llmtuner 走 HF 的 flex 集成，不带上游 FlexInnerAttention 的 autotune 配置）。
* **capture_scalar_outputs**：按上游条件在编译的 model part 含 token-choice MoE
  block（`iter_moe_layers` 非空，即 EP swap 后的 llmtuner MoE 栈）时设
  `torch._dynamo.config.capture_scalar_outputs=True`；dense 模型不动该全局量
  （逐位不变）；torch 无此 knob 时 loud-raise。

未移植（登记）：上游同文件的 `skip_fwd_side_effects_in_bwd_under_checkpoint`
（AC+compile 的 side-effect 重放分歧开关，llmtuner 未遇到其失败场景，需要时按上游
注释补）与 `FakeTensorMode.__init__` 的 `torch.compiler.disable` monkeypatch
（修上游 pytorch#178887，等上游修复即废弃的临时措施）。`CompileConfig.components`
未移植（llmtuner 只编译 model，loss 无 compile 通路）。

**已从 D 移除**（2026-09-24 批 4 移植）：`pipeline_with_first_stage_modules`——
多模态 first-stage 模块并入 stage 0，落为 `apply_pp` 的可选关键字参数
`first_stage_module_fqns: Sequence[str] | None`（默认 None，默认时切分与
state-dict 键逐位不变）+ `parallel/pipeline_parallel/apply.py::
prepend_first_stage_modules`（仅作用于自动生成的切分，把存在的 FQN 按序前插
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
`components/validate.py::Validator` 落 `trainer/validate.py`（2026-09-26 文件名对齐上游，原 validation.py）
（`Trainer.validate`/`should_validate`/`check_validation_feasibility` 的
薄委托背后）+
`config/training.py::ValidationConfig`（`training.validation_config`，默认
None 关闭，关闭时训练循环逐位不变；programmatic-only，同 `ema_config`）。
语义对齐：eval 模式 + `no_grad`、结束恢复 train；loss 按全局有效 token 数
归一化，token 计数走 dp mesh、loss 和走 dp×cp×tp loss mesh（与训练 loss 同一
对 mesh、同一归一化）；每次 pass 新建并关闭临时 dataloader（`repeat=False`
对应 `steps=-1`），不进 checkpoint、不动 `ntokens_seen`；训练循环内调用点在
checkpoint save 之后、profiler.step 之前，与上游同序。两条上游 bug fix 一并
移植：零 batch / 零有效 token 报 `ValueError`（上游 6c2dadbb3），dp>1 拒绝
`steps=-1`（上游 90b25912f，在 trainer 构造期、真实 dp degree 已知后检查）；
另对 random 无限语料的 `steps=-1` 同样 fail-fast。PP × validation 未支持：
llmtuner 的 PP loss 计算内嵌在 schedule 的训练步里，无 `pp_schedule.eval`
对应的 eval 通路，构造期 `NotImplementedError`（loud-raise，不静默跳过）。

**已从 D 移除**（2026-09-24 批 2 移植）：quantile-balanced MoE routing——
`QuantileBalancedTopKRouter` + `QuantileBalancer` + `register_moe_quantile_balancing_hook`
（biased top-(K+1) cutoff、1000-bin 直方图、分位数 mean-centred 覆写 bias），与
sign-based bias 互斥（同层 raise、跨层 hook raise），经 `ParallelConfig.
moe_quantile_balancing` 启用；MoE padding-mask 负载均衡——`MoE.set_padding_mask`
一次性暂存通道（HF layer 签名穿不了 mask），mask 只过滤负载均衡统计
（tokens_per_expert、aux loss f/p、quantile 直方图），不动 routing 执行，无 mask
逐位不变；CP/TP 由 `shard_padding_mask_for_cp/tp` 与 token 流同序切分。

**已从 D 移除**（2026-09-24 批 3a 移植）：在线 EMA——`llmtuner/components/optimizer/ema.py`
（515 行上游 `components/optimizer/ema.py` 的语义移植）：`EMA` 复用
`OptimizersContainer` 的 flat FQN state-dict 契约，`decay = 2**(-1/(half_life_fraction*num_updates))`
动态计划或固定 decay，firing count 由 trainer step 推导（resume 不重置 decay），
`step_bias` 支持阶段重编号，`start_step`/`update_every_n_steps` 门控，可选
`buffer_patterns` 浮点 buffer 跟踪（整型 buffer 拒绝）；checkpointer 增加 `ema`
state 键与 `_find_load_step(max_step=)`，llmtuner/config 完成三侧接线。上游
DTensor unwrap/rewrap 与 CUDA `torch._foreach_lerp_` 专项未移植（llmtuner 的 FSDP2
张量本身就是 DTensor，容器 state dict 直接交给 DCP）。

**已从 D 移除**（2026-09-24 批 1 移植）：`CastLinear`——lm_head compute-dtype
变换，落 `models/common/cast_linear.py`（`nn.Linear` 子类，state-dict FQN 不变），
经 `ModelConfig.compute_dtype` 启用，默认关闭；router `_debug_force_load_balance`
——落 `TokenChoiceTopKRouter` 同名构造参数，round-robin 语义与上游逐字一致；
PP per-stage seed——`trainer/seed.py` 的 `derive_distinct_seed`（上游
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
倒 SVG）未移植，llmtuner 的 AC 路径没有 dump folder 概念。该文件仍有一处未移植，是
依赖而非删减：`RegionAC` 需要 `torch_remat`（llmtuner 不依赖，且它的"模型声明
region"建立在 llmtuner 没有的 `Module.configure_remat_regions` 协议上）——配置
`mode='region'` 在 config 与 `apply_ac` 两处都是显式 `NotImplementedError`，解锁
条件写在报错与文件 docstring 里。原先同列的 `disable_dynamo_lru_cache`（SAC+PP 的
重编译 workaround）已于 2026-09-27 随 pp×AC 对齐一并移植，见下方六次增量。

**已从 D 移除**：`tools/validate.py`——上一版既写了它、又写"上游也没有这个路径，已从
表里移除"，自相矛盾。核实：上游 `torchtitan/tools/validate.py` **确实不存在**，这一行
没有意义，删掉。

## E —— 包面（`__init__.py` 与入口）

重组出口，不是移植内容：这些文件定义 llmtuner 的**公开 API 面**，上游对应物是同名
`__init__.py`（若存在）。改上游的导出列表时才需要看这里。全表 20 个（19 个
`__init__.py` + `__main__.py`），行数为 2026-09-27 实测。

| llmtuner | 行数 | torchtitan |
| --- | --- | --- |
| `__init__.py` | 35 | `__init__.py`（根面：`LLMTunerConfig` + `Trainer`） |
| `__main__.py` | 6 | `train.py` 的入口对等物 |
| `accelerator/__init__.py` | 73 | 上游无对应（PEP 562 懒加载包面） |
| `components/__init__.py` | 0 | 空文件 |
| `components/checkpointer/__init__.py` | 87 | `components/checkpointer/__init__.py` |
| `components/optimizer/__init__.py` | 39 | `components/optimizer/__init__.py` |
| `config/__init__.py` | 81 | 上游 config 包面（llmtuner 全量再导出 16 个配置类） |
| `datasets/__init__.py` | 64 | `components/data/__init__.py` |
| `datasets/multimodal/__init__.py` | 16 | `hf_datasets/multimodal/`（刻意不导入子模块） |
| `datasets/text/__init__.py` | 10 | `hf_datasets/`（刻意不导入子模块） |
| `models/__init__.py` | 0 | 空文件 |
| `models/common/__init__.py` | 53 | `models/common/__init__.py` |
| `parallel/__init__.py` | 49 | `distributed/__init__.py` |
| `parallel/context_parallel/__init__.py` | 27 | `distributed/context_parallel/__init__.py` |
| `parallel/expert_parallel/__init__.py` | 20 | 上游无对应（见 C 类） |
| `parallel/fully_shard/__init__.py` | 5 | 上游无对应子包（只再导出 `apply_fsdp`） |
| `parallel/pipeline_parallel/__init__.py` | 12 | 上游无对应 |
| `parallel/tensor_parallel/__init__.py` | 23 | 上游 `distributed/tensor_parallel.py` 已删除，后继是 `protocols/sharding.py` + 各模型 `*_sharding.py` 的声明面 |
| `trainer/__init__.py` | 27 | 上游无对应（配置再导出为兼容别名） |
| `utils/__init__.py` | 0 | 空文件 |

`__init__.py` 的 ratio 平均偏低（0.3 上下）是正常的——它们导出的是各自的公开面，不是
从上游抄结构。表里的数值是"文件行数"，不是 ratio。

**空文件**（0 行）：`components/__init__.py`、`models/__init__.py`、`utils/__init__.py`。

**两个子包**：`datasets/` 下按语料分 `text/` 和 `multimodal/`，其余 8 个模块平铺在
`datasets/` 根下。上游的 `hf_datasets/` 是 `components/data/` 的兄弟目录，llmtuner 曾用
`datasets/hf/` 镜像它（4 层，三个 0 字节的 `__init__.py`），先被溶解成单层，再于重组
时按语料分成两个子包。

划分依据是**内容**而非上游路径：`datasets/` 根下的模块与上游 `components/data/` 一一
对应，且都被两边共用（`loader.py` 默认 `TextCollator`、`multimodal/mm_collator.py`
复用 `collators.py` 的 `Collator`/`TrainerBatch`），所以不进任何一边；只有 `text.py`
和 5 个 `mm_*.py` 是语料专属。两个子包的 `__init__.py` 都**刻意不导入子模块**——惰性
导入契约靠这一点维持。

**上游路径对应关系因此不再一一成立**（`hf_datasets/multimodal/utils/image.py` 在
llmtuner 侧是 `datasets/multimodal/image.py`），本表的 llmtuner 列是唯一权威。

## TP/SP 对齐结论（2026-09-27）

本轮把 llmtuner 的 TP / sequence-parallel 与 TorchTitan `f35966713` 逐项核对。结论：
**数学与通信角色逐条等价，差异集中在"声明形态"、两处已修缺口与一处已登记的 D 类缺口**。

| 面 | llmtuner | 上游 | 判定 |
| --- | --- | --- | --- |
| 声明形态 | `parallel/tensor_parallel/tp.py` 的 `ShardingConfig` + `resolve_plan`（消费 HF `tp_plan` 的规格字符串） | `protocols/sharding.py::ShardingConfig` 挂在模块 `_sharding_config` 上，由 `Module._parallelize` 分发 | **等价（形态不同）**：两者都是"权重切分 + 激活布局"的声明，llmtuner 的词汇表来自 HF |
| colwise | `ColumnParallelLinear`：weight 切 dim 0，融合 all-gather 入、输出 feature-sharded | `hf_sharding.py:60 _hf_colwise_config`：weight/bias `S(0)`、out `S(-1)` | **等价** |
| rowwise | `RowParallelLinear`：weight 切 dim 1，融合 reduce-scatter 出（回到序列分片） | `hf_sharding.py:71 _hf_rowwise_config`：weight `S(1)`、bias `R`、out_src `P`、out_dst → SP placement | **等价**（上游 `P`→SP 的重分布正是 llmtuner 融合 RS 的位置） |
| 注意力边界 | `GatherSequenceFirst` + `ColwiseLinearNoGather`：父模块持有 gather，q/k/v 退化为 plain feature-sharded GEMM | `decoder_sharding.py:218 set_gqa_attention_sharding` + `_attach_flex_kernel`（SP 输入在注意力内部 gather 回 Replicate） | **等价** |
| 序列并行语义 | TP 即 SP：batch 先按 CP、再按 TP 切；`parallelism.enable_sequence_parallel=false` 直接 config-raise | `sp_enabled = tp_enabled and enable_sequence_parallel`（`parallel_dims.py:550`） | **有意分歧**：llmtuner 没有"激活全复制"的退化路径 |
| 序列切分顺序 | 先 CP（`models/hf/model.py:591`）后 TP（`:623`），TP 切在 CP 分片内 | `hf_sharding.py:52 _hf_sequence_parallel_placement()` = `PartitionSpec(DP, (CP, TP), None)` | **等价**：CP 外、TP 内的联合切分 |
| norm 权重 | q/k norm 保持复制（HF 4.57 起 plan 已不声明它们），梯度由 `Trainer._allreduce_replicated_tp_grads`（`trainer/trainer.py:470`）汇总 | `decoder_sharding.py:177 norm_config`：SP 时权重 `R`，"BWD AR 交给 FSDP" | **等价**（同 D14：上游归 FSDP、llmtuner 归 trainer，数值一致） |
| token 计数 / loss mesh | `trainer/batch.py:221` 计 `labels.numel() // (cp*tp)`；loss mesh 含 tp（`parallel/parallel_dims.py:220`） | loss mesh 只含 dp×cp（`parallel_dims.py:260`） | **耦合差异**：上游把 tp 的归约放进 vocab-parallel CE，llmtuner 的 head 是复制的、必须跨 tp 求和。两侧各自自洽，随 lm_head 缺口一同处理 |
| lm_head 与 loss | HF 的 `colwise_gather_output` 解析为 None → head 保持复制（全 vocab）+ 普通 CE；loss 侧参数已接线（按形状分派，复制下 no-op） | head `S(0)`/`S(-1)` vocab 分片 + core `cross_entropy_loss` 检测分片走 vocab-parallel CE | **D 类缺口，两步走的第二步未做**：loss 侧接线 2026-09-27 完成，head 真分片与"未接线即 loud-raise"待做，见上"D —— 真正缺失"表 |
| 注意力头整除 | 本轮之前只有 ulysses CP 路径检查 `% (tp*cp)` | `config/validation.py:150 head_shard_degree`：解析期即校验 `heads % (tp*cp)` | **本轮修复**：新增 `parallel/head_sharding.py`；`apply_tp` 查 `% tp`、ulysses CP 查 `% (tp*cp)`，合起来即上游那一次检查 |
| 未实现的 HF 规格 | `colwise_rep` / `rowwise_rep` / `local_*` / `gather` / `replicate` / `sequence_parallel` loud-raise | 这些是 DTensor 时代的 replicated-activation 布局，上游由 SPMD 声明承担 | **有意拒绝**：llmtuner 的 GEMM 是 SP 对偶 collective，没有全复制激活路径；本轮把报错改成指名 + 说明理由 |

**验证边界**：TP/SP 的数值等价需要多卡与 torch≥2.12（symm-mem、`spmd_types`、
`torch.distributed.pipelining`），本机（torch 2.2.2、CPU）不可达，可执行的只有声明层/
装配期单测与 2-rank gloo 等价性（design §8 第 10 项）。因此上表的"等价"是**代码级核对**
结论；多卡上的数值等价仍是待办，不能据本表声称已验证。

## 版本与漂移

- 本文最近一次人工审计工作树：llmtuner `11002c1`（+本轮改动），TorchTitan `b64103072`；
  详细验证记录见
  `llmtuner_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区）。
- 2026-09-26 增量审计：基线推进至 TorchTitan `9e159aed7`（审计时上游 HEAD 附近），
  llmtuner 工作树 HEAD `8c0ac4c`（+本轮文档改动）。`b64103072..9e159aed7` 间三个
  提交的处理结论：

  | 提交 | 结论 |
  |---|---|
  | `9e159aed7` TP projection 后端重构（#4704） | **语义已对齐，无代码动作**。通信角色不变量在 llmtuner 已成立：column 拥有 input collective（`ColumnParallelLinear` 融合 all-gather）、row 拥有 output collective（`RowParallelLinear` 融合 reduce-scatter）；共享输入多投影在父模块一次性 gather（`GatherSequenceFirst` + `ColwiseLinearNoGather`，同上游"父模块持有、子投影为 plain Linear"语义）。`_linear()` seam 服务 LoRA/量化（llmtuner 裁剪面，不移植）；`PartialBiasRowwiseLinear` 上游删除并并入 `RowParallelLinear`，llmtuner 同名类的 bias I→P 语义本就一致，保留（仅测试使用）。AsyncTensorParallelTransform 重写是上游 Module-registry 面的模块替换实现，llmtuner async TP 走 inductor `_micro_pipeline_tp` + symm-mem，机制不受影响；"转换后（LoRA/量化）投影不支持 async TP"的约束在 llmtuner 无对应面（两者均裁剪），不登记守卫。上游 `dist_gemm.py` 改名 `async_linear.py`，本文映射随之更新。 |
  | `847f98a6f` RegionAC AllToAll remat regions（#4837） | **随 RegionAC/DeepEP 缺口锁定，解锁条件不变**（torch_remat + CUDA deep_ep 核）。TokenDispatcher 变 Module 是 remat region 机制的载体，llmtuner 无消费方。可独立移植的语义——dispatch/combine 恒 SAVE——经核对**已在 llmtuner 成立**：selective AC 的 save set 含 `_c10d_functional.all_to_all_single`（`activation_checkpoint.py` 的 `comm_ops`），即 llmtuner AllToAllTokenDispatcher 用的原语，无需动作。 |
  | `090c0c931` graph_trainer none AC MemoryPolicy（#4476） | **实验目录，不适用**。`experiments/graph_trainer/` 无 llmtuner 对应面；等义语义 llmtuner 已有（`activation_checkpoint_mode='none'`）。 |
  上一轮审计（llmtuner `528dc9d` × TorchTitan `c6e416bbd`）引用的
  `llmtuner_torchtitan_alignment_audit_2026-09-21.md` 不在当前工作区。
- 2026-09-27 增量审计：基线推进至 TorchTitan `c8a3e7666`（审计时 HEAD），llmtuner 工作树
  `b6bbb33`（本轮改动落为 `142bb6d`）。`9e159aed7..c8a3e7666` 共 20 个提交、118 个文件，与本仓相关
  的只有 6 处（其余集中在 `rl/`、`quantization/`、`experiments/graph_trainer/`、
  `overrides/fused_mla.py`、`distributed/flex_shard/`、`config/transform/`，属范围外）。
  逐项结论：

  | 上游文件/符号 | 上游意图 | 分类 | llmtuner 处理 |
  |---|---|---|---|
  | `distributed/activation_checkpoint.py::FullAC` / `SelectiveAC` 的 `early_stop` `False`→`True`（#4836） | 性能：recompute 产出全部所需张量后即停，不再重放区域剩余算子，省 1-4% step time。上游 8×H100 实测数值不变（多数 exact，最大 loss 差 2.8e-6）、峰值 reserved 内存漂移 0；旧的 `False` 是上游 #1580 的 llama4 内存泄漏 workaround，已失效 | **A** | **已同步**：`parallel/activation_checkpoint.py` 两处改为 `early_stop=True`，并把 docstring 里"非默认旋钮"的理由改写为上游 #4836 的结论 |
  | `models/common/linear.py` 新增 `GroupedLinear`（`num_linears` 投影轴，w13 存 `[E,2,F,D]`）、`models/common/moe.py` 用它重写 `GroupedExperts`、`models/common/moe_sharding.py` 把单一 `inner_experts` 配置拆成 `w13`/`w2` 两份、`distributed/fsdp.py` 专家放置改为 `_linear_param_shard_placements(include_unstacked_grouped=True)`（即 `Shard(weight.ndim-2)`） | 新能力面：把融合的 w13 投影做成带 projection 轴的 `GroupedLinear`（载体是 blockwise 量化与 LoRA），FSDP 相应沿矩阵行而非默认 dim 0 切 | **B** | **无需动作（表示等价）**：llmtuner 的专家是 packed 3-D（`gate_up_proj (E,2F,D)`、`down_proj (E,D,F)`），`Shard(ndim-2)` 与 llmtuner 的 `Shard(1)` 切的是同一段——`[E,2F,D]` 的第 i 个 chunk 与 `[E,2,F,D]` 沿 F 的第 i 个 chunk 重合，`down_proj` 两侧同为 `Shard(1)`；dense 侧 llmtuner 的 fused QKV 是 2-D `[r*H,D]`，默认 `Shard(0)` 即切输出维，正确。`num_linears` 轴服务量化/LoRA，两者都在 llmtuner 裁剪面内 |
  | `models/common/token_dispatcher.py`、`distributed/deepep/deepep.py` | 注释改名 `GroupedExperts`→`RoutedExperts` | C | 无动作（纯注释，无语义） |
  | `experiments/transformers_modeling_backend/moe_replacement.py`、`state_dict_adapter.py` | Module-registry / state-dict adapter 面的 MoE 构建适配 | C | llmtuner 无该面（swap 直拷 HF 权重），不移植 |
  | `models/common/multimodal.py`、`distributed/flex_shard/`、`quantization/`、`rl/`、`experiments/graph_trainer/`、`overrides/fused_mla.py`、`config/transform/*` | 范围外 | C | 不适用 |

  本轮环境与验证（与 design doc §7 同轮记录）：Python 3.11.5（conda base，`python -V`）
  / torch 2.2.2 / CPU gloo。
  该 torch 缺的是**一组**新 API，不是单个包：`spmd_types==0.2.5` 装了但 import 失败
  （缺 `torch.distributed._local_tensor`）、`torch.distributed.tensor` 无公开
  `DTensor`、无 `torch.distributed._composable.fsdp`、无 `torch.nn.attention`
  （flex_attention）、无 `torch.distributed.pipelining`、无 `torch.OutOfMemoryError`、
  无 CUDA。实测：25 个 integration 脚本 **2 passed / 23 failed**，23 个失败全部是上述
  API 缺失（10 × `DTensor` ImportError、7 × `torch.nn.attention`、4 ×
  `spmd_types`、2 × `_composable.fsdp`），没有一个失败来自本仓逻辑
  ——唯一通过的两个是 `reduce_equivalence`（只依赖 gloo）与
  `vocab_parallel_loss_equivalence`（只依赖 `components/loss.py`）。
  CPU 单测 143 passed / 9 failed / 59 skipped，9 个 failed 全是
  `torch.OutOfMemoryError` 与 `torch.distributed.pipelining` 缺失。因此本轮
  **没有**复跑任何等价性测试，可运行的只有静态门禁与不依赖上述面的 CPU 单测。
  要复跑 ②（vocab-sharded head）与 ③（PP）的等价性，需要 torch≥2.12 + 多卡；
  ② 的 first half（loss 调用点 vocab-aware）不需要多卡，但需要 trainer 可导入，
  即同样受这批 API 阻断。
  同轮另修一处 llmtuner 内部缺陷：`62df262` 把 `parallel/parallelize_hf.py` 改名为
  `parallel/parallelize.py` 时漏改两个 test 侧 import
  （`tests/unit_tests/cpu/parallel/test_activation_checkpoint.py`、
  `tests/integration_tests/ep_fsdp_equivalence.py`）。该缺陷此前一直被环境门禁跳过掩盖，
  在 torch≥2.12 的环境里会是 `ModuleNotFoundError`；已于 `142bb6d` 修复。
- 2026-09-27 二次增量（同日，走查驱动）：基线再前进到 TorchTitan `f35966713`
  （`c8a3e7666` 之后 14 个提交、177 个文件，`git -C <torchtitan> log --oneline c8a3e7666..f35966713`）。
  与本仓相关的面逐项核对如下（其余在 `rl/`、`experiments/`、`quantization/`、
  `models/qwen3_5|6|8`、`torchtitan_recipes/tests`，范围外）：

  | 上游改动 | 分类 | llmtuner 处理 |
  |---|---|---|
  | `trainer.py`、`train.py`、`components/loss.py` **未变** | —— | 无动作；`llmtuner_trainer_walkthrough.md` 引用的行号据此复核通过 |
  | `training_engine.py`（42 行）、`components/validate.py`（25 行）：PP 编排、`max_num_documents` 的传入点 | **A/B（PP、校验面）** | 走查已覆盖：校验器的 PP 分支是 llmtuner 的 loud-raise 缺口（D18）、`max_num_documents` 的裁剪理由经复核成立（D19）；PP 损失函数的双驱动接线缺陷已修（D17，`make_schedule_loss_fn`），见 `llmtuner_trainer_walkthrough.md` §11.3 |
  | `distributed/{parallel_dims,pipeline_parallel,utils,fsdp,activation_checkpoint}.py`，`distributed/context_parallel/` 包重整为 `context_parallel.py` | **B（并行面，走查进行中）** | 属 `parallel/` 批次（`llmtuner_trainer_walkthrough.md` §12 第 2 项），已开走查：`activation_checkpoint.py` 于 2026-09-27 对齐（pp×AC 放行 + `_disable_dynamo_lru_cache` 移植，见六次增量）；`parallel_dims.py` / `fsdp.py` / `context_parallel.py` / `utils.py` 仍待走，`stages.py` 的 stage 表与 `matrix.py` 守卫同批复核（stage 顺序与上游 `parallelize_hf_transformers` 的 sharding→AC→compile→FSDP 一致；上游 PP 的 `head_shard_degree` 式解析期校验在本表「TP/SP 对齐结论」已登记）。本轮只确认本表引用的 `torchtitan/distributed/parallel_dims.py:260` 与 `set_determinism`（`torchtitan/distributed/utils.py:118`）语义未变 |
  | `config/{parallelism,validation,configs}.py`：`max_num_documents` 的 CUDA graph 校验（`torchtitan/config/validation.py:49`） | C | llmtuner 无图通路（D10），不适用 |

  本轮验证（与 `llmtuner_trainer_walkthrough.md` 同轮）：CPU 单测
  **143 passed / 59 skipped / 9 failed**，9 个失败与前一轮同因（7 个
  `torch.OutOfMemoryError`、2 个 `torch.distributed.pipelining` 缺失），都在未触碰的文件；
  新增 1 个受 `pipelining` 门禁的用例
  （`tests/unit_tests/cpu/parallel/test_pipeline.py:413`），本机跳过、torch≥2.12 环境执行。
  文档 `文件:行号` 引用经机械校验（文件存在 + 行号在范围内）全部命中，脚本见
  `llmtuner_torchtitan_alignment_workflow.md` §7.1。
- 2026-09-27 三次增量（TP/SP 对齐走查）：逐项判定见上「TP/SP 对齐结论」。本轮落地：
  1. **注意力头整除守卫**（对齐上游 `config/validation.py:150 head_shard_degree`）：新增
     `llmtuner/parallel/head_sharding.py`；`apply_tp` 校验 `heads % tp`，ulysses CP 校验
     `heads % (tp*cp)`（原来的内联检查改为调用同一函数，拒绝语义与消息要点不变）。
     此前纯 TP（cp=1）没有任何守卫：`shard_weight` 只看特征维，而 8 个 KV 头 × head_dim 128
     = 1024 特征恰好能被 `tp=16` 整除，错误要等 HF 的 head reshape 才暴露。测试：
     `tests/unit_tests/cpu/parallel/test_head_sharding.py`（7 例，**无**环境门禁，本机可跑）、
     `test_tp.py` 新增 2 例（守卫在触碰 mesh 之前触发；整除时不被误伤）、
     `test_cp.py` 原有 3 例继续覆盖 `tp*cp` 路径。
  2. **plan 读取统一**：新增 `tp.py::model_tp_plan`（`tp_plan` 属性优先、回退 `_tp_plan`），
     `resolve_plan` 与 `apply_tp` 的 MoE 规格探测共用它。`HFTransformerModel.tp_plan` 在内层
     `_tp_plan` 为空时回退到内层 `tp_plan` 属性——HF 只用属性暴露 plan 的模型不再被读成
     "无 plan"（那样 `apply_tp` 会静默一个投影都不切）。测试：`test_hf_wrapper.py` 新增 1 例
     （属性回退 + 属性优先的取舍各钉一次）。
  3. **HF `*_rep` 类规格的报错改写**：`colwise_rep` / `rowwise_rep` / `local_*` / `gather` /
     `replicate` / `sequence_parallel` 仍然 loud-raise（llmtuner 没有 replicated-activation
     TP 路径，静默改成 `colwise` 会让相邻算子拿到它不预期的布局），但报错现在指名规格、列出
     可接受词汇并给出理由。**影响面**：Apertus / GLM-4V / Phi-4-multimodal / Llama-4 /
     FlexOlmo 的 HF `base_model_tp_plan` 带这些规格，这些家族要么改写 plan、要么不开 TP。
     测试：`test_tp.py` 6 例 parametrize + 1 例 typo 仍拒绝。
  4. **文档修订**：三处"上游 `distributed/tensor_parallel.py` 已删除、无后继"改为指向真正的
     后继（`protocols/sharding.py` + `models/common/decoder_sharding.py` +
     `experiments/transformers_modeling_backend/hf_sharding.py`），并新增本文「TP/SP 对齐结论」。
  同轮顺带修掉一个会误报的用例：`tests/unit_tests/cpu/parallel/test_tp.py` 的 Qwen3 用例断言
  transformers 4.57 已不再声明的 `replicated_with_grad_allreduce`（在 torch≥2.12 环境会失败），
  改为把该规格注入真实 plan 来钉住分支，恢复版本无关性。
  本轮验证：`test_head_sharding.py` 7 passed（本机）；`test_tp.py` / `test_cp.py` /
  `test_hf_wrapper.py` 在本机被 `spmd_types` 门禁跳过，改以注入最小 torch/spmd 垫片后脱门禁运行，
  三处分别为 20 passed / 3 passed（其余因 fake PG 与 DeviceMesh 的 2.2 差异 errored）/
  1 passed，新增用例全绿。
- 2026-09-27 四次增量（删除 batch-invariant 特性面）：删除 `llmtuner/utils/batch_invariant.py`
  及全部消费点。依据是上游语义：torchtitan 的开关由 `debug.batch_invariant` 驱动
  `distributed/utils.py::set_batch_invariance`（注册 ATen override + 设 NCCL 环境），但
  `trainer.py` 对 SFT 直接 `raise ValueError("Batch-invariant mode is not needed in
  supervised learning.")`——该特性只服务 `rl/`。llmtuner 是 SFT trainer，且
  `set_batch_invariant_mode` 全仓无生产调用者，原模块自述"开关可设但未安装 kernel"的
  TODO 即其唯一存在理由。删除面：`batch_invariant.py`（整文件）；`hf_wrapper.py` /
  `cp_kernel.py` 的 `separate_full_blocks=not is_in_batch_invariant_mode()` 改为常量
  `True`（即此前的默认值；torch 2.10 无该旋钮时 `models/common/attention/masks.py` 仍会 pop 掉）；
  `components/loss.py` 的 `_GatherVocabShards` / `_gather_vocab_shards` 与
  `compute_logprobs` 的 gather 分支（vocab-parallel 的 `reduction="none"` 路径保留）；
  `test_cp.py` 的 `test_full_length_mask_tracks_batch_invariant_mode`；
  `vocab_parallel_loss_equivalence.py` 的 gather 值/梯度等价块；
  `cp_ulysses_equivalence.py` 的同名调用点；以及本文、符号指南、设计文档与
  `llmtuner/README.md` 的登记行。
- 2026-09-27 五次增量（lm_head 缺口第一步 + config 走查）：
  1. **vocab-parallel loss 四处接线**（D 类 `lm_head` 缺口的第一步，纯 no-op）：
     `Trainer.loss_vocab_kwargs()`（原 `_loss_vocab_kwargs`；TP mesh + 模型自身 HF config 的 `vocab_size`，
     新增 `HFTransformerModel.vocab_size` 属性）驱动 `Trainer.loss_sum`（原 `_loss_sum`）、
     `chunked_lm_head_cross_entropy`、PP 的 `scalar_loss_fn`（2026-09-29 去私有化，原名 `_scalar_loss_fn`）与 Validator 路径；
     选择仍按形状（`components/loss.py`），所以 lm_head 复制的今天每条路径都走
     普通 CE，逐位不变。测试：`test_chunked_loss.py` 增 1 例（分片参数下值与三份
     梯度不变）、`test_trainer.py` 增 2 例（缺 TP 轴/缺词表 → 空 kwargs）、
     `test_pipeline.py` 增 1 例（schedule loss 同样收参数且仍为 no-op）、
     `test_hf_wrapper.py` 增 1 例（属性取内层 config，缺字段返回 None）。
  2. **CP `context_parallel_load_balancer` 默认对齐上游**：`"headtail"` → `None`
     （上游 `config/parallelism.py` 默认 None，headtail 由 recipe 显式打开，其
     `test_config_manager.py` 亦断言默认 None）。均衡分片改变每个 rank 参与的
     token 集合，不该由默认替用户决定；`test_config.py` 的默认断言与 ulysses
     配对用例同步改写（拒绝仍是"组合"拒绝）。
  3. **非 CLI 字段不再出现在 `--help`**：新增 `config/cli.py`——`arch_overrides`
     （dict）、`param_groups`（嵌套 dataclass 列表）、`purge_exempt`（callable）
     三个字段，HfArgumentParser 会为它们建 flag 但拒绝一切取值（实测
     `invalid dict value` / `invalid ParamGroupConfig value` / `invalid Callable
     value`），既不能用又占帮助面。`init=False` 是 HfArgumentParser 唯一的跳过
     钩子，于是给解析器一份生成视图（子类 + `init=False` + 沿用基类默认值），
     `isinstance`/`__post_init__`/默认值全部不变；`PARSER_GROUPS` 一并搬到该模块，
     组集合与顺序自此单一来源。测试：`test_config.py` 增 2 例（三个 flag 不在
     option 行、视图解析后仍是真组且默认值在位）——实测 flag 数 127 → 124。
  4. **`test_config.py` 门禁收窄**：`require_env('dtensor')` → `require_env('pipelining')`
     （真正的硬依赖是 `ParallelConfig.__post_init__` 里的 `get_schedule_class` 导入，
     不是 DTensor）；checkpointer 常量改为用例内局部导入 + `skip_without('dcp')`
     （`tests/caps.py` 新增的用例级守卫）；两处 `llmtuner.trainer.LLMTunerConfig`
     改为 `llmtuner.config.LLMTunerConfig`（配置测试不该为此拖入引擎层）。
     `symm_mem` 用例改为能力判定（CC≥9 设备上 skip），不再假设开发机是 CPU。
  5. **复核后不改的两项**：`pipeline_parallel_schedule_csv`——上游同样在装配期
     校验（`distributed/pipeline_parallel.py:360`，先查文件存在、再 `_load_csv`），
     移进 `__post_init__` 反而偏离，保持现状；`fsdp_symm_mem_scope`——上游用
     `scope=None` 表达"关"且 `tyro.conf.Suppress` 不进 CLI，llmtuner 用
     `enable_fsdp_symm_mem=False` + scope 两字段，**默认语义等价**，见符号指南该行。
  6. 顺带发现（第六轮已修）：`train.py` docstring 称"YAML/JSON 文件可位置传入"，
     实测 `HfArgumentParser.parse_args_into_dataclasses(['x.json'])` 直接报
     "Some specified arguments are not used by the HfArgumentParser"（该版本
     transformers 只把 `parse_json_file`/`parse_yaml_file` 做成"整体替换 argv"的
     方法，没有 argv 路径入口），docstring 与实现不符；现改为如实说明"只有旗标，
     没有配置文件通道"，并写明上游用的是 `--module`/`--config` 选择配置函数。
  本轮验证：`test_config.py` 在垫片环境下 38 passed / 1 failed（唯一失败是垫片
  `get_schedule_class` 不抛 ValueError 造成的既有假阴性，HEAD 同）；四处新增用例
  在本机分别随 `test_chunked_loss.py`（12 passed）与脱门禁的
  `test_trainer.py`/`test_pipeline.py`/`test_hf_wrapper.py` 通过（+2/+1/+1）；
  全量 CPU 套件 151 passed / 59 skipped / 9 failed（失败集与基线一致）。
- 2026-09-27 六次增量（pp × AC 对齐，`parallel/` 批次第一项）：
  1. **AC 进入 PP 路径**。上游 `experiments/.../parallelize.py` 的 PP 分支把
     `ac_config` 交给每个 model part 自己的 `parallelize` 调用（
     `pipeline_hf_transformers` 里 `m.parallelize(..., ac_config=ac_config, ...)`），
     而它的 pipeline recipe 全部登记 `SelectiveAC.Config()`；llmtuner 此前在
     `parallelize_hf_transformers` 入口对 `pp > 1` 直接拒绝 AC
     （`matrix.pp_activation_checkpoint`，文案自述"尚未接线"），等于拒绝上游的默认
     组合。现在 `stages.py` 的 `ac` 行 `on_pp=True`，PP 的 per-chunk runner 表加上
     `ac`，两条路径共用同一个 `apply_ac` 闭包（AC 参数不可能在两边漂移）；
     `matrix.pp_activation_checkpoint` 与其表行一并删除。逐 chunk 应用是安全的：
     `split_model_into_stages` 保留 `layers` 容器（层号沿用原索引），`apply_ac`
     折的正是本 chunk 持有的那些层，与上游 `model.get_submodule("layers")` 同构。
  2. **`disable_dynamo_lru_cache` 移植**（上一步解锁的前置条件）。上游在每个
     policy 的 `apply()` 开头调用它，修的是 AC+PP+Flex 下"第二个 microbatch
     以动态 shape 重编译 → 同一区域存在两张合法图 → dynamo 默认 latest-wins
     可能选到期望多一个 symint 的那张，而 SAC 缓存的 inductor-HOP 输出没有它"
     （pytorch/pytorch#166926）。llmtuner 现按上游位置调用（包装模式在折层前、
     `memory_budget` 在设全局量前），但走 `has("dynamo_lru_cache")` 能力门
     （新增注册项）：本机 torch 2.2.2 的 `torch._C._dynamo.eval_frame` 存在而无
     `_set_lru_cache`，此时记一条 info 后继续，不因缺 workaround 拒绝整轮训练。
  3. 测试：`test_assembly_stages.py` 的 `PP_STAGE_ORDER` 断言改为
     `("tp","ac","compile","fsdp")`；`test_matrix.py` 删掉该行用例（表与模块函数
     一一对应的断言自动跟随）；`test_activation_checkpoint.py` 增 4 例——PP 分支
     逐 chunk 折层且 `stage.submod` 重绑、`mode='none'` 在 PP 上仍是 no-op、
     workaround 在 knob 存在时置 `False` / 缺失时不动它。
  4. 文档：`llmtuner/README.md` 的"被明确拒绝的组合"表把 AC 从 `pp > 1` 行移除；
     符号指南 AC 行与能力表、设计文档能力表同步。
  5. 顺带修正 `train.py` docstring 里"YAML/JSON 可位置传入"的不实说法（见五次增量
     第 6 条）：现在是"只有旗标"，并说明上游的配置入口是 `--module`/`--config`
     选配置函数，不是配置文件。
  本轮验证：`test_matrix.py` + `test_assembly_stages.py` 13 passed、
  `test_capabilities.py` 8 passed、全量 CPU 套件 151 passed / 59 skipped / 9 failed
  （失败集与基线一致，本机）。新增的 4 例 AC 用例在本机**不可执行**：`test_activation_checkpoint.py`
  经 `parallelize_hf_transformers` 链式 import `parallel/fully_shard/fsdp.py`，后者要
  `torch.distributed._composable.fsdp.FSDPModule` 与 `torch.distributed.fsdp.CPUOffloadPolicy`
  （torch≥2.4/2.6），而本机 2.2.2 连 AC 模块自身都进不去（模块级
  `torch.ops.aten.mm.dtype`）。故改用"打桩 `apply_ac` + 假 ParallelDims"的编排级检查：
  `apply_ac` 对 PP chunk 恰好调用一次、调用顺序为 `pp → ac → compile → fsdp → schedule`、
  `PP_STAGE_ORDER` 为 `("tp","ac","compile","fsdp")`、`submod` 已重绑（本机实测通过，
  脚本不留仓）；4 例用例本身为代码审阅，需 torch≥2.5 的 CI 执行。真值等价
  （AC×PP 的 loss/grad）仍需 torch≥2.12 + 多卡，与 2-rank gloo 的 `pp_equivalence.py`
  同属未闭合项。
- 2026-09-28 七次增量（`parallel_dims.py` 复核，`parallel/` 批次第二项）：
  逐项对上游 `distributed/parallel_dims.py`。结论：**结构等价，三处差异全部有意**
  ——(a) `loss` mesh 含 `tp`（上游是 `dp*cp`；llmtuner 的 TP 端到端切序列，每个 rank
  的 loss 和只覆盖 `T/tp`，见 "TP/SP 对齐结论" 与 `lm_head` 行）；(b) 上游 `_validate`
  用 `assert`，llmtuner 用带消息的 `ValueError`（`-O` 下 assert 会消失）；(c) 上游
  `DistributedTopology` / `enable_sequence_parallel` / `_real_pp_group_for_fake_spmd`
  与 SPMD-only 的 `unfold_dp_axis*` / `get_activated_mesh` / `resolve_mesh` /
  `get_dense_tp_mesh` 在 llmtuner 无对应（前者是 debug 后端与 SP 开关，后者只服务
  spmd_types，已计入 2026-09-23 死代码清理）。本轮把其中两件从"文档说明"变成"可执行
  契约"：
  1. `_validate_meshes` 的尺寸表提取为 `_expected_mesh_sizes()`：不再只有建过 mesh
     才可达，因此 `loss == batch * cp * tp`（唯一含 tp 的轴）可以在没有进程组的机器上
     钉住——这正是最容易被后来者"顺手改成上游那样"的一行。`test_parallel_dims.py` 新增
     1 例断言该式与非 tp 轴不变。
  2. 三处把 loss mesh 说成"dp+cp"的旧注释改为事实（`models/common/aux_loss.py`、
     `models/common/moe/block.py`、`test_parallel_dims.py` 的用例 docstring）；代码侧
     `trainer`/`validate` 的门本来就是 `dp_cp_enabled or tp_enabled`，与表一致。
  3. 新增 D 类登记：上游的单进程 fake-SPMD debug 后端（`backend="fake"` /
     `real_pp_fake_spmd`，torch 2.2 无此 backend）。
  本轮验证：`test_parallel_dims.py` 脱门禁后 9 passed（新增用例在列），唯一 error 是
  需要真 gloo 进程组的用例被沙箱 `Cannot resolve 127.0.0.1` 挡住；全量 CPU 套件
  151 passed / 59 skipped / 9 failed（失败集与基线一致）；ruff 干净。
- 2026-09-28 八次增量（`fully_shard/fsdp.py` 复核，`parallel/` 批次第三项）：
  逐项对上游 `distributed/fsdp.py`（含 `experiments/transformers_modeling_backend/` 与
  `models/common/decoder.py` 的调用面），并读 `resolve_fsdp_mesh`／
  `resolve_sparse_fsdp_mesh`／`disable_fsdp_gradient_division`／
  `enable_fsdp_symm_mem`／`get_fsdp_reshard_after_forward_policy`／prefetch 段的对应实现。
  结论：**装配逻辑等价，差异四类全部有意**——
  (a) **mesh 表示**：上游把「多轴 storage mesh + `DataParallelMeshDims(shard=dp_shard[,cp],
  replicate=dp_replicate)`」交给 `fully_shard`（它的参数是 SPMD DTensor，必须显式声明轴），
  llmtuner 的参数是普通 tensor，改为 `resolve_fsdp_mesh` 重建 1-D/2-D 专用子网，torch 按形状
  默认读法得到同一组轴；专家侧同理（`(dp_replicate?, efsdp)` 对
  `DataParallelMeshDims(shard="efsdp", replicate="dp_replicate")`）。
  (b) **专家计数来源**：上游读 `moe.num_experts`（SPMD DTensor 的逻辑总数），llmtuner 读
  `moe.router.num_experts`（EP swap 后 `inner_experts` 只剩本地切片 total/ep）；两者比较的
  仍是同一个 total，阈值语义不变。
  (c) `linear_param_shard_placements`（上游把 stacked/grouped 权重按 `Shard(ndim-2)` 切矩阵行）
  在 llmtuner 无对应物：llmtuner 只有 packed 3-D 专家权重（`w1`/`w3` `(E,F,D)`、`w2`
  `(E,D,F)`）与 2-D `nn.Linear`，前者的 `Shard(1)` 就是输出维、后者的默认 `Shard(0)` 就是
  输出维；且 llmtuner 没有任何 `num_linears` 式融合 Linear（上游该 helper 只对这类权重生效）。
  (d) `apply_fsdp_to_multimodal_encoder`（vision tower 整体分片）无消费者：llmtuner 的
  `models/common/multimodal.py` 只有 span/gather 融合算子，没有编码器模块。
  本轮四处动作：
  1. **命名统一（degree/size）**：`apply_fsdp_to_decoder` 的参数保持 `ep_size`，**不跟**
     上游改名成 `ep_degree`——llmtuner 的配置面统一拼 `*_size`（`expert_parallel_size` /
     `data_parallel_shard_size`，见 `config/parallel.py` 的字段说明），`ep_size` 与之同词根。
     同一次把 `_fsdp_shard_degree` 改名 `fsdp_shard_size`、
     `require_heads_divisible_by(degree=, divisor=)` 改为 `(size=, axis=)`（两个调用点
     `apply_tp` / ulysses CP 同步）；其余 prose 里的 "degree" 只作概念词保留。
  2. **私有面收敛**：把 4 个其实被跨模块生产代码使用的 `_` 符号转正 ——
     `iter_moe_layers`（`moe/balancing.py` / `hf/model.py` / `compile.py` 三处）、
     `iter_fsdp_modules`（`fully_shard/apply.py`）、`resolve_top_k` /
     `resolve_score_func`（`expert_parallel/convert.py`），并给 `fully_shard/fsdp.py` 与
     `expert_parallel/probe.py` 补 `__all__` 明确公共面（fsdp.py 的表面 = 上游 `__all__`
     减去 llmtuner 有意不带的两个入口）。只被单测引用的 `get_default_save_ops` /
     `yarn_inv_freq`（原 `_get_default_save_ops`/`_yarn_inv_freq`）于二十次增量改为公开：
     本仓规则是模块级 helper 默认公开，`_` 只留给框架协议名与同名校验核。
  3. docstring/注释校正：删掉 `apply_fsdp_to_decoder` 里「上游 `dp_mesh_dims` 入口从未接线」
     这句含混说法，改为写明 mesh 表示差异的成因（上游参数是 DTensor，llmtuner 不是）；MoE
     分支的 NOTE 补一句上游的 stacked-Linear 覆盖与 llmtuner 无需覆盖的理由；`Shard(1)`
     覆盖旁注明它就是上游 `Shard(ndim-2)` 切的那一段。
  4. 新增可执行契约
     `test_efsdp_placement.py::test_packed_expert_weights_shard_their_output_dim_at_index_one`：
     钉住三个 packed 专家权重的形状与「输出维在第 1 轴」。此前这行等价性只写在文档里，改布局
     （挪动专家轴或 up/down 投影的特征轴）会让 flat `Shard(1)` 静默切错段。
  本轮验证：`test_efsdp_placement.py` 脱门禁后 **9 passed**（含新增用例；其中 3 个需要 size-1
  gloo 进程组的用例在沙箱外执行）、`test_fsdp_contract.py` 脱门禁 **12 passed**。两者都依赖
  一个临时 harness（torch 2.2 既没有 FSDP2 surface 也不支持 1-D 命名 mesh 切片），shim 只
  模拟 FSDP2 的模块接口与 `edp_mesh["efsdp"]` 的取值，不实现分片本身，故真机分片行为仍需
  torch≥2.12 + 多卡复跑（见 symbol guide §12）。全量 CPU 套件 151 passed / 59 skipped /
  9 failed（失败集与基线一致）。
- 2026-09-28 九次增量（`context_parallel/` 输入分片复核，`parallel/` 批次第四项）：
  逐项对上游 `distributed/context_parallel.py`（上游此前的 `context_parallel/` 包已重整为
  单文件；kernel 侧在 `models/common/cp_attention.py`）。结论：**语义等价，差异三类，全部
  登记**——
  (a) **表示**：上游 `shard_tensors(input_dict, input_shardings, permutation)` 走 `SpmdType`
  声明 + `spmd.shard(R→S(seq_dim))`；llmtuner 没有声明面，改为
  `shard_batch_for_cp/tp`、`shard_padding_mask_for_cp/tp`、`shard_attention_mask_for_cp`
  五个显式入口，置换与切分一并委托 torch 私有的 `_context_parallel_shard`（它接
  `load_balancer`，permute+shard 在同一步内完成）。两端对「先置换、再按 seq 轴切」的语义
  一致，mask 只切 Q 轴、KV 保持全长也一致。
  (b) **负载均衡器形态**：上游是 `ContextParallelLoadBalancer` 抽象类 + `HeadTail` /
  `PTRR` 两个实现（构造期吃 `seq_len` 与 `attention_metadata`，`generate_permutation()`
  产出全局置换）；llmtuner 用字符串 `context_parallel_load_balancer`（`"headtail"` /
  `None`）现场构造 torch 的 `_HeadTailLoadBalancer`，`seq_len % (2*cp)` 的校验两边相同。
  `"ptrr"` 仍是登记缺口（配置期 + 装配期两道 loud-raise）：它要从 BlockMask 推置换，而
  llmtuner 切 batch 时还没有 mask（mask 由 wrapper 在 forward 内按 positions 构建）。
  (c) **校验位置**：上游 `shard_tensors` 每次调用查 `shape[seq_dim] % cp == 0`、以及所有
  CP 张量同 seq len/同 device；llmtuner 把 `max_seq_len % cp` 放在配置解析期
  （`config/root.py::__post_init__`），运行期交给 torch 的 helper。后者更早、更强（trainer
  路径的 T 恒为 `max_seq_len`），但没有覆盖「调用方传入与 `max_seq_len` 不同的 T」这种绕过
  配置的用法——登记为已知差异，本轮不补冗余运行期检查（本机 torch 2.2 无 CP helper，该分支
  无法验证，且会与 `require_torch_cp` 的调用顺序纠缠）。
  本轮只做复核与登记、无代码改动（同日的命名/私有面调整为八次增量的一部分）：
  `cp_kernel.py` ↔ 上游 `models/common/cp_attention.py` 的逐项对照、
  `accelerator/dist_utils.py` ↔ 上游 `distributed/utils.py` 仍待走。
- 2026-09-28 十次增量（`cp_kernel.py` 与 `accelerator/dist_utils.py` 复核，`parallel/`
  批次第五项，收尾）：两项复核 + 一处补移植，结论全登记——
  (a) **`context_parallel/cp_kernel.py` ↔ 上游 `models/common/cp_attention.py`：语义等价。**
  kv_allgather 走的是**同一个** torch 自定义算子 `flex_cp_allgather`（上游 HF 后端的
  `_wrap_flex_kernel_cp` 用的就是它，`(b, heads, seq, dim)` 的 dim 2、进程组名捕获、
  backward 由算子自带 reduce-scatter），Q 保持 token 分片、mask 为 Q 分片/KV 全长；
  ulysses 的 token↔head all-to-all 与上游 `UlyssesCPInnerAttention` 是同一置换（scatter
  head、gather seq，反向互为转置），mask 分派（packed 全长 vs 单文档重建）与九次增量
  记录的 wrapper 语义一致。**一处显式差异**：上游**原生**模型的
  `KVAllGatherCPFlexInnerAttention.Config` 暴露 `reduce_dtype: float32|bfloat16`
  （`backward_options={"op_dtype": ...}`，默认 fp32），llmtuner 委托 torch 算子因而没有该
  旋钮——但上游 **HF 后端**（llmtuner 真正对应的那一支）同样没有，故属「与 HF 后端对齐、
  与原生模型路径不同」，不是什么待补参数；`llmtuner_torchtitan_symbol_guide.md` 中
  「dtype 可配（默认 fp32）」的旧措辞已在本次改正。
  (b) **`accelerator/dist_utils.py` ↔ 上游 `distributed/utils.py`：无能力缺口。**
  两者血缘不同（llmtuner 那支 vendored 自 OpenMMLab `mmengine.dist`，上游等价物在
  `accelerator/collectives.py` 与调用点的 `accelerator/dist.all_reduce`），故本次按能力
  而非形状对照，逐项终态：`dist_sum`/`dist_max`/`dist_mean`/`dist_sum_tensor` → llmtuner
  用 `all_reduce` 在调用点（trainer/validator 的 loss·token 归约；`components/metrics.py`
  完全不碰 `torch.distributed`），不重建命名薄封装（`collectives.py` 模块 docstring 同步
  改写）；`set_pg_timeouts`/`clip_grad_norm_` → 已移植且已复核（EP 裁剪按物理本地 expert
  参数分组并跨 EP 归约，免去上游「每个参数都是带 `"ep"` 轴的 DTensor」断言；dense-only
  路径与上游逐行同构）；`init_distributed` + `DistributedTopology`（`fake` /
  `real_pp_fake_spmd`）→ 唯一的真缺口，已挂 D 表（本机 torch 2.2.2 无 `backend="fake"`，
  不可验证）。`batch_invariant`/`bf16x9` 属四次增量已删特性面，不重复。
  (c) **补移植（唯一代码改动）**：上游 `set_determinism` 中两件可移植件落到
  `Trainer.seed_everything`（原 `_seed_everything`，2026-09-29 去私有化）—— `PYTHONHASHSEED = str(seed % 2**32)`（本进程读不到，但
  之后 spawn 的 dataloader worker 会读，故必须在此设置，与上游同拼写）与
  `detect_anomaly`（`TrainingConfig.detect_anomaly`，默认 `False` 逐位不变；
  `torch.autograd.set_detect_anomaly(True, check_nan=False)` + 上游同文告警，
  `check_nan=False` 的理由同上游：NaN/Inf 梯度检查走 `aten._is_any_true`，无 DTensor
  sharding 策略，分片参数上会崩，而记录栈这一半保留）。`collectives.py` 的 `__all__`
  故意不再导出确定性符号——该模块的 docstring 声明「每个符号都源自上游」，而
  `set_determinism` 在 llmtuner 无同名对应物，落点就是 seed 助手。
  测试：`tests/unit_tests/cpu/test_trainer.py` 新增两条（`PYTHONHASHSEED` 导出、
  `detect_anomaly` 的 `check_nan=False` 契约），取消 gate 后 33 passed（基线 31），
  失败集不变。
  本轮后 `parallel/` 批次的复核面（`stages.py`/`matrix.py`/`activation_checkpoint.py`/
  `parallel_dims.py`/`fully_shard/fsdp.py`/`context_parallel/`/`cp_kernel.py`/
  `accelerator/dist_utils.py`）全部走完，余下只有需要多卡的 TP/SP 布局根治（D14）与
  D18 的 PP×校验实现。
- 2026-09-28 十一次增量（`components/` 批次开篇：checkpointer + profiler 走查）：
  (a) **checkpointer（`components/checkpointer/`）逐文件对照，语义已对齐，改动只一处死代码。**
  对照方式：AST 归一化（剥 docstring/注释后 `ast.unparse`）逐文件 diff，再人工核对语义。
  `base.py`：策略方法（`_parse_step`/`_find_load_step`/`_purge_stale_checkpoints`/
  `_states_to_load`/`_create_checkpoint_id`/`ModelWrapper`/`purge_thread`/`shares_storage`（当时名 `_shares_storage`））
  与上游逐行同构；`dcp.py`：`_save`/`_load_checkpoint`/`dcp_save`/`_save_last_step`/
  `_flattened_model_states_sd` 与上游逐行同构（含 async 三段模式、`exclude_from_loading`
  与 staged 目录保留）；`torch_checkpointing.py`：策略与 `_save_last_step` 的 HF 导出
  路径（`sharded/` 子目录 + `pre_finalize_callback` 里的 barrier+consolidate）同构；
  `filesystem.py` 与上游 `tools/filesystem.py` 1:1（ratio 1.000），`utils.py::canonical_fqn`
  同构。**登记的有意差异（均为适配而非缺口）**：
  (1) `BaseCheckpointManager` 新增 `_initialized` 门（上游无）：上游靠各类自己的
  `hasattr(self, "staging_future"/"_manager")` + `getattr(self, "save_future", None)` 容错
  部分构造；llmtuner 的 `dcp.CheckpointManager.__init__` 会在 HF 选项上 raise，为此用一个
  显式标志统一挡住 `load`/`save`/`close`/`maybe_wait_for_*`，语义等价、形状不同；
  (2) 新增 `enable`（配置驱动）：上游的 manager 只在配置了 checkpointer 时才构造，llmtuner
  总是构造一个 `CheckpointConfig`，所以 `__init__`、`_should_save`、`_should_prewarm`
  三处按 `enable` 短路；
  (3) dcp 的 4 处 `assert` 改显式 `raise`（ValueError/TypeError）——同上游语义，但 `-O`
  下不会消失；HF 选项在无 `sd_adapter` 时报的错误更具体（llmtuner 不随包发布 adapter，
  三个 HF 选项都是登记缺口而非可用开关）；
  (4) `base._should_purge` 的 `dist.get_rank() == 0` → `dist_utils.is_main_process()`
  （非分布式下前者会 raise）；
  (5) 配置校验全量搬到 `config/checkpoint.py::__post_init__`（13 条逐条对齐），额外把
  上游 `dcp.Config.async_mode` 与 `training_engine` 的 `create_seed_checkpoint` 收进同一
  dataclass；上游那条 `initial_load_model_only` 无 `initial_load_path` 的告警**故意不移植**
  （llmtuner 每次运行都会构造默认 config，含 `--help`，该告警会无端触发）；
  (6) 命名：`_FilesystemCheckpointStorage`/`_async_save_config` 去私有化为
  `FilesystemCheckpointStorage`/`async_save_config`（测试需要这两个 seam），符合本仓
  "非必需不加 `_`" 的取向；`EXPORT_DTYPE_MAP` 与 `models/common/cast_linear.py` 的
  `TORCH_DTYPE_MAP` 是同一张 3 键表的两次书写（上游只有 `config/__init__.py` 一份），
  登记为已知重复，不为此把 components 反向依赖 models；
  (7) 上游 `dcp.py` 的 `SaveDone` 是**零引用死类**，不移植。
  **该模块本次唯一的代码改动**：`TorchCheckpointingManager` 的 `staging_future` 是死成员
  （`__init__` 置 None、`_close` 读一次，全仓无任何赋值点；上游该类根本没有这个成员，
  它在 dcp 里才有对应物），连同 `_close` 里那段空转一起删除。
  `torch_checkpointing` 后端仍未装、未跑（可达性说明见该文件 docstring），故只做静态复核。
  (b) **profiler（`components/profiler.py` ↔ `observability/profiler.py`）：对齐，并修一处
  设备面缺口。** 策略、目录布局、OOM 处理（`caused_by_oom` 防环 + 隐式链）、
  `MemoryProfiler` 的频率/命名/协议 4 全部同构。**发现的缺口**：上游的 activity 列表是
  "CUDA 可用加 CUDA、否则 XPU 可用加 XPU"，llmtuner 只加了 CUDA 分支，且 docstring 写了
  "设备只有 cuda/cpu 两种"——但 `accelerator/device.py` 的 `DEVICE_PRIORITY` 是
  npu/cuda/musa/mlu/xpu 五值，`xpu` 可达，于是 XPU 运行的 trace 会退化成 CPU-only。已按
  resolved device 补 `xpu` 分支（其余设备仍 CPU-only，与上游一致），并改正该 docstring；
  新增用例 `test_the_trace_activity_follows_the_resolved_device`（cpu/cuda/xpu 三态钉住）。
  登记的有意裁剪：`leaf_folder`（只服务上游 torchft 的 per-replica 子目录，llmtuner 无副本
  概念）、CUDA-graph annotations（随 D10 无图路径）、`structured_logger` span、`active()`
  builder（llmtuner 构造处已知道 step 与 folder）；memory history 改经
  `accelerator/monitoring` 的 device 探针（上游非 CUDA 分支调 `torch.memory`，不存在）。
  (c) 测试/文档：`test_profiler.py` +1 例；`llmtuner_torchtitan_symbol_guide.md` §7.2/§7.3
  与 `llmtuner_trainer_walkthrough.md` §12 同步。`components/` 批次余下 metrics 与 optimizer
  两个模块。
- 2026-09-28 十二次增量（`components/` 批次收尾：metrics + optimizer 走查）：同样以 AST 归一化
  逐文件对照，结论如下。
  (a) **metrics（`components/metrics.py` ↔ `observability/metrics.py`）：逐项对齐，无能力缺口。**
  训练与校验两条日志的 key 集合、`tps`/`tflops` 公式、`time_metrics`/`memory` 派生、
  `should_log` 的 `step == 1 or step % log_freq == 0` 判据全部一致。差异四类，全部为已登记
  裁剪或适配：(1) 上游的自由函数 `compute_training_performance_metrics`（供 trainer 与实验
  共用）没有移植，llmtuner 把它内联进 `MetricsProcessor._derive` 的 `_Derived` dataclass；
  语义相同，形状不同；(2) MFU 抑制条件：上游看 `has_quantization`，llmtuner 看
  `gpu_peak_flops == 0 or num_flops_per_token == 0`（无量化路径，故用"设备未知就不报"等价替代）；
  (3) `_build_metric_logger` 去掉 `ft_enable`/`ft_replica_id`（无副本概念）、wandb 失败策略
  改为只吞 ImportError，rank 判定改用 `get_distributed_rank()`（单进程安全）；
  (4) `DeviceMemoryMonitor` 改吃 `device_type`、CPU 分支返回全零 `DeviceMemStats`、
  `_to_pct` 加除零守卫、`build_device_memory_monitor()` 在 CPU 上不打印容量；
  `ensure_pp_loss_visible` 增加 `not pp_enabled` 提前返回（上游该函数无此守卫，仅靠唯一调用点
  门控，语义等价），`_get_metrics_rank` 去私有化为 `get_metrics_rank`。模块 docstring 声称
  "从不调用 `torch.distributed`"经复核成立。
  **本次唯一代码改动**：删除 `MetricsProcessor.set_num_flops_per_token`——全仓零调用者，且其
  docstring 的理由（"metrics processor 先于模型构造、当时拿不到该数"）与 llmtuner 的实现相反
  （构造函数本来就收 `num_flops_per_token`）；上游也没有这个方法（它由 trainer 直接赋值属性）。
  (b) **optimizer（`components/optimizer/`）：三条登记差异（两条为本次新登记），其余对齐。**
  `optimizer.py` 的构造算法（逐 model part × 逐 optimizer 名的分组、pattern 首个命中即认领、
  `_build_impl_kwargs`、`_log_optimizer`、`step`、flat FQN state dict）与上游同构。
  **新登记**：上游第四种 `implementation="fused_opt_states_bf16"`（Adam step pre-hook 预建
  bf16 状态 + load post-hook 复原 dtype，CUDA fused 混合精度核）与
  `optimizer_factory_kwargs_by_name` 未移植，两条均入 D 表（前者不可验证：价值全在 CUDA
  fused 核且改写 checkpoint 状态 dtype；后者无消费者）。**补登记**：`_resolve_optimizer_factory`
  的第三个工厂 `DistMuon` 位于 `distributed/flex_shard/`，该目录早已作为范围外 C 类登记，
  此处把两者显式连上；`default_adamw`（上游 `lr=8e-4`/betas (0.9,0.95)/wd 0.1 的便捷构造）
  未移植，其上游调用者只有 torchft llama3 recipe 与 RL 示例（均在裁剪面内），训练路径由
  config 的 catch-all 默认组覆盖（betas 用 torch 默认，config 已声明为 NUMERIC 分叉）。
  **形状差异**：MoE 负载均衡/quantile hook 的注册点从容器挪到 `trainer/builder.py`（上游是
  各模型自己的代码注册），hook 本体在 `models/common/moe/balancing.py`；与上游"容器只提供
  `Optimizer.__init__` 的 hook 机制"一致。**优于上游的两处**：`_validate_params` 会点名未被
  认领的可训练参数并检出重复认领（上游只是一条 `assert expected == actual`）；`step` 的
  closure 断言改 `ValueError`。
  `utils.py`（ratio 0.996）：只剩 `isinstance(x, A | B)` 与 `zip(..., strict=False)` 之类
  现代化改写，无能力差异。`lr_scheduler.py`：WSD 公式逐行核对一致（同样的 0-based `+1`
  修正、`stable_steps = total + 1 - warmup - decay`、三种 decay 形状与 `min_lr_factor` 缩放）；
  差异为 `wsd_factor`（当时名 `_wsd_factor`）提到模块级 + `build_lr_scheduler` 自由函数取代上游嵌套闭包与
  `Config.build`，以及已登记的 `decay_ratio` 默认 0.0（上游 None）、`total_steps < 训练步数`
  拒绝、`load_state_dict({})` 空字典 no-op、`LRSchedulersContainer(total_steps=...)` 断言 seam。
  `ema.py` 属 2026-09-24 批 3a 已移植项，本轮无改动。
  (c) **走查中发现并修掉一个真 bug（本轮唯一行为改动）：`implementation="fused"` 在 CPU 上必崩。**
  取证：`torch.optim.AdamW([p], fused=True)` 在 CPU 上**构造期**即 `RuntimeError`
  （`requires ... supported devices: ['cuda','xpu','privateuseone']`），但 llmtuner（与上游同形）
  把 `fused` 放进每个 param group 而非构造参数，torch 只校验构造参数，于是 CPU 上构造通过、
  第一次 `step()` 才炸：`NotImplementedError: Could not run 'aten::_fused_adamw_' with arguments
  from the 'CPU' backend`。而 `implementation` 的默认值正是 `fused`，所以**文档里的 quickstart
  （单进程 `--steps 20`）在 CPU 上会在首步 optimizer.step 崩溃**；旧 config help 却写着
  "falls back to the for-loop kernel elsewhere"，测试 `test_the_implementation_flag_reaches_the_
  inner_optimizer` 还把 `group["fused"] is True` 钉住（因为它只构造、不 step，看不见崩溃）。
  上游不会遇到：torchtitan 是 GPU-only。**修法（保持上游形状、落实本仓自己的承诺）**：
  `_build_impl_kwargs` 先问 torch 该设备有没有 fused 核（`torch.optim.adam.
  _get_fused_kernels_supported_devices()`，取不到时回退 `{"cuda","xpu"}`），没有就把
  `fused` 降级为 for-loop 并记一行 info；`fused` 是"内核偏好"而非硬要求（它是默认值），
  且在 CPU 上 for-loop 与 foreach 实测逐位一致（5 步 AdamW，`torch.equal` 全真）。CUDA/XPU 上
  行为与上游逐字相同。测试：改写 `test_the_implementation_flag_reaches_the_inner_optimizer`
  为按设备判定，新增降级用例与**回归用例**（默认 config 在 CPU 上真的 `step()` 一次）。
  (d) 验证：全量 CPU 套件 152 passed / 59 skipped / 9 failed（失败集同基线）；optimizer
  子套件 43 passed（2 个 `test_optimizer_config` 失败仍是缺 `torch.distributed.pipelining`）。
- 2026-09-28 十三次增量（`components/` 全树收口：`loss.py` + `tokenizer.py` 走查）：这两个文件
  不在 `walkthrough` §12 第 3 项的子清单里（loss 由 vocab-loss 批次覆盖过半、tokenizer 属
  A1），因此是回答"components 是否完全对齐"时才补走的。结论：
  (a) **`components/tokenizer.py` ↔ 上游同名文件：对齐。** AST 归一化后 11 处差异全是
  `Configurable`/嵌套 `Config` 的移除（`BaseTokenizer`/`HuggingFaceTokenizer`/
  `MultiModalTokenizer` 改成显式构造参数）、`open(..., 'r')` 之类现代化，以及上游 PR-1540 的
  `assets/tokenizer` 弃用提示（llmtuner 无该历史路径）。`apply_chat_template` 注入 bos/eos
  kwargs 与 `add_generation_prompt=True` 是已登记的上游 backend tokenizer 同源改动。
  (b) **`components/loss.py` ↔ 上游同名文件：数值路径逐行对齐，结构差异四条（三条本轮新登记）。**
  逐行核对的是最要紧的那段：vocab-parallel `forward` 的三个 TP all-reduce（max → sumexp →
  gather）、shard 边界公式（`chunk_size = ceil(V/tp)` + 双侧 `min(V, ...)`）、`out_of_range`
  掩码、`backward` 的融合导数（`grad_update = out_of_range - 1` 与
  `(grad_input + exp(log_probs)) * grad_output`）、`vocab_parallel_entropy`（当时名 `_vocab_parallel_entropy`）的 `isneginf`
  守卫与 stack 后单次 all-reduce、`compute_logprobs` 的两条分支、`mse_loss` 的
  `float().detach()` 与 sum 归约——与上游逐行一致（含 `reduction="none"` 的 `[T]` 语义）。
  **新登记差异**：(1) 上游 `cross_entropy_loss` 暴露 `reduction: sum|none`，llmtuner 固定
  `"sum"`（`"none"` 只有 `LossParallelCrossEntropy.apply` 与 `compute_logprobs` 用），无消费者；
  (2) 上游 `ChunkedLossWrapper` 支持**多输出** tuple pred/labels（dMTP 类模型）并返回
  `(loss, metrics)`（`_combine_chunk_metrics` 逐 chunk 合并），llmtuner 的
  `chunked_lm_head_cross_entropy` 只接单个 `(T, H)`、只返回求和 loss；(3) 上游用预分配缓冲的
  `GradAccumulator`（就地拷贝），llmtuner 用 list + `torch.cat`（多一次 `T*H` 拷贝）。
  另登记：上游的 `spmd_typecheck` 静态断言不移植（llmtuner 无 spmd 曲面），
  `_LossParallelCrossEntropy`/`_VocabParallelEntropy` 去私有化；`compute_logprobs`/`mse_loss`
  在 llmtuner **无生产调用者**（上游分别只服务 `rl/` 与 flux 的 `MSELoss`，均在裁剪面内），
  仓内唯一引用是 `tests/integration_tests/vocab_parallel_loss_equivalence.py`，按"移植曲面"保留；
  chunk 循环期间不改 lm_head 的 FSDP reshard/grad-sync 早前已登记为性能差异，此处不重复。
  (c) 结论：`llmtuner/components/` 六个单元（checkpointer、metrics、profiler、optimizer、
  loss、tokenizer）已全部走查完毕；未对齐项只剩逐条登记的差异与 D 表条目
  （`fused_opt_states_bf16`、`optimizer_factory_kwargs_by_name`、范围外的 `DistMuon`，以及
  未安装/未跑过的 `torch_checkpointing` 后端）。本轮无代码改动，套件数字不变
  （152 passed / 59 skipped / 9 failed）。
- 2026-09-28 十四次增量（`walkthrough` §12 第 4 项开篇：`datasets/` 核心走查）：`components/data/`
  与 `llmtuner/datasets/` 的六个核心文件逐一做了 AST 归一化对照 + 逐行读数值相关段，结论是
  **忠实移植**，差异全部是已登记的"去 Configurable/去 microbatch 类型"那一类。
  (a) `packing.py`（3 hunk / 28 行）：只有两处 `Config.build()` → `build_*_packing()` 自由函数，
  grain 图本身逐字相同——`length_struct`/`padding_struct`（`labels` 填 `IGNORE_INDEX`、`padding_mask`
  填 `True`）、`meta_features=("labels","positions")`、`seed`/`shuffle_bins`/`num_packing_bins`/
  `max_sequences_per_bin`、`DocumentAwareConcatThenSplitIterDataset/Iterator`（2026-09-29 去私有化，原名带 `_` 前缀；含 remainder
  `get_state/set_state`）、`next_document_chunk_end`/`packing_output_is_full`（两者均 2026-09-29 去私有化，原名带 `_`）全部无差异。
  (b) `collators.py`（4/21）：`TextCollator` 的载荷逐字相同（zeros + `torch.cat` + 超长 raise +
  `positions[num_tokens:].remainder_(max_context_length)` + `num_valid_tokens=(labels !=
  IGNORE_INDEX).sum()`）。差异只有类型面：上游 `TrainingMicrobatch`/`TokenizedTrainingMicrobatch`
  两个 dataclass → llmtuner 的 `TrainerBatch: TypeAlias = dict[str, Any]`，字段名 `input` 不变；
  `num_rows_per_microbatch()` → `num_rows_per_batch()`。**一处需要登记的设备面差异**：
  `HAS_PIN_MEMORY` 上游取 `torch.accelerator.is_available()`（"机器上有加速器"），llmtuner 取
  `should_use_pin_memory()`（"**本进程解析出的**设备是加速器"）——在"有 GPU 但 `--use_cpu`"的
  机器上前者为 True、后者为 False；pin 内存只对加速器拷贝有意义，故 llmtuner 的判定更贴合实际，
  且 CPU-only 机器上不会去 pin。
  (c) `dataset.py`（6/93）与 `sources.py`（6/63）：同样只有 Configurable → dataclass + 自由函数的
  形状变化。`dataset.py` 的 DP 分片数学被提成 `shard_for_dp`（`divmod` + `min(rank, remainder)`
  的错位切分逐字相同，`DatasetConcat` 内联的那份也改调它，公式一致）、子策略
  （`shuffle=False, repeat=False, dp_rank=0, dp_world_size=1`）、mix 的 `seed + index` 派生、
  `MapDataset.mix`/`IterDataset.mix` 分派全部一致；`sources.py` 的索引 JSONL 解析、
  `split_dataset_by_node`、streaming 不支持精确 resume 的拒绝、`load_dataset` 的一等字段与
  kwargs 冲突校验（提成 `reject_duplicated_hf_fields`，当时名 `_reject_duplicated_hf_fields`）全部一致。
  (d) `types.py`（3/27）与 `loader.py`（3/23）：`DatasetBuildContext`/`DatasetIterationPolicy`
  新增 `__post_init__` 校验（已登记）；`TrainingMicrobatch` 类层次删除符合 microbatch 分叉；
  llmtuner 另加 `Batch` dataclass（`input_ids`/`labels`，合成路径用；放这里是为了避免
  models 层反向 import 数据源，模块 docstring 已说明）。loader 的
  `dataset.batch(collator.num_rows_per_batch(), drop_remainder=repeat, batch_fn=collator)` 与
  `ThreadPrefetchIterDataset` 逐字相同，差异只是"数据集图由调用方建好再传进来"（C 类单入口设计）。
  (e) `datasets/text/processors.py` ↔ 上游 `hf_datasets/text_datasets.py`：SFT 信号的三条关键规则
  **逐字一致**——超长丢弃（`len(tokens) - 1 > max_context_length`，按 `-1` 而非全长）、prompt 段
  labels 置 `IGNORE_INDEX`（`labels[:max(prompt_len - 1, 0)]`）、renderer 路径用
  `~loss_mask[1:]` 掩码；`DATASETS` 注册表三项（c4/c4_test/c4_validation）同源。llmtuner 新增
  `make_local_jsonl{,_sft,_multiturn}` 三个工厂（C 类，服务 CLI 的本地语料）与 `renderer` 参数
  直通。**交叉验证**：llmtuner 的 tokenizer 抄的是上游 `experiments/transformers_modeling_backend/
  tokenizer.py` 的 `apply_chat_template`（默认 `add_generation_prompt=True`），而上游 SFT 数据路径
  用的是基础 `components/tokenizer.py` 的 tokenizer（HF 默认 False）——所以 llmtuner 的
  `ChatProcessor` 必须显式传 `add_generation_prompt=False` 才与上游渲染一致，代码里正是这么写的，
  该写法与它依赖的那个默认值同源。
  (f) 环境与验证边界：`tests/unit_tests/cpu/datasets/` 的 3 个文件（含 57 例的
  `test_data_pipeline.py`）整体被 `require_env('grain')` 门控，本机未装 grain 故**全部 skip**；
  pyproject 钉 `grain==0.2.18`，而本机 PyPI 镜像只有 `0.2.3`（`pip install --dry-run` 甚至误报
  可装，`pip download` 直接失败），版本不符，故**不在本机装**——装错版本会让这批用例的结论失去
  意义。因此本轮的 `datasets/` 结论是**代码级对照**（AST 归一化 + 逐行读），不是运行期证据；
  运行期等价性仍需在装了 grain 0.2.18 的环境复跑这 57 例。
  (g) `datasets/` 余下未走：`random_data.py` 与 `build.py`（都是已登记的 C 类）、
  `text/renderer.py`（已登记的可选导入适配层）、`multimodal/*`（下一轮）。`models/common/*`
  的数值等价性（§12 第 4 项的后半）尚未开始。
- 2026-09-28 十五次增量（`datasets/` 重构与优化）：
  (a) **去重三处，全部行为不变**（用归一化 AST diff 逐条确认改动范围）：
  (1) `lambda sample: sample is not None` 在 `text/text.py`（6 处）与 `multimodal/mm_datasets.py`
  （3 处）共 9 份，提成 `dataset.py::is_not_none` —— 每个 processor 都用返回 `None` 表示"丢弃这行"，
  过滤谓词本就只有一个；具名后九处引用读作同一契约，且谓词可 pickle。
  (2) `if isinstance(graph, grain.MapDataset): graph = graph.to_iter_dataset(read_options=...)`
  在 6 处重复（`packing.py` ×3、`loader.py`、`multimodal/mm_datasets.py`、`dataset.py` 的 mix 子节点），
  提成 `dataset.py::as_iter_dataset(graph, context=...)`；`loader.py` 那处原用局部 `read_options`，
  与它构造 context 时传入的是同一个对象，故语义一致。
  (3) `packing.py` 的 4 键 `length_struct` 字面量出现两次（concat-then-split 与 first-fit），
  提成模块内 `row_lengths(context)`（当时名 `_row_lengths`）；`text/text.py` 的三个 `make_local_jsonl*` 工厂各自重复
  `SingleDataset(source=IndexedJsonlSource(patterns=(path,)), ..., post_filters=...)`，提成
  `local_jsonl_recipe(path=, processor=)`（当时名 `_local_jsonl_recipe`）。全部是同值替换，无行为变化。
  (b) **一次"差一点改错"的记录（有价值，留档）**：对照上游时发现 `process_cc12_wd_sample`
  缺了上游的 `if image is None: texts=[text]` 兜底，看起来是"缺图行会崩"的 bug——实测确认
  `insert_vision_placeholders([None, "text"], [])` 会 `TypeError: sequence item 0: expected str
  instance, NoneType found`。但进一步读仓内用例发现**这是本仓有意为之**：
  `test_insert_vision_placeholders_rejects_a_slot_with_no_token_count` 与
  `test_process_cc12_wd_sample_raises_when_the_image_field_is_absent` 把"缺图 = 样本畸形 → 响亮失败"
  钉成了契约（上游则是静默降级为纯文本行）。故**不改行为**，只在 `process_cc12_wd_sample` 处
  留注释说明该分叉并指向用例；差异登记于 symbol guide §6。
  (c) 有意不动的重复：`process_cc12_wd_sample`/`process_obelics_sample`（当时名 `_process_obelics_sample`）那份九参数显式转发表
  与上游逐字相同（`_process_mm_sample` 的上游调用方也这么写），属"上游形状"，不为了 DRY 而偏离。
  (d) 验证：本机**无法运行** `tests/unit_tests/cpu/datasets/`——pyproject 钉 `grain==0.2.18`，
  而该版本在 PyPI 上只有 macOS **arm64** wheel（`macosx_11_0_arm64`），没有 x86_64，本机是 Intel
  mac 故装不上；能装的最新版 `0.2.3` 又缺 `ConcatThenSplitIterDataset`（默认 packing 用的就是它），
  拿它跑不能说明问题，故不装。`tests/caps.py` 的 `grain` 探针探测的是 `grain.python`（会连带 import
  jax），所以本机这批用例继续 skip。**本轮的验证手段**：逐文件归一化 AST diff（证明"只改了预期的
  那几处"）、`ruff check`（含 F821 未定义名检查）、以及把被修/被查函数的实际调用跑通——
  `process_cc12_wd_sample` 在纯文本行上的行为用 sys.modules 里的 grain 桩真跑过（确认兜底会去
  `insert_vision_placeholders` 的那条 TypeError 路径，也确认显式 raise 是当前行为）。
  运行期等价性仍需在装对 grain 的机器（Linux/arm64 mac）复跑这 57+ 例。
- 2026-09-28 十六次增量（`walkthrough` §12 第 4 项的 `models/common/*` 走查 + 一处真缺口修复）：
  (a) **逐文件 AST 归一化复核**（`models/common/` 全部同名文件；上游独有文件另计）：
  `moe.py`(17 hunks)、`aux_loss.py`(12)、`rope.py`(12)、`async_linear.py`(12)、`activation.py`(9)、
  `linear.py`(6)、`embedding.py`(6)、`feed_forward.py`(6)、`multimodal.py`(3)、
  `__init__.py`(2)、`token_dispatcher.py`(28)。逐处定性后**无新增代码缺口**，结论与登记见
  symbol guide §4；其中三组要点：
  (1) `moe.py`：结构性差异（`RoutedExperts` 只持 `GroupedExperts`+dispatcher；`tokens_per_expert_E`
  归 MoE；`expert_bias_E` 由 MoE 按 quantile 路由注册）与两处**有意分叉**——上游 MoE-under-TP 的
  三个 `_maybe_*_across_tp` 方法在 llmtuner 由块边界 AG/RS 对偶 + tp×ep 的 T/tp 分片直通替代；
  AC 下的 token 重复计数不去重（上游 `// 2` 是给 expert-usage 指标用的，llmtuner 的
  `sign(mean − x)` 对任何正的均匀缩放不变，也不记录该指标）。归约轴口径：
  上游 = `loss` mesh(dp×cp) + EP 时 dense-TP；llmtuner 显式枚举 `dp`+`cp`(+EP 时 `tp`)，
  同一 rank 集合（不能复用 llmtuner 的 `loss` mesh——它含 tp，见 `parallel_dims` 文档）。
  `MicrobatchWiseLoadBalanceLoss` 的 `('cp','tp' if EP else 'cp')` 与上游 `('cp','tp')` 在
  可达域内等价（上游在 tp>1 无 EP 时于 `MoE.forward` 断言拒绝）；丢掉的
  `spmd_local_context('dp')` 是类型检查上下文，llmtuner 必须在 ambient dense mesh 上查 tp group。
  (2) `aux_loss.py` / `rope.py` / `async_linear.py` / `activation.py` / `linear.py` /
  `embedding.py` / `feed_forward.py` / `multimodal.py`：去 Config/Module 协议与 spmd 注解的同型差异；
  `linear.py` 保留 `RouterGateLinear`/`PartialBiasRowwiseLinear`（后者仅测试使用），
  `Linear`/`ColumnParallelLinear`/`RowParallelLinear`/`GroupedLinear` 分别由 `nn.Linear`+TP realizer、
  `GroupedExperts`、`cast_linear.py` 承接；`activation.py` 只留 `SwiGLU`（`Sigmoid`/`Softmax`/
  `SqrtSoftplus` 由 router 的字符串打点替代，`SiTUGLU` 只在 Kimi 面）。
  (3) `token_dispatcher.py` 索引数学逐行核对：`_local_reorder`/`_permute`/`_unpermute`/`combine`
  与上游逐字一致（含 rank-major→expert-major 的 `input_starts[seg_ids] + arange −
  output_starts[seg_ids]`），`all_to_all_single`+`materialize` 对应上游编译分支的
  `all_to_all_single`/`wait_tensor`。**同轮修正一处文档错误**：`BaseEPTokenDispatcher.num_experts`
  被写成"每 rank 专家数"，实为全局数。
  (b) **上游独有三文件定性**：`param_init.py`（已删死代码）、`lora.py`（裁剪面：LoRA 由 HF/peft 提供）、
  `config_utils.py`（**新增登记**：上游 config 工厂层，llmtuner 无 config tree，其判定分别落在
  `expert_parallel/probe.py` / `parallel/matrix.py` / `models/hf/factory.py` /
  `models/hf/model.py` 的 `flex_supported`，逐函数映射见 symbol guide §10）。
  同轮定性上游实验目录里三个此前未登记的文件：`module_conversion.py`（把 HF 模块 `__class__`
  换成 `Module` 协议子类，好让 Module registry 的 `parallelize()` 生效）—— llmtuner 无
  `Module` 协议层，故无对应物，同类 `__class__` swap 技术用于
  `GatherSequenceFirst`/`TPMoeSequenceBoundary`/TP realizer；`config_registry.py`（实验用家族
  config 注册表）—— llmtuner 用 HF `AutoConfig` + `config/` 门面 + `hf/factory.py`；`__init__.py`
  的模型注册表 —— HF auto mapping（`resolve_model_class`）。
  (c) **代码改动：`num_flops_per_token` 重写为结构感知**（本轮唯一实现变更；原"统一近似"对
  MoE 系统性偏低——Qwen3-MoE 约 2×、DeepSeek-V3 约 1.8×，MLA 的 `head_dim` 还是 rope 切片）。
  新增纯函数 `flops_per_token(arch, *, seq_len)` 与上游同名 `quadratic_attention_flops_per_token`：
  MoE 层 = router + top_k 路由专家（active ratio）+ 全部 shared 专家；MLA 用 `q_lora`/`kv_lora`
  与 `qk_head_dim`/`v_head_dim`；`layer_types` 支持 full/sliding/chunked；稠密与 MoE 混栈按
  `first_k_dense_replace`/`mlp_only_layers`/`decoder_sparse_step`/`moe_layer_freq` 分层。
  **拒绝猜**：几何不可解析时返回 0（缺尺寸、MoE 宽度或层划分不明、`layer_types` 短于层数、
  参数面不可推导的层型如 `linear_attention`），延续原契约；未移植 `delta_rule_flops_per_token`。
  验证：DeepSeek-V3 真实 config 反推 active params = 3.64e10（发布值 37B，差值即未计入的
  norms/biases —— 一个既错 MLA 项又错专家权重的公式不可能落到这个数）；稠密路径与旧公式逐位
  相同；20 个新用例在 `tests/unit_tests/cpu/models/test_flops.py`（**不依赖 pipelining/wandb/flex
  门禁**，真实 HF config 驱动）全绿。
  (d) 顺带修掉两个**从未运行过**的坏用例：原 `test_metrics.py` 的 FLOPs 用例 monkeypatch
  `hf/model.py` 里从未导出的 `build_model_config_for` 并调 `num_flops_per_token`（两者都在
  `hf/factory.py`），而包装模块
  从不导出这两个名字；该模块又被 `require_env('wandb','pipelining','flex_attention')` 整体
  skip，故一直未暴露。FLOPs 用例已迁至新模块，`test_metrics.py` 删去该段与三个失去用途的 import。
  (e) 验证：`tests/unit_tests` = 172 passed / 59 skipped / 9 failed（失败集与上轮完全一致：
  7 例 profiler-OOM + 2 例 optimizer_config 缺 `torch.distributed.pipelining`）；`ruff check`
  通过。**未覆盖**：真多卡 EP/CP/PP 数值等价、CUDA/XPU fused kernel、`torch_checkpointing`
  后端、renderer 真库路径（环境不可达，见 §12 验证边界）。
- 2026-09-28 十七次增量（`models/` 目录与文件组织重构；**只搬家，不改行为**）：
  (a) **新布局**：`models/` 分成两半——`hf/`（HF 适配层，B 类）与 `common/`（模型词汇表）。`common/`
  里两个「文件家族」升级为子包，其余仍是一文件一节点：
  ```
  models/
    __init__.py            两半的导览（不 re-export）
    hf/                    __init__.py + factory.py + model.py + flops.py
                           + state_dict_adapter.py
    common/
      __init__.py          发现面索引（re-export，仍是 `from .models.common import MoE` 的入口）
      activation.py  async_linear.py  aux_loss.py  cast_linear.py  embedding.py
      feed_forward.py  linear.py  multimodal.py  rope.py  scatter_add.py
      attention/           __init__.py（纯文档，见下）+ qkv.py + masks.py
      moe/                 __init__.py（纯文档）+ block.py + router.py + experts.py
                           + dispatcher.py + load_balance.py + balancing.py
  ```
  旧→新逐条映射（代码、文档、用例同轮改完）：
  | 旧 | 新 |
  |---|---|
  | `models/hf_wrapper.py` | `models/hf/wrapper.py`（十八次增量再改名 `models/hf/model.py`） |
  | `models/hf_factory.py` | `models/hf/factory.py` |
  | `models/hf_state_dict_adapter.py` | `models/hf/state_dict_adapter.py` |
  | `models/common/moe.py`（MoE 本体） | `models/common/moe/block.py` |
  | `models/common/moe.py`（`RoutedExperts`） | `models/common/moe/experts.py` |
  | `models/common/grouped_experts.py` | `models/common/moe/experts.py` |
  | `models/common/routers.py` | `models/common/moe/router.py` |
  | `models/common/token_dispatcher.py` | `models/common/moe/dispatcher.py` |
  | `models/common/moe.py`（`MicrobatchWiseLoadBalanceLoss`） | `models/common/moe/load_balance.py` |
  | `models/common/balancing.py` | `models/common/moe/balancing.py` |
  | `models/common/qkv.py` | `models/common/attention/qkv.py` |
  | `models/common/masks.py` | `models/common/attention/masks.py` |
  (b) **三条组织约定**（写进各包 docstring）：① 子包索引只做导航，不 re-export——导入一个节点不得
  拖进它的兄弟（`dispatcher` 需要 EP collectives、`load_balance` 需要 SPMD mesh 上下文；
  `attention/__init__.py` 更必须保持惰性，因为 `masks.py` 在模块层 import
  `torch.nn.attention.flex_attention`，否则 `models.common` 在 CPU/无 flex 的 torch 上直接不可导入）；
  ② 发现面索引只有 `models/common/__init__.py` 一个（平名字可从那里取）；
  ③ `moe/block.py` 里的 `__getattr__` 懒导出 hack 删除（拆包后循环依赖消失，改为调用方直接
  import 叶子）。
  (c) **机械等价性证明**：写脚本把 `git show HEAD:<旧路径>` 的每一个顶层 `def`/`class` 去 docstring
  归一化后与它在新区里的位置逐一定比——**57 个顶层定义、0 个函数体变化**；唯二的「缺失」是
  `RoutedExperts`（已证 byte-identical，落在 `moe/experts.py`）与被删除的 `__getattr__`。
  行为不变的其余证据：`ruff check`（含 F821/I001）通过；用 `spmd_types`/flex 垫片把 26 个
  models 模块 + `parallel/*`/`components/metrics` 逐个 import，28/30 成功（两个失败是环境既有限制：
  `trainer.trainer` 缺 grain.experimental、`trainer.builder` 触发 transformers 5.9 在 torch 2.2.2
  下的内部 `NameError`，与本轮无关）；`tests/unit_tests` = 172 passed / 59 skipped / 9 failed
  （失败集与上一轮完全一致）。
  (d) **有意不动**：类名/函数名一个没改（它们与上游符号一一对应，是映射表的锚点）；测试模块
  文件名保持原样（`test_hf_wrapper.py` 等仍测同一模块，只是模块路径变了）；`models/hf/model.py`pper.py`
  的 import 段行数不变，故文档里 `wrapper.py:591/639` 这类行号引用仍指向原语义。

- 2026-09-28 十八次增量（`models/hf/` 收尾：按职责拆算术、按上游改名）：
  (a) **`factory.py` 拆出 `flops.py`**：前者 581 行里混着四件事（HF config 构造 / 类解析 /
  meta materialize / FLOPs 计算），后者的纯算术（`flops_per_token` +
  `quadratic_attention_flops_per_token` + 六个私有 helper，322 行）没有构建语义，独立成节点后
  `factory.py` 回到 259 行、职责单一。入口 `num_flops_per_token(cfg)` 仍留在 `factory.py`——它
  需要先解析出 config，再委托 `flops.flops_per_token`；`flops.py` 只依赖
  `transformers.configuration_utils`，不依赖 torch。
  (b) **`hf/wrapper.py` → `hf/model.py`**：上游对应文件就叫
  `experiments/transformers_modeling_backend/model.py`，改名后映射 1:1、类名不变
  （仍是 `HFTransformerModel`）。纯 rename，行号不变，故文档里 `model.py:591/639` 的引用语义不动。
  (c) 验证：以十七次增量提交为基线做「按定义」比对——`factory.py`+`flops.py` 16 个顶层定义、
  `wrapper.py`→`model.py` 7 个顶层定义，**0 个函数体变化**（纯搬家）；`ruff check` 通过；
  垫片环境下 9/9 目标模块可导入（含新 `flops.py`，且同一 Qwen3-MoE config 的 FLOPs 与拆分前
  逐位相同：1151827968）；`tests/unit_tests` 172 passed / 59 skipped / 9 failed（失败集不变）。
  `models/` 模块数 26→27，全仓 122→123 个 `.py`（99 实现模块）。

- 2026-09-28 十九次增量（D 表该批的 DSA 稠密 mask 落地 + 三项可读性/健壮性改进）：
  (a) **DSA 稠密 additive mask（移植）**：`models/common/attention/masks.py::build_dense_attention_mask`
  （纯张量：`[1,1,T,T]`，0 允许 / -inf 屏蔽，`block_causal` 与 flex 的 causal+same-document 同义），
  `models/hf/model.py::get_attention_masks` 对 `uses_dsa(config)`（`index_topk`）走该分支——上游
  `_build_dense_attention_mask` 的逐行等价物；`__init__` 里原来的构造期 `NotImplementedError` 换成
  `self.uses_dsa = uses_dsa(config)`。flex 路径不变：HF 的 flex 集成按 mask 类型分支，稠密张量当
  `score_mask`（上游注释同义）。**同轮显式拒绝 CP × DSA**（`_get_cp_attention_masks`：
  CP 的 mask 通道只接受 BlockMask，稠密张量要手工 Q 切分 + 过 load balancer，未验证不给近似）。
  验证：新用例把稠密 mask 与 `get_causal_mask_mod` / `get_document_mask_mod` 在同一索引网格上比对
  （`causal & same_document` 逐位相等），另有 dtype/shape/取值集合断言。
  (b) 附带改进（同轮，均为可读性/可验证性）：① `models/common/__init__.py` 改成**惰性索引**
  （PEP 562 + `_EXPORTS` 表 + TYPE_CHECKING 静态视图），兑现它自己文档里"命名一个节点不得导入它"
  的承诺——此前 `import llmtuner.models.common.rope` 会连带 MoE 栈与 `DTensor`；② `attention/masks.py`
  的 flex import 惰性化（`flex_ops()` 缓存），于是该模块在无 flex 的 torch 上可导入，DSA 稠密 mask
  也因此能在本机跑用例；③ `capabilities.is_compiling()` 收掉 `torch.compiler.is_compiling` 在
  torch 2.2 缺失的问题（rope 的 bounds check 调用点），避免纯 torch 模块在新老版本上二选一。
  **测试口径变化**：`test_masks.py` / `test_rope.py` / `test_multimodal.py` 三处 `require_env('spmd_types')`
  门禁被证实是惰性的（它们只依赖纯 torch 代码）故移除，新增 `test_models_index.py`（惰性索引），
  `test_masks.py` 增稠密 mask 段。顺带修掉一处此前被门禁掩盖的失效 import
  （`test_models_common.py` 仍在 `from llmtuner.models.common import grouped_experts`）。
  全量：**258 passed / 56 skipped / 9 failed**（失败集仍是那 9 项环境问题：7 profiler-OOM +
  2 optimizer_config 缺 pipelining）；相比本轮之前 172 passed，多出的用例全部是此前被门禁
  跳过、现在真跑起来的。
  (c) **同轮试做、随后整条撤下的部分（留档）**：单进程 fake 后端（`init_distributed` +
  `NGPU`/`FAKE_PP_RANK` + `DistributedTopology` 类型与 `parallelism.comm_backend` 配置字段）曾实现并在
  本机验证通过（`NGPU=8` → 单进程逻辑世界 8、DeviceMesh 可建；`FAKE_PP_RANK=2`×pp=4 → 物化逻辑 rank 4），
  但**按用户要求撤下**：不引入 `DistributedTopology`、不引入 `comm_backend`，`accelerator/dist_utils.py` /
  `config/parallel.py` / `trainer/builder.py` 回到原状，该能力继续按 D 表登记为缺口（上面那两行）。
  撤下的原因值得记下：它是本批唯一需要新增配置面与类型面的东西，而那两项的收益（单机模拟多卡）在
  本仓的验证边界里仍属"未覆盖"，不如保持接口不变。

- 2026-09-29 二十次增量（`_` 私有面：全仓审计 + 两批去私有化）：
  (a) **新规则**（写进 symbol guide 第五条横切约定）：模块级 helper / 数据类**默认公开**；
  前导 `_` 只保留三种——① 框架协议要求的名字（`scatter_add.py` 的 `_backward`/`_setup_context`
  是 `torch.library.custom_op` + autograd 的契约）；② 与"公开包装"配对的低层核心
  （`accelerator/dist.py` 的 `_broadcast_object_list`/`_all_gather_object`/`_gather_object`：
  同名的公开版返回 list，私有版收 out-list，去前缀会重名）；③ 类内 protected 方法
  （子类/同包协作者的 template-method 钩子）。类内 `self._x` 属性属于封装，不在本条范围。
  (b) **模块级：95 → 5 个前导 `_`**。审计脚本枚举全部模块级 `def`/`class` 后，按"跨模块使用
  → 必须公开"先处理了 6 个（`packing/{conversions,iterators}` 的 5 个 + `parallel/
  pipeline_parallel/apply.py::scalar_loss_fn`，见下一条十六次/二十次记录），随后按新规则把
  其余 90 个一并公开（含 32 个上游同名 helper：`_yarn_inv_freq`→`yarn_inv_freq`、
  `_maybe_check_max_pos`→`maybe_check_max_pos`、`_RouterGateLinearFunction`→
  `RouterGateLinearFunction`、`_get_default_save_ops`→`get_default_save_ops`、
  `_maybe_enable_async_tp`→`maybe_enable_async_tp`、`_get_pipeline_metadata`→
  `get_pipeline_metadata`、`_EMAParamOptimizer`→`EMAParamOptimizer`、
  `_BackendCheckpointStorage`→`BackendCheckpointStorage` 等），并顺带把两处"去前缀会读歪"
  的改成更准确的名字：`hf/flops.py::_MoE`→`MoEGeometry`（避免与模型类的 `MoE` 混淆）、
  `checkpointer/torch_checkpointing.py::_Backend`→`BackendConfig`（避免与 torch 的
  `Backend` 枚举混淆）、`components/metrics.py::_Derived`→`DerivedMetrics`。
  同时 `scatter_add.py` 中 `@register_fake` 的匿名 `def _` 命名为
  `deterministic_scatter_add_fake`。**上游同名 helper 的拼写分叉是有意的**：symbol guide §10
  的映射行仍以"llmtuner 公开名 ↔ 上游 `_name`"记录（`docs/` 里历史条目保留旧名，读法见本条）。
  (c) **方法级：109 → 103 个前导 `_`**。13 个被"别的模块"调用的私有方法里，6 个 llmtuner 自有的
  公开（`Trainer.seed_everything`/`example_model`/`loss_vocab_kwargs`/`loss_sum`/`param_context`、
  `FeedForward.split_gate_up`），5 个上游同名 protected 钩子按规则③保留
  （`checkpointer/base.py` 的 `_should_save`/`_purge_stale_checkpoints`/`_create_checkpoint_id`、
  `optimizer.py` 的 `_post_init`/`_validate_params`）。
  (d) **两处踩坑留档**：① 集成测试里各自定义的*本地参考实现* `_loss_sum` 一度被脚本误改
  （会造成 `loss_sum = loss_sum(...)` 自遮蔽），已回退——测试内的私有模型不在审计范围；
  ② 批量改名会顺带打中*第三方私有路径*：`from torch.distributed._functional_collectives import ...`
  被改成 `functional_collectives`（3 个文件 + 1 个测试模块导入失败），已逐条还原为
  `_functional_collectives`。教训：改名脚本必须排除 `模块.名字` 形式。
  (e) 验证：脚本复查"跨模块使用却仍带 `_` 的模块级符号" = 0（仅剩 `capabilities.py` 里对
  `disable_dynamo_lru_cache` 的字符串引用与另一个模块的同名 `resolve` 两处误报）；
  `ruff check`（F821 覆盖全部改名后引用）、`git diff --check`、`compileall` 通过；
  垫片环境下 44/49 目标模块可导入（5 个失败全是本机 `grain` 缺 `experimental` 的既有环境问题）；
  `tests/unit_tests` = 248 passed / 56 skipped / 9 failed（失败集不变）。

- 检查后续漂移：`git -C <torchtitan> log f35966713..HEAD -- torchtitan/`。
- 2026-09-23 映射修订：上游 `distributed/linear.py` 已删除、内容迁入
  `models/common/dist_gemm.py`（改名 `AsyncAllGatherLinear`/`AsyncLinearReduceScatter`，
  数学不变；上游 9e159aed7 再把该文件改名 `async_linear.py`），此后上游
  async_linear.py 同时对应 llmtuner 的 `parallel/tensor_parallel/linear.py`
  （autograd 原语）与 `models/common/async_linear.py`（模块层，llmtuner 2026-09-26
  同步改名），一对二；
  `distributed/tensor_parallel.py` 已随 DTensor 后端整体删除（`7e7f271e0`）；继任者
  不是另一个同名文件，而是把 TP 变成声明的三处：`protocols/sharding.py`
  （`ShardingConfig`、`in/out_src/dst_shardings`、`local_spmd`）、
  `models/common/decoder_sharding.py`（core 模型的 `norm_config` /
  `token_id_placement` / `set_gqa_attention_sharding`）与
  `experiments/transformers_modeling_backend/hf_sharding.py`（HF 模型的
  `set_hf_sharding_configs` 及其 `_hf_colwise_config` / `_hf_rowwise_config` /
  `_hf_sequence_parallel_placement`）；算子侧是 `models/common/async_linear.py`。
  逐项对应见本文「TP/SP 对齐结论」。
- 早前基线：llmtuner `528dc9d`，TorchTitan `c6e416bbd`。
- 表中的 ratio 除 A2 中明确标为 2026-09-21 复核的十行外，来自早期结构快照
  （llmtuner `f5be809` 附近），只用于解释来源，**不是当前工作树的实时相似度**。源码
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
"""llmtuner -> torchtitan 结构相似度。输出: ratio \t ratio2 \t llmtuner路径 \t 上游路径 \t 代码长度"""
import ast, difflib
from pathlib import Path

HP = Path("<你的>/TorchLLMTuner/llmtuner")
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
