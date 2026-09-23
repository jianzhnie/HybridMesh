# hpmesh -> torchtitan 对应关系表

[hpmesh](../hpmesh) 拿掉了 TorchTitan 的 `Configurable` 与 `Module` 两个抽象层，换来一个明显更短
的框架：89 个 Python 模块、约 22.4k 行，覆盖 TP / FSDP2 / CP / EP / PP 五条并行路径的装配、
训练循环、checkpoint 与等价性测试。

下面是对应关系表：

## 怎么用这张表

**先查表,再动手。** 这张表是唯一权威 —— 不要在 hpmesh 源码里加
`# upstream: <path> @ <sha>` 之类的来源标注,那种标注试过又被退了,因为
sha 会腐烂而表不会。

**分类不是装饰,是操作指令。** 把 A 类的高保真同步规则套到 B 类文件上会毁掉
设计;套到 C 类上会把项目**故意删掉**的抽象又拽回来。

**比结构用 AST,不要比 diff 行数。** `utils/filesystem.py` 的 diff 有几十行,
代码差异是 **0** —— 全是改写措辞。做法是剥掉 docstring、`ast.unparse`、
再 `difflib.SequenceMatcher`。下表 `ratio` 列就是这么来的。

| hpmesh                                   | torchtitan                                              |
| ---------------------------------------- | ------------------------------------------------------- |
| `parallel/**`                            | `distributed/`                                          |
| `components/metrics.py`                  | `observability/metrics.py`                              |
| `models/common/scatter_add.py`           | `ops/scatter_add.py`                                    |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` |
| `datasets/text/text.py`                  | `hf_datasets/text_datasets.py`                          |
| `utils/filesystem.py`                    | `tools/filesystem.py`                                   |

***

## 图例

- **A 移植 (vendored)** —— 从 torchtitan 移植。上游改了,hpmesh **应该**逐项核对。
  子类是它们的真实差异。
- **B 适配 (adapted)** —— 同一个想法、不同的形状。上游改了,**读意图、不要抄形状**。
- **C hpmesh 独有** —— 上游没有对应物(或同名不同源)。不要"对齐"它。
- **D 缺口** —— 上游有,hpmesh **真的没有**。是要补还是不要补,是决策不是疏漏。

分类按文件的**主要维护策略**划分；一个文件只能有一个主分类。文件内部若混合了移植
与适配逻辑，在“改写点”中另行说明，避免同一文件同时收到互相冲突的操作指令。

`ratio` 主要是 AST 相似度的历史快照,**历史行是"全量最佳匹配"**(脚本第 1 列),仅对 A/B
类有意义。**1.000 不代表逐字相同** —— 剥掉
docstring 后 `ast.unparse` 归一化了空白和引号。要判断"真逐字",看 ratio 为
1.000 且人工确认过的那两个。少数行的"改写点"列会额外给出**同名比较**值 ——
两者差得远时,最佳匹配多半是噪音,以同名值为准(见血的教训 6)。

2026-09-21 复核后从 A1 移入 A2 的十行使用当前工作树的**指定对应文件直接比较**值，
不是全量最佳匹配；这些值用于解释为何维护策略已经变化，不与历史排名混用。

***

## A1 —— 高保真移植（改动需逐位验证）

只有 `components/checkpointer/utils.py` 和 `utils/filesystem.py` 在该快照中经人工确认
属于“去 docstring 后结构等价”；其余行即使 ratio 很高也不是逐字复制。

| hpmesh                                 | torchtitan                              | ratio | <br />                                                          |
| -------------------------------------- | --------------------------------------- | ----- | :-------------------------------------------------------------- |
| `components/checkpointer/utils.py`     | `components/checkpointer/utils.py`      | 1.000 | <br />                                                          |
| `utils/filesystem.py`                  | `tools/filesystem.py`                   | 1.000 | <br />                                                          |
| `components/optimizer/utils.py`        | `components/optimizer/utils.py`         | 0.996 | <br />                                                          |
| `datasets/multimodal/mm_image.py`      | `hf_datasets/multimodal/utils/image.py` | 0.977 | <br />                                                          |
| `datasets/multimodal/mm_text_utils.py` | `hf_datasets/multimodal/utils/text.py`  | 0.971 | <br />                                                          |
| `datasets/multimodal/mm_video.py`      | `hf_datasets/multimodal/utils/video.py` | 0.967 | <br />                                                          |
| `components/tokenizer.py`              | `components/tokenizer.py`               | 0.907 | <br />                                                          |

<br />

## A2 —— 移植但已改写（比例中等，需逐处核对）

| hpmesh                                           | torchtitan                                              | ratio | 改写点                                                                                           |
| ------------------------------------------------ | ------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------- |
| `components/checkpointer/__init__.py`            | `components/checkpointer/__init__.py`                   | 0.757 | <br />                                                                                        |
| `components/checkpointer/base.py`                | `components/checkpointer/base.py`                       | 0.717 | <br />                                                                                        |
| `parallel/parallel_dims.py`                      | `distributed/parallel_dims.py`                          | 0.772 | hpmesh 扩展 world/loss/sparse mesh 视图，不能按旧 A1 结构覆盖                                           |
| `datasets/multimodal/mm_collator.py`             | `hf_datasets/multimodal/mm_collator.py`                 | 0.777 | 增加 MRoPE grid/run/长度校验，当前已是契约适配                                                          |
| `models/common/multimodal.py`                    | `models/common/multimodal.py`                           | 0.888 | 保留算法来源，但加入同步规避与更严格的 span/run 校验                                                      |
| `datasets/types.py`                              | `components/data/types.py`                              | 0.506 | 去 Configurable 后重塑 build context 与 iteration policy                                             |
| `parallel/tensor_parallel/linear.py`             | `models/common/dist_gemm.py`（原 `distributed/linear.py`，上游 e72fd863d 搬迁并改名 `Async*`，数学不变） | 0.511 | 保留 fused/fallback 数学意图，但运行时上下文和 autograd 形状已适配 hpmesh                                  |
| `components/checkpointer/dcp.py`                 | `components/checkpointer/dcp.py`                        | 0.735 | 本地/remote storage、HF export 与生命周期已重塑                                                       |
| `datasets/text/text.py`                          | `hf_datasets/text_datasets.py`                          | 0.767 | 路径与 processor 构造契约已适配                                                                       |
| `parallel/fully_shard/fsdp.py`                   | `distributed/fsdp.py`                                   | 0.815 | 多轴 mesh 重建、HF decoder 与 MoE placement 是 hpmesh 适配                                            |
| `models/common/rope.py`                          | `models/common/rope.py`                                 | 0.616 | 上游持续重构后结构已分叉；同步公式与边界修复，不同步 Module/缓存形状                                            |
| `datasets/packing.py`                            | `components/data/packing.py`                            | 0.065 | 自由函数外还增加文档容量、padding mask、长文档切分和可恢复 remainder，按语义维护                              |
| `models/common/scatter_add.py`                   | `ops/scatter_add.py`                                    | 0.711 | <br />                                                                                        |
| `models/common/param_init.py`                    | `models/common/param_init.py`                           | 0.700 | <br />                                                                                        |
| `datasets/sources.py`                            | `components/data/sources.py`                            | 0.700 | <br />                                                                                        |
| `parallel/pipeline_parallel/pipeline.py`         | `experiments/transformers_modeling_backend/pipeline.py` | 0.686 | `None` -> `nn.Identity`;每 stage 追加 `rotary_emb`                                               |
| `models/common/aux_loss.py`                      | `models/common/aux_loss.py`                             | 0.682 | <br />                                                                                        |
| `models/common/linear.py`                        | `models/common/linear.py`                               | 0.620 | <br />                                                                                        |
| `datasets/loader.py`                             | `components/data/loader.py`                             | 0.590 | 去 `Configurable`;`GrainDataLoader` 直接收参数,无 config 类                                           |
| `models/common/feed_forward.py`                  | `models/common/feed_forward.py`                         | 0.560 | **曾写完又被退**,不要在没有明确指令时重新引入                                                                     |
| `models/common/dist_gemm.py`                     | `models/common/dist_gemm.py`                            | 0.527 | <br />                                                                                        |
| `models/common/token_dispatcher.py`              | `models/common/token_dispatcher.py`                     | 0.441 | <br />                                                                                        |
| `components/profiler.py`                         | `observability/profiler.py`                             | 0.413 | <br />                                                                                        |
| `components/metrics.py`                          | `observability/metrics.py`                              | 0.381 | <br />                                                                                        |
| `models/common/masks.py`                         | `models/common/attention.py`                            | 0.380 | 拆出了 mask 部分                                                                                   |
| `components/optimizer/lr_scheduler.py`           | `components/optimizer/lr_scheduler.py`                  | 0.373 | 去 `Configurable`                                                                              |
| `datasets/multimodal/mm_datasets.py`             | `hf_datasets/multimodal/mm_datasets.py`                 | 0.310 | 去 `Configurable`;packing 改自由函数 `build_mm_sample_packing`                                      |
| `components/loss.py`                             | `components/loss.py`                                    | 0.297 | 去 loss 类层次，保留自由函数与 vocab-parallel 数学                                                           |
| `datasets/dataset.py`                            | `components/data/dataset.py`                            | 0.267 | 去 `Configurable`;三个节点类去 `Config` 后缀,构建走自由函数 `build_dataset`                                   |
| `components/checkpointer/torch_checkpointing.py` | `components/checkpointer/torch_checkpointing.py`        | 0.265 | <br />                                                                                        |
| `models/common/qkv.py`                           | `models/common/attention.py`                            | 0.242 | <br />                                                                                        |
| `components/optimizer/optimizer.py`              | `components/optimizer/optimizer.py`                     | 0.240 | 容器化改写;`OptimizerWrapper` 已删                                                                   |
| `models/common/moe.py`                           | `models/common/moe.py`                                  | 0.155 | <br />                                                                                        |
| `parallel/activation_checkpoint.py`              | `distributed/activation_checkpoint.py`                  | 0.374 | **FullAC + SelectiveAC 已移植**;RegionAC(需 `torch_remat`)、MemoryBudgetAC(需编译)未移植,理由见文件 docstring |
| `datasets/collators.py`                          | `components/data/collators.py`                          | 0.145 | <br />                                                                                        |

## B —— 适配层(读意图,不要抄形状)

这几个是 **torchtitan 每个模型一个文件** 的那种东西的**替代品**。照搬它们的形状
会破坏分片契约。

| hpmesh                                         | 替代掉的上游                                                                                                 | ratio       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------- |
| `models/hf_wrapper.py`                         | `experiments/transformers_modeling_backend/model.py` 的包装层;上游另有 `models/*/model.py` 各一份                 | 0.059       |
| `parallel/parallelize_hf.py`                   | `experiments/transformers_modeling_backend/parallelize.py` + 各 `models/*/parallelize.py`               | 0.089       |
| `parallel/tensor_parallel/tp.py`               | 各模型 TP plan；上游 `distributed/tensor_parallel.py` 已随 DTensor 后端删除、无后继文件。hpmesh 是**手写 plan realizer**，不是声明式 `_sharding_config` | 0.056       |
| `parallel/expert_parallel/apply.py` + `ep.py`  | `experiments/.../moe_replacement.py` + 各模型 EP parallelize；hpmesh 搬运 HF 权重而非重新初始化                 | 0.036-0.146 |
| `parallel/context_parallel/primitives.py`      | `models/common/cp_attention.py`；剥掉 attention 基类，只保留 redistribution                                   | 0.060       |
| `parallel/fully_shard/fsdp_wrap.py`            | 各 `models/*/parallelize.py` 的 FSDP driver；HF 五部件适配                                                     | 0.155       |
| `parallel/pipeline_parallel/pp.py`             | `distributed/pipeline_parallel.py`；hpmesh 直接消费 HF stage 部件                                             | 0.130       |
| `trainer/trainer.py`                           | `trainer.py`,基本重写                                                                                      | 0.065       |
| `trainer/config.py`                            | `config/configs.py`                                                                                    | 0.189       |
| `trainer/train.py`                             | `train.py`                                                                                             | 0.186       |
| `mesh.py`                                      | `distributed/parallel_dims.py` + `trainer.py` 中分散的 mesh 逻辑                                        | 0.111       |
| `models/common/grouped_experts.py`             | `models/common/grouped_experts.py` + `models/gpt_oss/moe.py`                                           | 0.119       |
| `utils/gc.py`                                  | `tools/utils.py` 的 GC helper，去 structured logger                                                     | 0.211       |

**注意** **`mesh.py`**:上游没有单一对应物 —— mesh 逻辑散在
`distributed/parallel_dims.py` 和 `trainer.py` 里,不是某个文件的移植。

***

## C —— hpmesh 独有(不要对齐上游)

| hpmesh                                     | 说明                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `utils/spmd_context.py`                    | `spmd_types` pip 包的**独立活跃适配层**，由 trainer 和 `models/common/*` 使用；不依赖 `spmd_shims.py`                                                                                                                                                                                                                  |
| `parallel/collectives.py`                  | 最优相似度 0.089,是独立实现                                                                                                                                                                                                                                                                                      |
| `parallel/context_parallel/apply.py`       | 0.058;CP 的编排层,上游无对应文件                                                                                                                                                                                                                                                                                  |
| `parallel/context_parallel/cp_kernel.py`   | 0.051;hpmesh 独有的 CP flex kernel                                                                                                                                                                                                                                                                        |
| `parallel/context_parallel/input_shard.py` | 0.078                                                                                                                                                                                                                                                                                                  |
| `utils/logger_utils.py`                    | 0.070,上游无对应                                                                                                                                                                                                                                                                                            |
| `utils/monitoring.py`                      | 与 `tools/utils.py` 0.107,独立实现(含 `get_peak_flops`)                                                                                                                                                                                                                                                      |
| `utils/checkpoint_keys.py`                 | 上游无                                                                                                                                                                                                                                                                                                    |
| `utils/device.py`                          | 上游无(0.382 是噪音,命中实验目录)                                                                                                                                                                                                                                                                                  |
| `utils/batch_invariant.py`                 | 上游无;上游把 batch-invariant 开关放在 `trainer.py`/`config/configs.py` 里,没有独立模块                                                                                                                                                                                                                                 |
| `models/common/flex_kernel.py`             | **上游没有这个文件**(上一版说"上游是 torchtitan `Module` 子类"是错的)。hpmesh 无 Module 协议,自持 `_sharding_config`                                                                                                                                                                                                             |
| `models/common/nn_modules.py`              | 上游有同路径文件,但 ratio **0.061** —— 上游那份是 `nn.X` + `Module` 的菱形继承包装,hpmesh 只留了 PP 占位所需的部分,不是移植(注:0.263 是它对 `param_init.py` 的最佳匹配,是噪音,不是同名比)                                                                                                                                                                  |
| `models/common/activation.py`              | 与上游同名但不同源；公式由 hpmesh 自持，不能按 A 类覆盖                                                                                                                                                                                                                                                            |
| `models/common/embedding.py`               | 与上游同名但不同源；包含 hpmesh 的 vocab-shard 契约                                                                                                                                                                                                                                                               |
| `datasets/random_data.py`                  | 合成语料,上游无                                                                                                                                                                                                                                                                                               |
| `datasets/build.py`                        | 工厂;上游把 `build()` 放在 config 上                                                                                                                                                                                                                                                                           |

**已清理悬空链**：`parallel/sharding.py` 与 `parallel/spmd_shims.py` 没有运行时
消费者，已在 2026-09-21 一并删除。`utils/spmd_context.py` 是独立活代码，不在删除组
内。`HFFlexKernel._sharding_config` 仍是无人读取的兼容字段；TP 的活跃实现继续是
`parallel/tensor_parallel/tp.py` 的 plan 引擎。

***

## D —— 真正缺失

| 上游                                                                                | 影响                                                                                       |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `distributed/compile.py`                                                          | **被裁剪成整体 `torch.compile(model)`**（PP 则每 chunk 一次）。裁掉的是四件互相独立的事：逐 block 编译、async TP `_micro_pipeline_tp`、`regional_inductor`、`capture_scalar_outputs`（后者是 token-choice MoE dispatch 的动态 shape 所需的） |
| `models/common/moe_sharding.py`                                                   | **比“缺一个文件”更深**。旧的未接线 `parallel/sharding.py` 形式已删除；hpmesh 没有 MoE 的 TP 声明或读取声明的运行引擎。它真正的载荷是 **MoE-under-TP**（routed 专家在 TP 轴分片、router 保持 Replicate），而 hpmesh 的 TP 对 `moe_tp_experts` 明确 raise。所以这是 **TP×MoE 组合维度整体没有**，不是漏文件。 |
| `components/quantization/`, `structured_logger/`, `protocols/`, `configurable.py` | **故意删除**,不是缺口。不要"补回来"                                                                    |
| `components/optimizer/ema.py`（2026-09 新增，515 行）                                    | 在线 EMA 模型平均；需 config/trainer/checkpointer 三侧接线，hpmesh 无任何消费者                                |
| quantile-balanced MoE routing（f8bb599a7，kimi_k3 在用）                               | `QuantileBalancedTopKRouter` + optimizer hook，跨 moe.py 与 optimizer.py                      |
| MoE padding-mask 负载均衡（d34a13fdf）                                                  | routing 统计与 aux loss 屏蔽 padding token；hpmesh `MoE.forward` 无 padding_mask 通道，接线需改 EP swap 后调用链 |
| `CastLinear`（150c4f73a 配套）                                                         | lm_head compute-dtype 变换；hpmesh 不带 `Linear` 类体系                                            |
| Ulysses CP × varlen/packed（baff3c681）                                              | redistribution 原语 hpmesh 已有，缺 varlen 内层 attention 路径；`apply_cp` 对该组合保持 fail-fast          |
| 多轮对话 SFT 的 renderer 路径（4a0d8dab3）                                                 | 依赖 `renderers==0.1.11` 与上游 `components/renderer.py`（Configurable 系）                    |

**已从 D 移除**:`distributed/activation_checkpoint.py` 的 `SelectiveAC` —— 于
2026-09-21 移植(见 A2)。至此该文件只剩两处未移植,都是环境依赖而非删减:
`RegionAC` 需要 `torch_remat`(hpmesh 不依赖,且它的"模型声明 region"建立在
hpmesh 没有的 `Module` 协议上)、`MemoryBudgetAC` 只在模型被 compile 后才有意义
(它本身没有策略代码,只是设两个 `torch._functorch.config` 全局量)。两者连同
`_disable_dynamo_lru_cache`(修的是 SAC+PP 的重编译交互,而 hpmesh 在 `pp > 1`
上直接拒绝 AC,够不到那个场景)都在文件 docstring 里写明了。

**已从 D 移除**:`tools/validate.py` —— 上一版既写了它、又写"上游也没有这个路径,
已从表里移除",自相矛盾。核实:上游 `torchtitan/tools/validate.py` **确实不存在**,
这一行没有意义,删掉。

***

## E —— 包面(`__init__.py` 与入口)

重组出口,不是移植内容:这些文件定义 hpmesh 的**公开 API 面**,上游对应物是
同名 `__init__.py`(若存在)。改上游的导出列表时才需要看这里。

| hpmesh                                   | 行数 | torchtitan                                 |
| ---------------------------------------- | -- | ------------------------------------------ |
| `__init__.py`                            | 30 | `__init__.py`                              |
| `__main__.py`                            | 6  | `train.py` 的入口对等物                          |
| `models/common/__init__.py`              | 61 | `models/common/__init__.py`                |
| `datasets/__init__.py`                   | 64 | `components/data/__init__.py`              |
| `parallel/__init__.py`                   | 35 | `distributed/__init__.py`                  |
| `parallel/context_parallel/__init__.py`  | 23 | `distributed/context_parallel/__init__.py` |
| `parallel/expert_parallel/__init__.py`   | 20 | 上游无对应(见 C 类)                               |
| `parallel/pipeline_parallel/__init__.py` | 12 | 上游无对应                                      |
| `trainer/__init__.py`                    | 26 | 上游无对应                                      |

`__init__.py` 的 ratio 平均偏低(0.3 上下)是正常的 —— 它们导出的是各自的公开面,
不是从上游抄结构。表里行的 value 是"文件行数",不是 ratio。

**空文件**(0 行,不必查表):`components/__init__.py`、`models/__init__.py`、
`parallel/fully_shard/__init__.py`、`parallel/tensor_parallel/__init__.py`、
`utils/__init__.py`。

**两个子包**:`datasets/` 下按语料分 `text/` 和 `multimodal/`,其余 8 个模块平铺在
`datasets/` 根下。上游的 `hf_datasets/` 是 `components/data/` 的兄弟目录,hpmesh
曾用 `datasets/hf/` 镜像它(4 层,三个 0 字节的 `__init__.py`),先被溶解成单层,
再于本次重组按语料分成两个子包。

划分依据是**内容**而非上游路径:`datasets/` 根下的模块与上游 `components/data/`
一一对应,且都被两边共用(`loader.py` 默认 `TextCollator`、
`multimodal/mm_collator.py` 复用 `collators.py` 的 `Collator`/`TrainerBatch`),
所以不进任何一边;只有 `text.py` 和 5 个 `mm_*.py` 是语料专属。两个子包的
`__init__.py` 都**刻意不导入子模块**——惰性导入契约靠这一点维持。

**上游路径对应关系因此不再一一成立**(`hf_datasets/multimodal/utils/image.py`
在 hpmesh 侧是 `datasets/multimodal/mm_image.py`),本表的 hpmesh 列是唯一权威。

## 版本与漂移

- 本文最近一次人工审计工作树：hpmesh `5749d19`（+本轮改动），TorchTitan `b64103072`；
  详细验证记录见
  [`hpmesh_torchtitan_alignment_audit_2026-09-23.md`](./hpmesh_torchtitan_alignment_audit_2026-09-23.md)。
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
  （hpmesh `58eb279` 附近），只用于解释来源，**不是当前工作树的实时相似度**。
  源码变化后应运行下方脚本重算，不能据旧 ratio 判定漂移。
- 2026-09-22 设备验证补充：Qwen3-8B 已按 TorchTitan 的 meta 构建 → FSDP →
  `to_empty` → checkpoint load 顺序完成 8 卡 HCCL、4096 序列的真实训练，并完成完整
  DCP save→resume。训练示例显式使用 `last_save_model_only=False`；上游默认的
  model-only 最终 checkpoint 只适合作为导出物，不能作为续训状态。
- 同日并行复核修正了 EP 不应计入 world-size 乘积的 config helper，以及 Torch 2.10
  functional-collective 的 TP fallback API。2-rank FSDP/TP/CP/EP-grad-norm 等价性通过；
  PP 1F1B 的多步轨迹仍有约 `8.5e-3` 最大偏差，保持未通过状态。

***

## 重现这张表

保存为 `recompute_map.py`,把两个根目录指向你自己的 checkout,然后
`python recompute_map.py | sort -t$'\t' -k1 -rn`。它是产出本表的**实际脚本**
(不是伪码),两分钟内跑完。

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
    if len(a) < 40:            # 空 __init__.py 等,跳过(见血的教训 2)
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

**必须先把 torchtitan 的候选集限定在** **`torchtitan/torchtitan/`** **下**,否则会匹配到
`experiments/rl/` 之类的噪音。

**性能提示(历史实测)**:不要用 `autojunk=False`,91 x 444 会跑十几分钟。
默认 `autojunk=True` 加那条上界剪枝就能在两分钟内跑完。

**不要用** **`diff`** **行数判断改动量**,`quick_ratio()` 也**不能**用来筛候选 ——
它是上界不是估计,会漏掉真正的对应物(实测漏过 `components/loss.py`)。

**第 2 列是次优匹配**。它比最优值还重要:两者**接近**说明这个文件的归属有歧义,
需要人工判(见血的教训 3、6);两者差距大,最优才是可信的。
