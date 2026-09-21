# hpmesh -> torchtitan 对应关系表

基线: **hpmesh** **`58eb279`** / **torchtitan** **`1c7ab8089`**(2026-09-21 全表重算,
torchtitan 工作区在该 sha 上无未提交改动)。表内所有 `ratio` 都是本次重算的
值。重算的结果是**大部分行复现了原值**,但**若干行改判、几条事实性说法是错的**,
另有已加的模块未进表(见文末"本次重算的改动")。

**例外**:`parallel/activation_checkpoint.py` 在 2026-09-21 当天补完 `SelectiveAC`
(原基线 0.149 -> 现在 0.374),`test_trainer.py` 那次 `trainer/config.py` 改动也没进
基线 sha,所以这两行(连同连带微动的 `parallelize_hf.py`)的 ratio 是对**当前
工作区**测的,不是对 `58eb279`。

回答的问题是:*torchtitan 改了文件 X,hpmesh 哪些文件必须跟着改?* 以及反过来
*这个 hpmesh 文件的上游是谁*。

***

## 怎么用这张表

**先查表,再动手。** 这张表是唯一权威 —— 不要在 hpmesh 源码里加
`# upstream: <path> @ <sha>` 之类的来源标注,那种标注试过又被退了,因为
sha 会腐烂而表不会。

**分类不是装饰,是操作指令。** 把 A 类的"逐字复制"规则套到 B 类文件上会毁掉
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

- **A 移植 (vendored)** —— 从 torchtitan 复制。上游改了,hpmesh **应该**跟着改。
  子类是它们的真实差异。
- **B 适配 (adapted)** —— 同一个想法、不同的形状。上游改了,**读意图、不要抄形状**。
- **C hpmesh 独有** —— 上游没有对应物(或同名不同源)。不要"对齐"它。
- **D 缺口** —— 上游有,hpmesh **真的没有**。是要补还是不要补,是决策不是疏漏。

`ratio` 是 AST 相似度,**表里的一律是"全量最佳匹配"**(脚本第 1 列),仅对 A/B
类有意义。**1.000 不代表逐字相同** —— 剥掉
docstring 后 `ast.unparse` 归一化了空白和引号。要判断"真逐字",看 ratio 为
1.000 且人工确认过的那两个。少数行的"改写点"列会额外给出**同名比较**值 ——
两者差得远时,最佳匹配多半是噪音,以同名值为准(见血的教训 6)。

***

## A1 —— 逐字复制(改动需逐位验证)

| hpmesh                                 | torchtitan                              | ratio | <br />                                                          |
| -------------------------------------- | --------------------------------------- | ----- | :-------------------------------------------------------------- |
| `components/checkpointer/utils.py`     | `components/checkpointer/utils.py`      | 1.000 | <br />                                                          |
| `utils/filesystem.py`                  | `tools/filesystem.py`                   | 1.000 | <br />                                                          |
| `components/optimizer/utils.py`        | `components/optimizer/utils.py`         | 0.996 | <br />                                                          |
| `parallel/parallel_dims.py`            | `distributed/parallel_dims.py`          | 0.988 | <br />                                                          |
| `datasets/multimodal/mm_image.py`      | `hf_datasets/multimodal/utils/image.py` | 0.977 | <br />                                                          |
| `datasets/multimodal/mm_text_utils.py` | `hf_datasets/multimodal/utils/text.py`  | 0.971 | <br />                                                          |
| `datasets/multimodal/mm_video.py`      | `hf_datasets/multimodal/utils/video.py` | 0.967 | <br />                                                          |
| `datasets/multimodal/mm_collator.py`   | `hf_datasets/multimodal/mm_collator.py` | 0.957 | <br />                                                          |
| `models/common/multimodal.py`          | `models/common/multimodal.py`           | 0.945 | <br />                                                          |
| `components/tokenizer.py`              | `components/tokenizer.py`               | 0.907 | <br />                                                          |
| `datasets/types.py`                    | `components/data/types.py`              | 0.897 | <br />                                                          |
| `parallel/tensor_parallel/linear.py`   | `distributed/linear.py`                 | 0.893 | <br />                                                          |
| `components/checkpointer/dcp.py`       | `components/checkpointer/dcp.py`        | 0.862 | <br />                                                          |
| `datasets/text/text.py`                | `hf_datasets/text_datasets.py`          | 0.852 | <br />                                                          |
| `parallel/fully_shard/fsdp.py`         | `distributed/fsdp.py`                   | 0.846 | <br />                                                          |
| `models/common/rope.py`                | `models/common/rope.py`                 | 0.823 | <br />                                                          |
| `datasets/packing.py`                  | `components/data/packing.py`            | 0.820 | 配方做自由函数;选择器落在 `DataloaderConfig.packing`,上游落在模型 config registry |

<br />

A2 —— 移植但已改写(比例中等,需逐处核对)

| hpmesh                                           | torchtitan                                              | ratio | 改写点                                                                                           |
| ------------------------------------------------ | ------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------- |
| `components/checkpointer/__init__.py`            | `components/checkpointer/__init__.py`                   | 0.757 | <br />                                                                                        |
| `components/checkpointer/base.py`                | `components/checkpointer/base.py`                       | 0.717 | <br />                                                                                        |
| `models/common/scatter_add.py`                   | `ops/scatter_add.py`                                    | 0.711 | <br />                                                                                        |
| `models/common/param_init.py`                    | `models/common/param_init.py`                           | 0.700 | <br />                                                                                        |
| `datasets/sources.py`                            | `components/data/sources.py`                            | 0.700 | <br />                                                                                        |
| `parallel/pipeline_parallel/pipeline.py`         | `experiments/transformers_modeling_backend/pipeline.py` | 0.686 | `None` -> `nn.Identity`;每 stage 追加 `rotary_emb`                                               |
| `models/common/aux_loss.py`                      | `models/common/aux_loss.py`                             | 0.682 | <br />                                                                                        |
| `models/common/linear.py`                        | `models/common/linear.py`                               | 0.620 | <br />                                                                                        |
| `datasets/loader.py`                             | `components/data/loader.py`                             | 0.590 | 去 `Configurable`;`GrainDataLoader` 直接收参数,无 config 类                                           |
| `models/common/feed_forward.py`                  | `models/common/feed_forward.py`                         | 0.560 | **曾写完又被退**,不要在没有明确指令时重新引入                                                                     |
| `parallel/spmd_shims.py`                         | `distributed/spmd_types.py`                             | 0.532 | **见 C 类说明**                                                                                   |
| `models/common/dist_gemm.py`                     | `models/common/dist_gemm.py`                            | 0.527 | <br />                                                                                        |
| `models/common/token_dispatcher.py`              | `models/common/token_dispatcher.py`                     | 0.441 | <br />                                                                                        |
| `components/profiler.py`                         | `observability/profiler.py`                             | 0.413 | <br />                                                                                        |
| `components/metrics.py`                          | `observability/metrics.py`                              | 0.381 | <br />                                                                                        |
| `models/common/masks.py`                         | `models/common/attention.py`                            | 0.380 | 拆出了 mask 部分                                                                                   |
| `components/optimizer/lr_scheduler.py`           | `components/optimizer/lr_scheduler.py`                  | 0.373 | 去 `Configurable`                                                                              |
| `utils/spmd_context.py`                          | `distributed/spmd_types.py`                             | 0.350 | 同上                                                                                            |
| `parallel/sharding.py`                           | `protocols/sharding.py`                                 | 0.340 | 换掉了 Module 协议                                                                                 |
| `datasets/multimodal/mm_datasets.py`             | `hf_datasets/multimodal/mm_datasets.py`                 | 0.310 | 去 `Configurable`;packing 改自由函数 `build_mm_sample_packing`                                      |
| `components/loss.py`                             | `components/loss.py`                                    | 0.297 | 见上,docstring 关于 upstream 的说法是错的(已于 2026-09-21 修正)                                             |
| `datasets/dataset.py`                            | `components/data/dataset.py`                            | 0.267 | 去 `Configurable`;三个节点类去 `Config` 后缀,构建走自由函数 `build_dataset`                                   |
| `components/checkpointer/torch_checkpointing.py` | `components/checkpointer/torch_checkpointing.py`        | 0.265 | <br />                                                                                        |
| `models/common/qkv.py`                           | `models/common/attention.py`                            | 0.242 | <br />                                                                                        |
| `components/optimizer/optimizer.py`              | `components/optimizer/optimizer.py`                     | 0.240 | 容器化改写;`OptimizerWrapper` 已删                                                                   |
| `trainer/config.py`                              | `config/configs.py`                                     | 0.189 | <br />                                                                                        |
| `trainer/train.py`                               | `train.py`                                              | 0.186 | <br />                                                                                        |
| `models/common/activation.py`                    | (同名,上游 `models/common/activation.py`)                   | 0.168 | **同名不同源**:最佳匹配是 `param_init.py`(噪音),同名仅 0.124,各自写的                                            |
| `parallel/fully_shard/fsdp_wrap.py`              | `models/deepseek_v3/parallelize.py`                     | 0.155 | **B 类典型**:把 HF 模型套到 Decoder 形状上                                                               |
| `models/common/moe.py`                           | `models/common/moe.py`                                  | 0.155 | <br />                                                                                        |
| `parallel/activation_checkpoint.py`              | `distributed/activation_checkpoint.py`                  | 0.374 | **FullAC + SelectiveAC 已移植**;RegionAC(需 `torch_remat`)、MemoryBudgetAC(需编译)未移植,理由见文件 docstring |
| `datasets/collators.py`                          | `components/data/collators.py`                          | 0.145 | <br />                                                                                        |
| `models/common/embedding.py`                     | (同名,上游 `models/common/embedding.py`)                    | 0.141 | 同 `activation.py`,同名仅 0.138,同名不同源                                                             |
| `parallel/pipeline_parallel/pp.py`               | `distributed/pipeline_parallel.py`                      | 0.130 | <br />                                                                                        |
| `models/common/grouped_experts.py`               | `models/gpt_oss/moe.py`                                 | 0.119 | 上游的 `GroupedExperts` 在 `models/common/` 与 `gpt_oss` 各有一份                                      |
| `mesh.py`                                        | (散在 `distributed/parallel_dims.py` + `trainer.py`)      | 0.111 | 上游无单一对应物,见 B 类                                                                                |
| `utils/monitoring.py`                            | `tools/utils.py`                                        | 0.107 | 独立实现(含 `get_peak_flops`);名字毫不相干,是 AST 匹配找出来的                                                  |

## B —— 适配层(读意图,不要抄形状)

这几个是 **torchtitan 每个模型一个文件** 的那种东西的**替代品**。照搬它们的形状
会破坏分片契约。

| hpmesh                                         | 替代掉的上游                                                                                                 | ratio       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------- |
| `models/hf_wrapper.py`                         | `experiments/transformers_modeling_backend/model.py` 的包装层;上游另有 `models/*/model.py` 各一份                 | 0.059       |
| `parallel/parallelize_hf.py`                   | `experiments/transformers_modeling_backend/parallelize.py` + 各 `models/*/parallelize.py`               | 0.089       |
| `parallel/tensor_parallel/tp.py` + `linear.py` | `distributed/tensor_parallel.py` 等;hpmesh 是**手写 plan + 融合 GEMM**,不是声明式 `_sharding_config`              | 0.056       |
| `parallel/expert_parallel/*`                   | `experiments/.../moe_replacement.py`;**不共享代码**,是另一套实现                                                  | 0.036-0.146 |
| `parallel/context_parallel/*`                  | `distributed/context_parallel/`(`api.py`);但 `primitives.py` 的上游是 `models/common/cp_attention.py`,见 C 类 | 0.024-0.078 |
| `parallel/pipeline_parallel/pipeline.py`       | `experiments/.../pipeline.py`;差异是 `None` -> `nn.Identity`、每 stage 追加 `rotary_emb`                      | 0.686       |
| `trainer/trainer.py`                           | `trainer.py`,基本重写                                                                                      | 0.065       |
| `trainer/config.py`                            | `config/configs.py`                                                                                    | 0.189       |
| `trainer/train.py`                             | `train.py`                                                                                             | 0.186       |

**注意** **`mesh.py`**:上游没有单一对应物 —— mesh 逻辑散在
`distributed/parallel_dims.py` 和 `trainer.py` 里,不是某个文件的移植(见 C 类)。

***

## C —— hpmesh 独有(不要对齐上游)

| hpmesh                                     | 说明                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `parallel/spmd_shims.py`                   | 0.532 —— **是** **`spmd_types`** **pip 包的薄壳,不是 torchtitan 的**。`spmd_types` 装在 site-packages。不要把它"对齐"到 `distributed/spmd_types.py`                                                                                                                                                                       |
| `utils/spmd_context.py`                    | 同上,配合 `spmd_shims` 的上下文管理                                                                                                                                                                                                                                                                              |
| `parallel/collectives.py`                  | 最优相似度 0.089,是独立实现                                                                                                                                                                                                                                                                                      |
| `parallel/context_parallel/apply.py`       | 0.058;CP 的编排层,上游无对应文件                                                                                                                                                                                                                                                                                  |
| `parallel/context_parallel/cp_kernel.py`   | 0.051;hpmesh 独有的 CP flex kernel                                                                                                                                                                                                                                                                        |
| `parallel/context_parallel/input_shard.py` | 0.078                                                                                                                                                                                                                                                                                                  |
| `parallel/context_parallel/primitives.py`  | 0.060 —— **有真实上游**:`models/common/cp_attention.py`。上游那里是 `KVAllGatherCPFlexInnerAttention` / `UlyssesCPFlexInnerAttention` 两个 `(CPInnerAttention, FlexInnerAttention)` 子类,hpmesh 没有 `FlexInnerAttention`,所以类层次被剥掉、只剩两个 redistribution 本身。**低 ratio 在这里不代表无源** —— 剥掉基类后 AST 结构必然对不上,这条是"改了形状的移植",不是"独有" |
| `parallel/expert_parallel/apply.py`        | 0.146                                                                                                                                                                                                                                                                                                  |
| `parallel/expert_parallel/ep.py`           | 0.036;HF block -> 原生 MoE 的**权重搬运**,上游 `moe_replacement.py` 是重新初始化,不是同一件事                                                                                                                                                                                                                               |
| `mesh.py`                                  | 0.111;上游 mesh 逻辑散在 `distributed/parallel_dims.py` + `trainer.py`                                                                                                                                                                                                                                       |
| `utils/logger_utils.py`                    | 0.070,上游无对应                                                                                                                                                                                                                                                                                            |
| `utils/monitoring.py`                      | 与 `tools/utils.py` 0.107,独立实现(含 `get_peak_flops`)                                                                                                                                                                                                                                                      |
| `utils/checkpoint_keys.py`                 | 上游无                                                                                                                                                                                                                                                                                                    |
| `utils/gc.py`                              | 与 `tools/utils.py` 0.211                                                                                                                                                                                                                                                                               |
| `utils/device.py`                          | 上游无(0.382 是噪音,命中实验目录)                                                                                                                                                                                                                                                                                  |
| `utils/batch_invariant.py`                 | 上游无;上游把 batch-invariant 开关放在 `trainer.py`/`config/configs.py` 里,没有独立模块                                                                                                                                                                                                                                 |
| `models/common/flex_kernel.py`             | **上游没有这个文件**(上一版说"上游是 torchtitan `Module` 子类"是错的)。hpmesh 无 Module 协议,自持 `_sharding_config`                                                                                                                                                                                                             |
| `models/common/nn_modules.py`              | 上游有同路径文件,但 ratio **0.061** —— 上游那份是 `nn.X` + `Module` 的菱形继承包装,hpmesh 只留了 PP 占位所需的部分,不是移植(注:0.263 是它对 `param_init.py` 的最佳匹配,是噪音,不是同名比)                                                                                                                                                                  |
| `datasets/random_data.py`                  | 合成语料,上游无                                                                                                                                                                                                                                                                                               |
| `datasets/build.py`                        | 工厂;上游把 `build()` 放在 config 上                                                                                                                                                                                                                                                                           |

**已知悬空**:`parallel/sharding.py` 的 `ShardingConfig` 与
`models/common/flex_kernel.py` 的 `HFFlexKernel._sharding_config` 目前
**没有任何读取者** —— 唯一会读它们的 `set_hf_sharding_configs` 已随
`hpmesh/parallel/hf_sharding.py` 在 `cc6e217` 被删除(`moe_swap.py`、
`placements.py`、`spec.py`、`moe_probe.py` 同批)。这几个符号留着是给将来的
分片层用的,不是活代码。TP 现在的实现是 `parallel/tensor_parallel/tp.py` 的
plan 引擎,与它们无关。

悬空是**成链**的,不止那几个符号:`parallel/sharding.py`(去 docstring 后 2902 字符,文件本身 6865 字节)唯一的
导入者是 `parallel/spmd_shims.py`(0.532,是 `spmd_types` pip 包的薄壳),
而 **`spmd_shims.py` 自己没有任何导入者** —— 已实测:`import hpmesh`、
`import hpmesh.parallel`、`import hpmesh.parallel.expert_parallel` +
`hpmesh.parallel.tensor_parallel.tp` 之后,`sys.modules` 里两个模块都不存在。
`spmd_shims.py` 的 docstring 说 `resolve_placements` 是 "its only real
consumer",这句现在是反的:它自己才是唯一的消费者链,而那条链的顶端没人接。
所以"要么接线,要么删"面对的其实是**这两个文件一起**(`sharding.py` 单独留着
没有意义)。注:`spmd_context.py` 是**活的**(trainer、`models/common/*` 都在
用),它和 `spmd_shims.py` 只是名字像 —— 后者是上层 shim,不要一起删。

***

## D —— 真正缺失

| 上游                                                                                | 影响                                                                                       |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `distributed/compile.py`                                                          | **被裁剪成一个 `torch.compile(model)`**(`parallelize_hf.py:158`,PP 每 chunk 一次在 `pp.py:255`)。裁掉的是四件互相独立的事:逐 block 编译、async TP `_micro_pipeline_tp`、`regional_inductor`、`capture_scalar_outputs`(后者是 token-choice MoE dispatch 的动态 shape 所需的) |
| `models/common/moe_sharding.py`                                                   | **比"缺一个文件"更深**。hpmesh 有这套**形式**(`parallel/sharding.py` 的 `ShardingConfig` + `resolve_placements`),但没有 MoE 的**声明**,也没有引擎 —— `set_moe_sharding_config` 要求基类有 `sharding_config` 属性 + 一个读 `in_src/in_dst/out_src/out_dst_shardings` 的 `parallelize_module`,hpmesh 两个都没有。而且它真正的载荷是 **MoE-under-TP**(routed 专家按 `Shard(1)/Shard(2)` 上 TP 轴、router 保持 Replicate),而 hpmesh 的 TP **明确拒绝** MoE 专家(`tp.py:293`,`moe_tp_experts` 直接 raise:没有 fused-expert realizer)。所以这不是一个模块的缺口,是 **TP×MoE 这个组合维度整体没有** |
| `components/quantization/`, `structured_logger/`, `protocols/`, `configurable.py` | **故意删除**,不是缺口。不要"补回来"                                                                    |

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
| `components/optimizer/__init__.py`       | 34 | `components/optimizer/__init__.py`         |
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

- torchtitan `1c7ab8089` 之后 hpmesh 未跟进的:`git -C <torchtitan> log <sha>..HEAD -- torchtitan/`
- 本表基线 hpmesh `58eb279`;其父提交 `53a2ead` 起,`hpmesh/parallel/context_parallel/`
  做过一轮 CP 测试与文档整理。

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

**性能提示(本次实测)**:不要用 `autojunk=False`,91 x 444 会跑十几分钟。
默认 `autojunk=True` 加那条上界剪枝就能在两分钟内跑完。

**不要用** **`diff`** **行数判断改动量**,`quick_ratio()` 也**不能**用来筛候选 ——
它是上界不是估计,会漏掉真正的对应物(实测漏过 `components/loss.py`)。

**第 2 列是次优匹配**。它比最优值还重要:两者**接近**说明这个文件的归属有歧义,
需要人工判(见血的教训 3、6);两者差距大,最优才是可信的。
