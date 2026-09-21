# hpmesh -> torchtitan 对应关系表

基线: **hpmesh `5fc0c45`** / **torchtitan `1c7ab8089`**(2026-09-21 核对,
torchtitan 工作区在该 sha 上无未提交改动)。

回答的问题是:*torchtitan 改了文件 X,hpmesh 哪些文件必须跟着改?* 以及反过来
*这个 hpmesh 文件的上游是谁*。

---

## 怎么用这张表

**先查表,再动手。** 这张表是唯一权威 —— 不要在 hpmesh 源码里加
`# upstream: <path> @ <sha>` 之类的来源标注,那种标注试过又被退了,因为
sha 会腐烂而表不会。

**分类不是装饰,是操作指令。** 把 A 类的"逐字复制"规则套到 B 类文件上会毁掉
设计;套到 C 类上会把项目**故意删掉**的抽象又拽回来。

**找漏网的用 copyright header,不要 grep "torchtitan"。**
`components/checkpointer/utils.py` 里有 "torchtitan" 字样吗?没有。它是最逐字的
文件之一(ratio 1.000)。判据是文件头那三行 Meta 版权声明,hpmesh 里 40 个文件
带它。

**比结构用 AST,不要比 diff 行数。** `utils/filesystem.py` 的 diff 有 53 行,
代码差异是 **0** —— 全是改写措辞。做法是剥掉 docstring、`ast.unparse`、
再 `difflib.SequenceMatcher`。下表 `ratio` 列就是这么来的。

**目录不对应。** 上游路径经常跨目录:

| hpmesh | torchtitan |
|---|---|
| `parallel/**` | `distributed/` |
| `components/metrics.py` | `observability/metrics.py` |
| `models/common/scatter_add.py` | `ops/scatter_add.py` |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/transformers_modeling_backend/pipeline.py` |
| `datasets/text/text.py` | `hf_datasets/text_datasets.py` |
| `utils/filesystem.py` | `tools/filesystem.py` |

---

## 图例

- **A 移植 (vendored)** —— 从 torchtitan 复制。上游改了,hpmesh **应该**跟着改。
  子类是它们的真实差异。
- **B 适配 (adapted)** —— 同一个想法、不同的形状。上游改了,**读意图、不要抄形状**。
- **C hpmesh 独有** —— 上游没有对应物。不要"对齐"它。
- **D 缺口** —— 上游有,hpmesh **真的没有**。是要补还是不要补,是决策不是疏漏。

`ratio` 是 AST 相似度,仅对 A/B 类有意义。**1.000 不代表逐字相同** —— 剥掉
docstring 后 `ast.unparse` 归一化了空白和引号。要判断"真逐字",看 ratio 为
1.000 且人工确认过的那三个。

---

## A1 —— 逐字复制(改动需逐位验证)

| hpmesh | torchtitan | ratio |
|---|---|---|
| `components/checkpointer/utils.py` | `components/checkpointer/utils.py` | 1.000 |
| `utils/filesystem.py` | `tools/filesystem.py` | 1.000 |
| `components/optimizer/utils.py` | `components/optimizer/utils.py` | 0.996 |
| `parallel/parallel_dims.py` | `distributed/parallel_dims.py` | 0.988 |
| `datasets/multimodal/mm_image.py` | `hf_datasets/multimodal/utils/image.py` | 0.977 |
| `datasets/multimodal/mm_text_utils.py` | `hf_datasets/multimodal/utils/text.py` | 0.971 |
| `datasets/multimodal/mm_video.py` | `hf_datasets/multimodal/utils/video.py` | 0.967 |
| `datasets/multimodal/mm_collator.py` | `hf_datasets/multimodal/mm_collator.py` | 0.957 |
| `models/common/multimodal.py` | `models/common/multimodal.py` | 0.945 |
| `components/tokenizer.py` | `components/tokenizer.py` | 0.907 |
| `datasets/types.py` | `components/data/types.py` | 0.897 |
| `parallel/tensor_parallel/linear.py` | `distributed/linear.py` | 0.893 |
| `parallel/fully_shard/fsdp.py` | `distributed/fsdp.py` | 0.846 |
| `parallel/activation_checkpoint.py` | `distributed/activation_checkpoint.py` | — |
| `datasets/packing.py` | `components/data/packing.py` | 0.829 | 配方做自由函数;选择器落在 `DataloaderConfig.packing`,上游落在模型 config registry |
| `models/common/rope.py` | `models/common/rope.py` | 0.823 |

**注意 `components/optimizer/utils.py`**:docstring 说 "unchanged in logic",
但 ratio 是 0.996 不是 1.000 —— **docstring 的自述不可信**,以 ratio 为准。
`components/loss.py` 的 docstring 更离谱:它声称 `next_token_targets` 是
"kept from upstream",而上游**根本没有这个符号**(`grep -rn next_token_targets`
在 torchtitan 全仓零命中)。这条已修正;同一段还多声称了
`vocab_shard_bounds` 是从上游"保留"的,它同样只存在于 hpmesh。

---

## A2 —— 移植但已改写(比例中等,需逐处核对)

| hpmesh | torchtitan | ratio | 改写点 |
|---|---|---|---|
| `components/checkpointer/dcp.py` | `components/checkpointer/dcp.py` | 0.862 | `sd_adapter` 插槽的接线 |
| `datasets/text/text.py` | `hf_datasets/text_datasets.py` | 0.853 | 去掉了 `Configurable` |
| `models/common/scatter_add.py` | `ops/scatter_add.py` | 0.711 | |
| `models/common/param_init.py` | `models/common/param_init.py` | 0.700 | |
| `models/common/aux_loss.py` | `models/common/aux_loss.py` | 0.682 | |
| `datasets/loader.py` | `components/data/loader.py` | 0.627 | 去 `Configurable`;`GrainDataLoader` 直接收参数,无 config 类 |
| `models/common/linear.py` | `models/common/linear.py` | 0.620 | |
| `datasets/dataset.py` | `components/data/dataset.py` | 0.607 | 同上;三个节点类去 `Config` 后缀,构建走自由函数 `build_dataset` |
| `models/common/feed_forward.py` | `models/common/feed_forward.py` | 0.560 | **曾写完又被退**,不要在没有明确指令时重新引入 |
| `components/checkpointer/__init__.py` | `components/checkpointer/__init__.py` | 0.757 | |
| `components/checkpointer/base.py` | `components/checkpointer/base.py` | 0.717 | |
| `models/common/dist_gemm.py` | `models/common/dist_gemm.py` | 0.527 | |
| `models/common/token_dispatcher.py` | `models/common/token_dispatcher.py` | 0.441 | |
| `datasets/multimodal/mm_datasets.py` | `hf_datasets/multimodal/mm_datasets.py` | 0.368 | 去 `Configurable`;packing 改自由函数 `build_mm_sample_packing` |
| `datasets/sources.py` | `components/data/sources.py` | 0.762 | |
| `components/profiler.py` | `observability/profiler.py` | 0.413 | |
| `components/metrics.py` | `observability/metrics.py` | 0.384 | |
| `components/optimizer/lr_scheduler.py` | `components/optimizer/lr_scheduler.py` | 0.373 | 去 `Configurable` |
| `components/loss.py` | `components/loss.py` | 0.297 | 见上,docstring 关于 upstream 的说法是错的(已于 2026-09-21 修正) |
| `components/optimizer/optimizer.py` | `components/optimizer/optimizer.py` | 0.240 | 容器化改写;`OptimizerWrapper` 已删 |
| `components/checkpointer/torch_checkpointing.py` | `components/checkpointer/torch_checkpointing.py` | 0.265 | |
| `models/common/masks.py` | `models/common/attention.py` | 0.380 | 拆出了 mask 部分 |
| `models/common/qkv.py` | `models/common/attention.py` | 0.242 | |
| `parallel/spmd_shims.py` | `distributed/spmd_types.py` | 0.532 | **见 C 类说明** |
| `utils/spmd_context.py` | `distributed/spmd_types.py` | 0.350 | 同上 |
| `parallel/sharding.py` | `protocols/sharding.py` | 0.340 | 换掉了 Module 协议 |
| `models/common/grouped_experts.py` | `models/gpt_oss/moe.py` | 0.136 | 上游的 GroupedExperts 在 `models/common/` 与 gpt_oss 各有一份 |
| `parallel/fully_shard/fsdp_wrap.py` | `models/deepseek_v3/parallelize.py` | 0.155 | **B 类典型**:把 HF 模型套到 Decoder 形状上 |
| `parallel/pipeline_parallel/pp.py` | `distributed/pipeline_parallel.py` | 0.130 | |

**只有路径对应、内容其实无关的**:`models/common/activation.py`(0.124)、
`models/common/embedding.py`(0.138)在两个仓里都是同名文件,但 AST 相似度低于
同仓库的其他文件 —— **是各自写的,不是移植的**。归到这里是为了说明"同名不等于
同源",不要把它们的名字当依据。

---

## C —— hpmesh 独有(不要对齐上游)

| hpmesh | 说明 |
|---|---|
| `utils/spmd_shims.py` | **是 `spmd_types` pip 包的薄壳,不是 torchtitan 的**。`spmd_types` 装在 site-packages。不要把它"对齐"到 `distributed/spmd_types.py` |
| `utils/spmd_context.py` | 同上,配合 `spmd_shims` 的上下文管理 |
| `parallel/collectives.py` | 最优相似度 0.088,是独立实现 |
| `parallel/context_parallel/apply.py` | 0.063;CP 的编排层,上游无对应文件 |
| `parallel/context_parallel/cp_kernel.py` | 0.054;hpmesh 独有的 CP flex kernel |
| `parallel/context_parallel/input_shard.py` | 0.078 |
| `parallel/context_parallel/primitives.py` | 0.084 |
| `parallel/expert_parallel/apply.py` | 0.104 |
| `parallel/expert_parallel/ep.py` | 0.038;HF block -> 原生 MoE 的**权重搬运**,上游 `moe_replacement.py` 是重新初始化,不是同一件事 |
| `mesh.py` | 0.087;上游 mesh 逻辑散在 `distributed/parallel_dims.py` + `trainer.py` |
| `utils/logger_utils.py` | 上游无对应 |
| `utils/monitoring.py` | 与 `tools/utils.py` 0.107,独立实现(含 `get_peak_flops`) |
| `utils/checkpoint_keys.py` | 上游无 |
| `utils/gc.py` | 与 `tools/utils.py` 0.211 |
| `utils/device.py` | 上游无 |
| `utils/batch_invariant.py` | 上游无 |
| `models/common/flex_kernel.py` | 上游是 torchtitan `Module` 子类;hpmesh 无 Module 协议,改成普通 `nn.Module` 自持 `_sharding_config` |
| `models/common/nn_modules.py` | 上游 `models/common/nn_modules.py` 存在,但 hpmesh 只留了 PP 占位所需的部分 |
| `datasets/random_data.py` | 合成语料,上游无 |
| `datasets/collators.py` | 0.545 |
| `datasets/build.py` | 工厂;上游把 `build()` 放在 config 上 |

**已知悬空**:`parallel/sharding.py` 的 `ShardingConfig` 与
`models/common/flex_kernel.py` 的 `HFFlexKernel._sharding_config` 目前
**没有任何读取者** —— 唯一会读它们的 `set_hf_sharding_configs` 已随
`hpmesh/parallel/hf_sharding.py` 在 `cc6e217` 被删除(`moe_swap.py`、
`placements.py`、`spec.py`、`moe_probe.py` 同批)。这几个符号留着是给将来的
分片层用的,不是活代码。TP 现在的实现是 `parallel/tensor_parallel/tp.py` 的
plan 引擎,与它们无关。

---

## B —— 适配层(读意图,不要抄形状)

这几个是 **torchtitan 每个模型一个文件** 的那种东西的**替代品**。照搬它们的形状
会破坏分片契约。

| hpmesh | 替代掉的上游 |
|---|---|
| `models/hf_wrapper.py` | `experiments/transformers_modeling_backend/model.py` 的包装层;上游另有 `models/*/model.py` 各一份 |
| `parallel/parallelize_hf.py` | `experiments/transformers_modeling_backend/parallelize.py` + 各 `models/*/parallelize.py` |
| `parallel/tensor_parallel/tp.py` + `linear.py` | `distributed/tensor_parallel.py` 等;hpmesh 是**手写 plan + 融合 GEMM**,不是声明式 `_sharding_config` |
| `parallel/expert_parallel/*` | `experiments/.../moe_replacement.py`;**不共享代码**,是另一套实现 |
| `parallel/context_parallel/*` | `distributed/context_parallel/` |
| `parallel/pipeline_parallel/pipeline.py` | `experiments/.../pipeline.py`;差异是 `None` -> `nn.Identity`、每 stage 追加 `rotary_emb` |
| `trainer/trainer.py` | `trainer.py`(0.057,基本重写) |
| `trainer/config.py` | `config/configs.py`(0.235) |
| `trainer/train.py` | `train.py`(0.186) |

**注意 `mesh.py`**:上游没有单一对应物 —— mesh 逻辑散在
`distributed/parallel_dims.py` 和 `trainer.py` 里,不是某个文件的移植(见 C 类)。

---

## C —— hpmesh 独有(不要对齐上游)

| hpmesh | 说明 |
|---|---|
| `utils/spmd_shims.py` | **是 `spmd_types` pip 包的薄壳,不是 torchtitan 的**。`spmd_types` 装在 site-packages。不要把它"对齐"到 `distributed/spmd_types.py` |
| `utils/logger_utils.py` | 上游无对应 |
| `utils/monitoring.py` | 与 `tools/utils.py` 0.107,实为独立实现(含 `get_peak_flops`) |
| `utils/checkpoint_keys.py` | 上游无 |
| `utils/gc.py` | 与 `tools/utils.py` 0.211 |
| `utils/device.py` | 上游无 |
| `utils/batch_invariant.py` | 上游无 |
| `models/common/flex_kernel.py` | 上游是 torchtitan `Module` 子类;hpmesh 无 Module 协议,改成普通 `nn.Module` 自持 `_sharding_config` |
| `models/common/nn_modules.py` | 上游 `models/common/nn_modules.py` 存在,但 hpmesh 只留了 PP 占位所需的部分 |
| `datasets/random_data.py` | 合成语料,上游无 |
| `datasets/collators.py` | 0.545 |
| `datasets/build.py` | 工厂,上游把 `build()` 放在 config 上 |
| `components/optimizer/__init__.py` 等 | |

**已知悬空**:`parallel/sharding.py` 的 `ShardingConfig` 与
`models/common/flex_kernel.py` 的 `HFFlexKernel._sharding_config` 目前
**没有任何读取者**——唯一会读它们的 `set_hf_sharding_configs` 已随
`hpmesh/parallel/hf_sharding.py` 在 `cc6e217` 被删除。这两个符号留着是给
将来的分片层用的,不是活代码。TP 现在的实现是
`parallel/tensor_parallel/tp.py` 的 plan 引擎,与它们无关。

---

## D —— 真正缺失

| 上游 | 影响 |
|---|---|
| `distributed/activation_checkpoint.py` 的 `SelectiveAC` | **只有 `FullAC` 被移植**(`parallel/activation_checkpoint.py`,自己的 docstring 写明了)。`mode` 是扩展点,`"none"`/`"full"` 之外的值**报错而不是静默不 checkpoint** —— 这是有意的 |
| `tools/validate.py` | 上游**也没有这个路径**,已从表里移除 |
| `distributed/compile.py` | 只有整模型 `torch.compile(model)`,没有逐 block 编译 / regional inductor / `capture_scalar_outputs` |
| `models/common/moe_sharding.py` | hpmesh 没有专家分片声明层;EP 走 `expert_parallel/ep.py` |
| `components/quantization/`, `structured_logger/`, `protocols/`, `configurable.py` | **故意删除**,不是缺口。不要"补回来" |

---

```
hpmesh / torchtitan
```

## E —— 包面(`__init__.py` 与入口)

重组出口,不是移植内容:这些文件定义 hpmesh 的**公开 API 面**,上游对应物是
同名 `__init__.py`(若存在)。改上游的导出列表时才需要看这里。

| hpmesh | torchtitan |
|---|---|
| `__init__.py`(30 行) | `__init__.py` |
| `__main__.py`(6 行) | `train.py` 的入口对等物 |
| `models/common/__init__.py`(67 行) | `models/common/__init__.py` |
| `datasets/__init__.py`(74 行) | `components/data/__init__.py` |
| `parallel/__init__.py`(35 行) | `distributed/__init__.py` |
| `parallel/context_parallel/__init__.py`(37 行) | `distributed/context_parallel/__init__.py` |
| `parallel/expert_parallel/__init__.py`(20 行) | 上游无对应(见 C 类) |
| `parallel/pipeline_parallel/__init__.py`(12 行) | 上游无对应 |
| `components/optimizer/__init__.py`(42 行) | `components/optimizer/__init__.py` |
| `trainer/__init__.py`(26 行) | 上游无对应 |

**空文件**(0 行,不必查表):`components/__init__.py`、`models/__init__.py`、
`parallel/fully_shard/__init__.py`、`parallel/tensor_parallel/__init__.py`、
`utils/__init__.py`。

**两个子包**:`datasets/` 下按语料分 `text/` 和 `multimodal/`,其余模块平铺在
`datasets/` 根下。上游的 `hf_datasets/` 是 `components/data/` 的兄弟目录,hpmesh
曾用 `datasets/hf/` 镜像它(4 层,三个 0 字节的 `__init__.py`),先被溶解成单层,
再于本次重组按语料分成两个子包。

划分依据是**内容**而非上游路径:`datasets/` 根下的 9 个模块与上游
`components/data/` 一一对应,且都被两边共用(`loader.py` 默认 `TextCollator`、
`multimodal/mm_collator.py` 复用 `collators.py` 的 `Collator`/`TrainerBatch`),
所以不进任何一边;只有 `text.py` 和 5 个 `mm_*.py` 是语料专属。两个子包的
`__init__.py` 都**刻意不导入子模块**——惰性导入契约靠这一点维持。

**上游路径对应关系因此不再一一成立**(`hf_datasets/multimodal/utils/image.py`
在 hpmesh 侧是 `datasets/multimodal/mm_image.py`),本表的 hpmesh 列是唯一权威。

---

## 版本与漂移

- torchtitan `1c7ab8089` 之后 hpmesh 未跟进的:`git -C <torchtitan> log <sha>..HEAD -- torchtitan/`
- hpmesh 基线的父提交是 `5fc0c45`;`hpmesh/parallel/expert_parallel/ep.py`、
  `pyproject.toml`、`models/common/moe.py` 的 transformers 5.9 改造已在
  `f2b501a` 提交,**不在工作区**。

---

## 重现这张表

两个脚本都在 `$CLAUDE_JOB_DIR/tmp/` 之外要重建的话:

```python
# 1) 结构匹配:剥 docstring -> ast.unparse -> SequenceMatcher
import ast, difflib
from pathlib import Path

def strip_src(p):
    t = ast.parse(Path(p).read_text())
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
```

然后对每个 hpmesh 文件与每个 torchtitan 文件算 `SequenceMatcher(None, a, b).ratio()`。
**必须先把 torchtitan 的候选集限定在 `torchtitan/` 下**,否则会匹配到
`experiments/rl/` 之类的噪音(见下表"血的教训")。

**不要用 `diff` 行数判断改动量**,`quick_ratio()` 也**不能**用来筛候选 ——
它是上界不是估计,会漏掉真正的对应物(实测漏过 `components/loss.py`)。

---

## 血的教训

1. **defaultdict 的别选错**:用 `default_factory=dict` 那版才算对了
   `parallel/parallel_dims.py` 的 0.988;用别的会得到 0.28。
2. **空 `__init__.py` 会污染 basename 匹配**:hpmesh 有 10 个空
   `__init__.py`,按 basename 匹配会命中 30+ 个上游同名文件。先按大小过滤。
3. **不要按名字猜上游**:`utils/monitoring.py` 与 `tools/utils.py` 名字毫不相干,
   是 AST 匹配找出来的。
4. **别信 docstring 里的"kept from upstream"** —— `components/loss.py` 关于
   `next_token_targets` 的说法就是错的。
