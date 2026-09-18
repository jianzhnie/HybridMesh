# hpmesh 结构设计与优化

> 本文只谈**结构**：分层、依赖方向、目录、以及哪些代码该删。
> 数值/迁移的路线图在 `hybridmesh_design.md`（阶段 0/1 已完成），本文是它的**结构续篇**。

所有数字都是 AST 实测（`hpmesh/**/*.py`），不是估计。

---

## 0. 一句话结论

hpmesh 的**分层方向已经是对的**（`trainer -> parallel -> models` 单向，反向零依赖），
但**层内部没有边界**：`parallel/` 是一个 9 文件的平铺目录，混装了「并行维度」
「并行基础设施」「torchtitan 遗产」三类东西；`models/` 里有一个 2846 行的孤岛。

结构问题里最值钱的两个，都不需要改代码即可证明：

1. **39% 的包体从 trainer 不可达**，其中 2846 行（`models/common/`）是有测试、有文档、
   有类型注解的**已完成能力**，只是没有接线。
2. **`parallel/` 里有一个 125 行的死文件**（`sharding.py`）和一个**同名不同义的类**
   （`ShardingConfig` 在 `sharding.py:32` 和 `tp.py:128` 各有一份）。

---

## 1. 现状实测

### 1.1 规模与分层

```
40 个模块, 7406 行 (hpmesh/)

trainer   (4 模块,  758 行)   config / trainer / train / __init__
parallel  (12 模块, 2852 行)  tp / cp_ep / pp / fsdp / fsdp_wrap / linear /
                              parallelize_hf / parallel_dims / pipeline /
                              spmd_types / sharding / __init__
models    (17 模块, 3641 行)  hf_wrapper + common/{16 modules, 3146 行}
utils     (4 模块,  209 行)
mesh.py   (1 模块,   90 行)   <-- 悬在包的根上
```

### 1.2 可达性：26 个可达，14 个不可达

从真实入口（`hpmesh/__main__.py` + `hpmesh/trainer/train.py`）做 AST 可达性分析：

| 不可达模块 | 行数 | 有测试? |
|---|---|---|
| `models/common/token_dispatcher.py` | 547 | 是（`ep_equivalence.py`） |
| `models/common/moe.py` | 430 | 是 |
| `models/common/rope.py` | 389 | 是（25 个） |
| `parallel/pipeline.py` | 296 | 是（8 个） |
| `models/common/aux_loss.py` | 250 | 是（28 个） |
| `models/common/dist_gemm.py` | 241 | 是（15 个） |
| `models/common/qkv.py` | 185 | 是（19 个） |
| `models/common/multimodal.py` | 165 | 是（20 个） |
| `models/common/grouped_experts.py` | 132 | 是 |
| `models/common/embedding.py` | 74 | 是 |
| `models/common/activation.py` | 72 | 是 |
| `models/common/flex_kernel.py` | 43 | **否** |
| `models/common/param_init.py` | 42 | **否** |
| `models/common/scatter_add.py` | 38 | **否**（只被 token_dispatcher 用） |
| **合计** | **2904** | 可达部分仅 4502 行 |

**这不是"废弃代码"**，这是**已完成但未接线的能力**。区别很重要：
- `parallel/pipeline.py`（296 行）是 PP stage 切分的完整前半段，阶段 4 要用；
- `models/common/*`（2846 行）是从 torchtitan 移植、逐位对拍过的一整层；
- 删掉它们 = 把未来的工作量翻倍。**但它们的存在方式有问题**（见 §2.1）。

### 1.3 死 API：`spmd_types.py` 的 17 个导出里 9 个没人用

| 符号 | 真实使用 |
|---|---|
| `spmd_mesh_group` | 9（`cp_ep` / `moe` / `embedding`） |
| `spmd_sparse_mesh` | 4（`moe`） |
| `current_spmd_mesh` | 6（`dist_gemm`） |
| `set_current_spmd_mesh` | 14（全部在 **tests/**） |
| `spmd_axes` / `_per_axis_types` | 4 / 6 |
| `maybe_set_sparse_mesh` | 1 |
| `spmd_mesh_size` | 1 |
| `spmd_dense_mesh` | 3 |
| **`plain_tensor_to_dtensor_state_dict`** | **0** |
| **`dtensor_to_plain_tensor_state_dict`** | **0** |
| **`set_spmd_meshes`** | **0**（只有 `moe.py` / `test_aux_loss.py` 的**文档字符串**提到它） |
| **`annotate_input_spmd_types`** | **0** |
| **`annotate_replicated_parameters`** | **0** |
| **`spmd_validate_redistributions`** | **0** |
| **`spmd_redistribute_per_axis`** | **0** |
| **`spmd_distribute_tensor`** | **0** |
| **`spmd_local_context`** | **0** |

9 个死导出里，7 个是「类型检查/权重转换」家族——它们服务于 `Module.parallelize()` 那套
**已被删除的抽象**。留着它们，读者无法区分"这是 API"和"这是遗迹"。

### 1.4 `parallel/sharding.py` 是死文件

125 行，唯一的外部引用是 `spmd_types.py:74` 的**函数内 import**，而那个调用者
（`plain_tensor_to_dtensor_state_dict`）自己也是死的。删掉调用者，这个文件就孤立了。

更糟的是**同名类冲突**：

```
parallel/sharding.py:32   class ShardingConfig   # DTensor placement 声明（死）
parallel/tp.py:128        class ShardingConfig   # TP 切分 kind（活）
```

同一个包里两个 `ShardingConfig`，语义完全不同。任何 grep "ShardingConfig 在哪定义"
都会得到两个答案。

### 1.5 层倒置：`models/common` 反向依赖 `parallel/`

```
models/common/aux_loss.py     -> hpmesh.parallel.parallel_dims.ParallelDims
models/common/dist_gemm.py    -> hpmesh.parallel.linear.{AllGatherLinear, LinearReduceScatter}
models/common/dist_gemm.py    -> hpmesh.parallel.spmd_types.current_spmd_mesh
models/common/moe.py          -> hpmesh.parallel.spmd_types.{spmd_mesh_group, spmd_sparse_mesh}
models/common/embedding.py    -> hpmesh.parallel.spmd_types.spmd_mesh_group
```

反向（`parallel/ -> models/`）是**零**。所以依赖图是：

```
      trainer
         |
         v
      parallel              <-- 并行维度（apply_tp 等） + 并行基础设施（fsdp, spmd）
         ^
         |  (反向!)
         |
    models/common           <-- 模型组件
```

`models/common` 不是"模型层"，它是**可并行化的模型组件层**——它知道 mesh 的存在。
这个耦合是**真实需求**（MoE 要知道 EP 组多大），不是设计失误。
但它必须被命名和约束，否则下一步就会有人在 `models/common/rope.py` 里 import `Trainer`。

### 1.6 六个文件 `cfg.*` 只读一个布尔或度数

```
parallel/cp_ep.py         -> cfg.cp
parallel/fsdp_wrap.py     -> cfg.parallel.fsdp_reshard_after_forward, cfg.parallel.enable_fsdp_symm_mem
parallel/pp.py            -> cfg.pp
parallel/tp.py            -> cfg.tp
parallel/parallelize_hf.py-> cfg.compile
parallel/parallel_dims.py -> ParallelismConfig
```

六个文件为了 6 个字段，各自 import 了整个 `HybridMeshConfig`。
后果：`parallel/` 无法脱离 `trainer/` 单独测试；换配置系统要改 6 处；
且 `apply_tp(model, pg, tp_size, tp_rank)` 这种**本就该是纯函数**的东西被迫收一个大对象。

注意 `tp` / `cp` / `pp` / `ep` **在 `parallel/*.py` 里各只被读 1 次**——而
`parallel_dims`（它自己就是从 config 派生的）已经把同样的数字算好了。**同一个事实存了两份。**

### 1.7 三个空 `__init__.py`

`models/__init__.py`、`models/common/__init__.py`、`utils/__init__.py` 都是 0 字节。
`models/common` 有 16 个模块，对外却没有任何稳定名字——调用方只能写全路径。

---

## 2. 目标结构

### 2.0 分层规则（三条，可被测试钉死）

```
L0  utils/                    无 hpmesh 依赖
L1  models/common/            可依赖 L0 + parallel/dist_context（见 I2）
L2  parallel/dist_context.py  可依赖 L0
    parallel/fsdp.py          可依赖 L0 + dist_context
L3  parallel/{tp,cp_ep,pp,fsdp_wrap}.py   可依赖 L1 + L2
    models/hf_wrapper.py      可依赖 L1 + L0
L4  parallel/plan.py + models/registry.py 声明层
L5  trainer/                  可依赖所有；**没有任何东西依赖它**
```

**I1（新增，可测）**：`models/common/*` 只能 import `torchtitan` 无关的 `parallel/` 子模块中的
**`dist_context`**（`parallel_dims`、`linear` 两个例外见 §3.2），
且**永不** import `parallel/{tp,cp_ep,pp,fsdp_wrap}` 或 `trainer/`。
**I2（新增，可测）**：`parallel/` 永不 import `trainer/`。
**I3（保留）**：`models/` 与 `parallel/` 之间无环——今天成立，用测试固化。

### 2.1 目标目录

```diff
  hpmesh/
+ ├── dist_context.py          # NEW  mesh 作用域 + 轴查询。从 parallel/spmd_types.py 升上来
+ ├── mesh.py                  # (从包根移进来也行，见 §3.3)
  ├── trainer/
  │   ├── config.py            # 529 行 —— 阶段 3 再拆
  │   ├── trainer.py
  │   └── train.py
  ├── parallel/
- │   ├── spmd_types.py        # 529 -> 分裂：状态/轴查询 -> dist_context；其余 9 个死函数删除
- │   ├── sharding.py          # 125 -> 删除（死文件 + 同名类冲突）
  │   ├── parallel_dims.py     # 533
  │   ├── linear.py            # 298
  │   ├── fsdp.py              # 440
  │   ├── fsdp_wrap.py         # 112
  │   ├── tp.py                # 248
  │   ├── cp_ep.py             # 213
  │   ├── pp.py                # 26   <- 与 pipeline.py 合并
  │   ├── pipeline.py          # 296
  │   └── parallelize_hf.py    # 62
  ├── models/
  │   ├── hf_wrapper.py        # 495
  │   └── common/              # 16 模块, 2846 行 —— 结构不动，只加边界
  └── utils/
```

**关键：`models/common/` 内部结构不动。** 它已经是"一个文件一个概念"的整齐切分
（attention 相关在 `qkv`/`flex_kernel`/`masks`，MoE 在 `moe`/`grouped_experts`/
`token_dispatcher`/`scatter_add`），加目录只会增加 import 深度，不增加清晰度。
它的问题**不是切分**，是**不可达**——那是接线问题（阶段 3/4），不是布局问题。

---

## 3. 三个**需要动代码**的结构改动

### 3.1 【改动 1】`parallel/sharding.py` 删除 + 9 个死 API 切除

- **删** `parallel/sharding.py`（125 行）——连同它的唯一（死）调用者。
- **删** `spmd_types.py` 的 9 个死导出（§1.3 表）：`plain_tensor_to_dtensor_state_dict`、
  `dtensor_to_plain_tensor_state_dict`、`set_spmd_meshes`、`annotate_input_spmd_types`、
  `annotate_replicated_parameters`、`spmd_validate_redistributions`、
  `spmd_redistribute_per_axis`、`spmd_distribute_tensor`、`spmd_local_context`。

**收益**：约 **-400 行**，且消灭同名 `ShardingConfig`。
**风险**：零——全部实测 0 使用。
**注意**：`moe.py:380` 和 `test_aux_loss.py:448` 的**文档字符串**提到 `set_spmd_meshes`
来解释"没有 mesh 时退化"。删函数的同时要把这两处改写成"没有 `dist_context` 时退化"，
否则文档会指向一个不存在的符号。（这正是阶段 0 遇到的同类问题。）

### 3.2 【改动 2】`parallel/spmd_types.py` 升为顶层 `dist_context.py`

拆成两段，对应两种完全不同的问题：

```python
# hpmesh/dist_context.py
"""运行时的进程组作用域 —— 唯一的共享可变状态。

这个文件是 hpmesh 里唯一允许"隐式到作用域"的东西，所以它被刻意放在
最外层并保持很小：任何下层的代码都能问"这个 mesh 轴存在吗"，而不必
把 mesh 沿调用链传下去。

  mesh 状态    set_current_spmd_mesh / current_spmd_mesh / _spmd_mesh_stack
  轴查询       spmd_mesh_group / spmd_mesh_size / spmd_dense_mesh /
               spmd_sparse_mesh / maybe_set_sparse_mesh
  layout 解析  spmd_axes / _per_axis_types          <- 待阶段 3 归并进 plan
"""
```

**为什么升到顶层**：`models/common` 的 5 处反向依赖里，**4 处是 `spmd_types`**。
把 `models/common/*` 指向顶层 `dist_context` 而不是 `parallel.spmd_types`，会得到：

```
      trainer  ->  parallel  ->  models
                    |
                    v
              dist_context  <---  models/common   (依赖，不是倒置)
```

依赖图**从"倒置"变成"菱形"**，这是正确的形状：`dist_context` 是两边共享的**下界**，
而不是 `models` 依赖 `parallel`。`L1 可以依赖 dist_context` 这条规则因此是**自洽的**。

**剩下 1 处**（`dist_gemm.py -> parallel/linear.py`）**保留不动**：
`AllGatherLinear` 是 TP 的 GEMM 实现，`dist_gemm` 是它的融合版本——
它们是同一个机制的两种写法，合并或搬移只会掩盖这个事实。

**`aux_loss.py -> parallel_dims`**：这是**局部 import**（不是模块级），因为 `aux_loss`
在 CPU 上被单测时不需要 `parallel_dims`。保持局部 import，在规则里写成"允许的例外"，
并在文档字符串里说明原因。

### 3.3 【改动 3】`parallel/pipeline.py` 合并进 `pp.py`

```python
# parallel/pp.py  今天: 26 行, apply_pp 直接 raise NotImplementedError
# parallel/pipeline.py  今天: 296 行, 完整可用的 stage 切分
```

两个文件描述**同一个概念的两半**，且 `pp.py` 的文档字符串已经在指路
（"``pipeline.py`` decides the layer assignment"）。合成一个文件后，
阶段 4 要做的就只剩"补 schedule"这一件事，而不是"先找到东西在哪"。

### 3.4 【改动 4，可选】`mesh.py` 归位

`hpmesh/mesh.py` 悬在包根上，同时 import `parallel/parallel_dims` 和 `trainer/config`。
它**应该**在那里——它是组装层，管"从 config 造出 mesh"。
但它读起来像基础设施。两个选项：

| 选项 | 做法 | 代价 |
|---|---|---|
| A（推荐） | 不动 | 零 |
| B | 移到 `hpmesh/trainer/mesh.py` | 与 `dist_context` 的"mesh 在哪"再次产生两个答案 |

**取 A。** 一个 90 行的文件挂在根上不是问题；**再造一个 mesh 概念才是**。

---

## 4. 两个**暂不改**的结构问题（附理由）

好的设计文档必须说明哪些没做、为什么。

### 4.1 `parallel/` 的 6 处 config 倒置 —— 留到阶段 2/3

§1.6 那个问题是真的，但**现在改会做错**。正确解法是阶段 2 的 SEAM 2：

```python
def apply_tp(model, *, plan: ParallelPlan):
    if tp_size() == 1:        # <- 从 dist_context 读，不再读 cfg.tp
        return model
```

一旦 `tp_size()` / `cp_size()` 从 `dist_context` 读，「这个维度开没开」和
「这个维度多大」**变成同一个查询**，`I5（degree==1 时 no-op）从纪律变成事实**，
6 处 config import 自然消失。在阶段 2 之前单独改，只会把 `cfg.tp` 换成别的参数，
问题原样保留。

### 4.2 `trainer/config.py` 529 行 —— 不是一个文件的问题

它同时装着：`ModelArguments`（模型架构）、`ParallelismConfig`（533 行的
`parallel_dims.py` 也依赖它）、16 个 `pipeline_parallel_*` 字段、
一个 flat-view property 层（`cfg.dp` / `cfg.lr` / ...）。

**这是 torchtitan 的形状，不是 hpmesh 的。** 但拆它需要先回答
"`ParallelismConfig` 属于 trainer 还是 parallel"——而那个答案**取决于 §4.1 的
改动先落地**。阶段 3 之后，`ParallelismConfig` 可以整体移进 `parallel/`，
`config.py` 才会塌缩成真正的"训练配置"。

**所以顺序是：结构 1/2/3（本文） -> 阶段 2（SEAM 2）-> 阶段 3（SEAM 1 + config 搬家）。**

---

## 5. 执行顺序与验收

每一改独立可跑、可对拍。**不允许跨改混做。**

| # | 改动 | 行数 | 风险 | 验收 |
|---|---|---|---|---|
| 1 | 删 `sharding.py` + 9 个死 API | -400 | **零** | `pytest tests/` 185 passed 不变；`ruff` 干净；`python -m hpmesh --steps 2` loss 不变 |
| 2 | `spmd_types.py` -> `dist_context.py` | 0 | 低 | 同上 + 新增分层测试（§6） |
| 3 | `pipeline.py` 并入 `pp.py` | 0 | 零 | `tests/test_pipeline.py` 8 个仍过 |
| 4 | `parallel/__init__` 导出 `dist_context` | +5 | 零 | —— |

**改动 1 的对拍**：与阶段 0 同法——`git worktree add /tmp/... HEAD` 跑基线，
两边 `--steps 2` 的 loss 必须逐位相同（`4.858671` / `4.855821`）。
改的是纯删除，任何数值变化都说明删错了。

**改动 2 的对拍**：EP/CP 等价脚本结果**不变**（`3.271e-07` / `2.384e-07`）。
纯移动 + 重命名，数值必须逐位相同。

---

## 6. 用测试钉死分层

结构规则写在文档里等于没写。四条可执行断言：

```python
# tests/test_layering.py

def test_L1_models_never_imports_the_trainer():
    """models/ 不能知道 trainer/ 的存在。"""
    # AST: models/**/*.py 的 import 集合与 hpmesh.trainer.* 无交集

def test_L2_parallel_never_imports_the_trainer():
    """parallel/ 不能知道 trainer/ 的存在。"""
    # AST: parallel/**/*.py 与 hpmesh.trainer.* 无交集

def test_L3_models_common_touches_only_dist_context():
    """models/common 反向依赖 parallel 必须经由 dist_context，两个例外要有注释。"""
    ALLOWED = {"hpmesh.dist_context", "hpmesh.parallel.linear",
               "hpmesh.parallel.parallel_dims"}

def test_L4_no_cycles():
    """models 与 parallel 之间无环。"""
```

**为什么是 AST 而不是 grep**：`moe.py:380` 的文档字符串里就写着 `set_spmd_meshes`。
grep 会把它当成依赖。这一点在本次调查里已经踩过一次。

---

## 7. 结论

| 问题 | 现状 | 处理 |
|---|---|---|
| 分层方向 | **已正确**（单向） | 不动，用测试固化 |
| 分层边界 | 无 | 加 `dist_context` + 4 条测试 |
| 死代码 | `sharding.py` 125 行 + 9 个死 API | **改动 1** |
| 同名类冲突 | 2 个 `ShardingConfig` | **改动 1** |
| 概念被劈成两半 | `pp.py` / `pipeline.py` | **改动 3** |
| 39% 不可达 | 2904 行 | **不是布局问题**，是阶段 3/4 的接线 |
| config 倒置 | 6 处 | **留到阶段 2**（否则做错） |
| `config.py` 529 行 | 真的过大 | **留到阶段 3**（依赖阶段 2） |

**核心判断**：hpmesh 现在缺的不是"更好的目录结构"——目录基本是对的。
它缺的是**边界**：哪些层可以依赖哪些层、哪些代码是 API、哪些是遗迹。
本文的改动 1-4 都是在补边界，而不是在搬家。
