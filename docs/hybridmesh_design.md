# Hybrid Parallel training over a Torch DeviceMesh

## 设计原则

hpmesh 在设计时拿掉 TorchTitan 的抽象, 希望能直接对接 HuggingFace 的 API, 适配
HuggingFace 的模型. 拿掉的抽象层包括:

- `configurable.py` 的配置抽象 —— 全套 `Configurable` 类
- `module.py` 的模块抽象 —— 全套 `Module` 类

拿掉这两个抽象层后, hpmesh 直接使用非常简单直观的 `config.py` 类来传递参数,
所有的函数和类都变得更加简单直观, 不再需要复杂的配置类和模块类.

本文结合 `/Users/robin/work_dir/torchtitan/torchtitan` 与当前已完成的 hpmesh
框架, 给出更优的原型设计. 结论先放在这里:

> **拿掉 `Configurable` / `Module` 只是把复杂度搬走了, 没有消灭它.**
> torchtitan 用两个显式抽象换来了两样东西 —— "配置如何到达模块" 和
> "模块如何描述自己的并行方式". 这个框架目前把前者转嫁给了配置对象本身
> (`parallel/` 下每个 `apply_*` 都接收整个 `HybridMeshConfig`, 尽管最多只用 2 个字段),
> 把后者转嫁给了**模块类的身份** (`parallel/tp.py` 靠 `isinstance(module, nn.Linear)`
> 扫描模型来猜哪些投影该切).
>
> 后者是真正的技术债: 它是"框架反向依赖具体模型内部结构", 而这个仓库
> (`models/common/` 里 21 个模块的移植 + `moe_swap.py` + `moe_probe.py` +
> 两个并存且互相冲突的 `HFModelWrapper`) 正在真实地支付这笔债.
>
> 本文主张一次**收敛 (convergence)**: 不引入任何新抽象层, 只引入**两个缝 (seam)**
> —— 模型侧用一份数据 (而非类身份) 声明如何切分, 运行时侧用一个线程局部的
> 分布式上下文取代到处传递 mesh / group. 收敛后删除被取代的死代码,
> 把两套模型包装归一.

---

## 1. 现状画像

### 1.1 已经建成的部分 (实测)

| 维度 | 状态 | 证据 |
|---|---|---|
| 配置 | 分组 dataclass, 无 `Configurable` | `trainer/config.py` |
| 拓扑 | `ParallelDims` + 4 维 `DeviceMesh` | `mesh.py`, `parallel/parallel_dims.py` |
| FSDP2 | torchtitan `fsdp.py` 原样 vendored | `parallel/fsdp.py` + `fsdp_wrap.py` |
| TP | 声明式 plan + 融合 GEMM | `parallel/tp.py`, `parallel/linear.py` |
| CP | 两个 redistribute 策略 | `parallel/cp_ep.py` |
| EP | all-to-all dispatcher + MoE | `models/common/{moe,token_dispatcher,grouped_experts}.py` |
| 模型组件 | 21 个 torchtitan 模块中 10 个已落地 | `models/common/*.py` |
| PP | **stage 切分已完成 (未接线), 缺 schedule** | `parallel/pipeline.py` (保留), `parallel/pp.py` (stub) |
| 验证 | **206 个 CPU 单测通过**; EP 等价 rel `3.271e-07`; CP 等价 `2.384e-07`, Ulysses 往返 `0` | `tests/` |

这套骨架是健康的. 下面的问题不是"没做完", 而是"做完了的部分正在被自身的形状反噬".

### 1.2 一个可以量化的诊断

统计每个 `apply_*` 实际读了多少配置字段:

```
apply_tp        cfg.tp                                      -> 1 个字段
apply_cp_ep     cfg.cp, cfg.ep                              -> 2 个字段
apply_pp        cfg.pp                                      -> 1 个字段
apply_fsdp      cfg.parallel.enable_fsdp_symm_mem,
                cfg.parallel.fsdp_reshard_after_forward     -> 2 个字段
```

**没有任何一个函数需要超过 2 个字段, 但每一个都接收整个 `HybridMeshConfig`.**

代价是具体的、可验证的 —— 实测 `trainer/config.py` 的反向依赖方:

```
mesh.py                        from .trainer.config import HybridMeshConfig
bundle.py                      from .trainer.config import HybridMeshConfig
parallel/parallelize_hf.py     from ..trainer.config import HybridMeshConfig
parallel/tp.py                 from ..trainer.config import HybridMeshConfig
parallel/fsdp_wrap.py          from ..trainer.config import HybridMeshConfig
parallel/pp.py                 from ..trainer.config import HybridMeshConfig
parallel/cp_ep.py              from ..trainer.config import HybridMeshConfig
```

底层的并行模块反向依赖 trainer 层. 一个用来承载 CLI 参数的 god-object,
成了整个框架的类型中枢 —— 结果是**为了给 `apply_tp` 传一个整数, 整个配置对象
都必须被构造出来**, 单元测试也因此必须伪造一个 `HybridMeshConfig`.

> 诚实地说: `models/common/` 目前**没有**这个问题 (它只 import `torch` / `spmd_types`
> 与同目录模块, 对 trainer 的引用仅存在于文档字符串). 但它的签名是
> `register_aux_loss_zero_hook(optimizer, model_parts, parallel_dims)` 这种形状,
> 意味着"配置 + 运行时"已经在往模型组件里渗. 本设计的 I1 是把**当前恰好成立的
> 事实固化成不变量**, 而不是在描述一个已存在的缺陷.

看起来像是文件组织问题, 实质是"传参"这种最朴素的机制撑不住 5 个并行维度之后,
必然会退化成的形状.

### 1.3 第二个诊断: 框架在识别具体模型

移植 `qkv.py` 时我给 `QKVLinear` 加了一个 `_project` 接缝, 理由是让
`AllGatherFusedQKVLinear` 能只替换 matmul. 这个接缝本身是对的, 但它暴露了一个
更深的问题 —— **框架必须知道某个 `nn.Linear` 到底是 qkv 还是普通投影, 只能靠猜**:

```python
# 现存的猜测方式 (实测位置)
if isinstance(module, nn.Linear): ...              # parallel/tp.py:229      [活, 要解决]
("embed_tokens", "wte") / ("norm", "ln_f")         # bundle.py:73,81         [活, 要归并]
if hasattr(experts, "gate_up_proj"): ...           # moe_probe.py:231,269    [删]
if hasattr(block, "experts"): ...                  # moe_probe.py, moe_swap.py [删]
"model.layers" / "model.model.layers" / "layers"   # _utils.py:10, moe_swap.py:280 [删]
```

每一条都是在**反推 HF 模型的内部结构**. 而且它们同时是 hpmesh 的**价值**
(这就是"对接 HF"的本体) 和它的**脆弱点** —— 所以处理方式不是消灭它们, 而是
**把它们收敛到一处**: 名字解析归 `HFTransformerModel` (§5.1), 切分目标归
`ParallelPlan` (§4 SEAM 1).

分布很说明问题: **5 处里有 3 处落在 §1.4 判定要删的被取代代码里**
(`moe_probe` / `moe_swap` / `_utils`). 所以阶段 0 的删除是"免费的减债".
剩下两处里, `tp.py:229` 是唯一一处"框架主动扫描模型来找切分目标"的地方 ——
那正是 SEAM 1 要解决的.

### 1.4 第三个诊断: 不可达的代码

按"从 trainer 出发能否到达"做的精确可达性分析 (AST 解析相对导入, 非子串匹配):

| 模块 | 可达 | 原因 | 处理 |
|---|---|---|---|
| `models/hf_wrapper.py` | 否 | 只有 `tests/test_hf_wrapper.py` 引用 | 保留 (§1.5 要升级它) |
| `models/spec.py` (`HPModelSpec`) | 否 | 仅 `rope.py` 文档字符串提了一句 | **删除** (被 `registry.py` 取代) |
| `models/moe_swap.py` + `moe_probe.py` | 否 | `moe_swap` -> `moe_probe`, 无外部入口 | **删除** |
| `parallel/hf_sharding.py` + `placements.py` | 否 | 无 hpmesh 内部引用 | **删除** |
| `parallel/_utils.py` | 否 | 无引用 | **删除** |
| `parallel/pipeline.py` | 否 | `pp.py` 只提了名字, 没 import | **保留** —— 它是 PP 的前半段 (§1.6) |
| `models/common/embedding.py` | 否 | 自身 docstring 写明 "currently unused" | **保留** —— 待 TP 接上 `Shard(0)` |
| `protocols/` | —— | 空目录 | **删除** |

按项目规则「Deprecated files should be removed, not updated」, **被取代的**应当删除.
但"不可达"不等于"可删除" —— 见下.

### 1.5 诊断: 两套模型包装并存

```
bundle.py:HFModelWrapper                 <- 活的 (trainer -> bundle -> build_bundle)
models/hf_wrapper.py:HFTransformerModel  <- 死的 (只有 tests/test_hf_wrapper.py)
```

两个类**职责重叠、互不引用**, 一个活着一个死了. 而活着的那个能力更弱
(无 flex attention 装配、无 CP 支持、无 logits 转储), 死掉的那个恰好是设计更好的一份.

> 这不是"还没合并", 这是**两条设计路线在同一个仓库里并行生长**. 设计文档的第一件事
> 就是裁决它们 (§5.1).

### 1.6 一句重要的区分: "不可达" != "可删除"

§1.4 那张表按"从 trainer 是否可达"分类, 但**不可达有两种, 处理方式相反**:

| 不可达的原因 | 例子 | 处理 |
|---|---|---|
| 被取代 (superseded) | `hf_sharding`/`placements` 被 `spmd_types` 取代; `spec.py` 被 `registry.py` 取代 | **删除** |
| 还没接线 (not yet wired) | `pipeline.py` (PP stage 切分, 296 行, 完整可用); `models/common/embedding.py` (vocab-parallel embedding) | **保留**, 由对应阶段接上 |

`pipeline.py` 尤其容易被误判: 它可独立使用, 只依赖 torch 的
`distributed.pipelining`, 甚至已经带着 `ScheduleDualPipeV` /
`ScheduleZBVZeroBubble` / `get_schedule_class` 的导入和一个 `get_mesh` 回调.
缺的只是 `pp.py` 把它接进 `apply_pp`. **删掉它等于把阶段 4 的工作量翻倍.**

> 这条区分是本设计里唯一的"读代码时要小心"的地方 —— 只按可达性做删除判断,
> 会把未接线的能力一起埋掉.

---

## 2. 设计目标与不变量

在给出结构之前, 先把"什么算更好"写死, 否则无法验收.

**目标 (Go)**

- G1 **一个模型**: 任意 HF `AutoModelForCausalLM` 可训练, 不改模型代码即可并行.
- G2 **一条数据通路**: 配置只沿一个方向流动 (`CLI -> Config -> 顶层显式传参`).
- G3 **每个 `apply_*` 只依赖它的契约**, 不依赖 `trainer` 层, 也不依赖具体模型类.
- G4 **一次加一个维度**: 每个维度是独立文件 + 独立单测 + 独立对拍脚本.
- G5 **可对拍**: 任何非计算改动必须逐位一致 (本仓库已有的标准).

**不变量 (Invariants) —— 这些是"不许再退化回去"的红线**

- I1 `models/common/` 不得 import `trainer/`. (当前只差一步就破了)
- I2 任意并行模块不得 import 具体模型模块 (`qkv`, `moe`, ...); 只能 import
  `torch` / `nn` / `parallel/*` / 协议.
- I3 模型侧向框架暴露的信息, 只能是**数据** (dict / dataclass / 张量),
  不能是"调用框架的内部函数".
- I4 新增/重写的注释与文档字符串只用 ASCII.
- I5 `apply_X` 在对应 degree == 1 时必须是 no-op, 且**不改变返回类型**.
- I6 每个维度必须有: CPU 单测 (逻辑) + 多卡等价脚本 (数值).

---

## 3. 目标结构

不新增抽象层, 只新增两个**缝**:

```
+---------------------------------------------------------------+
|  CLI (HfArgumentParser)          trainer/config.py            |
|                        HybridMeshConfig (仍是唯一配置入口)     |
+---------------------------------------------------------------+
                              |  组装层读取; 不向下传
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
+---------------+   +-------------------+   +-----------------+
| parallel/tp   |   | parallel/cp_ep    |   | parallel/pp     |
| apply_tp(m,   |   | apply_cp_ep(m,    |   | apply_pp(m,     |
|  *, plan)     |   |  *, plan)         |   |  *, plan)       |
+---------------+   +-------------------+   +-----------------+
        |                     |                     |
        +---------------------+---------------------+
       每个 apply_* 读自己那一片 (plan.tp / plan.cp / plan.pp_split)
                              |
              [SEAM 1]  模型的并行声明 (纯数据)
                              |
+---------------------------------------------------------------+
|  models/registry.py                                           |
|    ModelSpec(                                                  |
|      name       : str                                          |
|      config_cls : type                                         |
|      plan       : ParallelPlan          <-- 纯数据, 无逻辑       |
|      attn_mask_type : str                                      |
|      state_dict_adapter : type | None                          |
|    )                                                           |
+---------------------------------------------------------------+
                              |
+---------------------------------------------------------------+
|  models/hf_wrapper.py    THE one HF wrapper                    |
|    forward(input_ids, *, positions, attention_masks) -> logits |
|    named_children() -> tok_embeddings / layers / norm / lm_head |
+---------------------------------------------------------------+
                              |
              [SEAM 2]  分布式运行时上下文 (线程局部)
                              |
+---------------------------------------------------------------+
|  parallel/context.py                                           |
|    dist_context(mesh)  # contextmanager                         |
|    tp_group() / cp_group() / dp_group() / ep_group()  -> pg|None|
|    tp_size()  / cp_size()  / pp_size()               -> int     |
+---------------------------------------------------------------+
                              |
+---------------------------------------------------------------+
|  models/common/*   (rope, masks, qkv, aux_loss, moe, dist_gemm)|
|    通过 tp_group() 等取值, 不接收 cfg, 不 import trainer        |
+---------------------------------------------------------------+
```

### 3.1 为什么是两个缝而不是两个抽象层

`Configurable` / `Module` 是**继承型**抽象: 你要用框架, 就必须继承它、注册它、
让它能 `traverse()` 你. 两个缝是**数据型 + 作用域型**:

- SEAM 1 是**一份数据**. 模型说"我这些 module path 按 colwise 切", 框架负责切.
  模型不需要 import 框架的任何东西, 也不需要继承任何基类.
- SEAM 2 是**一个作用域**. 进入 `dist_context` 之后, `tp_group()` 自然可答;
  离开后是 `None`. 不需要把 mesh 沿调用链传 8 层, 也不需要模块持有 group 状态.

这正好回答了最初的设计原则: 拿掉抽象之后, 应该用**数据 + 作用域**去补位,
而不是放任复杂度以"到处传参"和"靠类身份猜"的形式重新长出来.

---

## 4. 核心接口

### SEAM 1: 模型的并行声明

```python
# hpmesh/parallel/plan.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import torch


def _cut(t: torch.Tensor, dim: int, *, size: int, rank: int) -> torch.Tensor:
    """Keep this rank's slice of ``t`` along ``dim``. No collective: every rank
    starts from the same full weight, so each just drops the rest."""
    if t.shape[dim] % size != 0:
        raise ValueError(f"dim {dim} (size {t.shape[dim]}) not divisible by {size}")
    return torch.chunk(t.detach(), size, dim=dim)[rank].contiguous()


@dataclass(frozen=True)
class Colwise:
    """Output features split. The transpose is NOT cosmetic: the realizer stores
    [in, out/tp] because that is what the fused all-gather GEMM consumes, and
    that is exactly what ColwiseLinear does today. `cut` keeps that contract, so
    swapping the realizer does not change the weight layout."""
    def cut(self, weight, *, size, rank):    # [out, in] -> [in, out/tp]
        return _cut(weight.t().contiguous(), 1, size=size, rank=rank)


@dataclass(frozen=True)
class Rowwise:
    """Input features split: HF's [out, in] cut on dim 1, consumed as-is."""
    def cut(self, weight, *, size, rank):    # [out, in] -> [out, in/tp]
        return _cut(weight, 1, size=size, rank=rank)


@dataclass(frozen=True)
class Fused:
    """Fused QKV. Must cut on the HEAD dim, not the flat feature dim, or a rank
    ends up holding a fraction of a head rather than whole heads."""
    num_groups: int
    def cut(self, weight, *, size, rank): ...


@dataclass
class ParallelPlan:
    """What the MODEL declares. Pure data -- no tensors, no collectives, no logic."""

    tp: dict[str, Colwise | Rowwise | Fused] = field(default_factory=dict)
    """module path pattern -> how to cut it. Glob patterns, matched deepest-first."""

    ep: set[str] = field(default_factory=dict)
    """module paths holding expert weights that EP should shard."""

    pp_split: list[list[str]] = field(default_factory=list)
    """layer FQNs per pipeline stage."""

    cp: Literal["ulysses", "kv_all_gather"] | None = None
```

关键点: `ParallelPlan` **不是抽象基类** —— 它是一份数据. 模型侧**不 import 它**,
只是在 `ModelSpec.plan` 里放一个 dict 形状的声明:

```python
# 模型作者 (或 HP 适配层) 只需要写这个:

def hf_tp_plan(*, fused_qkv: bool = False) -> ParallelPlan:
    """The projections every HF llama-family decoder shares. Derived from HF's own
    ``model._tp_plan`` when present; this is the fallback for when it is not.

    ``fused_qkv`` is where the fused layout shows up as DATA instead of as a
    ``hasattr(linear, "wqkv")`` probe: with it, the three projections are declared
    as one ``Fused`` entry, and the engine knows to cut on the head dim.
    """
    if fused_qkv:
        return ParallelPlan(tp={"*.qkv_proj": Fused(num_groups=...)}, ...)
    return ParallelPlan(
        tp={
            "*.q_proj":    Colwise(),
            "*.k_proj":    Colwise(),
            "*.v_proj":    Colwise(),
            "*.o_proj":    Rowwise(),
            "*.gate_proj": Colwise(),
            "*.up_proj":   Colwise(),
            "*.down_proj": Rowwise(),
            "*.lm_head":   Colwise(),
        },
    )
```

**这就是那四个 `hasattr` / `isinstance` 的替代品.** 框架不再猜"这个 `nn.Linear`
是不是 qkv", 模型直接说了. 而这份声明可以用 HF 自己的 `model._tp_plan` 自动生成,
所以对接新模型仍然是一行.

### SEAM 2: 分布式运行时上下文

```python
# hpmesh/parallel/context.py

from __future__ import annotations
from contextlib import contextmanager
from threading import local

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

_TLS = local()
_OFF = object()


@contextmanager
def dist_context(mesh: DeviceMesh | None):
    """Make the current mesh answerable for the duration of the block.

    Nestable and restorable: a nested ``dist_context(None)`` (e.g. a single-rank
    reference pass inside a distributed test) hides the outer mesh, and the
    outer one comes back on exit. Without the save/restore, that reference pass
    would silently keep using the outer mesh's groups.
    """
    prev = getattr(_TLS, "mesh", _OFF)
    _TLS.mesh = mesh
    try:
        yield
    finally:
        _TLS.mesh = prev


def _group(name: str) -> dist.ProcessGroup | None:
    """The multi-rank process group for `name`, or None when inactive/absent.

    None (rather than a size-1 group) keeps every caller's "not active" branch
    honest: a collective on a size-1 group is a silent no-op, so a missing mesh
    would otherwise hide behind a run that merely computes the wrong answer.
    """
    mesh = getattr(_TLS, "mesh", None)
    if mesh is None or name not in (mesh.mesh_dim_names or ()):
        return None
    group = mesh.get_group(name)
    return group if group.size() > 1 else None


def _size(name: str) -> int:
    """Mesh axis size, or 1 when the axis is absent -- so `size() == 1` is the
    same question as "is this dimension off", and §6's no-op guard is free."""
    mesh = getattr(_TLS, "mesh", None)
    if mesh is None or name not in (mesh.mesh_dim_names or ()):
        return 1
    return mesh.size(name)


def tp_group():  return _group("tp")
def cp_group():  return _group("cp")
def ep_group():  return _group("ep")
def dp_group():  return _group("dp")

def tp_size():   return _size("tp")
def cp_size():   return _size("cp")
def ep_size():   return _size("ep")
def pp_size():   return _size("pp")
```

> 这个接口**基本已经存在了**, 只是散落在两处: `parallel/spmd_types.py` 的
> `spmd_mesh_group(axis)` / `spmd_mesh_size(axis)` 和 `parallel/cp_ep.py` 的
> `cp_group()`. 本设计做的是**把它们提到一个地方并统一命名**, 同时把"进入上下文"
> 从**从未被调用的** `set_spmd_meshes()` 改成 trainer 显式包一层. 今天
> `dist_gemm.py` 调 `current_spmd_mesh()` 永远拿到 `None`, 于是永远走未融合的
> 回退路径 —— 这是**静默的**, 正是这个 seam 要修的东西.

---

## 5. 收敛点: 模型层

### 5.1 裁决: 保留 `models/hf_wrapper.py`, 删除 `bundle.py` 的 wrapper

| | `bundle.py:HFModelWrapper` | `models/hf_wrapper.py:HFTransformerModel` |
|---|---|---|
| 存活 | **活** (trainer 在用) | **死** (只有测试) |
| 名字解析 | 每次调用 `_resolve()` 线性扫 | `__init__` 里解析一次存名字 |
| 层列表 | `.layers` 直通 | `.layers` + property setter (PP 可原地换 stage) |
| flex attention | 无 | `get_attention_masks()` (causal / block_causal) |
| CP 支持 | 无 | `set_cp_mesh()` 坐标标记 |
| 数值对拍 | 无 | `HF_BACKEND_LOGIT_DUMP` |
| logits 返回 | **返回 loss** | **返回 logits** |

**裁决: 取后者为唯一实现.** 理由不是"它功能多", 而是**返回类型决定分层**:

- 返回 `logits` 是正确的边界. loss 是训练策略 (是否 shift、是否 z-loss、
  是否带 aux loss 加权), 属于 trainer; 塞进 wrapper 会把 trainer 的一半逻辑
  搬进模型层.
- `get_attention_masks` 是 CP 与 packed-document 训练**必需**的, 而 CP 已经建好了
  (`parallel/cp_ep.py`), 没有 mask 装配它的等价脚本就跑不起来.
- PP 需要 property setter 来原地替换 stage.

`bundle.py` 只保留**构建职责** (`build_bundle`), 类名统一为 `HFTransformerModel`,
`HFModelWrapper` 这个名字从仓库消失.

### 5.2 归一后的接口

```python
# hpmesh/models/hf_wrapper.py

class HFTransformerModel(nn.Module):
    """One HF causal LM, presented under the names the parallel layer reads.

    Two jobs, both about translation:

    1. INPUT: HF nests the decoder under ``model.model`` and spells its parts
       differently across families (``embed_tokens``/``wte``, ``norm``/``ln_f``).
       The parallel layer wants flat names. Resolving once here -- instead of at
       every call site -- is what keeps that translation in ONE place.
    2. OUTPUT: return raw logits. Loss is the trainer's business.
    """

    def __init__(self, hf_model: nn.Module, *, attn_mask_type: str = "causal"): ...

    # -- 平行层读取的五个名字 ------------------------------------------------
    @property
    def tok_embeddings(self) -> nn.Module: ...
    @property
    def layers(self) -> nn.ModuleList: ...
    @layers.setter
    def layers(self, value) -> None: ...          # PP 原地换 stage
    @property
    def norm(self) -> nn.Module: ...
    @property
    def lm_head(self) -> nn.Module | None: ...
    @property
    def enable_weight_tying(self) -> bool: ...    # 按 identity 判, 不按 config 标志

    def get_attention_masks(self, positions: torch.Tensor): ...   # -> BlockMask

    def forward(self, input_ids, *, positions=None, attention_masks=None) -> Tensor:
        """Return logits (T, V). Deliberately NOT the loss: shifting labels and
        choosing an objective is training policy, and it belongs to the trainer.
        Once the wrapper returns logits, the parallel layer never has to know
        which loss is in use."""
```

### 5.3 并行声明的位置

`ParallelPlan` 由**模型注册表**提供, 与模型一起注册:

```python
# hpmesh/models/registry.py

@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_cls: type
    plan: ParallelPlan           # <-- SEAM 1
    attn_mask_type: str = "causal"
    state_dict_adapter: type | None = None


MODELS = {
    "llama":  ModelSpec("llama",  LlamaConfig,  plan=hf_tp_plan, ...),
    "qwen3":  ModelSpec("qwen3",  Qwen3Config,  plan=hf_tp_plan, ...),
    "gpt_oss": ModelSpec("gpt_oss", ..., plan=hf_tp_plan, ...),
}
```

这里顺手收掉了 `models/spec.py` 里那个死的 `HPModelSpec` (它带着
`pipelining_fn` / `post_optimizer_build_fn` 两个 hpmesh 用不上的 torchtitan 残留).

---

## 6. 一个 step 的完整分层视图

把上面的东西拼起来, 训练循环应该长这样 (注意配置的流动方向是**单向且显式**的):

```python
class Trainer:
    def __init__(self, cfg: HybridMeshConfig):
        self.rank, self.local_rank, self.world_size = init_distributed()
        self._seed_everything(cfg.seed, deterministic=cfg.deterministic)
        self.device = ...

        self.parallel_dims = ParallelDims.from_config(cfg.parallel, self.world_size)
        self.mesh = build_mesh(self.parallel_dims)

        spec = MODELS[cfg.model_name]
        self.model = HFTransformerModel(build_hf_model(spec, cfg, device=self.device))

        # ---- SEAM 2 建立一次: 之后每个 apply_* 自己从上下文里读度数 ----
        with dist_context(self.mesh):
            self.model = apply_tp(self.model, plan=spec.plan)
            self.model = apply_cp_ep(self.model, plan=spec.plan)
            self.model = apply_pp(self.model, plan=spec.plan)
            if cfg.compile:
                self.model = torch.compile(self.model)
            # FSDP 是唯一的例外: 它要构造 dp_replicate x dp_shard x cp x tp 的
            # 多维 storage mesh, 单靠一个 dp_group() 表达不了, 所以它仍然接收
            # parallel_dims. 这是真实需求, 不是没收拾干净.
            self.model = apply_fsdp(
                self.model,
                self.parallel_dims,
                reshard=cfg.parallel.fsdp_reshard_after_forward,
                symm_mem=cfg.parallel.enable_fsdp_symm_mem,
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, ...)

    def train_step(self, step: int) -> float:
        batch = self._dp_slice(self._make_batch(step))
        self.optimizer.zero_grad(set_to_none=True)
        with dist_context(self.mesh):                          # <-- SEAM 2
            logits = self.model(batch.input_ids, positions=batch.positions)
            loss = cross_entropy(logits, batch.labels)          # 训练策略留在 trainer
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())
```

**关键简化: `apply_*` 连度数参数都不需要了.** 因为 `tp_size()` 在没有 tp 轴时
返回 `1`, "这个维度开没开"和"这个维度多大"是同一个查询 —— I5 (degree==1 时 no-op)
从一条需要人工遵守的纪律, 变成**从上下文里读出来的事实**:

```python
def apply_tp(model, *, plan: ParallelPlan):
    if tp_size() == 1:
        return model                       # I5 自然成立, 不需要调用点判断
    for pattern, sharding in plan.tp.items():
        ...
```

每个 `apply_*` 接收**同一个** `ParallelPlan`, 但只读自己那一片 (`plan.tp` /
`plan.cp` / `plan.pp_split`). 这比"每个函数收一个不同的子对象"更简单:
调用方不必记住谁要哪一片, 而"这个函数只看这一片"由实现本身保证 ——
一个只读 `plan.tp` 的函数无法偷偷依赖 `plan.ep`.

注意这两个缝的**分工不是对称的**, 这是刻意设计:

| | 谁传给谁 | 为什么 |
|---|---|---|
| SEAM 1 (`plan`) | **调用方显式传参** | 它是**意图** "我声明这些投影这么切", 必须能被测试直接构造 (`apply_tp(m, plan={...})`), 也必须能与上下文不一致 (例如测一条单 rank 路径) |
| SEAM 2 (`context`) | **隐式作用域** | 它是**环境** "此刻有哪些进程组", 沿调用链穿透到底层 `models/common/*`, 显式传递会让每个函数都多三个参数 |

把 `plan` 也做成上下文 = 丢失显式性; 把 context 也做成参数 = 丢失穿透性.
所以是**一个显式 + 一个隐式**, 不是"两个都隐式".

对比现状的三点改进:

1. `parallel/*` 不再反向 import `trainer/config.py` (I1) ——
   `parallel/` 下现有的 5 个 (`parallelize_hf`, `tp`, `fsdp_wrap`, `pp`, `cp_ep`)
   全部消失. (`mesh.py` 与 `bundle.py` 仍然接收 config —— 它们本来就是组装层,
   不在 I1 的约束范围内.)
2. `apply_tp` 不再 `isinstance(nn.Linear)` 扫模型猜目标, 而是读 `plan.tp` (I2).
3. `dist_context` 让 `dist_gemm` / `aux_loss` / `moe` 在没有 mesh 时**明确**退化,
   在有 mesh 时**真的生效** —— 今天第二条永远不成立.

---

## 7. 迁移路线

每一阶段独立可跑、可对拍. **不允许**跨阶段混做 —— 这是唯一的节奏约束.

### 阶段 0: 收敛与删除 (零风险, 先做)

- 删除 §1.4 表中**被取代**的死模块 (`hf_sharding`, `placements`, `spec.py`,
  `moe_swap.py`, `moe_probe.py`, 空的 `protocols/`).
- 修掉指向它们的文档字符串引用 (`parallelize_hf.py`, `models/common/embedding.py`,
  `models/common/rope.py`, `pyproject.toml` 的注释, `README.md` 的代码地图).
- **保留** `pipeline.py` (PP 的第一半, 见 §1.6) 与 `models/common/embedding.py`.
- **不动** `hf_wrapper.py` / `trainer.py` / `bundle.py` / 任何测试 —— 阶段 0 是纯删除.
  (所以原计划里"把 `test_hf_wrapper.py` 并入 `test_core.py`"挪到阶段 1: 那一步的前提是
  CPU 上能真的跑 `HFTransformerModel`, 那要等阶段 1 的 attention 回退.)
- **验收**: `pytest tests/` 由 206 降到 181 —— 差值 25 恰好等于被删的
  `test_hf_sharding`(5) + `test_moe_probe`(6) + `test_moe_swap`(14); 其余 6 个文件
  逐测试点名一一对应, `ruff` 干净, 行数净减 (实测 -2014 / +11).
- **对拍**: 无数值变化. 单设备 `--steps 2` 与删除前逐位相同
  (`4.858892` / `4.855443`).
- **勘误**: §1.4 表里的 `_utils.py` 实际不在树里 (也从未提交过), 无需删除;
  `models/spec.py` 的真实引用者只有 `rope.py` 的文档字符串, 写成"被 `registry.py` 取代"
  是超前表述 —— `registry.py` 要到阶段 3 才存在.

### 阶段 1: 统一模型层 (收敛 §5.1)

- `HFTransformerModel` 成为唯一 wrapper; `forward` 返回 logits.
- trainer 承担 loss 计算 (从 `labels` 移位 + 交叉熵).
- `bundle.py` 删除 (构建职责并入 `trainer`), `tests/test_hf_wrapper.py` 并入
  `tests/test_core.py`.
- **验收前必须解决的前置**: 今天 `HFTransformerModel` 在**无 CUDA 的机器上直接跑不起来**:

  ```
  InductorError: NotImplementedError: torch.compile on current platform is
  not supported for CPU.  target: flex_attention
  ```

  (flex attention 走 inductor, inductor 没有 CPU 后端.) 所以把 trainer 切到
  `HFTransformerModel` 的同时, 必须加一个 `"_attn_implementation"` 回退:
  无 CUDA 时用 `"sdpa"`. 这也解释了为什么原计划里"阶段 0 就把 `test_hf_wrapper.py`
  并入 `test_core.py`"走不通 —— 那次合并已经顺带把回退带进来了.
- **验收 (已修正)**: 原计划的"loss 逐位一致"**不成立**, 因为回退把 attention
  从 flex 换成 sdpa, 数值必然变. 阶段 1 的正确门禁是:
  1. CPU 上 `HFTransformerModel` 的前向能跑通 (回退生效);
  2. 回退到 sdpa 后, 单设备 `--steps N` 的 loss **收敛**(单调下降且量级正常),
     不是逐位相等;
  3. 在 **CUDA 机器**上补一次带 flex 的逐位对拍 —— 那台机器上回退不触发,
     "wrapper 返回 loss -> trainer 算 loss" 这条改动本身必须零数值差异.
- **理由**: 把"接口重构"和"attention 后端切换"混在一次对拍里, 会得到一个
  既无法归因、又必然失败的验收标准.

### 阶段 2: SEAM 2 — 分布式上下文

- 新建 `parallel/context.py`; `spmd_mesh_group` 与 `cp_ep.cp_group` 统一到它.
- trainer 在 `train_step` 里包一层 `dist_context`.
- 删掉从未被调用的 `set_spmd_meshes()`.
- **验收**: 新增单测 —— 无上下文时 `tp_group() is None`, 有上下文时返回对应 pg.
- **对拍**: EP/CP 等价脚本结果**不变** (`3.271e-07` / `2.384e-07`).
  注意此时 `dist_gemm` 会**第一次真的走到融合路径**, 所以这条对拍必须有 TP>1 的配置.

### 阶段 3: SEAM 1 — 并行声明

- 新建 `parallel/plan.py` + `models/registry.py`.
- `apply_tp` 改为读 `plan`; 移除 `isinstance(nn.Linear)` 扫描.
- `parallel/{tp,cp_ep,pp,fsdp_wrap}.py` 逐个去掉 `from ..trainer.config import`.
- **验收**: I1 / I2 用一条单测钉死 (import 图断言, 见 §8).
- **对拍**: TP 等价脚本, 逐位一致.

### 阶段 4: PP

- 唯一还缺的维度. `pipeline.py` 的 stage 切分已在 `pp.py` 里描述, 缺的是 schedule.
- 先做最简单的 1F1B, 用 `_all_reduce_loss` 同款的"每 rank 必须进入"纪律.
- **验收**: PP=2 等价脚本, 对比 PP=1.

### 阶段 5: 把声明式 model 适配变成可验证的

- `ParallelPlan` 增加 `validate(model)`, 在 `apply_tp` 前检查每个 pattern **恰好命中**
  预期数量的 module; 未命中 = 报错, 不是静默跳过.
- **理由**: 今天 `_match` 没命中就什么都不做, 于是"TP 没生效"和"TP 生效了"
  在日志上无法区分. 这是本项目里最危险的一类静默失败.

---

## 8. 不变式如何被机器守住

不变量写在文档里等于没写. 三条可执行的断言:

```python
# tests/test_layering.py
"""Architecture tests: the rules in §2 that nothing else would catch.

These do not test behaviour. They test that the LAYERING still holds, because a
reverse dependency compiles, imports and runs perfectly -- it only hurts later,
when the two layers can no longer be changed independently.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> list[str]:
    """Absolute-ish module strings this file imports."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):        # import ... / from ... import
            out.append(node.module or "")
    return out


def test_models_do_not_import_trainer():
    """I1: the model layer must not depend on the trainer layer."""
    for path in Path("hpmesh/models").rglob("*.py"):
        for mod in _imports(path):
            assert "trainer" not in mod, f"{path} imports {mod}"


def test_parallel_does_not_import_concrete_models():
    """I2: parallelism must not know about qkv / moe / rope / grouped_experts."""
    banned = ("qkv", "moe", "rope", "activation", "grouped_experts")
    for path in Path("hpmesh/parallel").rglob("*.py"):
        for mod in _imports(path):
            assert not any(b in mod for b in banned), f"{path} imports {mod}"


def test_models_do_not_import_the_framework():
    """I3: the model side declares by DATA (or by nothing), never by calling in.

    `rope.py` and `qkv.py` today import only torch -- they are copyable into any
    project. This is the property that makes "directly wrap HuggingFace" work,
    so it is worth pinning rather than assuming.
    """
    allowed = ("torch", "spmd_types", "hpmesh.models")   # same layer is fine
    for path in Path("hpmesh/models").rglob("*.py"):
        for mod in _imports(path):
            if mod.startswith("hpmesh.") or mod in ("hpmesh",):
                assert mod.startswith(allowed), f"{path} imports {mod}"


def test_every_apply_is_a_noop_when_its_axis_is_off():
    """I5: the same trainer must run from 1 device to a full mesh.

    With SEAM 2 in place the degree is read from the context, so "off" is just
    an empty context -- the guard is structural, not a parameter the caller has
    to remember to pass.
    """
    model = tiny_model()
    with dist_context(None):
        assert apply_tp(model, plan={}) is model
        assert apply_cp_ep(model, plan=ParallelPlan()) is model
```

> 注意 `test_models_do_not_import_the_framework` 的意义: 它把 §3.1 的那句
> "模型不需要 import 框架的任何东西" 从主张变成了可执行的断言. 如果哪天
> `qkv.py` 开始 `from hpmesh.parallel import tp_group`, 这条测试会变红 ——
> 而那正意味着 SEAM 2 被用错了地方 (模型组件不该依赖运行时上下文, 只该依赖
> 传给它的张量和 group).

配合已有的数值门禁 (EP / CP 等价脚本 + loss 逐位对拍), 形成三层:

| 层 | 门禁 | 抓住什么 |
|---|---|---|
| 结构 | `test_layering.py` | 分层退化 (反向依赖) |
| 逻辑 | 206 个 CPU 单测 | 组件正确性 |
| 数值 | `ep_equivalence.py` / `cp_equivalence.py` / loss 对拍 | 分布式正确性 |

---

## 9. 与 torchtitan 的对照

最后回答"拿掉抽象之后, 我们用什么补位".

| torchtitan 的机制 | 它真正解决的问题 | hpmesh 的替代 |
|---|---|---|
| `Configurable` + `Config.build()` | 配置如何到达被构造的模块 | **dataclass 显式传参** (已做) + `ModelSpec.plan` (数据) |
| `Configurable.traverse()` | 从顶层配置找到嵌套的模型配置 | 不需要 —— 模型配置本就是普通 dataclass |
| `config_utils` (448 行) | 配置的解析/覆盖/校验管线 | `HfArgumentParser` + dataclass `__post_init__` |
| `Module.parallelize()` | 模块如何声明自己的并行方式 | **`ParallelPlan` 纯数据** (SEAM 1) |
| `Module.remat_region_name()` | 激活重算的切分点 | `torch.utils.checkpoint` 参数, 或不做 |
| `Module._init_self_buffers()` | 元设备上的 buffer 初始化 | 不存在 —— HF 模型自带 buffer |
| `spmd.local_map` / `assert_type` | DTensor 的运行时类型检查 | `dist_context()` 的作用域 + `None` 语义 (SEAM 2) |
| `ModelSpec.traverse` | 让 override 树能到达模型配置 | `ModelSpec.plan` (同样是一份数据, 但不需继承) |

**一句话**: torchtitan 用"让模块自己描述自己"解决耦合; hpmesh 用"让模型交出声明,
让框架持有作用域"解决同一问题. 前者要求模型继承框架, 后者不要求 —— 这正是
"能直接对接 HuggingFace" 这个目标所要求的形状.

---

## 10. 风险与不做的部分

**风险**

- R1 **PP 是唯一真正的缺口**, 且是五维里最难的一维 (schedule + 微批 + P2P).
  阶段 4 之前, `pp > 1` 会明确抛 `NotImplementedError`, 这是对的.
- R2 阶段 1 (loss 搬到 trainer) 会触碰梯度路径. 必须以逐位 loss 为准入,
  否则整个"HF 的 loss 就是我们的 loss"这一层保障会失效.
- R3 阶段 2 会让 `dist_gemm` 首次真的生效 —— 之前它一直走回退路径.
  这意味着**之前没有测过的代码路径会被点亮**, 必须有 TP>1 的对拍覆盖.

**明确不做**

- 不重新引入任何基类 / 注册表式继承 / `traverse()`.
- 不移植 `config_utils.py` (448 行, 40 处 `Configurable`)。
- 不做 `nn_modules.py` 那类零行为包装 (`class Conv1d(nn.Conv1d, Module)`).
- 不为 HP 模型新建并行实现 —— 复用 HF 的 `_tp_plan` / `_pp_plan`, 只做翻译.
- 不实现 torchao / DeepEP / HybridEP 后端 (需要 GPU-only 第三方库, 本机装不了也测不了;
  它们改变的是"token 怎么跨 rank", 不是路由契约).

---

## 附: 一句话总结

> hpmesh 已经证明了"拿掉 `Configurable` 和 `Module` 之后, 分布式训练框架可以更简单"。
> 但它还没解决这两个抽象**原本承担的那两个问题**: 配置如何到达模块, 模块如何声明并行。
> 本设计用**一份数据 (`ParallelPlan`)** 和**一个作用域 (`dist_context`)** 补位,
> 不引入新的继承层次。补完之后, `models/common/` 才真正独立于 trainer,
> `apply_*` 才真正不认识具体模型 —— 而这正是让框架能"直接对接 HF"的前提。
