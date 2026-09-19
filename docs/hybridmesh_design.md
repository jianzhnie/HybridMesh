# HP 并行训练框架设计：两个缝

> 本文只谈**设计**：现在缺什么、补什么、怎么验收。
> 结构审计（哪些代码该删、目录怎么分）见 `hpmesh_structure.md`。
>
> 所有数字都是对 `hpmesh/**/*.py` 的 AST 实测，不是估计。

---

## 0. 结论

hpmesh 拿掉了 TorchTitan 的 `Configurable` 与 `Module` 两个抽象层，换来了一个
明显更短的框架。但那两个抽象**原本承担的两件事**没有被替代，它们以更隐晦的形式
重新长了出来：

| torchtitan 用抽象解决的 | hpmesh 今天用什么顶 | 长成了什么形状 |
|---|---|---|
| 配置如何到达模块 | 到处传整个 `HybridMeshConfig` | `parallel/` 反向依赖 `trainer/` |
| 模块如何声明自己的并行方式 | **模块类的身份**（`isinstance` 扫描） | 框架反向依赖具体模型内部结构 |

**本文主张一次收敛：不引入任何新抽象层，只引入两个缝。**

- **SEAM 1（数据）**：模型用一份纯数据声明"哪些投影该怎么切"，取代 `isinstance` 扫描。
- **SEAM 2（作用域）**：一个线程局部的分布式上下文，取代沿调用链传 mesh / group。

两个缝的分工刻意不对称：**一个显式传参，一个隐式作用域**（理由见 §4.3）。
收敛之后删掉被取代的死代码，把两套并存的模型包装归一。

---

## 1. 设计原则

拿掉抽象不等于拿掉复杂度，只等于**把复杂度换成另一种形式**。hpmesh 选择的形式是：

**目标 (Go)**

| | 目标 | 为什么它值得写死 |
|---|---|---|
| G1 | 任意 HF `AutoModelForCausalLM` 不改模型代码即可并行 | 这是"直接对接 HuggingFace"的全部内容 |
| G2 | 配置只沿一个方向流动（`CLI -> Config -> 顶层显式传参`） | 单向才不会出现"改 A 要动 B" |
| G3 | 每个 `apply_*` 只依赖它的契约 | 不依赖 `trainer` 层，也不依赖具体模型类 |
| G4 | 一次加一个维度，每个维度独立文件 + 独立单测 + 独立对拍脚本 | 五维并行无法一次想清楚 |
| G5 | 任何非计算改动必须逐位一致 | 已有标准，见 `CLAUDE.md` 的验证要求 |

**不变量 (Invariants) —— 不许再退化回去的红线**

| | 不变量 |
|---|---|
| I1 | `models/common/` 不得 import `trainer/` |
| I2 | 任意并行模块不得 import 具体模型模块（`qkv` / `moe` / `rope` / ...） |
| I3 | 模型侧向框架暴露的信息只能是**数据**，不能是"调用框架的内部函数" |
| I4 | 新增/重写的注释与文档字符串只用 ASCII |
| I5 | `apply_X` 在对应 degree == 1 时必须是 no-op，且不改变返回类型 |
| I6 | 每个维度必须有：CPU 单测（逻辑）+ 多卡等价脚本（数值） |

---

## 2. 现状实测

### 2.1 规模与分层

```
50 个模块, 8310 行

trainer     4 模块, 1025 行   config / trainer / train / __init__
models     17 模块, 3404 行   hf_wrapper + common/{16 modules, ~2900 行}
parallel   17 模块, 3030 行   tensor_parallel/ fsdp2/ pepeline_parallel/ cp_ep
                              context_parallel/ deepep/ + parallel_dims 等
components  3 模块,  419 行   loss / checkpointer
datasets    2 模块,   97 行
utils       4 模块,  209 行
mesh.py     1 模块,   90 行   <-- 悬在包的根上
```

**分层方向已经是对的**：`trainer -> parallel -> models` 单向，反向零依赖。
问题不在方向，在**层内没有边界**。

### 2.2 诊断一：64% 的包体从 trainer 不可达

从真实入口（`hpmesh/__main__.py` + `hpmesh/trainer/train.py`）做 AST 可达性分析：

```
可达    16/50 模块
不可达  34 模块 / 5308 行 = 64%
```

按目录拆开看，问题集中在两个地方：

| 目录 | 模块 | 行数 | 可达模块 | 可达行数 |
|---|---|---|---|---|
| `models/` | 17 | 3404 | 2 | 793 |
| `parallel/` | 17 | 3030 | 2 | 658 |
| `trainer/` | 4 | 1025 | 4 | 1025 |
| `components/` | 3 | 419 | 1 | 94 |
| `utils/` | 4 | 209 | 3 | 209 |

**这 64% 不是废弃代码，是"已完成但未接线"的能力**：`models/common/` 有 25 个
rope 测试、28 个 aux_loss 测试、逐位对拍过的 EP/CP 等价脚本。删掉它们等于把
后续阶段的工作量翻倍。**它们的问题不是布局，是接线。**

> 判别"不可达"的两种含义，是读这份代码时唯一需要小心的点：
>
> | 不可达的原因 | 例子 | 处理 |
> |---|---|---|
> | **被取代** (superseded) | `hf_sharding` / `placements` / `moe_probe` / `moe_swap` | **删除** |
> | **还没接线** (not yet wired) | `pepeline_parallel/pipeline.py`（296 行，完整可用）；`models/common/embedding.py`（vocab-parallel embedding） | **保留**，由对应阶段接上 |

### 2.3 诊断二：并行模块反向依赖 trainer

实测 `parallel/` 里读配置字段的位置：

```
parallel/cp_ep.py:208                   cfg.cp
parallel/tensor_parallel/tp.py:218      cfg.tp
parallel/fsdp2/fsdp_wrap.py:98          cfg.parallel.fsdp_reshard_after_forward
parallel/parallelize_hf.py:78           cfg.compile
```

**没有任何一个函数需要超过 1 个字段，但每一个都接收整个 `HybridMeshConfig`。**
后果是具体的：

- `parallel/` 无法脱离 `trainer/` 单独测试；
- `apply_tp(model, mesh, cfg)` 这种**本就该是纯函数**的东西被迫收一个大对象；
- 单元测试必须伪造一个 `HybridMeshConfig` 才能测一行权重切分。

而 `tp` / `cp` 这些数字，`parallel_dims`（它自己就是从 config 派生的）**已经算好了**。
同一个事实存了两份。

### 2.4 诊断三：框架靠 `isinstance` 猜模型结构

```python
# parallel/tensor_parallel/tp.py:229
if isinstance(module, nn.Linear):
    spec = _match(sharding_plan, module_path)
```

这是**唯一一处"框架主动扫描模型来找切分目标"**，也恰好是 HP 对接价值的本体与
脆弱点的交汇处：框架必须知道某个 `nn.Linear` 到底是 qkv 还是普通投影，只能靠
"它匹配了哪个 glob 模式"来猜。

注意 `_match` 用的是 HF 自己的 `_tp_plan`（一份纯数据）作为默认 plan —— **方向是对的**，
缺的是让模型侧明确声明、并让"没命中"可报错（见 §5 阶段 5）。

### 2.5 诊断四：`torchtitan` 遗产与同名冲突

| 问题 | 实测 |
|---|---|
| 死文件 | `parallel/sharding.py`（125 行），唯一引用者是 `spmd_types.py:74` 的**函数内 import**，而那个调用者自己也是死的 |
| 同名类 | `ShardingConfig` 在 `sharding.py:32`（DTensor placement 声明，死）与 `tensor_parallel/tp.py:128`（TP 切分 kind，活）各有一份 —— 任何 grep 都会得到两个答案 |
| 死 API | `spmd_types.py`（529 行）的 17 个导出里 **11 个代码使用数为 0**，全部是「类型检查 / state-dict 转换」家族，服务于**已被删除的 `Module.parallelize()` 抽象** |

---

## 3. 目标结构

不新增抽象层，只新增两个缝：

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
| tensor_paralle|   | cp_ep             |   |  pp (待建)      |
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

---

## 4. 两个缝

### 4.1 SEAM 1：模型的并行声明

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
    [in, out/tp] because that is what the fused all-gather GEMM consumes. `cut`
    keeps that contract, so swapping the realizer does not change the layout."""
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

    ep: set[str] = field(default_factory=set)
    """module paths holding expert weights that EP should shard."""

    pp_split: list[list[str]] = field(default_factory=list)
    """layer FQNs per pipeline stage."""

    cp: Literal["ulysses", "kv_all_gather"] | None = None
```

关键点：`ParallelPlan` **不是抽象基类** —— 它是一份数据。模型侧**不 import 它**，
只是在 `ModelSpec.plan` 里放一个 dict 形状的声明：

```python
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

**这就是 `tp.py:229` 那个 `isinstance` 的替代品**：框架不再猜"这个 `nn.Linear` 是不是
qkv"，模型直接说了。而这份声明可以用 HF 自己的 `_tp_plan` 自动生成，所以对接新模型
仍然是一行。

### 4.2 SEAM 2：分布式运行时上下文

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
    same question as "is this dimension off", and I5's no-op guard is free."""
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

这个接口**基本已经存在了**，只是散落在两处：`parallel/spmd_types.py` 的
`spmd_mesh_group(axis)` / `spmd_mesh_size(axis)`，和 `parallel/cp_ep.py` 的
`cp_group()`。本设计做的是**把它们提到一个地方并统一命名**，同时把"进入上下文"从
**从未被调用的** `set_spmd_meshes()` 改成 trainer 显式包一层。

> 今天 `dist_gemm.py` 调 `current_spmd_mesh()` 永远拿到 `None`，于是永远走未融合的
> 回退路径 —— 这是**静默的**，正是这个缝要修的东西。

### 4.3 为什么是两个缝，而不是两个抽象层

`Configurable` / `Module` 是**继承型**抽象：你要用框架，就必须继承它、注册它、
让它能 `traverse()` 你。两个缝是**数据型 + 作用域型**：

- SEAM 1 是**一份数据**。模型说"我这些 module path 按 colwise 切"，框架负责切。
  模型不需要 import 框架的任何东西，也不需要继承任何基类。
- SEAM 2 是**一个作用域**。进入 `dist_context` 之后，`tp_group()` 自然可答；
  离开后是 `None`。不需要把 mesh 沿调用链传 8 层。

两个缝的分工**刻意不对称**：

| | 谁传给谁 | 为什么 |
|---|---|---|
| SEAM 1 (`plan`) | **调用方显式传参** | 它是**意图**："我声明这些投影这么切"。必须能被测试直接构造（`apply_tp(m, plan={...})`），也必须能与上下文不一致（例如测一条单 rank 路径） |
| SEAM 2 (`context`) | **隐式作用域** | 它是**环境**："此刻有哪些进程组"。它沿调用链穿透到底层 `models/common/*`，显式传递会让每个函数都多三个参数 |

把 `plan` 也做成上下文 = 丢失显式性；把 context 也做成参数 = 丢失穿透性。
所以是**一个显式 + 一个隐式**，不是"两个都隐式"。

### 4.4 一个 step 的完整分层视图

```python
class Trainer:
    def __init__(self, cfg: HybridMeshConfig):
        self.parallel_dims = build_parallel_dims(cfg, self.world_size)
        self.mesh = build_mesh(self.parallel_dims)

        spec = MODELS[cfg.model_name]
        self.model = HFTransformerModel(build_model_config_for(cfg))

        # ---- SEAM 2 建立一次: 之后每个 apply_* 自己从上下文里读度数 ----
        with dist_context(self.mesh):
            self.model = apply_tp(self.model, plan=spec.plan)
            self.model = apply_cp_ep(self.model, plan=spec.plan)
            if cfg.compile:
                self.model = torch.compile(self.model)
            # FSDP 是唯一的例外: 它要构造 dp_replicate x dp_shard x cp x tp 的
            # 多维 storage mesh, 单靠一个 dp_group() 表达不了, 所以它仍然接收
            # parallel_dims. 这是真实需求, 不是没收拾干净.
            self.model = apply_fsdp(self.model, self.parallel_dims, ...)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, ...)

    def train_step(self, step: int) -> float:
        batch = self._dp_slice(self._make_batch(step))
        self.optimizer.zero_grad(set_to_none=True)
        with dist_context(self.mesh):                          # <-- SEAM 2
            logits = self.model(batch.input_ids, positions=batch.positions)
            loss = cross_entropy_loss(logits, batch.labels)     # 训练策略留在 trainer
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())
```

**关键简化：`apply_*` 连度数参数都不需要了。** 因为 `tp_size()` 在没有 tp 轴时返回
`1`，"这个维度开没开"和"这个维度多大"变成**同一个查询** —— I5（degree==1 时 no-op）
从一条需要人工遵守的纪律，变成**从上下文里读出来的事实**：

```python
def apply_tp(model, *, plan: ParallelPlan):
    if tp_size() == 1:
        return model                       # I5 自然成立, 不需要调用点判断
    for pattern, sharding in plan.tp.items():
        ...
```

每个 `apply_*` 接收**同一个** `ParallelPlan`，但只读自己那一片。这比"每个函数收一个
不同的子对象"更简单：调用方不必记住谁要哪一片，而"这个函数只看这一片"由实现本身保证
—— 一个只读 `plan.tp` 的函数无法偷偷依赖 `plan.ep`。

---

## 5. 迁移路线

每一阶段独立可跑、可对拍。**不允许跨阶段混做 —— 这是唯一的节奏约束。**

### 阶段 0：收敛与删除（零风险，先做）

- 删除被取代的死模块：`parallel/sharding.py`、`parallel/hf_sharding.py`、
  `parallel/placements.py`、`models/spec.py`、`models/moe_swap.py`、
  `models/moe_probe.py`、空的 `protocols/`。
- 切掉 `spmd_types.py` 的 11 个零使用导出（§2.5），一并消灭同名 `ShardingConfig`。
- 修掉指向它们的文档字符串引用（`moe.py:380`、`test_aux_loss.py:450` 都还在提
  `set_spmd_meshes`）。
- **保留** `pepeline_parallel/pipeline.py`（PP 的前半段）与 `models/common/embedding.py`。
- **验收**：`ruff` 干净；pytest 通过数只减少被删测试的数量；行数净减。
- **对拍**：无数值变化。单设备 `--steps 2` 与删除前**逐位相同**。

### 阶段 1：统一模型层 — 已完成

- `HFTransformerModel` 成为唯一 wrapper；`forward` 返回 logits（loss 归 trainer）。
- `bundle.py` 删除，构建职责并入 `build_model_config_for`。
- **必须解决的前置**：`HFTransformerModel` 在**无 CUDA 的机器上跑不起来** —— flex
  attention 走 inductor，而 inductor 没有 CPU 后端。解法是 `_flex_supported()`：
  无 CUDA 时把 `_attn_implementation` 设为 `"sdpa"`。
- **回退不是"算术降级"**，真相比这微妙：HF 的 sdpa 包装器**只要 mask 存在就忽略
  `is_causal`，改从 mask 推因果**。所以把 `BlockMask` 传下去会静默关掉 mask。正确做法
  是**什么都不传**，让 HF 走它自己的路径。这段逻辑被收进一个显式的缝：
  `HFTransformerModel._apply_attention(positions, masks)` —— 角色固定（永远经某个
  attention 实现，永远喂它一个描述可注意范围的 mask），而"某个后端想要什么形状的 mask"
  在这里被隔离。真正失去的能力只有 **packed-document masking**，且这个缺口不静默：
  位置出现回退时 `_apply_attention` 直接 `ValueError`。
- **验收**：trainer 的 `_loss` 与 HF 自己的 `.loss` **逐位相等**（`abs diff = 0.0`）；
  固定一个 batch 过拟合 200 步 `4.8516 -> 0.9474`。
- **欠账**：CUDA 上补 flex 逐位对拍（本机无 CUDA）。

### 阶段 2：SEAM 2 — 分布式上下文

- 新建 `parallel/context.py`；`spmd_mesh_group` 与 `cp_ep.cp_group` 统一到它。
- trainer 在 `train_step` 里包一层 `dist_context`。
- 删掉从未被调用的 `set_spmd_meshes()`。
- **验收**：新增单测 —— 无上下文时 `tp_group() is None`，有上下文时返回对应 pg。
- **对拍**：EP/CP 等价脚本结果**不变**（`3.271e-07` / `2.384e-07`）。
  注意此时 `dist_gemm` 会**第一次真的走到融合路径**，所以这条对拍必须有 TP>1 的配置。

### 阶段 3：SEAM 1 — 并行声明

- 新建 `parallel/plan.py` + `models/registry.py`。
- `apply_tp` 改为读 `plan`，移除 `tensor_parallel/tp.py:229` 的 `isinstance(nn.Linear)` 扫描。
- `parallel/{tp,cp_ep,pp,fsdp_wrap}.py` 逐个去掉 `from ..trainer.config import`。
- **验收**：I1 / I2 用 import 图断言钉死（§6）。
- **对拍**：TP 等价脚本，逐位一致。

### 阶段 4：PP

- 唯一还缺的维度。`pepeline_parallel/pipeline.py` 的 stage 切分已经可用，缺的是
  schedule（今天 `parallelize_hf_transformers` 对 `pp > 1` 直接 raise）。
- 先做最简单的 1F1B，用"每 rank 必须进入"的纪律对齐 loss。
- **验收**：PP=2 等价脚本，对比 PP=1。

### 阶段 5：让声明式适配可验证

- `ParallelPlan` 增加 `validate(model)`，在 `apply_tp` 前检查每个 pattern **恰好命中**
  预期数量的 module；未命中 = 报错，不是静默跳过。
- **理由**：今天 `_match` 没命中就什么都不做，于是"TP 没生效"和"TP 生效了"在日志上
  无法区分。这是本项目里最危险的一类静默失败。

---

## 6. 不变量如何被机器守住

不变量写在文档里等于没写。四条可执行的断言：

```python
# tests/test_layering.py
"""Architecture tests: the rules in §1 that nothing else would catch.

These do not test behaviour. They test that the LAYERING still holds, because a
reverse dependency compiles, imports and runs perfectly -- it only hurts later,
when the two layers can no longer be changed independently.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> list[str]:
    """Absolute-ish module strings this file imports. AST, not grep: a docstring
    mentioning a module name is not a dependency."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
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
    allowed = ("torch", "hpmesh.models")   # same layer is fine
    for path in Path("hpmesh/models").rglob("*.py"):
        for mod in _imports(path):
            if mod.startswith("hpmesh."):
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

> `test_models_do_not_import_the_framework` 的意义：它把 §4.3 那句"模型不需要 import
> 框架的任何东西"从主张变成了可执行的断言。如果哪天 `qkv.py` 开始
> `from hpmesh.parallel import tp_group`，这条测试会变红 —— 而那正意味着 SEAM 2 被
> 用错了地方（模型组件不该依赖运行时上下文，只该依赖传给它的张量和 group）。

配合已有的数值门禁，形成三层：

| 层 | 门禁 | 抓住什么 |
|---|---|---|
| 结构 | `tests/test_layering.py` | 分层退化（反向依赖） |
| 逻辑 | 204 个 CPU 单测 | 组件正确性 |
| 数值 | `ep_equivalence.py` / `cp_equivalence.py` / loss 对拍 | 分布式正确性 |

---

## 7. 与 torchtitan 的对照

最后回答"拿掉抽象之后，我们用什么补位"。

| torchtitan 的机制 | 它真正解决的问题 | hpmesh 的替代 |
|---|---|---|
| `Configurable` + `Config.build()` | 配置如何到达被构造的模块 | **dataclass 显式传参**（已做）+ `ModelSpec.plan`（数据） |
| `Configurable.traverse()` | 从顶层配置找到嵌套的模型配置 | 不需要 —— 模型配置本就是普通 dataclass |
| `config_utils`（448 行） | 配置的解析/覆盖/校验管线 | `HfArgumentParser` + dataclass `__post_init__` |
| `Module.parallelize()` | 模块如何声明自己的并行方式 | **`ParallelPlan` 纯数据**（SEAM 1） |
| `Module.remat_region_name()` | 激活重算的切分点 | `torch.utils.checkpoint` 参数，或不做 |
| `Module._init_self_buffers()` | 元设备上的 buffer 初始化 | 不存在 —— HF 模型自带 buffer |
| `spmd.local_map` / `assert_type` | DTensor 的运行时类型检查 | `dist_context()` 的作用域 + `None` 语义（SEAM 2） |
| `ModelSpec.traverse` | 让 override 树能到达模型配置 | `ModelSpec.plan`（同样是一份数据，但不需继承） |

**一句话**：torchtitan 用"让模块自己描述自己"解决耦合；hpmesh 用"让模型交出声明、
让框架持有作用域"解决同一问题。前者要求模型继承框架，后者不要求 —— 这正是
"能直接对接 HuggingFace"这个目标所要求的形状。

---

## 8. 风险与明确不做的部分

**风险**

| | 风险 | 说明 |
|---|---|---|
| R1 | **PP 是唯一真正的缺口**，且是五维里最难的一维 | schedule + 微批 + P2P。阶段 4 之前 `pp > 1` 明确抛异常，这是对的 |
| R2 | 阶段 3 会触碰权重切分路径 | 必须以 TP 等价逐位为准入 |
| R3 | 阶段 2 会让 `dist_gemm` 首次真的生效 | 之前它一直走回退路径，**没测过的代码会被点亮**，必须有 TP>1 的对拍覆盖 |
| R4 | `models/common/` 有 64% 不可达 | 删掉它等于把后续阶段的工作量翻倍。它的问题是**接线**，不是布局 |

**明确不做**

- 不重新引入任何基类 / 注册表式继承 / `traverse()`。
- 不移植 `config_utils.py`（448 行，40 处 `Configurable`）。
- 不做 `nn_modules.py` 那类零行为包装（`class Conv1d(nn.Conv1d, Module)`）。
- 不为 HF 模型新建并行实现 —— 复用 HF 的 `_tp_plan` / `_pp_plan`，只做翻译。
- 不实现 torchao / DeepEP / HybridEP 后端（需要 GPU-only 第三方库，本机装不了也
  测不了；它们改变的是"token 怎么跨 rank"，不是路由契约）。

---

## 附：一句话总结

> hpmesh 已经证明了"拿掉 `Configurable` 和 `Module` 之后，分布式训练框架可以更简单"。
> 但它还没解决这两个抽象**原本承担的那两个问题**：配置如何到达模块，模块如何声明并行。
> 本设计用**一份数据（`ParallelPlan`）**和**一个作用域（`dist_context`）**补位，
> 不引入新的继承层次。补完之后，`models/common/` 才真正独立于 trainer，`apply_*` 才
> 真正不认识具体模型 —— 而这正是让框架能"直接对接 HF"的前提。
