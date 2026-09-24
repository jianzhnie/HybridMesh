# Optimizer checkpoint 格式迁移

本文记录 `OptimizersContainer` 改动（commit `0fd6cbe`）中唯一的破坏性变更：
**磁盘上的 optimizer state 采用了新布局。**

## 原因

旧布局按*位置索引*（positional parameter index）给 optimizer state 编键。在 pipeline
并行下，每个 stage 的 optimizer 都从 0 开始给自己的参数编号，于是两个 stage 把同一个
键写进了同一个共享 checkpoint，其中一个被覆盖丢失。旧的 `OptimizerWrapper` 用
re-keying 到 FQN 的方式绕过，但只在显式要求时（`fqn_keying=True`），而非 FFN 路径仍
保留在那里根本不安全的 positional 布局。

`OptimizersContainer` 按构造就横跨所有 model part，没有可以回退的 positional 布局，
它的 state dict 永远是扁平的 FQN 键。

## 变更内容

| | 变更前 | 变更后 |
|---|---|---|
| state 键 | `state/weight` → 嵌套 `{exp_avg, exp_avg_sq, step}` | `state/weight/exp_avg`、`state/weight/exp_avg_sq`、`state/weight/step` |
| param group 键 | `param_groups/0/lr`（一个共享 group） | `param_groups/weight/lr`（`state_dict()` 对每个 group 实际报告的形式） |

第二行对 checkpoint 可移植性是真实改进：FQN 键的 param group 对参数无歧义，而位置
索引 `0` 不是。

## 影响

**`0fd6cbe` 之前写出的 checkpoint 无法载入其后的构建。** 没有也不计划提供转换脚本：
映射只能从*旧* checkpoint 自己的 metadata 恢复，而这是一个 pre-1.0 研究框架。

从旧 checkpoint 恢复的运行会响亮失败（DCP 匹配不到不存在的键），而不是带着冷启动
optimizer 静默训练。删除或重新导出旧 checkpoint。

## 模型权重不受影响

只有 optimizer 子树移动了。`model/...` 键、trainer 的 `train_state` 计数器和
dataloader cursor 均未变化，模型权重的导出和重载与之前完全一致。

## 验证

- 默认运行逐位复现变更前基线（`loss` 4.85817 / 4.85671 / 4.85931 / 4.85672，
  `grad_norm` 0.5583 / 0.5612 / 0.5572 / 0.5487），参数为 `--steps 4 --seed 42
  --deterministic`。`fused` 默认值在 CPU 上与 for-loop kernel 逐位一致；在 CUDA 上
  fused kernel 是不同实现，末位预期有差异。
- PP checkpoint 往返逐位精确：`tests/integration_tests/pp_checkpoint_equivalence.py`
  报告 `max abs diff = 0.000e+00`。
- `tests/integration_tests/pp_equivalence.py`（1F1B 与 Interleaved1F1B）均通过。
- `tests/integration_tests/cp_wiring_equivalence.py` 通过，包括 `preprocess_inputs`
  seam。

## 这次改动暴露的一个 bug

复查 CP 路径时发现一个独立缺陷：**`attn_mask_type` 没有配置通路。**
`trainer/config.py` 里没有任何地方设置它，只有两个等价性测试手工赋值。它在 mask 处
（`hf_wrapper.py`、`context_parallel/apply.py`）回退到 `"causal"`。

这很要紧，因为每个非 random 语料都是 packed 的：`datasets/build.py` 总是把样本送进
`ConcatThenSplitPackingConfig`，一行里装多个文档，attention 不得跨越文档边界。后果：

- 在 **flex**（CUDA 路径）上，未设置的 flag 会静默构造 causal-only mask，跨文档边界
  做 attention——不报错，模型就是错的；
- 在 **sdpa**（CPU）上，wrapper 的 packed-sequence 守卫会 raise，失败是响亮的，但
  报出来的是"packing requires CUDA"，没有点出真正原因。

`build_model_config_for` 现在从 dataset selector 推导该 flag（合成 random 语料用
`"causal"`，其余用 `"block_causal"`），因此它不可能与 trainer 载入的语料不一致。
这里刻意不提供 CLI 开关：packing 今天不可独立配置，而一个可能与数据矛盾的开关本身
就是 bug，不是修复。

**为什么既有测试没抓到它：**`cp_wiring_equivalence.py` 直接构建模型并自己设置该
flag，所以它测试的是孤立的 mask 机制，而不是真实运行经过的 seam。新增的
`test_the_mask_type_follows_the_corpus_rather_than_being_configured` 钉住了这个
seam。
