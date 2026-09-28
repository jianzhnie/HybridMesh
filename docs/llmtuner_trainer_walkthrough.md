# llmtuner 训练主路径走查（对齐 torchtitan）

按"一次训练从入口到退出"的顺序走查 llmtuner，每一步给出两侧位置与**结论**。
结论只用四档：`对齐`、`等价但落点不同`、`有意裁剪`、`待确认/未对齐`。
引用一律写成 `文件:行号`（工作区内相对路径），便于机械校验。

- 范围：训练主路径（入口 → 配置 → 装配 → 循环 → 单步 → 前反向 → 报告 → 退出）。
- 不在范围：`parallel/`、`components/`、`datasets/`、`models/` 的内部实现，按
  [`llmtuner_upstream_map.md`](./llmtuner_upstream_map.md) 的 A/B/C/D 分类走各自批次。
- 上游参照：torchtitan `f35966713`（本地 `/Users/jianzhengnie/work_dir/torchtitan`）。
  本文走查期间上游从 `c8a3e7666` 前进到 `f35966713`（14 个提交、177 个文件）。与本仓相关的
  面已逐项核对：`trainer.py`、`train.py`、`components/loss.py` **未变**；
  `training_engine.py`（42 行）、`components/validate.py`（25 行）、
  `distributed/{parallel_dims,pipeline_parallel,utils,fsdp,activation_checkpoint}.py` 与
  `config/{parallelism,validation,configs}.py` 有改动，集中在 PP/CP/拓扑与并行校验面
  （其余在 `rl/`、`experiments/`、`models/qwen3_5|6|8`、`torchtitan_recipes/tests`，范围外）。
- **引用校验**：本文全部 `文件:行号` 引用经机械校验（文件存在 + 行号在范围内）命中，上游侧
  引用另做符号级核对（本轮新引用的行号逐条打印核对过内容）。脚本与用法见
  [`llmtuner_torchtitan_alignment_workflow.md`](./llmtuner_torchtitan_alignment_workflow.md) §7.1。
  **改动任何被引用的源码后必须重跑该脚本**：行号会随插入/删除漂移，而续引写法
  （``（`:207`）``）脚本查不到，只能人工复核。
- **验证边界**：本机 torch 2.2.2 低于项目要求的 torch≥2.12，`tests/unit_tests/cpu/test_trainer.py`
  被能力门禁跳过（`missing: dtensor, pipelining, spmd_types`）。本文是**静态走查**，
  不含运行时或数值等价性验证。

## 0. 前提：上游这一版把引擎与循环拆成了两个类

| 上游 | 职责 |
|---|---|
| `torchtitan/training_engine.py:65` `TrainingEngine` | 分布式运行时；模型/优化器/ckpt/profiler 构建（`initialize` `:230`）；单步原语 `prepare_step` `:446`、`forward_backward_microbatch` `:470`、`optimizer_step` `:654` |
| `torchtitan/trainer.py:41` `Trainer` | 数据、校验、报告策略与循环：`microbatch_generator` `:301`、`train_step` `:331`、`train` `:450`、`close` `:518` |
| `torchtitan/train.py:22` `main` | 入口、seed-checkpoint 分支、进程组销毁 |

llmtuner 是**一个合成的 `Trainer` 类**，外加四个拆出去的同级模块：

| llmtuner | 上游对应 |
|---|---|
| `llmtuner/trainer/trainer.py:195` `Trainer` | 上游 `Trainer` + `TrainingEngine` 两个类的并集 |
| `llmtuner/trainer/builder.py:57` `build_trainer_state` | `torchtitan/training_engine.py:176` `__init__`、`:200` `_initialize_distributed_runtime`、`:230` `initialize`、`:252/:351/:369/:390` 四个 `_initialize_*` |
| `llmtuner/trainer/batch.py:120` `batch_generator`、`:190` `microbatch` | `torchtitan/trainer.py:301` `microbatch_generator` + `TrainingMicrobatch` + 上游 dataloader 层 |
| `llmtuner/trainer/pp_steps.py:22` `pp_microbatches`、`:62` `pp_forward_backward_body` | `torchtitan/training_engine.py:470`（PP 微批组装）与 `:624` `_pp_forward_backward_body` |
| `llmtuner/trainer/validate.py:26/:77/:114` | `torchtitan/trainer.py:483` 的 validator 调用点 + `components/validate.py` |
| `llmtuner/trainer/train.py:99` `main` | `torchtitan/train.py:22` `main` |

**结论：`等价但落点不同`。** 语义边界一致（引擎/循环职责划分相同），实现边界不同
（上游用继承 `Trainer(TrainingEngine.Config)` 组合，llmtuner 用单类 + 过程式装配）。

## 1. 入口

- **llmtuner**：`llmtuner/__main__.py` → `llmtuner/trainer/train.py:99 main` → `:78 parse_config()`
  （`HfArgumentParser` 解析九个 group，再 `LLMTunerConfig.from_groups` 组装）→ `:101 Trainer(cfg)`
  → seed-checkpoint 分支 `:102-121` → `:122 trainer.train()`。
- **上游**：`torchtitan/train.py:22 main` → `ConfigManager().parse_args()` → `config.build()`
  → seed-checkpoint 分支 → `trainer.train()`；`try/except/else` 里 `trainer.close()` 与
  `destroy_process_group()`。
- **结论：`对齐`**（seed-checkpoint 的两条前置校验都在：上游用 `assert`，llmtuner 用
  `ConfigError`，`llmtuner/trainer/train.py:107-115`）。归属差异见 D13。

## 2. 配置

- **llmtuner**：九个 group（`llmtuner/trainer/train.py:65-75`）由 `LLMTunerConfig.from_groups`
  组装，graft 表在 `llmtuner/config/root.py`。
- **上游**：`torchtitan/trainer.py:48` `Trainer.Config(TrainingEngine.Config)`，各子系统配置为
  嵌套 dataclass，组合校验集中在 `__post_init__` 与 `validate_model_training_config`。
- **结论：`等价但落点不同`。** 校验位置随之不同：上游集中在 Config，llmtuner 把组合校验
  下放到 `llmtuner/parallel/matrix.py`（在 §3 第 2 步触发）。

## 3. 装配：`Trainer.__init__` → `builder.build_trainer_state`

llmtuner 的装配顺序是**契约**，写在 `llmtuner/trainer/builder.py:1-20`：① 进程组/rank →
度数；② 组合守卫；③ 确定性种子；④ mesh；⑤ 模型 + 并行；⑥ 优化器/lr/EMA/hook；
⑦ dataloader/checkpointer/metrics。

| 关注点 | llmtuner | 上游（`torchtitan/training_engine.py`） | 结论 |
|---|---|---|---|
| 分布式初始化 + rank | `llmtuner/trainer/builder.py:59-65` | `:200` `_initialize_distributed_runtime` | 对齐 |
| 并行度数 | `llmtuner/trainer/builder.py:70` | `ParallelDims.from_config` | 对齐 |
| 校验可行性前置拒绝 | `llmtuner/trainer/builder.py:76-88` | 无对应前置检查 | 等价但更严 |
| 种子/确定性 | `llmtuner/trainer/builder.py:97-103` + `llmtuner/trainer/trainer.py:240` | `torchtitan/distributed/utils.py:118` `set_determinism`、`:141-156` | 见 D15 |
| mesh | `llmtuner/trainer/builder.py:109-121` | `ParallelDims` + 各 `get_mesh` | 对齐 |
| EP×ckpt、chunked-loss×PP 守卫 | `llmtuner/trainer/builder.py:129-149` | 由 `Config` 校验 | 等价但落点不同 |
| 模型（meta 或实机） | `llmtuner/trainer/builder.py:150-160` | `:252` `_initialize_model` | 对齐 |
| 并行化 | `llmtuner/trainer/builder.py:166-179` | `model.parallelize` / `model.pipeline` | 等价但落点不同 |
| PP 结果落位 + loss sentinel | `llmtuner/trainer/builder.py:180-195` | `:390` `_initialize_forward_backward` | 对齐 |
| 优化器容器 | `llmtuner/trainer/builder.py:201-203` | `:351` `_initialize_optimizer` | 对齐 |
| aux-loss/MoE 平衡 hook | `llmtuner/trainer/builder.py:249-264` | `_register_optimizer_hooks` | 等价但落点不同 |
| checkpointer（`states`） | `llmtuner/trainer/builder.py:291-305` | `:369` `_initialize_checkpointer` | 对齐；llmtuner 多带 dataloader 读位置 |
| `step/ntokens_seen = 0` | `llmtuner/trainer/builder.py:309-310` | `:176` 段（`__init__` 内） | 对齐 |
| metrics + PP 可见性告警 | `llmtuner/trainer/builder.py:321-333` | `Trainer.__init__` 内 | 对齐 |
| GC handler 构建点 | `llmtuner/trainer/trainer.py:957` | `:200` 段 | 等价但落点不同（llmtuner 有意延后到模型存在后） |

## 4. 主循环 `train()`

| 关注点 | llmtuner（`llmtuner/trainer/trainer.py`） | 上游 | 结论 |
|---|---|---|---|
| 恢复 | `:924` `checkpointer.load(cfg.checkpoint.load_step)` | `torchtitan/trainer.py:456`、`torchtitan/training_engine.py:708` | 对齐 |
| 相对步 | `:935` `first_step_of_this_process = self.step + 1` | `torchtitan/trainer.py:496` `num_completed_steps - loaded_step == 1` | 对齐 |
| profiler | `:946` `with Profiler(...)`、`:1009` `profiler.step()` | `torchtitan/trainer.py:464/:491/:507` | 等价但落点不同（上下文管理器 vs enter/exit） |
| GC | `:957-960` 循环内 `gc_handler.run(self.step)` | `torchtitan/training_engine.py:446` 段（`prepare_step` 内） | 对齐 |
| 循环条件 | `:904` `self.step < cfg.steps` | `torchtitan/trainer.py:515` | 对齐 |
| 数据耗尽 | `:964-976` `DataloaderExhaustedError` → 弃步并回退计数 | `torchtitan/trainer.py:474` 同（不计数） | **已修，见 D2** |
| 保存 | `:995-996` `checkpointer.save(...)` | `torchtitan/trainer.py:478`、`torchtitan/training_engine.py:714` | 对齐 |
| 校验 | `:1003` `should_validate` → `validate` | `torchtitan/trainer.py:483` | 对齐 |
| 超时下调 | `:1017` `set_pg_timeouts(...)` | `torchtitan/trainer.py:496` | 对齐（llmtuner 多传 `device=`） |
| 步数维护 | `:959`、`:975` | `torchtitan/training_engine.py:654` `optimizer_step` 末尾自增 | **已修，见 D2** |
| 收尾日志 | 无 | `torchtitan/trainer.py:509` rank0 sleep + "Training completed" | 有意裁剪 |
| 结构化追踪 | 无 | `sl.log_trace_*`、`sl.set_step(...)` | 有意裁剪（见 D9） |

## 5. 单步 `train_step()`

上游把这一步拆成"数据/报告在 `Trainer.train_step`，原语在 `TrainingEngine`"，
llmtuner 全部内联在一个函数里（`llmtuner/trainer/trainer.py:528-830`）。

| 关注点 | llmtuner | 上游 | 结论 |
|---|---|---|---|
| 梯度清零 | `:553` `zero_grad(set_to_none=True)` | `torchtitan/training_engine.py:446` 段（`set_to_none=disable_cuda_graphs`） | 等价（llmtuner 无图路径） |
| lr 快照 | `:559` | `torchtitan/trainer.py:393` | 对齐（都在 scheduler.step 之前） |
| 读满整个窗口 | `:614-625` | `torchtitan/trainer.py:341-350` | 对齐（切分粒度见 D16） |
| 分母归约 | `:629-637` 只在 `dp_mesh` 上 | `torchtitan/trainer.py:360-366` | 对齐 |
| aux-loss 分母 | `:644` | `torchtitan/training_engine.py:446` 段 | 对齐 |
| 有限性标志 | `:654` 每步新建局部量 | `torchtitan/training_engine.py:492` engine 字段 | 等价但落点不同 |
| 累积 loss | `:655-679` 只在要记日志时累加 | `torchtitan/trainer.py:376-389` | 对齐 |
| HSDP 复制组归约 | `:656-665` 延迟到最后一个 accumulation 组 | `torchtitan/training_engine.py:502` | **已修，见 D12** |
| TP 复制参数梯度 | `:470`、`:684` `_allreduce_replicated_tp_grads()` | 上游无同名步骤（理由见 D14） | 等价（设计使然） |
| 梯度裁剪 | `:699` `clip_grad_norm_(...)` | `torchtitan/training_engine.py:654` 段 | 对齐 |
| 有限性归约 | `:714-733` | `torchtitan/training_engine.py:674/:680` | 对齐 |
| 断言 | `:833` `_check_finite` → `torch._assert_async` | `torchtitan/training_engine.py:654` 段 | 对齐 |
| staging 等待 | `:741` | `torchtitan/training_engine.py:654` 段 | 对齐 |
| 优化器/lr/EMA | `:743/:748/:752` | `torchtitan/training_engine.py:654` 段 | 对齐 |
| 报告 | `:813-830` 返回 dict，由循环 `:978` 打印 | `torchtitan/trainer.py:402-447` 在 `train_step` 内打印 | 等价但落点不同 |
| loss 语义 | 累加**未归一化 sum**，`:767` 报告时除 | `torchtitan/components/loss.py:321-322` 在 loss_fn 内除 | 等价，见 D1 |
| loss mesh 选择 | `:597-604` `dp_cp_enabled or tp_enabled` | `torchtitan/trainer.py:405-406` 只看 `dp_cp_enabled` | 等价（llmtuner 的 loss mesh 含 tp，见 D14） |
| `n_tokens_seen` | `:803-806` 在 loss mesh 上 sum | `torchtitan/trainer.py:423-430` | 对齐 |

## 6. 前反向

| 关注点 | llmtuner | 上游 | 结论 |
|---|---|---|---|
| 分支 | `llmtuner/trainer/trainer.py:332` `forward_backward_step` | `torchtitan/training_engine.py:470` | 对齐 |
| spmd 上下文 | `llmtuner/trainer/trainer.py:386` `spmd_context(self.parallel_dims)` | `torchtitan/training_engine.py:601` 段的 `get_spmd_context(..., spmd_typechecking=)` | 有意裁剪（见 D11） |
| 类型检查抑制 | 无 | `torchtitan/training_engine.py:601` 段 `spmd.no_typecheck()` | 有意裁剪（`llmtuner/models/common/rope.py:8` 已记录） |
| 非 PP body | `llmtuner/trainer/trainer.py:372-416` | `torchtitan/training_engine.py:601` | 等价但落点不同（见 D1） |
| loss 计算 | `llmtuner/trainer/trainer.py:419-446` `_loss_sum` | `torchtitan/components/loss.py:321-322` | 等价但落点不同 |
| chunked loss | `llmtuner/trainer/trainer.py:387-402` | `torchtitan/training_engine.py:252` 段的 `ChunkedLossWrapper` | 等价但落点不同 |
| PP body | `llmtuner/trainer/pp_steps.py:62` | `torchtitan/training_engine.py:624` | 对齐 |
| `_param_context` | `llmtuner/trainer/trainer.py:510` | 上游无对应（AC 在并行层） | 等价 + 注释已更正（见 D3） |

## 7. PP 微批的切分点

- **上游**：`train_step` 直接读 `[accumulation][num_pp_microbatches]` 二维微批
  （`torchtitan/trainer.py:341-350`），每个微批自己 `to_input_dict` + `preprocess_inputs`
  （`torchtitan/training_engine.py:517` 段）；微批数是 `num_pp_microbatches` 属性。
- **llmtuner**：`train_step` 每组只读 1 个 batch，`llmtuner/trainer/pp_steps.py:22`
  在 batch 内部切成 PP 微批。
- **结论：`等价但落点不同`（见 D16）**，两边最终都由 dataloader 配置决定微批数。

## 8. 数据读取与计数

- **llmtuner**：`llmtuner/trainer/batch.py:120` `batch_generator`（计时 +
  `metrics.add_tokens(labels.numel())`）→ `:190` `microbatch`
  （`:224` `ntokens_seen += labels.numel() // (cp * tp)`）→ `:158` `count_valid_tokens`。
- **上游**：`torchtitan/trainer.py:301` `microbatch_generator` 计时 →
  `torchtitan/training_engine.py:470` 段累计 `ntokens_seen += num_tokens_per_microbatch_per_dp_rank // cp`。
- **结论：`等价`（见 D7）。** 两边都按"本 rank 的序列份额"折算并在同一 mesh 上求和，
  都能重建全局 token 数；口径表达式不同，且累加点不同（上游在预处理时，llmtuner 在读批时）。
- **`count_valid_tokens` 口径（`llmtuner/trainer/batch.py:158`）：`对齐`。** 合成
  `Batch` 路径先按行重移位（`next_token_targets`，`:179`）再数非 `IGNORE_INDEX`；
  dict/Grain 路径优先读 collator 的 `num_valid_tokens`（`:181`），缺失时按同一公式重算
  （`:186`）。上游同口径：`int((labels != IGNORE_INDEX).sum())`
  （`torchtitan/components/data/collators.py:115`）。
- **loader 契约（`llmtuner/trainer/batch.py:49/:75/:98`）：`对齐`（两侧各自自洽）。**
  `build_dataloader`（`:75`）交给 Grain 的是 **per-rank** token 数（Grain 自己在
  `dp_world_size` 个 rank 间切行）；`data_iterator`（`:98`）交给
  `RandomTokenDataLoader` 的是 **global** `batch_size` 加 `dp_rank/dp_world_size`，
  由 loader 自己切行（`llmtuner/datasets/random_data.py:142`、`:197`）。两条路径因此都要求
  全局 batch 能被 `dp_world_size` 整除；`batch_size_per_rank`（`:49`）是二者共同经过的
  **唯一**前置检查点（缺它时每个 rank 会静默读到更小的 batch）。与上游的形状差异：
  上游该数来自配置（`torchtitan/training_engine.py:577` 的
  `num_tokens_per_microbatch_per_dp_rank`），llmtuner 由
  `global_batch_size // dp_world_size * max_seq_len` 派生 —— 数值等价，只是 llmtuner 多一道
  显式入口校验。

## 9. 校验 / 检查点 / 指标 / profiler / GC / 超时

- **校验**：`llmtuner/trainer/validate.py:26/:77/:114`；上游 `components/validate.py` +
  `torchtitan/trainer.py:483`。**等价**，函数体走查见 §11.4；llmtuner 多装配期可行性拒绝，
  并因此 loud-raise 掉 PP×校验（D18）。
- **检查点**：`llmtuner/trainer/trainer.py:913-919` 暴露 `{step, ntokens_seen}`，与
  `torchtitan/training_engine.py:699-706` 逐字段一致（上游多一步 `sdc_replayer.reset_schedule()`）。
  **对齐**。
- **指标**：`llmtuner/components/metrics.py` 的 `MetricsProcessor`，循环里
  `llmtuner/trainer/trainer.py:978` 打印；上游在 `train_step` 内打印
  （`torchtitan/trainer.py:441`）。**等价但落点不同**。
- **profiler / GC / 超时**：见 §4。

## 10. 差异清单（本轮复查后定论）

`对齐` 的不再列出。判据一栏是**可复核的证据**，动作一栏是本轮实际改动。

| # | 条目 | 判据 | 档位 | 本轮动作 |
|---|---|---|---|---|
| D1 | loss 归一化落点 | 上游在 loss_fn 内除（`torchtitan/components/loss.py:321-322`）并返回归一化值，`accumulated_loss` 直接求和上报；llmtuner 图内除、返回未归一化 sum（`llmtuner/trainer/trainer.py:416`），`:767` 报告时再除。逐项推导：上游 `accumulated = Σ局部sum/G`、`global_avg = Σ accumulated`；llmtuner `loss = Σsum/G`、`global_avg = Σ loss` —— 同一个数；`local_avg` 两边都等于 `Σsum/局部tokens` | 等价 | `forward_backward_step` docstring 明确声明返回值未归一化（已改） |
| D2 | 被取消步的计数 | 上游 `num_completed_steps` 只在 `optimizer_step`（`torchtitan/training_engine.py:654`）末尾自增，数据耗尽时不计；llmtuner 原在循环顶 `+1` | 真实差异 | 弃步分支回补（`llmtuner/trainer/trainer.py:975`），使 `state_dict` 不会跳过未更新的步 |
| D3 | `_param_context` | AC 在 llmtuner 并行层（`llmtuner/parallel/parallelize.py:169` 的 `apply_ac` 阶段），与上游把 `ac_config` 交给 `model.parallelize` 同构；`_param_context` 上游无对应，当前是 `nullcontext`，被三个 body 共用且测试可替换 | 等价 | 更正 docstring：它**不是** AC 的位置（已改） |
| D7 | `ntokens_seen` 口径 | llmtuner `labels.numel() // (cp*tp)`（`llmtuner/trainer/batch.py:224`）并在 `loss` mesh（含 tp，`llmtuner/parallel/parallel_dims.py:220`）上求和；上游 `num_tokens_per_microbatch_per_dp_rank // cp` 并在 `dp×cp` 的 loss mesh 上求和（`torchtitan/distributed/parallel_dims.py:260`、`:317`）。两者都重建全局 token 数 | 等价 | 无（口径差异源于 D14 的 TP 设计） |
| D9 | 结构化日志 | 上游 `sl.log_trace_*`/`sl.set_step` 遍布 trainer/engine；llmtuner 只有 `utils/logger_utils` | 有意裁剪 | 无（可观测性面） |
| D10 | SDC replay / CUDA graphs | 上游 `torchtitan/training_engine.py:390` `_initialize_forward_backward` + `:736` `close` 的 graph teardown；llmtuner 无图路径 | 有意裁剪 | 无（`zero_grad` 的 `set_to_none=True` 因此恒等价） |
| D11 | `spmd.no_typecheck()` / `spmd_typechecking` | 上游 `_non_pp_forward_backward_body` 包住 `loss.backward()`（`torchtitan/training_engine.py:601`）；llmtuner 有意去掉，理由已记录在 `llmtuner/models/common/rope.py:8` | 有意裁剪 | 无 |
| D12 | HSDP 复制组 all-reduce 延迟 | 上游只在最后一个 accumulation 组打开 `set_requires_all_reduce`（`torchtitan/training_engine.py:502`）；llmtuner 原先每个组都归约（正确但多通信） | 真实差异（性能面） | 已移植（`llmtuner/trainer/trainer.py:656-665`，配合 `enumerate` 的累加下标） |
| D13 | 进程组销毁归属 | llmtuner 在 `llmtuner/trainer/trainer.py:1032/:1051` `close()` 内销毁；上游在 `torchtitan/train.py` 入口销毁 | 等价但落点不同 | 无 |
| D14 | TP 复制参数梯度归约 | **两侧都需要这次求和，差别只在谁做。** 上游把 norm 权重在 SP 下标为 `R`，注释写明 backward all-reduce 交给 FSDP（`torchtitan/models/common/decoder_sharding.py:180`："Weight is unsharded@TP: R if SP (pending BWD AR handled by FSDP), else I."），激活布局由 `enable_sp` 决定（`torchtitan/models/llama3/sharding.py:51-53`）——即上游用**声明式放置**表达"每个 TP rank 只覆盖自己的序列分片、梯度需跨 TP 求和"；llmtuner 的 TP 端到端序列并行、norm 作用在分片序列上（`GatherSequenceFirst` 只装在 attention 模块，`llmtuner/parallel/tensor_parallel/apply.py:174-182`），同一份 token-局部梯度改在 trainer 里显式 all-reduce（`llmtuner/trainer/trainer.py:470`、调用点 `:684`，"求和不是平均"的理由在 `:485-486`） | 等价（同一求和，责任方不同） | 无（要取消只能给 llmtuner 引入等价的放置/自动归约机制，属 `parallel/tensor_parallel` 的 A/B 类改造，不是 trainer 层） |
| D15 | 确定性面 | 上游 `set_determinism` 除种子外还设 `cudnn.deterministic/benchmark`、`fill_uninitialized_memory=False`、`CUBLAS_WORKSPACE_CONFIG`、`PYTHONHASHSEED`、`detect_anomaly`（`torchtitan/distributed/utils.py:141-156`、`:181-195`、`:244-245`）；llmtuner 原只有 `use_deterministic_algorithms` | 真实差异 | 已移植（`llmtuner/trainer/trainer.py:244-275`）：四项确定性开关、`PYTHONHASHSEED = str(seed % 2**32)`（为之后 spawn 的 loader worker）、`detect_anomaly`（`TrainingConfig.detect_anomaly`，`set_detect_anomaly(True, check_nan=False)`，理由同上游：NaN/Inf 检查走 `aten._is_any_true`，无 DTensor 策略）。仍差 DTensor mesh-aware RNG tracker（上游用于分片初始化，llmtuner 走 HF 自身初始化，故不适用）与 `warn_only` 开关（llmtuner 固定 `False`，更严） |
| D16 | PP 微批切分点 | 上游从 dataloader 直接读 PP 微批（`torchtitan/trainer.py:341-350`）；llmtuner 每 accumulation 组读 1 个 batch 再切（`llmtuner/trainer/pp_steps.py:22`） | 等价但落点不同 | 无 |
| D17 | PP 损失函数的双驱动接线 | 两个驱动器送 `global_valid_tokens` 的通道不同：公开 `step` 经 `loss_kwargs` 转发进 `loss_fn(output, target, **loss_kwargs)`（上游正是这么接的），私有 `_step_microbatches` 没有该参数、只能读 `schedule._llmtuner_global_valid_tokens`。原实现在 `build_pipeline_schedule` 里把 schedule 的 `_loss_fn` 换成只收 `(pred, labels)` 的 lambda，于是公开路径一被走到就 `TypeError`（torch 会多传 `global_valid_tokens` 关键字）；单元测试用的假 schedule 自己读 `loss_kwargs`，掩盖了这条通路 | 真实差异（潜伏 bug） | 已修：抽出 `make_schedule_loss_fn`（`llmtuner/parallel/pipeline_parallel/apply.py:93`），kwarg 优先、属性兜底；两条通路都在调用前置好属性（`llmtuner/trainer/pp_steps.py:117`）；新增受 `pipelining` 门禁的用例钉住两侧一致（`tests/unit_tests/cpu/parallel/test_pipeline.py:413`） |
| D18 | PP×校验 | 上游校验器有完整 PP 分支，用 `pp_schedule.eval(arg_mbs=, kwarg_mbs=, target_mbs=, losses=)` 前向（`torchtitan/components/validate.py:164` 取微批数、`:234` 驱动 eval）；llmtuner 在装配期直接拒绝该组合（`llmtuner/parallel/matrix.py:117`，由 `llmtuner/trainer/validate.py:53` 调用），`llmtuner/trainer/validate.py` 体里没有任何 PP 分支 | 真实差异（功能缺失，loud-raise） | 未实现。理由经复核后**部分更正**：`matrix.pp_validation` 原写"没有 eval-only 管线通路"，但 torch 的 schedule 有 `eval`（上游正在用）；准确的原因是 llmtuner 的 PP 接缝只接了训练驱动器（行切微批 + D17 那套分母注入），eval 驱动器要另配一套。在补齐之前保持 loud-raise（不得静默跳过） |
| D19 | `max_num_documents` 未穿透到 `preprocess_inputs` | 上游把 `max_num_documents` 传给 `preprocess_inputs`（`torchtitan/training_engine.py:530`、`:569`）用于 CUDA graph 的定长 varlen 元数据（`torchtitan/config/validation.py:49` 在开启图且未设时 loud-raise）；llmtuner 无图通路（D10），走 flex `BlockMask`（`llmtuner/models/hf/wrapper.py:639`） | 有意裁剪（本轮已核实理由成立） | 无。`max_num_documents` 仍在打包面生效（`llmtuner/datasets/packing/build.py:55`）；`create_varlen_metadata_for_document` 存在但训练通路无调用方（只有 `tests/unit_tests/cpu/models/test_masks.py` 引用），即"传进去只会被丢弃"属实 |

**仍未闭合的一条**：D15 里的 DTensor RNG tracker。判定为"不适用"，理由写在上表，
但**没有运行期证据**（本机无 torch≥2.12 多卡环境）。若后续要严格对齐种子生成，
这是唯一还需要的改造点。

## 11. 第二批走查：`batch.py` / `pp_steps.py` / `validate.py` 的函数体

### 11.1 `count_valid_tokens`（`llmtuner/trainer/batch.py:158`）——`对齐`

两条路径同一口径：按行重移位后数非 `IGNORE_INDEX`（`:179`），或直接采信 collator 的
`num_valid_tokens`（`:181`），缺失时按同一公式重算而不静默兜底为 0（`:186`）。上游
`torchtitan/components/data/collators.py:115` 是同一公式。

### 11.2 loader 契约（`llmtuner/trainer/batch.py:49/:75/:98`）——`对齐`（两侧各自自洽）

per-rank token 数（Grain）与 global batch + 行切片（synthetic）是两种 loader 各自的契约，
`batch_size_per_rank`（`:49`）是二者唯一的共同前置检查点。详见 §8 的三条补充结论。

### 11.3 `pp_forward_backward_body` 的两个驱动器（`llmtuner/trainer/pp_steps.py:108`）——**真实差异，本轮已修（D17）**

- 上游只用公开 `step`（`torchtitan/training_engine.py:637`），分母经 `loss_kwargs` 进入
  loss fn（`torchtitan/components/loss.py:321`）。
- llmtuner 优先用私有 `_step_microbatches`（理由：公开 `step` 会重切已经切好的列表），
  私有驱动器没有 `loss_kwargs` 入口，改由 `schedule._llmtuner_global_valid_tokens` 承载分母。
- 修前：`schedule._loss_fn` 被换成只收两个参数的 lambda，公开通路一旦被走到就会
  `TypeError`。修后：`make_schedule_loss_fn`（`llmtuner/parallel/pipeline_parallel/apply.py:93`）
  kwarg 优先、属性兜底，两条通路共用同一个 loss fn。
- 报告口径不变：schedule 侧 loss 已除 G，`llmtuner/trainer/pp_steps.py:143` 再乘回 G，
  让调用方继续持有未归一化 sum（与 D1 的非 PP 路径一致）。

### 11.4 `validate_body`（`llmtuner/trainer/validate.py:114`）——`等价`

- **归约口径与训练一致**：token 数取自**未分片**的 batch、只在 `dp` 轴 reduce
  （`llmtuner/trainer/validate.py:120`）；loss 和在各 rank 自己的分片上、在 `loss` 视图上
  reduce（`:123`/`:126`，含 tp 是 D14 使然）。上游同构（`torchtitan/components/validate.py:200`、
  `:287`）。
- **归一化落点相同**：都在累加结束后一次除
  （`llmtuner/trainer/validate.py:217`；`torchtitan/components/validate.py:293`）。
- **两个 loud error 逐条对应**：零 batch（`llmtuner/trainer/validate.py:198`；
  `torchtitan/components/validate.py:272`）、零有效 token（`:207`；`:282`）。
- 与上游的差异只有两处，且都不在数值面：不传 `max_num_documents`（D19）、无 PP 分支（D18）。

## 12. 尚未走查（下一批）

1. ~~`llmtuner/config/`（九个 group 的字段级对齐与 `from_groups` graft 表）~~ ——
   已完成（2026-09-27 五次增量：字段级复核 + `cli.py` 视图 + CP 默认值对齐）。
2. `llmtuner/parallel/`（`stages.py` 的 stage 表 vs 上游 `parallelize` 顺序；`matrix.py`
   守卫 vs 上游配置校验；TP 的 SP 布局 vs 上游 `enable_sp`，即 D14 的根治方案）——
   **已走完**：stage 表/守卫已复核（2026-09-27 六次增量，`activation_checkpoint.py` 对齐），
   `parallel_dims.py` 七次增量、`fully_shard/fsdp.py` 八次增量、`context_parallel/`
   输入分片九次增量、`context_parallel/cp_kernel.py` ↔ 上游
   `models/common/cp_attention.py` 与 `accelerator/dist_utils.py` ↔ 上游
   `distributed/utils.py` 十次增量（2026-09-28，连带补上 `PYTHONHASHSEED` 与
   `detect_anomaly`）。余下只有 TP/SP 布局的根治（D14）与 PP×校验（D18），两者都需要
   多卡或新 torch，见第 5 条。
3. `llmtuner/components/`（checkpointer/metrics/profiler/optimizer）—— **全树已走完**（2026-09-28
  十一次增量：checkpointer 与 profiler，前者 AST 归一化逐文件对照并删掉
   `TorchCheckpointingManager` 的死成员 `staging_future`，后者补上 XPU activity 分支；
  十二次增量：metrics 与 optimizer，前者删掉零调用者的 `set_num_flops_per_token`，
   后者新登记两条 D 类缺口——`fused_opt_states_bf16` 与 `optimizer_factory_kwargs_by_name`，
   并把 `DistMuon` 与范围外的 `distributed/flex_shard/` 显式连上；同轮修掉 `fused`
   在 CPU 上首步崩溃的真 bug——按设备解析该 flag，无 fused 核时降级为 for-loop；
   十三次增量：补走不在本清单里的 `loss.py` 与 `tokenizer.py`，前者数值路径逐行对齐、
   结构差异四条登记，后者仅 Config 移除类差异）。
4. `llmtuner/datasets/`、`llmtuner/models/common/*` 的数值等价性 —— 进行中：
   `datasets/` 的六个核心文件（`types`/`collators`/`loader`/`packing`/`dataset`/`sources`）
   与 `text/text.py` 已代码级对照完毕（2026-09-28 十四次增量，见 upstream map），结论是忠实
   移植 + 已登记的形状差异；余 `multimodal/*`、`random_data.py`、`build.py` 未走。
   `models/common/*` 已于同日走完（十六次增量，见 upstream map）：`models/common/` 的全部同名文件
   逐文件 AST 归一化复核，无新增代码缺口，结论与逐处定性见 symbol guide §4；上游独有三文件
   （`param_init.py` / `lora.py` / `config_utils.py`）定性登记；`token_dispatcher.py` 的
   `_local_reorder`/`_permute`/`_unpermute`/`combine` 索引数学逐行核对通过（并修掉
   `BaseEPTokenDispatcher.num_experts` 的一处文档错误）；同轮把 `num_flops_per_token`
   重写为 MoE/MLA/sliding 感知（唯一实现变更，测试迁到不受门禁影响的
   `tests/unit_tests/cpu/models/test_flops.py`）。**该批的运行期证据仍缺**：`tests/unit_tests/cpu/datasets/`
   整体被 `require_env('grain')` 门控，本机未装 grain（pyproject 钉 `0.2.18`，本机镜像只有
   `0.2.3`，版本不符故不装），57 例全 skip。
5. D18 的实现（PP×校验），以及任何运行期/数值等价性验证——都需要 torch≥2.12 + 多卡，
   本机不可达（§0 的"验证边界"）。
