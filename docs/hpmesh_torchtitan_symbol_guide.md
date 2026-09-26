# hpmesh → TorchTitan 函数与类迁移指导手册

本文回答一个具体问题：修改 `hpmesh` 的某个重要函数、类或公共方法时，应该去
TorchTitan 的哪里理解原始意图，哪些代码可以同步，哪些只能借鉴，以及当前移植是否可信。

文件级分类以 [`hpmesh_upstream_map.md`](./hpmesh_upstream_map.md) 为唯一权威；本文在它
之上增加符号级导航。两份文档冲突时，先修正文件级表，再更新本文。不要在源码中加入固定
SHA 的 `# upstream:` 注释。

## 1. 结论与使用规则

当前审计结论：核心训练链路 TP、FSDP2、CP、EP、PP、checkpoint、数据装配和训练循环均有
明确来源或明确的 hpmesh 独立设计。已发现的差异大多是有意移除 `Configurable`、
TorchTitan `Module` 和声明式 `_sharding_config` 后产生的形状变化，而不是算法漂移。

正确性状态采用四档；它描述的是对应行所列契约，不代表该文件在所有设备和并行组合下都已
验证：

- **通过**：所列逻辑与上游等价，且有单测、等价性测试或实际容器验证。
- **通过（适配）**：实现形状不同，但契约和数学语义已验证。
- **受限**：实现正确，但只覆盖 hpmesh 明确支持的组合；表中会写出边界。
- **悬空/决策项**：当前无调用者或缺少完整执行引擎，不应假装已经支持。

操作规则：

1. A 类先比较同名符号，再比较所在文件；可以同步 bug fix，但要保留 hpmesh 的参数入口。
2. B 类只同步不变量、错误检查和数学意图，不复制 TorchTitan 的类层次。
3. C 类没有可同步对象；只用 hpmesh 测试和调用图判断。
4. D 类不是"漏了一个函数"，通常是组合能力或依赖缺失，需要单独设计。
5. 上游新增 `Config`、`build()`、`Module.Config`、`sharding_config` 时，不可机械迁入。

验证上游基线为 2026-09-22 的 TorchTitan `c6e416bbd`；2026-09-23 已审计至
`b64103072`（记录见
`hpmesh_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区））。
最新 `vllm-ascend-env` 容器已实际完成 8 卡 HCCL Qwen3-8B、4096 序列、真实 HF 权重和
真实 SFT 数据的 FSDP2+Full AC 训练，并完成完整 DCP checkpoint 的 save→resume：从
step 1 恢复 optimizer、scheduler、dataloader 和 train state 后完成并保存 step 2。
镜像使用 Torch/torch-npu 2.10；这些是明确路径的设备证据，不代表全套组合均已验证。
2026-09-21 的设备验证记录文件不在当前工作区，相关结论以本文件 §12 与
2026-09-23 审计记录为准。

## 2. 顶层装配与训练

| hpmesh 重要符号 | TorchTitan 对应实现 | 主要差异 | 结论/维护动作 |
|---|---|---|---|
| `trainer.train.parse_config`, `main` | `torchtitan/train.py` | hpmesh 直接构造一个集中式 dataclass 配置；上游构造 Configurable 树；group 嫁接表在 `HybridMeshConfig.from_groups`（2026-09-26 起，自 train.py 内联迁入） | 通过（适配）；同步启动顺序和全局运行时设置，不同步配置树 |
| `hpmesh.config.HybridMeshConfig` 及各子 config | `config/configs.py` 与各组件嵌套 `Config` | hpmesh 的 SEAM 0：全部字段集中；上游字段分散在组件 | 通过（适配）；新增功能必须先落到这里 |
| `HybridMeshConfig.auto_fill_model` | 上游模型 registry/config build | hpmesh 用 HF `AutoConfig` 填充；上游选原生模型 config | 通过；本地模型与 Hub 配置分别测试 |
| `parallel.parallel_dims.build_parallel_dims`, `build_mesh` | `distributed/parallel_dims.py` + `trainer.py` | 上游无单一对应函数；hpmesh 把解析与 mesh 构造分开 | 通过（适配） |
| `accelerator.dist_utils._init_dist_pytorch` | `train.py` 的 PG 初始化段 | 2026-09-24 起为 trainer 的 PG 引导（原 `mesh.init_distributed` 已并入）：trainer 直接调它而非 `init_dist` 门面，避开后者的 `mp.set_start_method('spawn')` 副作用；厂商加速器 backend 由设备层推导，CUDA 路径才消费 `backend` 实参 | 通过；后端特有行为需实际设备验证 |
| `Trainer.__init__` | `torchtitan/trainer.py::Trainer.__init__` | hpmesh 直接接收 HF wrapper、容器和自由函数；没有 Configurable build | 通过（适配） |
| `Trainer.batch_generator` | `Trainer.next_batch`/post-dataloading 路径 | hpmesh 把 dataloader exhausted、CP/TP shard 和设备搬运集中处理 | 通过（适配） |
| `Trainer.forward_backward_step`, `train_step` | `Trainer.train_step` 及 PP/non-PP 分支 | hpmesh 显式支持梯度累积、chunk loss、PP loss；非日志 step 不保留 loss graph | 通过；有 graph 释放回归测试 |
| `ntokens_seen` 计数 | 上游 `training_engine.py` 的计数段 | 每 rank 只计 `labels.numel() // (cp*tp)` 的本地份额，`train_step` 在 dp×cp×tp loss mesh 上求和还原语料总量；2026-09-23 修复（此前 CP/TP>1 时虚高 cp×tp 倍，对应上游 ec953b360 的口径修正） | 通过（适配）；绝对值尚无多 rank 端到端断言 |
| `Trainer._allreduce_replicated_tp_grads` | 上游 SPMD/TP placement 自动归约 | hpmesh 手写 TP plan，复制参数必须显式 SUM | 通过（适配）；新增 TP module 类型时必须更新识别集合 |
| `Trainer.state_dict`, `load_state_dict` | 上游 trainer state Stateful | hpmesh 只保存训练步等最小状态 | 通过 |
| `Trainer.train`, `close` | 上游同名方法 | 生命周期更短；仍保证 profiler/checkpointer/logger drain | 通过 |
| `Trainer.validate`, `should_validate`, `_check_validation_feasibility` | `components/validate.py::Validator`（含上游 6c2dadbb3 零 batch/零有效 token 报错、90b25912f dp>1 拒绝 `steps=-1`） | 2026-09-24 移植：`training.validation_config`（`ValidationConfig`，freq/steps/dataset，默认 None 关闭且逐位不变）；eval 模式 + `no_grad`，结束后恢复 train；loss 按全局有效 token 归一化，token 走 dp mesh、loss 走 dp×cp×tp loss mesh，与训练同语义；`steps=-1` 对 random 无限语料亦拒绝；PP 组合无 eval 管线通路，构造期 fail-fast（上游走 `pp_schedule.eval`，hpmesh 的 PP loss 内嵌在 schedule 训练步里，未验证） | 通过（适配）；多 rank 归约语义与 PP 组合待目标设备验证 |

## 3. Hugging Face 模型适配层

这是 B 类核心，不能按上游每模型一个 `model.py` 的形状重写。

| hpmesh 重要符号 | TorchTitan 对应实现 | 主要差异 | 结论/维护动作 |
|---|---|---|---|
| `build_model_config`, `build_model_config_for` | `experiments/transformers_modeling_backend/model.py` config 构造 | hpmesh 同时支持离线 architecture、Hub id、本地 checkpoint | 通过（适配） |
| `_unwrap_text_config` | 上游 VLM text config 选择 | hpmesh 把组合模型收敛成统一文本 decoder 契约 | 通过 |
| `_resolve_model_class` | 上游模型 registry | hpmesh 使用 HF auto mapping，不维护模型注册表 | 通过（适配） |
| `HFTransformerModel.__init__` | transformers backend wrapper + 各原生 Decoder | 暴露 `tok_embeddings/layers/norm/lm_head/rotary_emb` 五部件；不复制参数注册 | 通过（适配） |
| GQA 构造校验 | `models/common/attention.py::GQAttention.Config.__post_init__` | hpmesh 在 wrapper 边界校验 head 正数和 `Q heads % KV heads == 0` | 通过；Transformers 5.14 本身会漏掉后一项 |
| `_uses_dsa` + DSA 构造拒绝 | 上游 `_uses_dsa` + `_build_dense_attention_mask` 稠密 additive mask | 2026-09-24 起 wrapper 构造期对 `index_topk`（DSA 特征）fail-fast，不再静默走 flex BlockMask；稠密 mask 路径本身仍是 D 类缺口 | 通过（fail-fast 侧已对齐） |
| `experts_implementation` 旋钮 | 上游 `TitanMoeModelConfig.experts_implementation` + wrapper 应用 | 2026-09-24 起 `ModelConfig.experts_implementation`（默认 `native`）经 config 门面传到 HF config，wrapper 校验"可设置或 raise"（上游同语义），非法值先 raise；EP>1 无意义（swap 整块替换） | 通过 |
| `named_children` | 上游 `Decoder` 的自然子树 | HF CausalLM 多套一层 `model`，hpmesh 只改遍历视图，不改 state_dict FQN | 通过；FSDP/TP/PP 合约测试覆盖 |
| `tp_plan` | HF `_tp_plan` + 上游 sharding config | hpmesh 重写路径前缀供手写 plan 引擎消费 | 通过（适配） |
| `preprocess_inputs` | 上游 post-dataloading process | 合并 batch、构造 mask、先 CP 后 TP 切序列 | 通过；CP×TP 等价测试覆盖 |
| `get_attention_masks`, `_apply_attention` | `models/common/attention.py` 及 transformers backend | hpmesh 对 packed corpus 构造 BlockMask，对 CPU SDPA 明确拒绝错误语义 | 通过（适配） |
| `forward` | transformers backend wrapper forward | 首 stage 接 token，后续 PP stage 接 hidden states；统一 logits 输出 | 通过；PP stage chaining 测试覆盖 |
| `num_flops_per_token` | 各模型 FLOPs 估算 + observability | hpmesh 从 HF config 推导统一近似 | 受限：用于 MFU，不是模型数值路径；新 attention/MoE 结构需扩展 |

## 4. models/common

### 4.1 数学算子与初始化

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `activation.ActivationFn`, `SwiGLU` | `models/common/activation.py` 的对应激活 | 同名但历史来源不完全相同；公式有单测，**通过**，不要按 AST 强行替换 |
| `feed_forward.compute_ffn_hidden_dim` | 同名函数 | 去 Config，舍入公式一致，**通过** |
| `FeedForward.forward`, `SigmoidGatedFeedForward.forward` | 同名类 | hpmesh 接受现成 `nn.Module` 投影；上游由嵌套 Config 构建，**通过（适配）** |
| `param_init.skip_param_init`, `depth_scaled_std` | ~~`models/common/param_init.py`~~ | **2026-09-25 已删除**：parity vendored 死代码，hpmesh 用 HF 自带 `_init_weights`，无消费者 |
| `Embedding.forward` | 上游同名文件仅供概念比较 | hpmesh 是 C 类独立实现并支持 vocab shard bounds；不是同名移植。2026-09-23 移植上游 #4637 同源修复：vocab-parallel 分支把全局 `padding_idx` 映射为本地坐标，只有持有该行的 shard 传入，修复越界崩溃与他 shard 行梯度被静默抑制，**通过** |
| `scatter_add.deterministic_scatter_add` 及 autograd hooks | `ops/scatter_add.py` | 路径不同，算法来源明确；前后向测试覆盖，**通过** |
| `grouped_experts.GroupedExperts.forward` | `models/common/grouped_experts.py` 与 `models/gpt_oss/moe.py` | hpmesh 统一 grouped-mm/fallback，并承载 HF 权重形状，**通过（适配）** |
| `cast_linear.CastLinear`, `to_cast_linear` | `models/common/linear.py::CastLinear`（150c4f73a 配套） | 前向 input/weight/bias 转 `compute_dtype` 后 `F.linear`，参数保原 dtype（autograd 回 cast）；`nn.Linear` 子类 + 同 `Parameter` 重绑定，state-dict FQN 与 tying 不变；经 `ModelConfig.compute_dtype` 启用，默认关闭逐位回归，**通过（适配）** |

### 4.2 Attention、RoPE 与 mask

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `qkv.local_head_split` | `models/common/attention.py::local_head_split` | 去 SPMD 注解，reshape 语义一致，**通过** |
| `qkv.QKVLinear` | `FusedQKVLinear`/QKV 部分 | hpmesh 注入 plain linear 并用 state_dict hook 拆合 HF Q/K/V，**通过（适配）** |
| `QKVLinear._split_qkv_on_save/_merge_qkv_on_load` | 上游 fused QKV state hooks | hpmesh 额外兼容 DTensor gather 与原始 FQN，round-trip 测试覆盖，**通过**。上游 1e4b1f686 把 QKV 转换移入 HF adapters；hpmesh 不跟随——checkpoint 以 HF `wq/wk/wv` 名义存取是本地契约 |
| `RoPEConfig`, `RoPE`, `ComplexRoPE`, `CosSinRoPE` | `models/common/rope.py` | 去 Module/Config 协议，缓存为普通 buffer，**通过** |
| `_yarn_inv_freq` | 同名函数 | 已包含 YaRN `low==0/low==high` 和显式 factor 启用修复，**通过** |
| `_maybe_check_max_pos` | 上游 RoPE bounds check | async assert，compile 时跳过，**通过**。上游 7e7f271e0 已删除 DTensor positions 包装；hpmesh 本无此路径 |
| mask modifier 系列 | `models/common/attention.py` 对应 mask helpers | hpmesh 拆成 `masks.py`；公式一致，**通过** |
| `create_varlen_metadata_for_document` | 上游同名 helper | hpmesh 支持固定容量和动态路径，**通过** |
| `create_attention_mask` | `flex_attention.create_block_mask` 调用点 | hpmesh 缓存 compile，并兼容 Torch 2.10 缺少 `separate_full_blocks`，**通过（适配）** |

### 4.3 MoE 与路由

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `PartialBiasRowwiseLinear` | 上游 9e159aed7 已删除：bias 的 I→P 转换并入新 `RowParallelLinear`；hpmesh 同名类语义本就一致，保留（仅测试使用），**通过** |
| `RouterGateLinear`, `_RouterGateLinearFunction` | `models/common/linear.py` 同名实现 | 前向 FP32 输出、后向 FP32 GEMM；CUDA bf16 使用 `out_dtype`，其他设备安全提升，**通过** |
| `TokenChoiceTopKRouter.forward` | `models/common/routers.py` router | hpmesh 参数化而非 Config 构建，保留 softmax/sigmoid、group limit、route norm，**通过（适配）**。上游 e07084202 抽出可覆写 hooks，hpmesh 以 `_select_experts` 为覆写 seam，数学一致。2026-09-24 起 `_debug_force_load_balance` 调试开关已移植（构造参数，round-robin `(t*K+k)%E`，gating 值仍取真实 score，bias/group 限制均绕过——与上游逐字一致） |
| `RoutedExperts.forward`, `MoE.forward` | 上游同名逻辑 | hpmesh 专家权重是 EP swap 后的本地切片，不是上游 SPMD DTensor，**通过（适配）**。2026-09-24 起 `MoE.set_padding_mask` 一次性暂存通道（上游 d34a13fdf 同源）：mask（True=padding）只过滤负载均衡统计（`tokens_per_expert_E`、aux loss f/p、quantile 直方图），routing 决策/dispatch/expert compute 始终跑完整 token 流，无 mask 逐位不变；CP/TP 由 `shard_padding_mask_for_cp/tp` 与 token 流同序切分 |
| `QuantileBalancedTopKRouter`, `QuantileBalancer`, `register_moe_quantile_balancing_hook` | 上游 f8bb599a7 同名实现 | 训练时 biased top-(K+1)：前 K dispatch、第 K+1 个 biased 分为 cutoff；1000-bin int32 直方图（non-persistent）按 token 分片轴 all-reduce 后取 `top_k/num_experts` 分位数（bin 内插值），mean-centred 覆写 `expert_bias_E`；与 sign-based bias 互斥（同层构造 raise、跨层 hook raise、全 quantile 时 LB hook 自动不注册）；`ParallelConfig.moe_quantile_balancing` 启用，**通过（适配）** |
| `MoE.update_expert_bias` | 上游 expert bias 更新 | 在 optimizer step hook 执行；跨 PP part 汇总，**通过**。2026-09-24 起注册严格性与上游对齐：所有 MoE层 `load_balance_coeff` 混合配置（部分为 None）即 `ValueError`（上游 `_should_register_moe_balancing_hook` 同源），coeff 全 None 时不注册 hook（免每步无谓 collective） |
| `MicrobatchWiseLoadBalanceLoss` | 上游 load-balance loss | hpmesh 用 autograd carrier 注入并按有效 token 归一，**通过（适配）** |
| `aux_loss.AuxLoss.inject/collect_aux_loss_metrics` | `models/common/aux_loss.py` | 去全局 Module registry，使用显式寄存器与 step denominator；与上游逐符号一致（`reduce_mesh="dp"` ↔ hpmesh `"batch"` 是 mesh 命名适配），**通过** |
| `LocalTokenDispatcher` | `models/common/token_dispatcher.py` | 本地排序、dispatch/combine 与上游同意图，**通过** |
| `AllToAllTokenDispatcher` | 上游 EP dispatcher | hpmesh 直接操作本地专家切片和 PG，非 MinimalAsyncEP（上游已删除该实验），**通过（适配）** |
| `TorchAOTokenDispatcher` | 上游同名 dispatcher | 可选导入适配层：`_permute`/`_unpermute` 委托 torchao `permute_and_pad`（expert-major 重排 + token 组 pad 到 `pad_multiple`，EP=1 本地 padded permute 路径一并移植）；构造期 lazy import，未装 torchao loud-raise ImportError 带安装指引；`ParallelConfig.ep_token_dispatcher="torchao"` + `ep_torchao_pad_multiple` 接线，**通过（适配）**，数值**环境未覆盖**（无 torchao/CUDA），待 CUDA 目标设备复跑 |
| `DeepEPTokenDispatcher` / `HybridEPTokenDispatcher` | 上游同名 dispatcher | **登记缺口（loud-raise）**：CUDA-only（`deep_ep`/`hybridep` 内核）且 dispatch/combine 需上游 `distributed/deepep/` wrappers（1155 行，未 vendor），可选导入无法忠实表达契约；`ParallelConfig.ep_token_dispatcher` 选到即 NotImplementedError（含解锁条件：vendor wrappers + CUDA optional extra + CUDA 设备复跑），swap 入口防御性同语义；`AllToAllTokenDispatcher` 满足同一 dispatch/combine 契约 |
| `async_linear` 三个模块（2026-09-26 文件名对齐上游，原 dist_gemm.py） | `models/common/async_linear.py` | 保留 fused collective+GEMM，mesh 从 hpmesh context 获取。上游 e72fd863d 重构为组合式 `Async*Linear`、9e159aed7 再改为继承新 `ColumnParallelLinear`/`RowParallelLinear` 通信角色，hpmesh 保持子类式委托 `parallel/tensor_parallel/linear.py`，数学等价，**受限**：需 TP/CUDA 能力 |

### 4.4 多模态

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `get_vision_positions` | `models/common/multimodal.py` | 已采用一次性 `.tolist()`，避免逐 item CUDA sync；严格检查 run/token 数，**通过** |
| `scatter_vision_embeds` | 同名函数 | 原地 span fusion，消费数不一致即失败，**通过** |
| `build_vision_bank_indices`, `gather_vision_embeds` | 同名函数 | 去 `spmd.local()` 类型注解，tensor 语义一致，**通过** |

## 5. 并行层

### 5.1 Mesh 与公共 collectives

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `ParallelDims.from_config/_validate` | `distributed/parallel_dims.py` | hpmesh 使用稳定 `ValueError`，支持 `dp_shard=-1` 精确推导并校验 EP 整除，**通过** |
| `build_mesh` 与 mesh accessor | 同文件的 mesh 构造/flatten | hpmesh 额外建立 loss mesh，因 TP 端到端切序列，**通过（适配）** |
| `get_all_one_dimensional_meshes` | 同名上游方法 | 已排除 fake-backed axes，**通过** |
| `collectives.set_pg_timeouts` | 上游 trainer/comm timeout | hpmesh 独立实现，**通过（适配）** |
| 归约调用（train_step 的 loss/token 归约） | 上游 scattered reductions | 2026-09-24 起收敛为 `accelerator.dist.all_reduce` 在调用点直接使用（clone + in-place collective），原 `dist_sum`/`dist_max`/`dist_sum_tensor` 薄封装已删除；`reduce_equivalence.py` 验证 all_reduce 语义与 clone 调用惯例（trainer 的内联 clone 由 review 保证），**通过** |
| `clip_grad_norm_` | 上游 distributed grad clipping | hpmesh 额外按本地 expert/dense 参数分组并跨 EP 归约，支持 DP/TP/PP/EP，**通过（适配）** |

### 5.2 TP

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `AllGatherLinear`, `LinearReduceScatter` | `models/common/async_linear.py` 的 `AsyncAllGatherLinear`/`AsyncLinearReduceScatter`（原 `distributed/linear.py` → dist_gemm.py → async_linear.py，数学不变） | fused symmetric-memory autograd 实现；提供 functional collective fallback，**通过** |
| `all_gather_linear`, `linear_reduce_scatter` | 同上非融合语义 | CPU/gloo fallback，前后向是 collective 对偶，**通过** |
| `ColumnParallelLinear`, `RowParallelLinear`（2026-09-26 改名对齐上游，原 `ColwiseLinear`/`RowwiseLinear`） | 上游 `models/common/linear.py` 同名类（9e159aed7 起拥有各自 collective） | hpmesh 替换 HF `nn.Linear`，不使用 ParallelStyle；plan 规格字符串 `colwise`/`rowwise` 与 factory 不变，**通过（适配）** |
| `ColwiseLinearNoGather` | 无对应物（上游为父模块一次性 gather + plain Linear 子投影） | hpmesh 特有 realizer：输出保留 sequence shard；保持原名（hpmesh 特有，非改名对象），**通过** |
| `_resolve_plan`, `_match` | HF `_tp_plan` + 上游 sharding registry | 支持 colwise/rowwise/replicated；`colwise_gather_output` 当前保守保持 lm_head 复制，**通过（适配）** |
| `apply_tp` | transformers backend parallelize + 各模型 parallelize | 手写 pattern plan；明确拒绝 `moe_tp_experts`，**受限：TP×MoE 未实现** |

### 5.3 FSDP2

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `resolve_fsdp_mesh`, `resolve_sparse_fsdp_mesh` | `distributed/fsdp.py` mesh dims | hpmesh 把多轴 mesh 重建为 FSDP 可理解的 1D/2D mesh，**通过（适配）** |
| `apply_fsdp_to_decoder` | 同名上游函数 | 支持 HF ModuleList、MoE expert placement、prefetch。2026-09-23 移植上游 4b5023b80 同源修复：专家分片度经 `_fsdp_shard_degree` 只计 shard 轴，HSDP 下不再误选 `Shard(1)`，**通过（适配）** |
| `enable_fsdp_symm_mem` | 同名上游函数 | 2026-09-23 起支持 `scope="all"/"dense"/None`（上游 65e495dda），非法 scope 抛 ValueError；经 `fsdp_symm_mem_scope` config 字段（默认 "all"）对用户开放，**通过（适配）** |
| `disable_fsdp_gradient_division` | 同名上游 helper | global valid-token loss 自行缩放，故禁用 FSDP 平均，**通过** |
| `apply_fsdp` | 各模型 `parallelize.py` 的 driver | 固定 dtype 策略，非 NCCL 强制 SUM；兼容 Torch 2.10 类型缺失，**通过（适配）**。配置面收窄登记：`cpu_offload` 未接线（`fully_shard/apply.py` 恒 `False`），param/reduce dtype 固定（模型 dtype / fp32），所有 DP 轴为 1 时不装 MixedPrecisionPolicy（数值等价） |

### 5.4 CP、EP、PP 与 AC

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `apply_cp` | `distributed/context_parallel/api.py` + 模型 parallelize | 给 HF attention 注入 kernel；校验 backend、mesh 和 Ulysses heads；2026-09-25 起 ulysses×packed 不再 fail-fast（经 `set_cp_mesh(strategy=...)` 闩锁策略），**通过（适配）** |
| `shard_batch_for_cp/tp` | 上游 input sharding/post-dataloading | hpmesh 显式切 token tensors，保持 mask/positions 契约，**通过** |
| `CPFlexKernel`（KV all-gather / Ulysses 两条路径） | `models/common/cp_attention.py` 同名意图 | 剥掉上游 CPInnerAttention/FlexInnerAttention 类层，redistribution 与 kernel 合在 `context_parallel/cp_kernel.py`；KV all-gather 的 backward reduce-scatter dtype 可配（默认 fp32），Ulysses 为 seq↔head all-to-all，均与上游一致。2026-09-23 起独立的 `primitives.py` 已删除（零调用者的重复实现）；2026-09-24 起 ulysses 的 `_full_length_causal_mask` 复用 `masks.create_attention_mask`（与 wrapper 同 builder、同参数），本地 inspect 兼容副本已删；2026-09-25 起 ulysses 支持 packed/varlen——wrapper 全长透传文档 mask，kernel 按 mask Q 长度分派（上游 `UlyssesCPVarlenInnerAttention` 语义，varlen 元数据不随 token 分片） | **通过（适配）** |
| `swap_hf_moe_blocks` | transformers backend `moe_replacement.py` | 上游重新初始化，hpmesh 搬运 HF 权重；不是共享实现，等价性测试覆盖，**通过（适配）** |
| `apply_ep` | 上游模型 EP parallelize | 先 swap 再建立 dispatcher/组，**通过（适配）** |
| `generate_llm_fqn_per_model_part` | transformers backend `pipeline.py` | 加权切层公式一致，**通过** |
| `split_model_into_stages` | 同文件 stage split | 删除模块用 `Identity`，每 stage 保留 rotary，兼容 Torch 2.10 `PipelineStage`，**通过（适配）** |
| `apply_pp`, `build_pipeline_schedule` | `distributed/pipeline_parallel.py` | hpmesh 直接消费 HF 五部件契约，**通过（适配）** |
| `apply_pp(first_stage_module_fqns=...)`, `_prepend_first_stage_modules` | 同文件 `pipeline_with_first_stage_modules` | 额外顶层模块并入 stage 0：仅作用自动切分，存在的 FQN 按序前插，已占有/重复 FQN raise、缺失跳过，显式 `module_fqns_per_model_part` 给定时忽略并告警（同上游委托语义）；`split_model_into_stages` 配套把 wrapper `named_children()` 不呈现的额外顶层模块在非属主 stage 置 `Identity`（上游 "pruned on other stages" 语义），装五部件的容器经"包含已呈现部件"判定跳过。stage FQN 稳定、默认 None 逐位不变，**通过（适配）** |
| `apply_ac`, selective helpers, `_apply_memory_budget` | `distributed/activation_checkpoint.py` | FullAC/SelectiveAC 已移植，**通过**；MemoryBudgetAC 已移植为 `mode='memory_budget'` + `MemoryBudgetACConfig`（设 `torch._functorch.config.activation_memory_budget`，需 compile，torch 无 knob 时 loud-raise），见 §9.1；RegionAC 未移植（配置即 `NotImplementedError`）。FullAC 的 `determinism_check`/`debug` 旋钮未暴露（固定默认值），登记于此 |
| `apply_compile`, `_maybe_enable_async_tp`, `_maybe_regional_inductor_backend`, `maybe_regional_inductor` | `distributed/compile.py` 同名函数 | 四件全移植为 `parallel/compile.py` + `CompileConfig`（`training.compile_config`，默认全关 = 旧整体 compile 逐位不变）：逐 block compile 用 `Module.compile` 就地（`per_block=True`）；async TP 设 `_micro_pipeline_tp` + symm-mem 注册（按 group 名去重），配置期拒无 compile/tp=1，装配期对无 mesh/旧 torch loud-raise；regional_inductor 仅 `aot_eager`×flex 触发（wrapper `uses_flex_attention` 判定，annotation 在 `_flex_attention_hf`，inductor_configs 传空），flex×其他 backend `ValueError`、torch 无该模块 `NotImplementedError`；`capture_scalar_outputs` 按上游条件（`_iter_moe_layers` 非空）设置，dense 不动。上游的 `skip_fwd_side_effects_in_bwd_under_checkpoint` 与 FakeTensorMode monkeypatch 未移植（登记于 upstream map），**通过（适配）** |

## 6. 数据系统

| hpmesh 重要符号组 | TorchTitan 对应实现 | 差异与正确性 |
|---|---|---|
| `DatasetBuildContext`, `DatasetIterationPolicy` | `components/data/types.py` | 去 Configurable，参数校验已补齐。上游 ec953b360 把 `num_tokens_per_batch` 改名 `num_tokens_per_microbatch`；hpmesh 保持旧名且内部自洽，属故意分叉，**通过** |
| `TextSequence`, `SampleProcessor`, `SingleDataset` | `components/data/dataset.py` | 类去 `Config` 后缀；构建走自由函数，**通过（适配）** |
| `WeightedDataset`, `DatasetMix`, `DatasetConcat` | 同文件 config nodes | 数据组合语义保留，**通过** |
| `build_dataset` 与 `_build_*` | 上游各 config `.build()` | hpmesh 工厂替代对象构建协议，**通过（适配）** |
| source 类与 `build_source` | `components/data/sources.py` | 同上；HF streaming cursor 显式 Stateful，**通过** |
| `GrainDataLoader` | `components/data/loader.py` | 直接收参数，无 loader Config，state round-trip 保留，**通过** |
| `TextCollator` | `components/data/collators.py` | packed labels/positions 与 valid-token 计数契约。上游 d398a8fb9/ec953b360 已把 `batch` 改名 `microbatch` 并引入 `TrainingMicrobatch` 类型；hpmesh 保持 dict 版 `TrainerBatch`（labels 与 num_valid_tokens 已内含），语义等价，属故意分叉，**通过** |
| packing build 函数和 iterators | `components/data/packing.py` | registry 选择移到 `DataloaderConfig.packing`，算法保留；document-aware iterator、padding_mask 与可恢复 remainder 均已含上游 f23d7dfe2 载荷，**通过** |
| `TextProcessor`, `ChatProcessor` | `hf_datasets/text_datasets.py` | hpmesh 路径重组，处理语义一致；SFT prompt/response token 边界校验与上游 8108e201a 逐字一致，**通过** |
| `MultiModalCollator` | `hf_datasets/multimodal/mm_collator.py` | 已增加 MRoPE grid/run/长度校验，**通过** |
| image/text/video helpers | `hf_datasets/multimodal/utils/*` | A1 移植，路径扁平化；单测覆盖，**通过** |
| `MultiModalProcessor` 与 packing helpers | `hf_datasets/multimodal/mm_datasets.py` | 去 Configurable，packing 为自由函数，**通过（适配）** |
| `RandomTokenSource/RandomTokenDataLoader` | 无对应 | C 类合成数据；DP rank/world 校验已覆盖，**通过** |
| `datasets.build.build_dataloader` | 上游 config `.build()` 调度仅供概念比较 | C 类工厂，是 hpmesh 单入口设计，**通过（适配）** |

## 7. Components

### 7.1 Loss、optimizer、scheduler

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `cross_entropy_loss`, `_LossParallelCrossEntropy` | `components/loss.py` | 以 logits shape 选择 vocab-parallel；非法 label async 拒绝，**通过** |
| `vocab_shard_bounds`, `next_token_targets` | 上游公式散在 loss/训练器 | hpmesh 提取成共享 helper，**通过（适配）** |
| `chunked_lm_head_cross_entropy` | 上游 chunked CE | 自行 backward 以控制 logits 峰值，**通过**。允许不整除的短尾 chunk（sum 归约下数值等价）。性能差异登记：不合并 lm_head 的 FSDP reshard/grad-sync（上游在 chunk 循环期间禁用），chunked×FSDP 下每 chunk 多一次 all-gather/reduce-scatter，数值等价 |
| `compute_logprobs`, `mse_loss` | 上游对应 loss | 直接自由函数，无 BaseLoss。2026-09-23 起分片路径的 `return_entropy` 真正生效：entropy 经 `_vocab_parallel_entropy` 免 gather 计算（上游 a3d59d316 同源）；batch-invariant 模式先经 `_GatherVocabShards` 全量 gather（后向为切片），**通过**。严格性差异登记：`tp_group` 已给但 `global_vocab_size=None` 时静默走全词表路径（上游 raise），当前无调用者触发 |
| `OptimizersContainer` | `components/optimizer/optimizer.py` | 删除 OptimizerWrapper；多 PP part 容器直接实现 Optimizer/Stateful surface，**通过（适配）** |
| `init_optim_state` | `components/optimizer/utils.py` | 已支持部分参数已有 Adam state，并保持首次真实 step=1，**通过** |
| flat state dict helpers | 同文件 | FQN flat format，支持 nested state，**通过** |
| `_wsd_factor`, `LRSchedulersContainer`, `build_lr_scheduler` | `components/optimizer/lr_scheduler.py` | 去 Configurable，数学与 state 语义保留，**通过**。默认值分叉登记：上游 `decay_ratio=None`（默认）表示 warmup 后贯穿余程 decay；hpmesh 无 None，默认 `0.0` 表示永不 decay（config docstring 声明为有意设计）。另新增 `total_steps < training_steps` 拒绝（上游会跑出负 lr） |

### 7.2 Checkpoint

| hpmesh 符号 | TorchTitan 对应符号 | 差异与正确性 |
|---|---|---|
| `ModelWrapper` | `components/checkpointer/base.py` | 合并 PP parts，缓存稳定 storage 供 async staging，**通过** |
| `CheckpointStorage` | 上游 backend storage seam | hpmesh Protocol，不依赖 Configurable，**通过（适配）** |
| `BaseCheckpointManager` 生命周期方法 | 同名基类 | load/save/close、异步 drain、retention 集中在基类。2026-09-23 起 resume 优先于 initial_load_* 时记 info 日志（上游 810e62786），**通过** |
| `_parse_step/_find_load_step/_purge_stale_checkpoints` | 同名策略 | exact `step-N`、清理 staged/abandoned、保留豁免，**通过** |
| `dcp.CheckpointManager` | `components/checkpointer/dcp.py` | 本地/remote DCP、HF export guard。2026-09-23 起异步写总时长经 `save_future` done-callback 记 info 日志（上游 d9ca9e55a，以 info 行替代 structured scalar），**通过（适配）** |
| `TorchCheckpointingManager` | 同名 backend | optional dependency 延迟导入，保存统一经过 backend，**通过（适配）** |
| `canonical_fqn` | `components/checkpointer/utils.py` | A1，移除 checkpoint wrapper segment，**通过** |

### 7.3 Observability、profiler、tokenizer

| hpmesh 符号组 | TorchTitan 对应实现 | 差异与正确性 |
|---|---|---|
| `DeviceMemoryMonitor` | `observability/metrics.py` | 后端中立设备 API，**通过** |
| logger 类与 `LoggerContainer` | 同文件 | optional TensorBoard/W&B 延迟导入，**通过**；镜像需安装对应包。2026-09-23 起 `WandBLogger.log` 带 `commit=True`（上游 e0e35fe5a），防显式 step 被合并 |
| `MetricsProcessor` | 同名上游类 | 去 Configurable；按真实 step window 算吞吐/MFU，log frequency 构造时校验，**通过（适配）** |
| `get_metrics_rank`, `ensure_pp_loss_visible` | 上游 metrics rank/PP warning | hpmesh 明确 PP schedule 可见性，**通过** |
| `Profiler`, `MemoryProfiler` | `observability/profiler.py` | 去 Configurable，schedule 与 OOM 处理保留；`_caused_by_oom` 与上游 773e16e75 语义等价（含防环与隐式链），**通过** |
| `BaseTokenizer`, `HuggingFaceTokenizer` | `components/tokenizer.py` | A1；encode 强制 `add_special_tokens=False` 后自行处理 BOS/EOS。2026-09-23 起 `apply_chat_template` 接受 `Sequence[Mapping]`（上游 4a0d8dab3 多轮 SFT 配套），**通过**。2026-09-24 起 `apply_chat_template` 自动注入 `bos_token`/`eos_token` kwargs 与默认 `add_generation_prompt=True`（上游 backend tokenizer 同源）；SFT 全量渲染在 `datasets/text/text.py` 显式传 `add_generation_prompt=False` |
| `MultiModalTokenizer` | 同文件多模态 tokenizer | 组合 text/vision token 契约，**通过** |

## 8. Utils 与 C 类模块

| hpmesh 符号组 | TorchTitan 对应 | 结论 |
|---|---|---|
| `utils.filesystem.*` | `tools/filesystem.py` | A1，去 docstring 后 AST 等价，**通过** |
| `accelerator.spmd_context.*` | 意图接近 `distributed/spmd_types.py`，实际基于 pip `spmd_types` | C 类活代码；不要替换成上游 module protocol |
| `accelerator.device.*` | 无可靠同源 | C 类，统一 NPU/CUDA/MLU/MUSA/CPU 设备信息与 backend 选择，含 mmengine 风格厂商谓词 |
| `accelerator.monitoring.*` | 部分意图见 `tools/utils.py` | C 类，包含 peak FLOPS（含 MI350X）和 memory snapshot |
| `utils.gc.GarbageCollection` | `tools/utils.py` GC helper | 去 structured logger，**通过（适配）** |
| `utils.batch_invariant.*` | 上游 trainer/config 内的开关 | C 类提取，线程内全局状态；消费契约与上游一致（`hf_wrapper.py` 的 mask 构造读取同一开关），**通过** |
| `utils.logger_utils.*` | 无单一对应 | C 类日志格式与 rank helper；2026-09-24 起全仓模块 logger 统一经 `get_logger`（发射时 rank 过滤），文件输出参数随零调用删除 |
| `utils.checkpoint_keys` | 无文件对应 | C 类，打断 config→checkpointer 导入环 |

## 9. 缺口、悬空链与禁止误判项

### 9.1 真缺口

- `distributed/compile.py`：**已移植**（2026-09-24 批 8，`parallel/compile.py::apply_compile`
  + `CompileConfig`）。逐 block compile（`per_block`）、async TP（`_micro_pipeline_tp`
  + symm-mem，配置期/装配期双层 loud-raise）、regional_inductor（`aot_eager`×flex
  才 scoop，annotation 在 `_flex_attention_hf`）、`capture_scalar_outputs`（含
  token-choice MoE block 时设置，dense 不动）四件各自独立开关，默认全关即旧整体
  compile 逐位不变。未移植登记：`skip_fwd_side_effects_in_bwd_under_checkpoint`、
  FakeTensorMode monkeypatch、`components` 列表。见 §5 符号行与 upstream map。
- `models/common/moe_sharding.py`：**部分移除**（2026-09-25）。TP×MoE 组合能力的
  声明层与装配层已就位（B 类适配，见 upstream map"已从 D 移除（部分）"）：
  `apply_tp` 接受 `moe_tp_experts` 等规格并结构性地实现 MoE-under-TP——专家权重
  F 维原地切分、router Replicate、块边界 AG/RS;**tp×ep 同日起按上游语义放行**
  （TP 只切 dense、EP 独占 routed 专家、router Replicate;`apply_tp` 在 ep>1 时把
  MoE 块留给 swap，专家梯度排除由 `_tp_sharded_param_ids` 统一判定）;tp×ep×cp 与
  shared-expert×tp 保持 loud-raise。真多卡前后向等价性环境未覆盖，待 torch≥2.12
  复跑。符号对应：上游
  `expert_param_placement_sparse`（EP 轴 S(0) 声明）→ hpmesh EP swap 的 per-rank
  experts 切片（`parallel/expert_parallel/convert.py::_convert_block`)；上游
  `dense_param_placement(tp=R)` 的 router Replicate 声明 → hpmesh router 不切 +
  `_allreduce_replicated_tp_grads` 求和；上游
  `_moe_sharding_config` 的块边界 in/out 声明（ep=1 时 Replicate）→
  `tensor_parallel/tp.py::_TPMoeSequenceBoundary`（入口 `apply_tp` 在 `tensor_parallel/apply.py`）;ep>1 时的 sequence-parallel 布局
  → swap 后 MoE 直接消费/产出 T/tp 分片（无边界 collective);HF 侧
  `packed_colwise`/`moe_tp_experts` 规格 → `_shard_experts_for_tp`。
- RegionAC：依赖 `torch_remat` 包与上游 `Module.configure_remat_regions` 协议，hpmesh
  两者皆无，不引入该依赖；配置 `activation_checkpoint_mode='region'` 在 config 校验与
  `apply_ac` 两处均显式 `NotImplementedError` 并写明解锁条件。MemoryBudgetAC 已于
  2026-09-24 移植（`mode='memory_budget'`，需 compile，语义同上游）。
- 2026-09-23 审计新增登记（上游 `c6e416bbd..b64103072` 引入）：
  - `components/optimizer/ema.py`：在线 EMA 模型平均（1b9eef3bd，515 行）——**已移植**
    （批 3a，`hpmesh/components/optimizer/ema.py` + config/trainer/checkpointer
    三侧接线，checkpoint `ema` 键，见 §4 与 upstream map"已从 D 移除"）。
  - Quantile-balanced MoE routing（f8bb599a7）：**已移植**（批 2，
    `QuantileBalancedTopKRouter` + `QuantileBalancer` + quantile hook；与
    sign-based bias 互斥，`ParallelConfig.moe_quantile_balancing` 启用，见 §4.3）。
  - MoE padding-mask 负载均衡（d34a13fdf）：**已移植**（批 2，
    `MoE.set_padding_mask` 通道；mask 只过滤统计不动执行，无 mask 逐位不变，
    见 §4.3）。
  - `CastLinear`（lm_head compute-dtype 变换，150c4f73a 配套）。
  - Ulysses CP × varlen/packed（baff3c681）：**已移植**（2026-09-25，批 5）。B 类
    适配：不复制 `UlyssesCPVarlenInnerAttention` 类层次；wrapper 在 ulysses 策略下
    全长透传文档 mask（`set_cp_mesh(strategy=...)`），kernel 按 mask Q 长度分派；
    `apply_cp` 对该组合的 fail-fast 移除。2-rank 等价性测试
    `cp_ulysses_varlen_equivalence.py` 已写，本机 torch 2.2.2 无 flex，环境未覆盖
    待复跑。详见 upstream map"已从 D 移除"。
  - 多轮对话 SFT 的 renderer 路径（4a0d8dab3）：**已适配为可选路径**
    （2026-09-25，第 12 项）。B 类语义适配：不复制 Configurable 外形、不新增
    硬依赖。上游 `components/renderer.py::RenderersLibraryConfig.build` →
    hpmesh `datasets/text/renderer.py::build_chat_renderer`（renderer 名以 CLI
    字符串传入，lazy `importlib` 探测；`auto`/`default` 两个 renderer 同样
    loud-refuse）；上游 `RendererTokenizerWrapper` → hpmesh 同名类（逐字，
    适配 `HuggingFaceTokenizer`）；上游 `ChatProcessor.Config.renderer` /
    `_tokenize_with_renderer` → hpmesh `ChatProcessor(renderer=...)` /
    `_tokenize_with_renderer`（`build_training_sample(..., ensure_final_stop=True)`、
    mask 随 label 移位、超长丢弃、文本限定，语义逐字）；renderer 路径与
    chat-template 路径互斥（构造期二选一），renderer 在时不再要求 eos_id。
    接线：`DataloaderConfig.chat_renderer`/`messages_field`（仅
    `dataset=local_jsonl_sft`）→ `datasets/build.py` →
    `make_local_jsonl_sft_multiturn`。未装 `renderers` 时启用 ImportError
    带 `pip install renderers==0.1.11` 指引；默认关闭逐位不变。真实库数值
    未验证（本机无 renderers，单测用 sys.modules fake 模块覆盖），解锁条件：
    pyproject 加 optional extra 后复跑。
  - validation 循环：**已移植**（批 4，`Trainer.validate`/`should_validate` +
    `ValidationConfig`，上游 `components/validate.py::Validator` 对应物；上游
    6c2dadbb3 的零 batch/零有效 token 报错与 90b25912f 的 dp>1 拒绝
    `steps=-1` 两条校验一并移植，另对 random 无限语料的 `steps=-1` 同样
    fail-fast；PP × validation 无 eval 管线通路，构造期 NotImplementedError，
    见 §2 trainer 表）。
- 2026-09-24 复核新增登记：
  - `token_dispatcher.py` 的 TorchAO/DeepEP/HybridEP 三个 dispatcher：**已对齐**
    （2026-09-25，§9.1 第 13 项）。TorchAO 落为可选导入适配层
    `TorchAOTokenDispatcher`（`_permute`/`_unpermute` 委托 torchao
    `permute_and_pad`，构造期 lazy import，未装 loud-raise ImportError 带
    `pip install torchao` 指引），DeepEP/HybridEP 保持登记缺口（CUDA-only +
    上游 `distributed/deepep/` wrappers 未 vendor），配置期
    NotImplementedError 带解锁条件；`ParallelConfig.ep_token_dispatcher` /
    `ep_torchao_pad_multiple` 接线，默认 `alltoall` 逐位不变。torchao 数值
    **环境未覆盖**（本机无 torchao/CUDA，单测以 fake 模块覆盖 sentinel-row
    padding 契约），解锁条件：CUDA 目标设备装 torchao 复跑。
  - router `_debug_force_load_balance`：纯调试开关，有意不移植。
  - router `_debug_force_load_balance`：**已移植**（批 1，`TokenChoiceTopKRouter`
    同名构造参数，round-robin 语义逐字一致，见 §4.3）。
  - `CastLinear`：**已移植**（批 1，`models/common/cast_linear.py` +
    `ModelConfig.compute_dtype`，state-dict FQN 不变，默认关闭，见 §4.1）。
  - PP per-stage seed：**已移植**（批 1，`trainer/seed.py::derive_distinct_seed` +
    trainer 接线；DTensor RNG tracker 不移植）。
  - `pipeline_with_first_stage_modules`：**已移植**（批 4，`apply_pp` 的
    `first_stage_module_fqns` 参数 + `_prepend_first_stage_modules`；
    `split_model_into_stages` 配套置空非属主 stage 上的额外顶层模块，stage
    FQN 稳定，默认 None 逐位不变；当前无消费者，见 §5.4）。
  - transformers_modeling_backend 复核（同目录全量盘点，结论：其余功能均有
    等价支持或已登记裁剪）曾登记三项，**均已于 2026-09-24 对齐**：
    - DSA 模型：wrapper 构造期对 `index_topk` fail-fast（稠密 additive mask
      路径仍不实现，但静默错误语义已消除，见 §3）。
    - `experts_implementation` 旋钮：已移植（`ModelConfig` 字段 + wrapper
      应用，"可设置或 raise"上游同语义，见 §3）。
    - chat template 的 `bos_token`/`eos_token`/`add_generation_prompt` 自动
      注入：已移植到 `HuggingFaceTokenizer.apply_chat_template`（见 §7.3）。

### 9.2 有意删除

`Configurable`、TorchTitan `Module`、`protocols/`、`structured_logger/`、quantization
组件均是设计裁剪，不应为了"对应完整"重新加入。

### 9.3 已清理的悬空链

`parallel/sharding.py` → `parallel/spmd_shims.py` 因没有上层消费者已整体删除；
2026-09-23 第二轮排查又删除了三个零调用者文件（`context_parallel/primitives.py`、
`models/common/flex_kernel.py`（连同悬空的 `HFFlexKernel._sharding_config`）、
`models/common/nn_modules.py`）与 `parallel_dims.py` 的 7 个未用 API。`accelerator/spmd_context.py`
是独立活代码。

## 10. 全模块符号索引

下表是快速查找入口，覆盖当前 71 个非 `__init__.py` / `__main__.py` 实现模块。列出的为
顶层类/函数和重要公共方法；私有 helper 在前文涉及关键算法时单列。"同文件"指本文前述
路径变换后的 TorchTitan 文件。

| hpmesh 模块 | 重要符号 | 对应类别 |
|---|---|---|
| `components/checkpointer/base.py` | `ModelWrapper`, `CheckpointStorage`, `BaseCheckpointManager`, `purge_thread` | A2，同文件 |
| `components/checkpointer/dcp.py` | `CheckpointManager`, `AsyncMode` | A2，同文件 |
| `components/checkpointer/torch_checkpointing.py` | `TorchCheckpointingManager` 与 backend config helpers | A2，同文件 |
| `components/checkpointer/utils.py` | `canonical_fqn` | A1，同文件 |
| `components/loss.py` | CE、vocab CE、chunked CE、logprobs、MSE | A2，同文件 |
| `components/metrics.py` | monitor、logger、`MetricsProcessor` | A2，`observability/metrics.py` |
| `components/optimizer/lr_scheduler.py` | WSD factor、scheduler container/build | A2，同文件 |
| `components/optimizer/optimizer.py` | `OptimizersContainer` | A2，同文件 |
| `components/optimizer/utils.py` | optimizer state 初始化与 flat/FQN 转换 | A1，同文件 |
| `components/profiler.py` | `Profiler`, `MemoryProfiler` | A2，`observability/profiler.py` |
| `components/tokenizer.py` | tokenizer 三类 | A1，同文件 |
| `datasets/build.py` | `build_dataloader` | C，config build 替代品 |
| `datasets/collators.py` | `Collator`, `TextCollator` | A2，`components/data/collators.py` |
| `datasets/dataset.py` | dataset nodes 与 build 工厂 | A2，`components/data/dataset.py` |
| `datasets/loader.py` | `BaseDataLoader`, `GrainDataLoader` | A2，`components/data/loader.py` |
| `datasets/multimodal/mm_collator.py` | `MultiModalCollator` | A2，`hf_datasets/multimodal/mm_collator.py` |
| `datasets/multimodal/mm_datasets.py` | processor 与 sample packing | A2，`hf_datasets/multimodal/mm_datasets.py` |
| `datasets/multimodal/mm_image.py` | decode/resize/patch helpers | A1，`hf_datasets/multimodal/utils/image.py` |
| `datasets/multimodal/mm_text_utils.py` | padding 与 placeholder helpers | A1，`hf_datasets/multimodal/utils/text.py` |
| `datasets/multimodal/mm_video.py` | video load/process | A1，`hf_datasets/multimodal/utils/video.py` |
| `datasets/packing.py` | 两种 packing 与 Stateful iterator | A2，`components/data/packing.py` |
| `datasets/random_data.py` | synthetic source/loader | C |
| `datasets/sources.py` | JSONL/HF sources 与 cursor | A2，`components/data/sources.py` |
| `datasets/text/text.py` | text/chat processors | A2，`hf_datasets/text_datasets.py` |
| `datasets/types.py` | build context/iteration policy | A2，`components/data/types.py` |
| `parallel/parallel_dims.py`（含 `build_parallel_dims` / `build_mesh`，2026-09-25 自 `accelerator/mesh.py` 并入） | dims、mesh、distributed init | B，散在 parallel dims/trainer |
| `models/common/activation.py` | activation wrappers | C，同名不同源 |
| `models/common/aux_loss.py` | aux-loss carrier/registry/hooks | A2，同文件 |
| `models/common/async_linear.py` | TP-overlap projections/FFN | A2，同路径同名（上游 9e159aed7 改名，hpmesh 2026-09-26 跟随） |
| `models/common/embedding.py` | vocab-aware embedding | C，同名不同源 |
| `models/common/feed_forward.py` | FFN helpers/classes | A2，同文件 |
| `models/common/grouped_experts.py` | `GroupedExperts` | B，common + gpt_oss MoE |
| `models/common/linear.py` | router/partial-bias linear | A2，同文件 |
| `models/common/masks.py` | mask mods、varlen metadata | A2，`attention.py` 拆分 |
| `models/common/moe.py`（MoE 本体/experts/balance loss）+ `routers.py` + `balancing.py` | router、experts、MoE、balance loss、bias 更新钩子 | A2，同文件 |
| `models/common/multimodal.py` | vision/text fusion helpers | A2，同文件 |
| ~~`models/common/param_init.py`~~ | init context/std helper | 已于 2026-09-25 删除（死代码） |
| `models/common/qkv.py` | fused QKV 与 state hooks | A2，`attention.py` 拆分 |
| `models/common/rope.py` | RoPE 全家族 | A2，同文件 |
| `models/common/scatter_add.py` | deterministic scatter-add autograd | A2，`ops/scatter_add.py` |
| `models/common/token_dispatcher.py` | local/all-to-all dispatchers + TorchAO 可选导入适配层；DeepEP/HybridEP 登记缺口（config 期 loud-raise） | A2，同文件 |
| `models/hf_wrapper.py` + `models/hf_factory.py` | wrapper/forward 与 config 构建/类解析/meta materialize/FLOPs | B，transformers backend model |
| `parallel/activation_checkpoint.py` | full/selective AC | A2，distributed AC |
| `accelerator/collectives.py` | reductions、timeouts、grad norm | C；上游 collective 仅供意图比较 |
| `parallel/context_parallel/apply.py` | `apply_cp` | C，独立 HF 编排层 |
| `parallel/context_parallel/cp_kernel.py` | `CPFlexKernel` 与 seq/head autograd | C，CP flex attention 组合实现 |
| `parallel/context_parallel/input_shard.py` | CP/TP batch 和 mask sharding | C，独立输入分片层 |
| `accelerator/dist.py` | object collectives、all_reduce/gather、collect_results | C，vendored 自 OpenMMLab `mmengine.dist`（非 torchtitan 来源），已去 mmengine 化 |
| `accelerator/dist_utils.py` | init_dist 多 launcher（后端字符串由 `device.py` 单源驱动）、rank/group 查询、`cast_data_device` | C，同上 |
| `parallel/expert_parallel/apply.py` | `apply_ep` | B，模型 EP parallelize |
| `parallel/expert_parallel/swap.py`（编排）+ `probe.py`（探测）+ `convert.py`（转换） | HF MoE 探测、权重搬运与 swap | B，transformers backend `moe_replacement.py` |
| `parallel/fully_shard/fsdp.py` | FSDP engine、mesh 与 placement | A2，`distributed/fsdp.py` |
| `parallel/fully_shard/apply.py` | `apply_fsdp` HF driver | B，各模型 parallelize |
| `parallel/parallel_dims.py` | `ParallelDims` 与 mesh accessors | A2，distributed parallel dims |
| `parallel/parallelize.py`（2026-09-26 文件名对齐上游，原 parallelize_hf.py） | 五种并行的总装配 | B，transformers backend parallelize |
| `parallel/pipeline_parallel/pipeline.py` | FQN split 与 stage 构造 | A2，transformers backend pipeline |
| `parallel/pipeline_parallel/apply.py` | metadata、apply、schedule build | B，`distributed/pipeline_parallel.py` |
| `parallel/tensor_parallel/linear.py` | fused/fallback collective GEMM | A2，`models/common/async_linear.py`（原 `distributed/linear.py`，上游经 dist_gemm.py 搬迁改名） |
| `parallel/tensor_parallel/tp.py` + `apply.py` | HF plan realizer 与 `apply_tp` 入口 | B，各模型 TP plan（上游 `distributed/tensor_parallel.py` 已删除，无后继） |
| `config/`（顶层配置包） | 全部配置 dataclass | B，`config/configs.py` + 嵌套 Config |
| `trainer/train.py` | parse/main | B，根 `train.py` |
| `trainer/trainer.py`（+ `builder.py` 装配段 / `validation.py` / `pp_steps.py` / `batch.py`） | 完整训练生命周期 | B，根 `trainer.py` + `training_engine.py` |
| `utils/batch_invariant.py` | batch-invariant getter/setter | C，上游开关散在 trainer/config |

公开面约定：稳定面 = `hpmesh.HybridMeshConfig` / `hpmesh.Trainer` /
`hpmesh.config.*` / CLI；集成面与内部分级见 design doc §3.4。

组合判定约定：并行组合的支持/拒绝单一来源是 `parallel/matrix.py`
（每个组合一个普通函数 + 底部 `ENTRIES` 扁平表一行；config/assembly/probe
三阶段；配置期 `__post_init__` 原位调用，装配期与 probe 期由守卫点触发、
函数给出判定与文案）。

能力探测约定：torch 版本/环境探测集中在 `accelerator/capabilities.py`
（`has`/`require`，缺失经 `EnvironmentUnsupportedError` 报解锁指引）。

错误处理约定：fail-fast 类型层级在 `hpmesh/errors.py`——`ConfigError`
（配置非法，兼 `ValueError`）、`UnsupportedCombinationError`（组合拒绝，兼
`NotImplementedError`）、`EnvironmentUnsupportedError`（依赖缺失、文案带解锁
条件，兼 `NotImplementedError`）；可选包缺失保持 `ImportError`。
| `components/checkpointer/checkpoint_keys.py` | checkpoint state key 常量 | C |
| `accelerator/device.py` | 设备发现、backend 选择、厂商谓词、峰值显存查询 | C |
| `components/checkpointer/filesystem.py` | path/storage helpers | A1，`tools/filesystem.py` |
| `utils/gc.py` | `GarbageCollection` | B，`tools/utils.py` |
| `utils/logger_utils.py` | `get_logger`（彩色 formatter + 发射时 rank 过滤）、`get_distributed_rank` | C |
| `accelerator/monitoring.py` | device/memory/FLOPS helpers | C；部分意图可参考 `tools/utils.py` |
| `accelerator/spmd_context.py` | SPMD mesh 上下文 | C，pip `spmd_types` 适配 |

## 11. 上游同步检查清单

每次同步 TorchTitan 时按以下顺序执行：

1. `git -C <torchtitan> log <baseline>..HEAD -- torchtitan/`，先按本手册路径缩小范围。
2. A 类剥离 docstring 后比较 AST；不要用 diff 行数或 `quick_ratio()`。
3. B 类写出上游修复的不变量，例如"非法 label 必须在 collective 前失败"，再在 hpmesh
   的 seam 上实现，不复制其 Config/Module 外形。
4. C 类只检查调用者与测试，不根据同名或低 ratio 猜来源。
5. 对每个变更至少运行相关 CPU 单测、`ruff check`、`compileall`、`git diff --check`。
6. 并行语义变更必须增加多进程等价性测试；NPU/CUDA 专属 kernel 需要实际设备验证。
7. 更新文件级映射和本文的结论、限制及验证数字；不要把环境不支持写成代码已支持。

## 12. 已知验证边界

- 最新 `vllm-ascend` 镜像的 Torch 2.10 缺少新版 FSDP per-parameter mesh result，且
  Transformers 5.14 超出项目声明范围；EP×FSDP 等能力不能在该镜像完整验证。
- 镜像默认缺少部分项目依赖；临时补齐首批所需依赖后有 169 项通过，5 项因容器没有正确
  映射 Ascend driver/device、在 torch_npu 初始化阶段失败。未运行集合没有通过结论。
- CPU 单测不能证明 symmetric-memory、HCCL/NCCL、真实多卡 overlap 的性能与死锁安全；
  它们必须由 integration/equivalence 脚本和目标设备补足。
- AST 相似度只用于找候选，不是正确性证明。本文的"通过"来自不变量审计和测试证据。
- TorchTitan 的 `last_save_model_only=True` 默认值会生成不可续训的最终导出物；训练示例
  必须显式设为 `False`。不要把带 `.metadata` 的 model-only DCP 误判为完整训练
  checkpoint。
- 2026-09-22 的 Torch 2.10 复核发现 PP 1F1B 尚未通过轨迹等价性：首步 loss 一致，后续
  optimizer step 后偏离，4 step 最大约 `8.5e-3`。FSDP、TP、CP 和 EP grad norm 的对应
  2-rank 等价性通过；PP 在根因修复前不能归入"通过"。
- 2026-09-23 审计在 macOS 开发机上执行：torch 2.2.2 低于项目要求（>=2.12），缺
  `spmd_types`/`grain`，32 个测试文件收集即失败；当轮改动只有静态门禁、shim 级验证与
  vocab-loss 2-rank gloo 等价性覆盖，需在 torch>=2.12 环境重跑受影响套件。详见
  `hpmesh_torchtitan_alignment_audit_2026-09-23.md`（不在当前工作区）。
