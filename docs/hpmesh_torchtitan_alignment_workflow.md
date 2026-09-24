# hpmesh 对齐 TorchTitan 的 Agent 执行流程

本文指导 agent 持续对齐 `hpmesh` 与 TorchTitan，同时保持 hpmesh 的设计边界和运行
正确性。它是执行流程，不取代三份事实文档：

- [`hpmesh_upstream_map.md`](./hpmesh_upstream_map.md)：文件来源与 A/B/C/D 分类的唯一权威。
- [`hpmesh_torchtitan_symbol_guide.md`](./hpmesh_torchtitan_symbol_guide.md)：函数、类和方法的对应关系。
- [`hybridmesh_design.md`](./hybridmesh_design.md)：架构契约、装配顺序和支持边界。

若三份文档与源码冲突，以当前源码及可复现测试为准；确认事实后先修正文档，再继续迁移。

文中 `<hpmesh>` / `<torchtitan>` 指两个仓库的检出根目录（本仓库即 `<hpmesh>` 的
上一级；torchtitan 检出位置依机器而定）。

## 1. 目标与非目标

### 1.1 目标

1. 识别 TorchTitan 在上次审计基线之后的有效变化。
2. 按 A/B/C/D 分类将其移植、适配、忽略或登记为能力缺口。
3. 用单测、多进程等价性测试和目标设备测试证明数学语义与运行契约正确。
4. 同步维护文件映射、符号导航、设计边界和验证记录。

### 1.2 非目标

- 不追求目录结构、类层次或 diff 行数与 TorchTitan 一致。
- 不重新引入 `Configurable`、TorchTitan `Module`、配置树或模型 registry。
- 不把有意裁剪的 quantization、structured logger、protocol 等内容当成遗漏。
- 不用静默退化掩盖未支持的并行组合。
- 不在源码中添加会腐烂的固定 SHA `# upstream:` 注释。

## 2. 不可破坏的 hpmesh 契约

每轮对齐前必须确认并保持：

1. 配置沿 `CLI -> HybridMeshConfig -> 顶层显式参数` 单向传递。
2. 并行装配由普通函数完成，非 PP 路径顺序保持：

   ```text
   apply_tp -> apply_ep -> apply_cp -> apply_ac -> compile -> apply_fsdp
   ```

3. HF wrapper 继续暴露 `tok_embeddings / layers / norm / lm_head / rotary_emb`
   五部件契约；`named_children()` 不负责改写 state-dict FQN。
4. 模型和并行底层不得读取 trainer 的全局运行配置。
5. TP、CP、EP、PP 和 FSDP 的未支持组合必须 fail fast。
6. `accelerator/spmd_context.py` 是活跃运行路径；旧的悬空链
   `parallel/spmd_shims.py -> parallel/sharding.py` 已于 2026-09-21 删除，不要以
   任何形式复活或与之混淆。

## 3. 执行前准备

### 3.1 工作区保护

1. 查看 `git status --short`，记录用户已有修改和未跟踪文件。
2. 不回退、不覆盖、不格式化与本轮任务无关的修改。
3. 将任务拆成可以独立验证的小批次；不要同时重写数学逻辑、装配顺序和测试基线。
4. 编辑源码或文档时使用小范围 patch。

### 3.2 加载环境

```bash
cd <hpmesh>
source ./set_env.sh   # 存在时
```

目标设备测试使用项目指定的最新 `vllm-ascend-env` 容器。当前 agent 无法进入该容器时，
继续完成静态检查和 CPU 可运行测试，并明确把设备验证列为未完成，不能声称已经通过。

### 3.3 记录审计基线

```bash
git -C <hpmesh> rev-parse HEAD
git -C <torchtitan> rev-parse HEAD
git -C <torchtitan> log <上次-torchtitan-基线>..HEAD -- torchtitan/
```

把双方提交、日期、Python/PyTorch/Transformers 版本写入本轮验证记录。固定 SHA 只进入
审计记录和文档的版本章节，不进入源码注释。

## 4. 建立变更清单

对 TorchTitan 的每个变更文件执行：

1. 在 `hpmesh_upstream_map.md` 查找文件级对应关系。
2. 在 symbol guide 中定位受影响的函数、类及 hpmesh 调用者。
3. 用 `rg` 检查 hpmesh 当前调用图、测试和公开导出。
4. 记录上游改动真正维护的不变量，而不是先复制实现。
5. 为每项变更建立记录：

| 字段 | 内容 |
|---|---|
| TorchTitan 文件/符号 | 上游路径和符号 |
| hpmesh 文件/符号 | 本地路径和符号 |
| 主分类 | A / B / C / D |
| 上游意图 | bug fix、边界检查、性能或新能力 |
| 必须保持的不变量 | shape、dtype、FQN、collective、梯度或生命周期语义 |
| 计划动作 | 移植、适配、无需动作或单独立项 |
| 验证方式 | 单测、等价性、容器或目标设备测试 |

不要仅凭同名文件或最高 AST ratio 判断来源。AST 相似度用于找候选，不是正确性证明。

## 5. 按分类执行

### 5.1 A 类：高保真移植

适用于算法和结构主要来自 TorchTitan 的文件。执行顺序：

1. 比较同名符号，再比较文件整体。
2. 剥离 docstring 后用 AST 结构比较辅助定位差异。
3. 逐项核对签名、shape、dtype、异常、collective、autograd 和 state-dict 行为。
4. 同步有效 bug fix，但移除上游 `Configurable`、`Module` 等 hpmesh 不采用的外形。
5. 运行该模块单测及对应等价性测试。

推荐批次：

1. filesystem、初始化、RoPE、mask、packing 等纯函数。
2. loss、optimizer、scheduler、checkpoint。
3. QKV、MoE、dispatcher、TP linear 和 FSDP engine。

### 5.2 B 类：语义适配

适用于相同意图但实现形状不同的文件。只迁移不变量、错误检查和数学语义，不复制上游
类层次。重点模块：

- `models/hf_wrapper.py`
- `parallel/parallelize_hf.py`
- `parallel/tensor_parallel/tp.py`
- `parallel/expert_parallel/*`
- `parallel/context_parallel/cp_kernel.py`（CP redistribution 与 kernel；原 `primitives.py` 已于 2026-09-23 删除）
- `parallel/fully_shard/fsdp_wrap.py`
- `parallel/pipeline_parallel/pp.py`
- `trainer/*` 与 `accelerator/mesh.py`

实现前必须写明对应不变量，例如：

- TP 权重布局和前后向 collective 必须互为对偶。
- CP 分片输出和梯度必须等价于全序列参考。
- EP swap 必须精确保留 HF 权重、router 语义和本地 expert 范围。
- PP stage FQN 必须稳定，optimizer/checkpoint 键不能跨 stage 冲突。
- loss 必须按全局有效 token 数归一化。

### 5.3 C 类：hpmesh 独有

C 类没有可机械同步的上游实现，只能依据 hpmesh 的调用图、设计契约和测试判断是否修改。

- `accelerator/spmd_context.py` 是活代码。
- CP 编排、flex kernel、random dataset、build factory 和工具模块不能因低相似度或
  同名文件被强行覆盖。

### 5.4 D 类：真正缺口

D 类必须独立设计和验收，不能伪装成单文件同步。当前清单以
[`hpmesh_upstream_map.md`](./hpmesh_upstream_map.md) D 类表为准，长期项包括：

- TorchTitan compile 层中的逐 block compile、async TP 和 regional compile。
- TP×MoE 的声明、专家权重分片和执行引擎。
- RegionAC 和 MemoryBudgetAC。

对 D 类先写设计提案，说明依赖、契约、组合矩阵、失败模式和测试计划。未经明确授权，
不要在普通上游同步任务中扩大范围实现这些能力。

## 6. 推荐实施批次

按风险由低到高执行，每批独立验证：

1. **纯函数与工具**：filesystem、初始化、RoPE、mask、packing。
2. **components**：loss、optimizer、scheduler、checkpoint、metrics、profiler。
3. **datasets**：source、loader、packing、文本和多模态处理。
4. **模型数学层**：QKV、MoE、dispatcher、grouped experts、aux loss。
5. **并行原语**：TP linear、CP redistribution、EP all-to-all、FSDP mesh。
6. **总装配**：HF wrapper、trainer、PP、checkpoint resume 和混合拓扑。

一批失败时停止扩展下一批，先定位是代码错误、测试假设、依赖缺失还是环境 API 不兼容。

## 7. 正确性验证流水线

### 7.1 静态门禁

```bash
ruff check hpmesh tests
python -m compileall -q hpmesh
git diff --check
```

同时检查：

- 非入口实现模块仍被 symbol guide 覆盖。
- upstream map 中一个文件只有一个主要分类。
- 没有新增 `Configurable`、TorchTitan `Module` 或固定 SHA 来源注释。
- 底层模块没有新增对 trainer/run config 的反向依赖。
- 未支持组合仍有明确异常或拒绝路径。

### 7.2 CPU 单测

先运行受影响模块，再运行完整可运行集合。记录 `passed / failed / skipped /
deselected`，并单列 optional dependency 或 PyTorch API 不匹配导致的未运行项。

"可运行集合通过"不能写成"全套测试通过"。失败项不得在没有证据时归因于环境。

### 7.3 多进程等价性

数学或分布式语义变化至少覆盖：

| 领域 | 必须验证的内容 |
|---|---|
| TP | forward/backward、replicated gradient、vocab loss、TP×FSDP |
| CP | KV all-gather、Ulysses、packed、CP×TP、梯度归约 |
| EP | HF 权重搬运、all-to-all、aux loss、拒绝的 grad clip、EP×FSDP |
| PP | 1F1B、Interleaved1F1B、full/resume checkpoint |
| Loss | vocab-parallel、chunked loss、全局 token 归一化和跨 rank 可见性 |

等价性测试必须满足：

1. 分片执行与单卡/全量参考在明确容差内一致。
2. 同时检查输出、loss、关键参数梯度和必要的 state-dict/FQN。
3. 包含 non-vacuity 断言，避免全零输入或全零梯度造成虚假通过。
4. collective 前的输入校验必须在所有 rank 对称执行，避免局部异常导致其他 rank 挂死。

### 7.4 目标设备与容器验证

在最新 `vllm-ascend-env` 中验证：

- HCCL/NCCL collective 和多卡进程生命周期。
- NPU/CUDA 专属 fused kernel 与 fallback 的选择。
- symmetric-memory 路径。
- 多卡 overlap、超时和死锁安全。
- checkpoint 保存、退出、恢复后的 loss 轨迹。

若容器中的 PyTorch 缺少所需私有 API，记录为"环境未覆盖"，保留 fail-fast，不可为了
让测试变绿而绕过 placement、DTensor 或 collective 语义检查。

## 8. 结果记录模板

每个批次完成后追加一份记录（长期记录单独成文，命名
`hpmesh_torchtitan_alignment_audit_<日期>.md`，并在 upstream map 的版本章节登记）：

```markdown
## <批次名称>

- hpmesh 基线：`<sha>`
- TorchTitan 基线：`<sha>`
- 环境：Python / PyTorch / Transformers / backend
- 涉及分类：A / B / C / D

### 变更

| hpmesh 符号 | TorchTitan 符号 | 上游意图 | 本地处理 |
|---|---|---|---|

### 验证

| 命令 | passed | failed | skipped/deselected | 备注 |
|---|---:|---:|---:|---|

### 未验证或受限

- <设备、依赖、组合或 API 边界>

### 文档更新

- [ ] upstream map
- [ ] symbol guide
- [ ] design document
```

## 9. 完成门禁

只有同时满足以下条件，才能宣告一批对齐完成：

1. 上游变化已经逐文件、逐关键符号审计。
2. A/B/C/D 主分类明确且无冲突。
3. hpmesh 设计契约和装配顺序没有被破坏。
4. 新增或改变的能力具有对应单测或等价性测试。
5. 未支持组合保持 loud-raise。
6. CPU、容器和目标设备结果被分别记录，没有扩大结论。
7. 静态门禁通过，相关测试无未解释失败。
8. 三份权威文档已按事实同步更新。
9. 工作区中用户原有的无关修改保持不变。

若任一项未满足，状态写为"部分完成"或"受限"，列出剩余动作，不得用"已完全对齐"
代替具体证据。

## 10. Agent 最终交付格式

最终回复应简洁包含：

1. 已对齐的文件和核心语义。
2. 修复或保留的关键不变量。
3. 实际执行的验证命令和结果。
4. skipped/deselected/环境不兼容项。
5. 尚未支持的组合和 loud-raise 状态。
6. 更新后的文档链接。

不要仅报告"测试通过"或"已与上游一致"；必须说明通过了哪些测试、在哪个环境中通过，
以及哪些能力仍未验证。
