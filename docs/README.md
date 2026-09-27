# llmtuner 文档索引

六份文档，每份只有一个职责。**冲突时以当前源码与可复现测试为准**——确认事实后先改
文档，再继续迁移。

| 文档 | 回答什么问题 | 权威范围 |
|---|---|---|
| [`torchllmtuner_design.md`](./torchllmtuner_design.md) | 为什么这样设计、运行时契约是什么、哪些组合被拒绝 | 架构契约与当前运行边界（§7 记录环境门禁，§8 记录现状与已知缺口） |
| [`llmtuner_upstream_map.md`](./llmtuner_upstream_map.md) | 每个文件来自 torchtitan 的哪里、该按哪种策略维护 | 文件来源与 A/B/C/D 分类的**唯一权威**；版本与漂移章节记录每轮审计 |
| [`llmtuner_torchtitan_symbol_guide.md`](./llmtuner_torchtitan_symbol_guide.md) | 某个函数/类/方法该去上游哪里读、当前是否可信 | 符号级对应关系与正确性结论（§10 是全模块索引，§12 是验证边界） |
| [`llmtuner_trainer_walkthrough.md`](./llmtuner_trainer_walkthrough.md) | 训练主路径每一步对应上游哪里、结论是哪种 | 训练主路径（入口→装配→循环→单步→前反向→退出）的逐步对照与差异清单；A/B/C/D 分类仍以 `llmtuner_upstream_map.md` 为准 |
| [`llmtuner_torchtitan_alignment_workflow.md`](./llmtuner_torchtitan_alignment_workflow.md) | 下一轮对齐怎么执行、什么算完成 | Agent 执行流程、批次划分与完成门禁 |
| [`optimizer_checkpoint_format.md`](./optimizer_checkpoint_format.md) | 为什么 optimizer state 换了磁盘布局 | 单次破坏性格式变更的记录与影响面 |

## 阅读顺序

1. `llmtuner_upstream_map.md` 查文件的 A/B/C/D 分类——分类是操作指令，不是装饰。
2. `llmtuner_torchtitan_symbol_guide.md` 定位受影响的符号与已有结论。
3. `torchllmtuner_design.md` 确认要改的东西没有踩运行契约或支持边界。
4. 要按训练主路径读代码时，用 `llmtuner_trainer_walkthrough.md` 对位（含差异清单）。
5. 执行对齐任务时按 `llmtuner_torchtitan_alignment_workflow.md` 的批次与门禁走。
6. 只关心 checkpoint 兼容性时，直接读 `optimizer_checkpoint_format.md`。

## 维护规则

- 不把"环境未覆盖"写成"代码已支持"；未支持的组合保持 loud-raise，不静默退化。
- 不在源码里加固定 SHA 的 `# upstream:` 注释（SHA 会腐烂，表不会）。
- 一个文件只有一个主分类；文件内的混合逻辑写在该行的"改写点"列。
- 验证结果按日期 + 环境（Python / torch / backend）+ `passed / failed / skipped`
  记录，并区分 CPU、容器与目标设备。
- `docs/` 之外没有第二份事实来源：`README.md` 只做入门，不下结论。
