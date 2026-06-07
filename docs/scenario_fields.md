# 用户模拟器字段说明

本文档说明 `ScenarioSpec`（场景画像）各字段含义，以及它们与评测维度的关系。

## ScenarioSpec 字段

| 字段 | 类型 | 含义 | 注入位置 |
|------|------|------|----------|
| `id` | string | 场景唯一标识，如 `cooperative`、`faq_drill` | Case ID、日志 |
| `name` | string | 场景中文名 | Web UI 展示 |
| `user_goal` | string | 用户在本通话中的目标 | UserSim system prompt |
| `behaviour` | string | 行为剧本（不含标准答案） | UserSim system prompt |
| `target_nodes` | list[str] | 本场景应覆盖的流程节点 ID | Judge PCR 评测范围 |
| `target_constraints` | list[str] | 本场景应检验的约束键 | Judge HCR/TER 过滤 |
| `target_knowledge` | list[str] | 应触发的 FAQ 主题 | Judge KAR 评测范围 |
| `adversarial_events` | list[str] | 施压事件（打断、诱导等） | UserSim prompt + TER |
| `required_user_signals` | list[str] | 用户必须发出的信号 | UserSim prompt |
| `expected_termination` | string? | 预期收尾方式 | Judge TER |
| `forbid_reveal` | bool | 禁止向 UserSim 泄露剧本 | 隔离设计 |

## ScoreDetail 字段（Judge 明细）

| 字段 | 含义 |
|------|------|
| `criterion_id` | 判定项 ID，如 `pcr.S1`、`hcr.max_chars` |
| `label` | 判定项中文描述 |
| `passed` | 该项是否通过 |
| `deduction` | 扣分（0~1） |
| `turn_ids` | 关联的对话轮次 |
| `evidence_quote` | 原文证据片段 |
| `rationale` | 判定理由 |
| `confidence` | Judge 置信度 |
| `disagreement` | 多采样投票分歧度 |

## Case 通过判定

每个 case 独立判定，规则见 `configs/default.yaml > case_gate`：

- `weighted_total >= 0.75`
- `GSR >= 0.85`
- `HCR >= 0.80`
- 所有 HCR 硬约束明细 `passed=true`

Run 级通过 = **全部 case 通过**（`case_pass_rate == 1.0`）。

## 续拨与长期记忆

当 `orchestrator.retry_on_fail=true` 且 case 未通过时，系统最多续拨 `max_sessions` 次：

- 上次通话摘要写入 SUT / UserSim 的 `prior_memory`
- `DialogueTrace.prior_sessions` 记录历次会话摘要
- 最终以最后一次会话的评测结果作为 case 判定
