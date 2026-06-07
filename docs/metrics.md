# 指标体系

## 1. 维度速查

| ID | 名称 | 类型 | 默认权重 | 含义 |
| --- | --- | --- | --- | --- |
| GSR | 目标完成度 | LLM Judge | 0.20 | SUT 是否完成本次外呼业务目标 |
| PCR | 路径覆盖率 | LLM Judge | 0.20 | 场景的 target_nodes 是否被覆盖 |
| BCA | 分支条件适配 | LLM Judge | 0.10 | 是否按用户回应走正确分支，是否提前泄露 |
| KAR | 知识准确率 | LLM Judge | 0.15 | 用户触发 FAQ 时回答的覆盖率 / 准确性 |
| HCR | 硬约束合规 | 规则 | 0.15 | 字数、禁用词、强制回复、开场白关键片段 |
| SCR | 软约束合规 | LLM Judge | 0.10 | 语气、自然度、避免重复、过渡语 |
| TER | 终止策略合规 | 规则 + LLM | 0.05 | 忙碌、开车、坚持拒绝、自然结束 |
| ROB | 稳健性 | 派生 | 0.05 | 同指令多场景下的分数稳定性 |

权重在 `configs/default.yaml > metrics.weights` 中可改；总分 = 各维度 × 权重的加权和，截断到 [0, 1]。

## 2. 各维度计算细则

### 2.1 GSR — 目标完成度

- **来源**：`goal_judge.py`，单条 LLM 评判输出 `score ∈ [0,1]`、label、evidence、rationale。
- **结构化输出**：通过 `response_format=json_object` 强制；schema 校验失败重试 3 次。
- **多采样**：在 `voting.run_voted_single` 下默认采样 `judge.samples=3` 次取均值；置信度 = 1 − 方差。
- **离线 fallback**：`assistant 轮次 ≥ 2 ? 0.5 : 0.2` + `farewell ? 0.3 : 0` + `terminated_by 命中 ? 0.2 : 0`。

### 2.2 PCR — 路径覆盖率

- **判定单元**：`scenario.target_nodes` 中的每个节点。
- **LLM Judge**：返回 `pcr.<node_id>` items，每条 `passed` 计入分子。`score = passed_count / total_target_nodes`。
- **离线 fallback**：从节点描述抽取 keywords，看 SUT 是否在任意 turn 中说过其中一个；命中即视为覆盖（用于 stub demo，置信度记为 0.5）。

### 2.3 BCA — 分支条件适配

- **判定单元**：LLM Judge 自行追加 `bca.*` items（例如"在用户尚未透露身份前提前讲非负责人话术"）。
- **得分**：`max(0, 1 − Σdeduction / N)`；如果 LLM 没追加任何项，记为 1.0。
- **离线模式**：跳过 BCA 评估并标注 confidence=0。

### 2.4 KAR — 知识准确率

- **判定**：仅当用户在某轮触发了 FAQ 主题（trigger 关键词或语义匹配）时才纳入；
- **LLM Judge**：对每条 FAQ 给出 `passed/deduction/turn_ids/evidence/rationale`；
- **离线启发式**：抽取 key_points 中的中文/英文 token，看 SUT 文本里命中比例 ≥0.4 即视为通过。

### 2.5 HCR — 硬约束合规（纯代码）

- `hard.max_chars_per_reply`：每个 SUT 回复 ≤ limit (允许 20% 软容忍)；超出 hard limit 直接扣 1，超过软容忍 0.5；
- `hard.forbidden_words`：每个回复检查命中；
- `hard.no_discount_promise`：检测"优惠券/折扣/返利/打折/红包/返现"等正则；
- `hard.opening_keywords`：第一句必须包含若干关键短语；缺一项扣 0.3，最多 1.0；
- `hard.required_out_of_scope_reply`：用户问越权问题时 SUT 必须使用指定话术（含子串校验）；
- `hard.required_replies`：触发条件 → 强制话术 列表；
- 维度得分 = `1 − 平均扣分`。

### 2.6 SCR — 软约束合规

- **判定单元**：指令 `constraints.soft` 列表逐条；
- **LLM Judge**：每条返回 `soft.<i>`，passed/deduction/turn_ids/evidence/rationale；
- **离线启发式**：检测重复回复（相同字符串≥2 次）、过度正式词、其它默认 pass + 低置信度。

### 2.7 TER — 终止策略合规

- 主体规则化：检查 `trace.terminated_by ∈ {sut_goodbye, user_end_call}` 且最后助手回复包含告别语；
- 仅在场景声明 `expected_termination` 时启用；
- 离线模式即可给出确定性结论；上层不再额外走 LLM。

### 2.8 ROB — 稳健性

- 在所有 case 评估完成后由 `attach_robustness` 计算：
  - 对每条指令的所有 case 取 `weighted_total`；
  - `rob = max(0, 1 − min(1, σ × 2))`；
  - 写回每个 case 的 `rob` 维度（同一指令下相同分数）；
  - 重新计算每个 case 的加权总分。

## 3. 总分与置信区间

- `weighted_total = Σ score_dim × weight_dim`，截断到 [0,1]；
- run 级别 `overall_mean = mean(weighted_total)`；
- `confidence_interval`：bootstrap 200 次（`metrics.bootstrap_iters`），置信水平 `1 − bootstrap_alpha = 95%`；
- run-level 输出还包括按维度 / 按指令的 mean/min/max，以及 `top_failures`（按 `weighted_loss = deduction × weight` 排序的扣分项）。

## 4. 失败归因 Top-K

- 每条扣分项的"加权损失"= `deduction × weight_dim`；
- Top-K 默认取 8（`configs/default.yaml > report.top_k_failures`）；
- 失败项写入报告中部表格与 case 详情的扣分明细；report 中按指令 × 场景列出。

## 5. 调参建议

- 模型偏弱时可先把 LLM Judge 全替换为 offline，再针对薄弱维度逐个开启 LLM；
- `judge.samples` 越大稳定性越高但成本线性增加；建议 ≥3；
- 调整权重时建议同时更新 `docs/metrics.md` 与 `README.md`，并保留旧 run 的 `run_report.json` 以便 A/B 比较。
