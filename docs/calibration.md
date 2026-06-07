# 校准方法

## 1. 为什么需要校准

LLM Judge 的可靠性是这套评测系统最大的不确定性来源。我们通过"人工标注 + 一致率指标"来量化评测过程的可信度，并在每次系统升级（换模型、改 prompt、改权重）后回归评估。

## 2. 工作流

1. **生成 trace**：用 `python -m src.cli.eval_run` 跑出一个 run，确保 `traces/<run_id>` 存在。
2. **挑样本**：选取覆盖不同指令、不同场景、不同得分段的 ≥20 个 case 进行人工标注。
3. **撰写标签**：在 `data/calibration/<run_id>__<case_id>.json` 写入人工对每个维度的 `passed` 判定（参见示例文件）。
4. **运行 meta-eval**：

   ```bash
   python -m src.cli.meta_eval --run reports/<run_id>
   ```

   输出每个维度的 N（标注样本数）、accuracy 与 Cohen's kappa。
5. **沉淀结果**：把表格复制到本文件末尾的"历史校准记录"段落，并标注模型版本与日期。

## 3. 标注规范

- 标注前请阅读 `data/instructions/<id>.json` 与 `<id>.md`，确认该指令下的硬/软约束。
- 标注时按指令的视角判定 SUT 是否合规：
  - GSR：业务目标是否达成？挂断是否恰当？
  - PCR：本场景应覆盖的 `target_nodes` 是否在 trace 中能找到对应表述？
  - BCA：是否在用户尚未表态时就讲分支信息？是否选错分支？
  - KAR：仅当用户问到 FAQ 时评估；SUT 是否覆盖 key_points、是否捏造数字？
  - HCR：开场白字数、禁用词、强制话术等机器可判项，可参考报告里的规则结果。
  - SCR：是否礼貌、自然、不重复？是否给用户足够的发言空间？
  - TER：是否按场景预期收束（忙碌/开车/拒绝/正常结束）？

## 4. 一致率阈值（建议）

| 指标 | 阈值 | 含义 |
| --- | --- | --- |
| accuracy | ≥0.80 | judge 与人工同向 |
| Cohen's kappa | ≥0.60 | 排除偶然一致后仍较强一致 |

低于阈值的维度需要排查：是否 prompt 不清晰、采样数太低、规则与 judge 重叠。

## 5. 历史校准记录

| 日期 | Run ID | 模型 | N | GSR κ | PCR κ | BCA κ | KAR κ | SCR κ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 待补充 | | | | | | | | |

## 6. 示例文件

`data/calibration/1__cooperative.example.json` 提供了一份示例，演示了标签字段格式。请基于自己的 run 生成新的样本文件，文件名建议：

```
<run_id>__<case_id>.json
```

例如 `run_demo__1__cooperative.json`。
