# 人工标注校准集

本目录用于存放人工标注的样例，供 `meta_eval` CLI 用来评估 LLM judge 与人工的一致率。

## 文件格式

每条样例为一个独立的 JSON：

```json
{
  "case_id": "1__cooperative",
  "trace_path": "traces/run_20260522/1__cooperative.json",
  "labeller": "human-1",
  "labelled_at": "2026-05-22",
  "human_labels": {
    "gsr": {"passed": true, "rationale": "完成主要任务并以告别语收尾"},
    "pcr": {"passed": false, "rationale": "未覆盖 S2 节点（合同要求）"},
    "bca": {"passed": true},
    "kar": {"passed": true},
    "hcr": {"passed": false, "rationale": "开场白超过 30 字"},
    "scr": {"passed": true},
    "ter": {"passed": true}
  },
  "notes": "可选的自由文本备注"
}
```

## 标注指南摘要

| 维度 | 通过判定 | 关键参考 |
| --- | --- | --- |
| GSR 目标完成度 | 业务目标基本达成 | 是否传达了核心信息、用户是否被引导到正确分支 |
| PCR 路径覆盖 | 该场景应走的全部 target_nodes 都覆盖 | 看 SUT 是否依次说出节点要点 |
| BCA 分支条件适配 | 没有错误分支或提前剧透 | 是否在用户尚未透露分支条件时就给出特定分支信息 |
| KAR 知识准确率 | 用户触发 FAQ 时回答覆盖关键要点且不捏造 | 对照指令 Knowledge Points |
| HCR 硬约束 | 字数、禁用词、开场白等机器规则 | 与规则评估器对齐 |
| SCR 软约束 | 语气、自然度、不重复 | 主观判断 |
| TER 终止策略 | 是否按场景预期挂断 | 用户开车/坚持拒绝/任务完成等 |

## 标注流程

1. 在 `reports/<run_id>/run_report.json` 中找到目标 case，记下 `case_id` 与 `trace_path`。
2. 阅读 trace JSON 中的 turns，结合指令 spec（`data/instructions/<id>.json`）进行打分。
3. 在本目录新增一个 JSON 文件 `<run_id>__<case_id>.json`。
4. 运行 `python -m src.cli.meta_eval --run reports/<run_id>` 查看一致率。

完成 ≥20 条人工标注后，可计算 Cohen's kappa 并写入 `docs/calibration.md`。
