# 用户模拟器
- [x] 检查case是否符合要求，增加case的数量 — 新增指令3（商家运营）、case校验器、理论22 case
- [x] 搞清楚用户模拟器中各字段的含义 — `docs/scenario_fields.md` + Web UI 字段说明
- [x] 每个case独立判定通过/失败 — `CaseReport.passed` + `case_gate` 规则 + 通过率
- [x] 未通过续拨 + 长期记忆 — `max_sessions=2` + `prior_memory` 注入 SUT/UserSim
- [x] user-sut-judge 评测展示升级 — Trace 页三栏对照 + 报告中心一键复盘

# UI
- [x] 总览简洁界面 + 分步跳转 — Dashboard KPI/Case矩阵/快速入口 + 报告→Trace deep link

# 外呼
- [x] 检查case、增加case数量 — 3条外呼指令（骑手/课程平台/商家运营）
