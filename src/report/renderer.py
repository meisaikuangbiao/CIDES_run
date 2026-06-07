"""Render RunReport into self-contained HTML and Markdown."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..core.schemas import RunReport
from ..evaluators.aggregator import DIMENSION_NAMES


def _radar_data(run_report: RunReport) -> dict:
    dims = run_report.aggregate.get("dimensions", {})
    order = ["gsr", "pcr", "bca", "kar", "hcr", "scr", "ter", "rob"]
    theta = [DIMENSION_NAMES.get(k, k) for k in order if k in dims]
    r = [dims[k]["mean"] for k in order if k in dims]
    if r:
        theta.append(theta[0])
        r.append(r[0])
    return {
        "type": "scatterpolar",
        "r": r,
        "theta": theta,
        "fill": "toself",
        "name": "维度均值",
    }


def _md_cell(value: object, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|")
    return text[:limit]


def render_html(
    run_report: RunReport,
    *,
    template_dir: str | Path = Path("src/report/templates"),
    output_path: Optional[str | Path] = None,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    html = template.render(
        run_report=run_report,
        radar_data=json.dumps(_radar_data(run_report), ensure_ascii=False),
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
    return html


def render_markdown(
    run_report: RunReport,
    *,
    output_path: Optional[str | Path] = None,
) -> str:
    lines: list[str] = []
    agg = run_report.aggregate
    lines.append(f"# 对话指令评测报告 · {run_report.run_id}")
    lines.append("")
    lines.append(f"- 生成时间: {run_report.created_at}")
    lines.append(f"- 样本数: {agg.get('n_cases', 0)}")
    lines.append(
        f"- 总分（加权均值）: **{agg.get('overall_mean', 0):.3f}**, "
        f"95% CI: [{agg.get('confidence_interval', [0, 0])[0]:.3f}, "
        f"{agg.get('confidence_interval', [0, 0])[1]:.3f}]"
    )
    cfg = run_report.config or {}
    lines.append(
        f"- 模型: SUT={cfg.get('models', {}).get('sut', '-')}, "
        f"Judge={cfg.get('models', {}).get('judge', '-')}"
    )
    lines.append(
        f"- 后端: SUT={cfg.get('sut_backend', '-')}, "
        f"UserSim={cfg.get('user_sim_backend', '-')}, Judge={cfg.get('judge_backend', '-')}"
    )
    lines.append(
        f"- 参数: seed={cfg.get('seed', '-')}, max_turns={cfg.get('max_turns', '-')}, "
        f"judge_samples={cfg.get('judge_samples', '-')}, "
        f"require_evidence={cfg.get('judge_require_evidence', '-')}"
    )
    qg = agg.get("quality_gates", {})
    if qg:
        gate = "通过" if qg.get("passed") else "需复核"
        lines.append(
            f"- 质量门禁: **{gate}**；低置信={qg.get('low_confidence_count', 0)}，"
            f"fallback/警告={qg.get('fallback_or_warning_count', 0)}，"
            f"高分歧={qg.get('high_disagreement_count', 0)}"
        )
    lines.append("")
    lines.append("## 维度均值")
    lines.append("| 维度 | 均值 | 最低 | 最高 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for dim_id in ["gsr", "pcr", "bca", "kar", "hcr", "scr", "ter", "rob"]:
        info = agg.get("dimensions", {}).get(dim_id)
        if not info:
            continue
        lines.append(
            f"| {info['name']} ({dim_id}) | {info['mean']:.3f} | {info['min']:.3f} | {info['max']:.3f} |"
        )
    lines.append("")
    lines.append("## 按指令汇总")
    lines.append("| 指令 | 样本 | 均值 | 最低 | 最高 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for iid, info in agg.get("by_instruction", {}).items():
        lines.append(
            f"| {iid} | {info['n']} | {info['mean']:.3f} | {info['min']:.3f} | {info['max']:.3f} |"
        )
    lines.append("")
    lines.append("## 失败归因 Top-10")
    lines.append("| 指令 | 场景 | 维度 | 规则 | 扣分 | 加权损失 | 证据/原因 |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- |")
    for f in agg.get("top_failures", [])[:10]:
        evidence = _md_cell(f.get("evidence_quote") or f.get("rationale") or "", 100)
        lines.append(
            f"| {f['instruction_id']} | {f['scenario_id']} | "
            f"{f.get('dim_name', f['dim_id'])} | {f['label']} | "
            f"{f['deduction']:.3f} | {f['weighted_loss']:.3f} | {evidence} |"
        )
    if agg.get("fallback_or_warnings"):
        lines.append("")
        lines.append("## 低置信 / Fallback / 分歧")
        lines.append("| 类型 | Case | 维度 | 说明 |")
        lines.append("| --- | --- | --- | --- |")
        for item in agg.get("fallback_or_warnings", [])[:10]:
            lines.append(
                f"| fallback/warning | {item.get('case_id')} | {item.get('dim_id')} | "
                f"{_md_cell('; '.join(item.get('warnings') or []) or item.get('source'), 120)} |"
            )
        for item in agg.get("high_disagreement", [])[:10]:
            lines.append(
                f"| disagreement | {item.get('case_id')} | {item.get('dim_id')} | "
                f"{_md_cell(str(item.get('disagreement')) + ' · ' + str(item.get('label')), 120)} |"
            )
    lines.append("")
    lines.append(
        f"## Case 通过情况 ({agg.get('n_passed', 0)}/{agg.get('n_cases', 0)} 通过, "
        f"通过率 {agg.get('case_pass_rate', 0):.1%})"
    )
    lines.append("")
    lines.append("## Case 详情")
    is_real_dialogue = (run_report.config or {}).get("run_mode") == "real_dialogue"
    for case in run_report.cases:
        pass_label = "通过" if case.passed else "未通过"
        case_title = (
            f"对话 {case.trace.case_id} · 任务{case.trace.instruction_id}"
            if is_real_dialogue
            else f"{case.trace.instruction_id} · {case.trace.scenario_id}"
        )
        lines.append(
            f"### {case_title} "
            f"({pass_label}, 总分 {case.weighted_total:.3f}, "
            f"会话 {case.sessions_used}, 终止 {case.trace.terminated_by})"
        )
        if case.fail_reasons:
            lines.append(f"- 未通过原因: {'; '.join(case.fail_reasons[:5])}")
        lines.append("")
        for dim in case.dimensions:
            confidence = (
                f", 置信度 {dim.confidence:.2f}" if dim.confidence is not None else ""
            )
            source = f", 来源 {dim.source}" if dim.source else ""
            warn = f", 警告 {len(dim.warnings)}" if dim.warnings else ""
            lines.append(f"- **{dim.name} ({dim.id})**: {dim.score:.3f}{confidence}{source}{warn}")
            for d in dim.details[:6]:
                pass_str = "✓" if d.passed else "✗"
                turns_str = ",".join(str(x) for x in d.turn_ids[:5])
                ev = _md_cell(d.evidence_quote, 80)
                disagreement = (
                    f", disagreement={d.disagreement:.2f}"
                    if d.disagreement is not None
                    else ""
                )
                lines.append(
                    f"  - [{pass_str}] {d.label} (turns={turns_str}, ded={d.deduction:.2f}{disagreement}) — {_md_cell(d.rationale, 160)} {('| ' + ev) if ev else ''}"
                )
        lines.append("")
        lines.append("**对话 trace:**")
        lines.append("")
        for turn in case.trace.turns:
            speaker = "SUT" if turn.role == "assistant" else "USER"
            lines.append(f"- [{turn.index:02d} {speaker}] {turn.content}")
        lines.append("")
    text = "\n".join(lines)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
    return text


def write_run_artifacts(
    run_report: RunReport,
    out_dir: str | Path,
    *,
    template_dir: str | Path = Path("src/report/templates"),
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "run_report.json"
    json_path.write_text(run_report.model_dump_json(indent=2), encoding="utf-8")
    md_path = out_dir / "report.md"
    render_markdown(run_report, output_path=md_path)
    html_path = out_dir / "report.html"
    render_html(run_report, template_dir=template_dir, output_path=html_path)
    return {"json": json_path, "md": md_path, "html": html_path}
