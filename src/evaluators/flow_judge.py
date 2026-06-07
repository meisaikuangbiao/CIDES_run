"""LLM judge for path coverage (PCR) and branch condition adaptation (BCA)."""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..core.llm_client import LLMClient
from ..core.schemas import (
    DialogueTrace,
    DimensionScore,
    InstructionSpec,
    ScenarioSpec,
    ScoreDetail,
)
from .common import call_judge, coerce_details, format_trace_for_judge


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一名外呼对话流程的评审员。你将拿到指令中规定的流程节点、当前用户场景需要覆盖的目标节点，以及一段对话 trace。\n\n请评估：\n1. 每个 target_node 是否在 trace 中被 SUT 正确覆盖（说出了节点所要求的核心信息）。\n2. SUT 是否提前泄露了不该说的分支信息（例如在对方还没说自己是不是负责人时就讲了非负责人的话术）。\n\n请只输出严格 JSON 对象，不要 Markdown、注释、尾逗号或额外解释。字段：\n{\n  "items": [\n    {\n      "criterion_id": "pcr.S1" 或 "bca.skip_S2"等,\n      "label": "节点S1：身份确认",\n      "passed": true/false,\n      "deduction": 0~1.0,\n      "turn_ids": [整数轮次],\n      "evidence_quote": "...",\n      "rationale": "...",\n      "confidence": 0~1\n    }\n  ]\n}\n\n约束：\n- target_nodes 中每个节点都必须有一条对应的 pcr.X item。\n- 若 SUT 把后续步骤信息提前讲出（违反 BCA），单独追加一条 criterion_id 以 bca. 开头的 item。\n- evidence_quote 必须是 trace 中的原文片段（≤80字）。\n- rationale 用中文简短说明。\n- 不要输出 items 以外的字段。\n"""


def _format_nodes(spec: InstructionSpec, scenario: ScenarioSpec) -> str:
    lines: list[str] = ["流程节点列表："]
    for node in spec.flow_nodes:
        marker = " [TARGET]" if node.id in scenario.target_nodes else ""
        lines.append(f"- {node.id}: {node.desc}{marker}")
    if spec.flow_edges:
        lines.append("\n分支边：")
        for edge in spec.flow_edges:
            lines.append(f"- {edge.source} → {edge.target} when {edge.condition}")
    lines.append(f"\n场景 {scenario.id} 的目标节点: {scenario.target_nodes}")
    if scenario.behaviour:
        lines.append(f"场景行为概要: {scenario.behaviour[:200]}")
    return "\n".join(lines)


def evaluate_flow(
    spec: InstructionSpec,
    scenario: ScenarioSpec,
    trace: DialogueTrace,
    *,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: float = 0.2,
    require_evidence: bool = False,
) -> tuple[DimensionScore, DimensionScore]:
    if client is None:
        return _offline_flow(spec, scenario, trace)
    user_prompt = (
        _format_nodes(spec, scenario)
        + "\n\n对话 trace：\n"
        + format_trace_for_judge(trace)
        + "\n\n请输出 JSON。"
    )
    try:
        payload = call_judge(
            client,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            model=model,
            temperature=temperature,
            seed=seed,
        )
        details = coerce_details(
            payload,
            default_criterion="pcr",
            require_evidence=require_evidence,
        )
    except Exception as exc:
        logger.warning("flow judge failed, falling back offline: %s", exc)
        pcr, bca = _offline_flow(spec, scenario, trace)
        pcr.source = "fallback"
        pcr.warnings.append(f"LLM flow judge failed: {exc}")
        bca.source = "fallback"
        bca.warnings.append(f"LLM flow judge failed: {exc}")
        return pcr, bca
    pcr_details = [d for d in details if d.criterion_id.startswith("pcr")]
    bca_details = [d for d in details if d.criterion_id.startswith("bca")]
    pcr_details = _complete_pcr_details(spec, scenario, pcr_details)
    passed = sum(1 for d in pcr_details if d.passed)
    total = len(scenario.target_nodes) or len(pcr_details) or 1
    score = passed / max(1, total)
    pcr_score = DimensionScore(
        id="pcr",
        name="路径覆盖率",
        score=round(score, 4),
        raw_score=round(score, 4),
        details=pcr_details,
        source="llm",
    )
    missing = [
        node_id
        for node_id in scenario.target_nodes
        if not any(d.criterion_id == f"pcr.{node_id}" for d in details)
    ]
    if missing:
        pcr_score.warnings.append(
            f"Judge缺少target_nodes输出，已按未覆盖补齐: {', '.join(missing)}"
        )
    if not bca_details:
        bca_score = DimensionScore(
            id="bca",
            name="分支条件适配",
            score=1.0,
            raw_score=1.0,
            details=[],
            source="llm",
            confidence=0.8,
            warnings=["Judge未返回bca项；按未发现分支违规处理。"],
        )
    else:
        penalties = [d.deduction for d in bca_details]
        score = max(0.0, 1.0 - sum(penalties) / max(1, len(bca_details)))
        bca_score = DimensionScore(
            id="bca",
            name="分支条件适配",
            score=round(score, 4),
            raw_score=round(score, 4),
            details=bca_details,
            source="llm",
        )
    return pcr_score, bca_score


def _complete_pcr_details(
    spec: InstructionSpec,
    scenario: ScenarioSpec,
    details: list[ScoreDetail],
) -> list[ScoreDetail]:
    by_id = {d.criterion_id: d for d in details}
    out: list[ScoreDetail] = []
    nodes = {node.id: node for node in spec.flow_nodes}
    for node_id in scenario.target_nodes:
        key = f"pcr.{node_id}"
        if key in by_id:
            out.append(by_id[key])
            continue
        node = nodes.get(node_id)
        label = f"节点{node_id}" + (f"：{node.desc[:30]}" if node else "")
        out.append(
            ScoreDetail(
                criterion_id=key,
                label=label,
                passed=False,
                deduction=1.0,
                turn_ids=[],
                evidence_quote="",
                rationale="Judge未返回该目标节点的覆盖判定，按未覆盖计入PCR分母。",
                confidence=0.0,
            )
        )
    if not scenario.target_nodes:
        return details
    return out


def _offline_flow(
    spec: InstructionSpec, scenario: ScenarioSpec, trace: DialogueTrace
) -> tuple[DimensionScore, DimensionScore]:
    """Heuristic offline coverage: a node is considered covered if any keyword
    from its description appears in SUT turns."""
    assistant_text = " ".join(t.content for t in trace.turns if t.role == "assistant")
    details: list[ScoreDetail] = []
    for node in spec.flow_nodes:
        if node.id not in scenario.target_nodes:
            continue
        keywords = _extract_keywords(node.desc)
        hits = [k for k in keywords if k and k in assistant_text]
        passed = len(hits) > 0
        details.append(
            ScoreDetail(
                criterion_id=f"pcr.{node.id}",
                label=f"节点{node.id}：{node.desc[:30]}",
                passed=passed,
                deduction=0.0 if passed else 1.0,
                turn_ids=[
                    t.index for t in trace.turns
                    if t.role == "assistant" and any(k in t.content for k in keywords)
                ][:3],
                evidence_quote=", ".join(hits)[:80] if hits else "",
                rationale=(
                    f"匹配关键词：{hits[:3]}"
                    if hits
                    else "未观察到节点关键词，可能未覆盖。"
                ),
                confidence=0.5,
            )
        )
    if not details:
        details.append(
            ScoreDetail(
                criterion_id="pcr.unknown",
                label="无可评估目标节点",
                passed=True,
                deduction=0.0,
                turn_ids=[],
                evidence_quote="",
                rationale="scenario 没有指定 target_nodes",
                confidence=0.0,
            )
        )
    passed = sum(1 for d in details if d.passed)
    pcr_score = DimensionScore(
        id="pcr",
        name="路径覆盖率",
        score=round(passed / max(1, len(details)), 4),
        raw_score=round(passed / max(1, len(details)), 4),
        details=details,
        confidence=0.5,
        source="offline",
    )
    bca_score = DimensionScore(
        id="bca",
        name="分支条件适配",
        score=1.0,
        raw_score=1.0,
        details=[
            ScoreDetail(
                criterion_id="bca.offline_skip",
                label="离线模式不评估分支泄露",
                passed=True,
                deduction=0.0,
                turn_ids=[],
                evidence_quote="",
                rationale="离线启发式无法判定分支泄露，跳过。",
                confidence=0.0,
            )
        ],
        confidence=0.0,
        source="offline",
        warnings=["离线模式无法可靠评估BCA。"],
    )
    return pcr_score, bca_score


def _extract_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"[\s，。:：；;!?\-\*]+", " ", text)
    tokens = [tok for tok in cleaned.split(" ") if 2 <= len(tok) <= 10]
    return tokens[:5]
