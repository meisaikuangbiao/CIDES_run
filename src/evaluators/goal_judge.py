"""Overall task completion judge (GSR)."""
from __future__ import annotations

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
from .common import call_judge, format_trace_for_judge


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一名外呼对话整体表现评审员。请基于指令的任务目标、场景预期，以及 trace，判定 SUT 是否完成了本次通话目标。\n\n只输出严格 JSON，不要 Markdown、注释、尾逗号或额外解释：\n{\n  "score": 0~1 之间的小数,\n  "label": "未完成|部分完成|完成",\n  "evidence_quote": "...",\n  "rationale": "...",\n  "confidence": 0~1,\n  "turn_ids": [int]\n}\n\n评分参考：\n- 完成（score≥0.85）：业务目标达成且通话以恰当方式结束。\n- 部分完成（0.4~0.85）：完成主要信息传达但缺少确认或挂断不当。\n- 未完成（<0.4）：核心信息缺失或场景预期未达成。\n"""


def evaluate_goal(
    spec: InstructionSpec,
    scenario: ScenarioSpec,
    trace: DialogueTrace,
    *,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: float = 0.2,
    require_evidence: bool = False,
) -> DimensionScore:
    if client is None:
        return _offline_goal(spec, scenario, trace)
    user_prompt = (
        f"业务任务：{spec.task}\n"
        f"场景：{scenario.id} - {scenario.name}\n"
        f"场景目标：{scenario.user_goal}\n"
        f"预期收尾：{scenario.expected_termination or '按指令收尾'}\n\n"
        f"对话 trace：\n{format_trace_for_judge(trace)}\n\n请输出 JSON。"
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
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        evidence = str(payload.get("evidence_quote") or "")[:200]
        confidence = (
            float(payload["confidence"])
            if "confidence" in payload and payload["confidence"] is not None
            else None
        )
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
        rationale = str(payload.get("rationale") or "")[:400]
        warnings: list[str] = []
        if require_evidence and score < 0.85 and not evidence:
            evidence = "[Judge未提供原文证据]"
            rationale = (rationale + "；Judge输出缺少evidence_quote。").strip("；")
            confidence = min(confidence or 0.5, 0.5)
            warnings.append("Judge缺少失败证据，已降低置信度。")
        detail = ScoreDetail(
            criterion_id="gsr.overall",
            label=str(payload.get("label") or "任务完成度"),
            passed=score >= 0.85,
            deduction=round(1.0 - score, 4),
            turn_ids=[int(x) for x in (payload.get("turn_ids") or []) if str(x).isdigit() or isinstance(x, int)],
            evidence_quote=evidence,
            rationale=rationale,
            confidence=confidence,
        )
        return DimensionScore(
            id="gsr",
            name="目标完成度",
            score=round(score, 4),
            raw_score=round(score, 4),
            details=[detail],
            confidence=detail.confidence,
            source="llm",
            warnings=warnings,
        )
    except Exception as exc:
        logger.warning("goal judge failed, fallback offline: %s", exc)
        dim = _offline_goal(spec, scenario, trace)
        dim.source = "fallback"
        dim.warnings.append(f"LLM goal judge failed: {exc}")
        return dim


def _offline_goal(
    spec: InstructionSpec, scenario: ScenarioSpec, trace: DialogueTrace
) -> DimensionScore:
    """Heuristic GSR offline scoring used in stub mode."""
    if not trace.turns:
        return DimensionScore(
            id="gsr",
            name="目标完成度",
            score=0.0,
            raw_score=0.0,
            details=[
                ScoreDetail(
                    criterion_id="gsr.offline",
                    label="任务完成度（离线）",
                    passed=False,
                    deduction=1.0,
                    turn_ids=[],
                    evidence_quote="",
                    rationale="对话 trace 为空。",
                    confidence=0.2,
                )
            ],
            source="offline",
        )
    last_assistant = next(
        (t for t in reversed(trace.turns) if t.role == "assistant"), None
    )
    farewell = (
        last_assistant is not None
        and bool(
            re.search(
                r"(再见|稍后再打|今天先聊到这|多注意身体|祝.*顺利|挂断)",
                last_assistant.content,
            )
        )
    )
    valid_turns = sum(1 for t in trace.turns if t.role == "assistant" and t.content)
    coverage = 0.5 if valid_turns >= 2 else 0.2
    if farewell:
        coverage += 0.3
    if trace.terminated_by in {"sut_goodbye", "user_end_call"}:
        coverage += 0.2
    score = round(min(1.0, coverage), 4)
    label = "完成" if score >= 0.85 else ("部分完成" if score >= 0.4 else "未完成")
    return DimensionScore(
        id="gsr",
        name="目标完成度",
        score=score,
        raw_score=score,
        details=[
            ScoreDetail(
                criterion_id="gsr.offline",
                label=label,
                passed=score >= 0.85,
                deduction=round(1.0 - score, 4),
                turn_ids=[last_assistant.index] if last_assistant else [],
                evidence_quote=last_assistant.content[:80] if last_assistant else "",
                rationale=(
                    f"启发式：assistant 轮次={valid_turns}, "
                    f"含告别={farewell}, 终止={trace.terminated_by}"
                ),
                confidence=0.4,
            )
        ],
        confidence=0.4,
        source="offline",
    )
