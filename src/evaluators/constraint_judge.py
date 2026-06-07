"""Soft-constraint LLM judge (SCR)."""
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
from .common import call_judge, coerce_details, format_trace_for_judge


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一名对话风格的评审员。请基于指令的软约束列表，逐条判断 SUT 的整体表现：\n\n只输出严格 JSON，不要 Markdown、注释、尾逗号或额外解释：\n{\n  "items": [\n    {\n      "criterion_id": "soft.0",\n      "label": "软约束原文摘要",\n      "passed": bool,\n      "deduction": 0~1,\n      "turn_ids": [问题轮次],\n      "evidence_quote": "...",\n      "rationale": "...",\n      "confidence": 0~1\n    }\n  ]\n}\n\n规则：\n- 软约束按数组顺序对应 criterion_id soft.0、soft.1、…\n- 若违反，请引用原文证据；passed=false。\n- 若约束在 trace 中未被触发（例如打断、过渡），可以 passed=true 但 confidence 较低。\n- 不要新增不在约束列表中的项目。\n"""


def evaluate_constraints(
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
    if not spec.constraints.soft:
        return DimensionScore(id="scr", name="软约束合规", score=1.0, raw_score=1.0, details=[], source="rule")
    if client is None:
        return _offline_soft(spec, trace)
    soft_text = "\n".join(f"{i}. {s}" for i, s in enumerate(spec.constraints.soft))
    user_prompt = (
        f"软约束列表：\n{soft_text}\n\n对话 trace：\n{format_trace_for_judge(trace)}\n\n请输出 JSON。"
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
            default_criterion="soft",
            require_evidence=require_evidence,
        )
    except Exception as exc:
        logger.warning("constraint judge failed, fallback offline: %s", exc)
        dim = _offline_soft(spec, trace)
        dim.source = "fallback"
        dim.warnings.append(f"LLM soft-constraint judge failed: {exc}")
        return dim
    if not details:
        dim = _offline_soft(spec, trace)
        dim.source = "fallback"
        dim.warnings.append("LLM soft-constraint judge returned no details.")
        return dim
    penalties = [d.deduction for d in details]
    score = max(0.0, 1.0 - sum(penalties) / max(1, len(details)))
    return DimensionScore(
        id="scr",
        name="软约束合规",
        score=round(score, 4),
        raw_score=round(score, 4),
        details=details,
        source="llm",
    )


def _offline_soft(spec: InstructionSpec, trace: DialogueTrace) -> DimensionScore:
    assistant_turns = [t for t in trace.turns if t.role == "assistant" and t.content]
    details: list[ScoreDetail] = []
    for idx, rule in enumerate(spec.constraints.soft):
        passed = True
        rationale = "离线启发式未检测到违规。"
        turn_ids: list[int] = []
        evidence = ""
        if "重复" in rule:
            seen: dict[str, int] = {}
            for t in assistant_turns:
                seen[t.content] = seen.get(t.content, 0) + 1
            dups = {k: v for k, v in seen.items() if v >= 2}
            if dups:
                passed = False
                rationale = "出现重复回复（启发式：相同字符串≥2次）。"
                dup_text = next(iter(dups))
                turn_ids = [t.index for t in assistant_turns if t.content == dup_text][:3]
                evidence = dup_text[:60]
        if any(kw in rule for kw in ("语气", "随意", "口语")):
            formal = [t for t in assistant_turns if re.search(r"敬启|兹|您方|惠存|烦请", t.content)]
            if formal:
                passed = False
                rationale = "包含过于正式的用词。"
                turn_ids = [formal[0].index]
                evidence = formal[0].content[:60]
        details.append(
            ScoreDetail(
                criterion_id=f"soft.{idx}",
                label=rule[:30],
                passed=passed,
                deduction=0.0 if passed else 0.5,
                turn_ids=turn_ids,
                evidence_quote=evidence,
                rationale=rationale,
                confidence=0.4,
            )
        )
    penalties = [d.deduction for d in details]
    score = max(0.0, 1.0 - sum(penalties) / max(1, len(details)))
    return DimensionScore(
        id="scr",
        name="软约束合规",
        score=round(score, 4),
        raw_score=round(score, 4),
        details=details,
        confidence=0.4,
        source="offline",
    )
