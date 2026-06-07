"""FAQ / knowledge accuracy judge (KAR)."""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..core.llm_client import LLMClient
from ..core.schemas import (
    DialogueTrace,
    DimensionScore,
    InstructionSpec,
    ScoreDetail,
)
from .common import call_judge, coerce_details, format_trace_for_judge


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一名知识库准确性评审员。我会提供 FAQ 知识点以及对话 trace。\n\n请评估每条 FAQ：\n- 仅当 trace 中用户触发了该话题（出现 trigger 关键词或语义匹配）时，才纳入评估。\n- 若 SUT 的应答覆盖了 key_points 全部要点，passed=true；缺失一项 deduction=0.4，捏造/错误信息 deduction=1.0。\n- evidence_quote 引用 SUT 的关键回答片段。\n\n只输出严格 JSON，不要 Markdown、注释、尾逗号或额外解释：\n{\n  "items": [\n    {\n      "criterion_id": "kar.<topic>",\n      "label": "topic",\n      "passed": bool,\n      "deduction": 0~1,\n      "turn_ids": [int],\n      "evidence_quote": "...",\n      "rationale": "...",\n      "confidence": 0~1,\n      "triggered": bool\n    }\n  ]\n}\n\n若没有任何 FAQ 被触发，请输出 {\"items\": []}。"""


def evaluate_knowledge(
    spec: InstructionSpec,
    trace: DialogueTrace,
    *,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: float = 0.2,
    require_evidence: bool = False,
) -> DimensionScore:
    if not spec.knowledge:
        return DimensionScore(id="kar", name="知识准确率", score=1.0, raw_score=1.0, details=[], source="rule")
    if client is None:
        return _offline_kar(spec, trace)
    knowledge_lines: list[str] = []
    for i, kp in enumerate(spec.knowledge):
        triggers = ", ".join(kp.triggers) if kp.triggers else "(无显式触发词)"
        key_points = "; ".join(kp.key_points) if kp.key_points else ""
        knowledge_lines.append(
            f"{i}. topic={kp.topic} | triggers=[{triggers}] | key_points={key_points}"
        )
    user_prompt = (
        "知识点列表：\n"
        + "\n".join(knowledge_lines)
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
            default_criterion="kar",
            require_evidence=require_evidence,
            skip_untriggered=True,
        )
    except Exception as exc:
        logger.warning("knowledge judge failed, fallback offline: %s", exc)
        dim = _offline_kar(spec, trace)
        dim.source = "fallback"
        dim.warnings.append(f"LLM knowledge judge failed: {exc}")
        return dim
    triggered = [d for d in details]
    if not triggered:
        return DimensionScore(id="kar", name="知识准确率", score=1.0, raw_score=1.0, details=[], source="llm")
    penalties = [d.deduction for d in triggered]
    score = max(0.0, 1.0 - sum(penalties) / max(1, len(triggered)))
    return DimensionScore(
        id="kar",
        name="知识准确率",
        score=round(score, 4),
        raw_score=round(score, 4),
        details=triggered,
        source="llm",
    )


def _offline_kar(spec: InstructionSpec, trace: DialogueTrace) -> DimensionScore:
    user_text = " ".join(t.content for t in trace.turns if t.role == "user")
    assistant_text = " ".join(t.content for t in trace.turns if t.role == "assistant")
    details: list[ScoreDetail] = []
    for kp in spec.knowledge:
        triggers = [t for t in kp.triggers if t]
        triggered = any(t in user_text for t in triggers) if triggers else False
        if not triggered and any(t in assistant_text for t in triggers):
            triggered = True
        if not triggered:
            continue
        key_tokens = []
        for point in kp.key_points:
            key_tokens.extend(re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,12}", point))
        hits = [tok for tok in key_tokens if tok in assistant_text]
        coverage = len(set(hits)) / max(1, len(set(key_tokens))) if key_tokens else 0.0
        passed = coverage >= 0.4
        details.append(
            ScoreDetail(
                criterion_id=f"kar.{kp.topic[:16]}",
                label=kp.topic[:30],
                passed=passed,
                deduction=round(1.0 - coverage, 2),
                turn_ids=[
                    t.index for t in trace.turns
                    if t.role == "assistant" and any(h in t.content for h in hits)
                ][:3],
                evidence_quote=", ".join(sorted(set(hits)))[:80],
                rationale=f"启发式 token 覆盖率={coverage:.0%}",
                confidence=0.4,
            )
        )
    if not details:
        return DimensionScore(id="kar", name="知识准确率", score=1.0, raw_score=1.0, details=[], source="offline")
    penalties = [d.deduction for d in details]
    score = max(0.0, 1.0 - sum(penalties) / max(1, len(details)))
    return DimensionScore(
        id="kar",
        name="知识准确率",
        score=round(score, 4),
        raw_score=round(score, 4),
        details=details,
        confidence=0.4,
        source="offline",
    )
