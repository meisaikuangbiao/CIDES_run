"""Shared helpers for LLM-based judges."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from ..core.llm_client import ChatMessage, LLMClient
from ..core.schemas import DialogueTrace, ScoreDetail


logger = logging.getLogger(__name__)


def format_trace_for_judge(trace: DialogueTrace, *, max_chars: int = 8000) -> str:
    lines: list[str] = []
    total = 0
    for turn in trace.turns:
        speaker = "SUT" if turn.role == "assistant" else ("USER" if turn.role == "user" else "SYS")
        chunk = f"[{turn.index:02d} {speaker}] {turn.content}"
        if total + len(chunk) + 1 > max_chars:
            lines.append("[... truncated ...]")
            break
        lines.append(chunk)
        total += len(chunk) + 1
    return "\n".join(lines)


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a possibly noisy LLM response."""
    if not text:
        raise ValueError("Empty LLM response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found: {text!r}")
    candidate = cleaned[start : end + 1]
    return _loads_json_with_repair(candidate, original=text)


def _loads_json_with_repair(candidate: str, *, original: str) -> dict[str, Any]:
    attempts = [candidate]
    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
    if repaired not in attempts:
        attempts.append(repaired)
    # Some providers emit raw newlines inside string values. The strict=False
    # decoder accepts those while still rejecting structurally invalid JSON.
    last_err: Exception | None = None
    for item in attempts:
        try:
            return json.loads(item)
        except json.JSONDecodeError as exc:
            last_err = exc
            try:
                return json.JSONDecoder(strict=False).decode(item)
            except json.JSONDecodeError as strict_exc:
                last_err = strict_exc
    snippet = original.strip().replace("\n", "\\n")[:240]
    raise ValueError(f"Invalid judge JSON: {last_err}; snippet={snippet!r}")


def call_judge(
    client: LLMClient,
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    seed: Optional[int] = None,
    retries: int = 3,
    thinking: bool = False,
    reasoning_effort: Optional[str] = None,
) -> dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            attempt_seed = seed + attempt * 997 if seed is not None else None
            res = client.chat(
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                model=model,
                temperature=temperature,
                seed=attempt_seed,
                response_format={"type": "json_object"},
                cache=(attempt == 0),
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
            return extract_json(res.content)
        except Exception as exc:
            last_err = exc
            logger.warning("Judge attempt %s failed: %s", attempt + 1, exc)
    assert last_err is not None
    raise last_err


def coerce_details(
    payload: dict[str, Any],
    default_criterion: str,
    *,
    require_evidence: bool = False,
    skip_untriggered: bool = False,
) -> list[ScoreDetail]:
    raw_items = payload.get("items") or payload.get("details") or []
    details: list[ScoreDetail] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        if skip_untriggered and raw.get("triggered") is False:
            continue
        try:
            passed = bool(raw.get("passed", False))
            deduction = float(raw.get("deduction", 0.0 if passed else 1.0))
            deduction = max(0.0, min(1.0, deduction))
            evidence = str(raw.get("evidence_quote") or "")[:200]
            confidence = (
                float(raw["confidence"])
                if "confidence" in raw and raw["confidence"] is not None
                else None
            )
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
            rationale = str(raw.get("rationale") or "")[:400]
            if require_evidence and not passed and not evidence:
                evidence = "[Judge未提供原文证据]"
                rationale = (rationale + "；Judge输出缺少evidence_quote。").strip("；")
                confidence = min(confidence or 0.5, 0.5)
            details.append(
                ScoreDetail(
                    criterion_id=str(raw.get("criterion_id") or default_criterion),
                    label=str(raw.get("label") or default_criterion),
                    passed=passed,
                    deduction=deduction,
                    turn_ids=[int(x) for x in raw.get("turn_ids", []) if str(x).isdigit() or isinstance(x, int)],
                    evidence_quote=evidence,
                    rationale=rationale,
                    confidence=confidence,
                )
            )
        except Exception as exc:
            logger.warning("Skipping malformed judge item: %s (%s)", raw, exc)
    return details
