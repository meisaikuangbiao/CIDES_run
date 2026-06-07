"""Deterministic rule-based evaluators for the SUT's hard constraints.

Each evaluator returns a list of ``ScoreDetail``. The aggregator converts
them into a ``DimensionScore`` with a 0-1 score where 1.0 means perfect
compliance. Every detail carries:

* the criterion id (so the report can cite the exact rule);
* a list of turn indices where the violation occurred;
* an evidence quote and a human-readable rationale.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from ..core.schemas import (
    DialogueTrace,
    DimensionScore,
    InstructionSpec,
    ScenarioSpec,
    ScoreDetail,
    TurnRecord,
)


logger = logging.getLogger(__name__)


DISCOUNT_REGEXES = (
    re.compile(r"优惠券"),
    re.compile(r"折扣"),
    re.compile(r"返利"),
    re.compile(r"打.*?折"),
    re.compile(r"返现"),
    re.compile(r"红包"),
)


def _assistant_turns(trace: DialogueTrace) -> list[TurnRecord]:
    return [t for t in trace.turns if t.role == "assistant" and t.content]


def _user_turns(trace: DialogueTrace) -> list[TurnRecord]:
    return [t for t in trace.turns if t.role == "user" and t.content]


def _target_enabled(scenario: Optional[ScenarioSpec], criterion: str) -> bool:
    if scenario is None or not scenario.target_constraints:
        return True
    targets = set(scenario.target_constraints)
    aliases = {
        "hard.max_chars_per_reply": {"hard.max_chars_per_reply"},
        "hard.forbidden_words": {"hard.forbidden_words"},
        "hard.no_discount_promise": {"hard.no_discount_promise"},
        "hard.opening_keywords": {"hard.opening_keywords"},
        "hard.required_out_of_scope_reply": {"hard.required_out_of_scope_reply"},
        "hard.required_replies": {"hard.required_replies"},
        "termination": {"termination"},
    }
    return bool(aliases.get(criterion, {criterion}) & targets)


def check_max_chars(spec: InstructionSpec, trace: DialogueTrace) -> list[ScoreDetail]:
    limit = spec.constraints.hard.max_chars_per_reply
    if not limit:
        return []
    details: list[ScoreDetail] = []
    for turn in _assistant_turns(trace):
        chars = len(turn.content)
        passed = chars <= int(limit * 1.2)  # 20% tolerance for soft limit
        hard_passed = chars <= int(limit)
        deduction = 0.0 if hard_passed else (0.5 if passed else 1.0)
        details.append(
            ScoreDetail(
                criterion_id="hard.max_chars_per_reply",
                label=f"每轮回复≤{limit}字",
                passed=hard_passed,
                deduction=deduction,
                turn_ids=[turn.index],
                evidence_quote=turn.content[:80],
                rationale=(
                    f"第{turn.index}轮回复{chars}字，限制{limit}字。"
                    if not hard_passed
                    else f"第{turn.index}轮回复{chars}字，未超限。"
                ),
            )
        )
    return details


def check_forbidden_words(
    spec: InstructionSpec, trace: DialogueTrace
) -> list[ScoreDetail]:
    words = [w for w in (spec.constraints.hard.forbidden_words or []) if w]
    if not words:
        return []
    details: list[ScoreDetail] = []
    for turn in _assistant_turns(trace):
        hits: list[str] = []
        for word in words:
            if word and word in turn.content:
                hits.append(word)
        passed = not hits
        details.append(
            ScoreDetail(
                criterion_id="hard.forbidden_words",
                label="禁用词检查",
                passed=passed,
                deduction=0.0 if passed else 1.0,
                turn_ids=[turn.index],
                evidence_quote=turn.content[:80],
                rationale=(
                    f"第{turn.index}轮命中禁用词：{', '.join(hits)}"
                    if hits
                    else f"第{turn.index}轮未命中禁用词。"
                ),
            )
        )
    return details


def check_no_discount(
    spec: InstructionSpec, trace: DialogueTrace
) -> list[ScoreDetail]:
    if not spec.constraints.hard.no_discount_promise:
        return []
    details: list[ScoreDetail] = []
    for turn in _assistant_turns(trace):
        matches: list[str] = []
        for rx in DISCOUNT_REGEXES:
            m = rx.search(turn.content)
            if m:
                matches.append(m.group(0))
        passed = not matches
        details.append(
            ScoreDetail(
                criterion_id="hard.no_discount_promise",
                label="不承诺折扣/优惠券",
                passed=passed,
                deduction=0.0 if passed else 1.0,
                turn_ids=[turn.index],
                evidence_quote=turn.content[:80],
                rationale=(
                    f"第{turn.index}轮提到承诺/优惠：{', '.join(matches)}"
                    if matches
                    else f"第{turn.index}轮未涉及优惠承诺。"
                ),
            )
        )
    return details


def check_opening_keywords(
    spec: InstructionSpec, trace: DialogueTrace
) -> list[ScoreDetail]:
    keywords = list(spec.constraints.hard.opening_keywords or [])
    if not keywords:
        return []
    assistant = _assistant_turns(trace)
    if not assistant:
        return []
    first = assistant[0]
    missing = [kw for kw in keywords if kw and not _keyword_matches(kw, first.content)]
    passed = not missing
    deduction = 0.0 if passed else min(1.0, 0.3 * len(missing))
    return [
        ScoreDetail(
            criterion_id="hard.opening_keywords",
            label="开场白关键短语覆盖",
            passed=passed,
            deduction=deduction,
            turn_ids=[first.index],
            evidence_quote=first.content[:120],
            rationale=(
                f"开场白缺失关键片段：{', '.join(missing)}"
                if missing
                else "开场白覆盖了全部关键片段。"
            ),
        )
    ]


def _keyword_matches(keyword: str, content: str) -> bool:
    if keyword in content:
        return True
    # Old parsed specs may contain phrases such as "你好，请问是吗" after
    # removing ${rider_name}. Treat the missing slot as a short wildcard.
    if "是吗" in keyword:
        pattern = re.escape(keyword).replace("是吗", r"是.{0,20}吗")
        if re.search(pattern, content):
            return True
    compact_kw = re.sub(r"\s+", "", keyword)
    compact_content = re.sub(r"\s+", "", content)
    return bool(compact_kw and compact_kw in compact_content)


def check_out_of_scope_reply(
    spec: InstructionSpec, trace: DialogueTrace
) -> list[ScoreDetail]:
    required = spec.constraints.hard.required_out_of_scope_reply
    if not required:
        return []
    out_of_scope_keywords = (
        "年终奖",
        "团购",
        "换站点",
        "天气",
        "工资",
        "提成",
        "请假",
        "公司",
    )
    user_turns = _user_turns(trace)
    triggered = False
    triggered_user_turn: int | None = None
    for ut in user_turns:
        if any(k in ut.content for k in out_of_scope_keywords):
            triggered = True
            triggered_user_turn = ut.index
            break
    if not triggered:
        return []
    assistant = _assistant_turns(trace)
    next_assistant = next(
        (a for a in assistant if a.index > (triggered_user_turn or -1)), None
    )
    if next_assistant is None:
        return [
            ScoreDetail(
                criterion_id="hard.required_out_of_scope_reply",
                label="越权问题兜底话术",
                passed=False,
                deduction=1.0,
                turn_ids=[triggered_user_turn or 0],
                evidence_quote="用户提出越权问题但模型未回复",
                rationale="用户提出越权问题，但 SUT 未给出兜底话术。",
            )
        ]
    snippet = required[: max(8, min(len(required), 14))]
    hit = snippet in next_assistant.content or required[:8] in next_assistant.content
    return [
        ScoreDetail(
            criterion_id="hard.required_out_of_scope_reply",
            label="越权问题兜底话术",
            passed=hit,
            deduction=0.0 if hit else 1.0,
            turn_ids=[triggered_user_turn or 0, next_assistant.index],
            evidence_quote=next_assistant.content[:80],
            rationale=(
                f"使用了指定兜底话术：{snippet!r}"
                if hit
                else f"SUT 未按指令使用兜底话术，期望包含：{snippet!r}"
            ),
        )
    ]


def check_required_replies(
    spec: InstructionSpec, trace: DialogueTrace
) -> list[ScoreDetail]:
    rules = spec.constraints.hard.required_replies or []
    if not rules:
        return []
    details: list[ScoreDetail] = []
    user_turns = _user_turns(trace)
    assistant = _assistant_turns(trace)
    for rule in rules:
        triggers = rule.get("trigger") if isinstance(rule, dict) else None
        reply = rule.get("reply") if isinstance(rule, dict) else None
        if not triggers or not reply:
            continue
        triggers_list = triggers if isinstance(triggers, list) else [triggers]
        triggered_user: TurnRecord | None = None
        for ut in user_turns:
            if any(t and t in ut.content for t in triggers_list):
                triggered_user = ut
                break
        if triggered_user is None:
            continue
        next_assistant = next(
            (a for a in assistant if a.index > triggered_user.index), None
        )
        hit = bool(next_assistant and reply in next_assistant.content)
        details.append(
            ScoreDetail(
                criterion_id="hard.required_replies",
                label=f"触发词`{triggers_list[0]}`要求话术",
                passed=hit,
                deduction=0.0 if hit else 1.0,
                turn_ids=[triggered_user.index] + ([next_assistant.index] if next_assistant else []),
                evidence_quote=(next_assistant.content[:80] if next_assistant else ""),
                rationale=(
                    f"使用了规定话术：{reply}"
                    if hit
                    else f"未使用规定话术。期望：{reply}"
                ),
            )
        )
    return details


def aggregate_dim(
    dim_id: str,
    name: str,
    details: Iterable[ScoreDetail],
) -> DimensionScore:
    details_list = list(details)
    if not details_list:
        return DimensionScore(id=dim_id, name=name, score=1.0, raw_score=1.0, details=[])
    penalties = [d.deduction for d in details_list]
    avg_pen = sum(penalties) / len(penalties)
    score = max(0.0, 1.0 - avg_pen)
    return DimensionScore(
        id=dim_id,
        name=name,
        score=round(score, 4),
        raw_score=round(score, 4),
        details=details_list,
    )


def evaluate_hard_constraints(
    spec: InstructionSpec, trace: DialogueTrace, scenario: Optional[ScenarioSpec] = None
) -> DimensionScore:
    details: list[ScoreDetail] = []
    if _target_enabled(scenario, "hard.max_chars_per_reply"):
        details.extend(check_max_chars(spec, trace))
    if _target_enabled(scenario, "hard.forbidden_words"):
        details.extend(check_forbidden_words(spec, trace))
    if _target_enabled(scenario, "hard.no_discount_promise"):
        details.extend(check_no_discount(spec, trace))
    if _target_enabled(scenario, "hard.opening_keywords"):
        details.extend(check_opening_keywords(spec, trace))
    if _target_enabled(scenario, "hard.required_out_of_scope_reply"):
        details.extend(check_out_of_scope_reply(spec, trace))
    if _target_enabled(scenario, "hard.required_replies"):
        details.extend(check_required_replies(spec, trace))
    dim = aggregate_dim("hcr", "硬约束合规", details)
    dim.source = "rule"
    if scenario is not None and scenario.target_constraints:
        dim.warnings.append(
            "已按场景target_constraints过滤硬约束: "
            + ", ".join(scenario.target_constraints)
        )
    return dim


def evaluate_termination(
    spec: InstructionSpec,
    trace: DialogueTrace,
    scenario: Optional[ScenarioSpec] = None,
) -> DimensionScore:
    """Quick heuristic check for the TER dimension."""
    if scenario is not None and not _target_enabled(scenario, "termination"):
        return DimensionScore(
            id="ter",
            name="终止策略合规",
            score=1.0,
            raw_score=1.0,
            details=[],
            source="rule",
            warnings=["该场景未声明termination为目标约束，跳过TER。"],
        )
    if not spec.constraints.termination and not (scenario and scenario.expected_termination):
        return DimensionScore(id="ter", name="终止策略合规", score=1.0, raw_score=1.0, details=[], source="rule")
    user_turns = _user_turns(trace)
    triggers = {
        "busy_driving": ("开车", "在车", "驾驶", "上车", "太忙", "很忙"),
        "resistant": ("不送", "送不了", "做不了", "确实不行", "不行", "以后再说"),
        "out_of_scope": ("年终奖", "团购", "换站点", "工资", "公司"),
        "interrupt": ("等一下", "打断", "先听", "先问"),
    }
    scenario_id = scenario.id if scenario is not None else ""
    active_keys = [scenario_id] if scenario_id in triggers else list(triggers)
    fired: list[str] = []
    triggered_user_turn: int | None = None
    for ut in user_turns:
        for key in active_keys:
            kws = triggers[key]
            if any(kw in ut.content for kw in kws):
                fired.append(key)
                triggered_user_turn = ut.index
                break
        if fired:
            break
    if not fired:
        return DimensionScore(
            id="ter",
            name="终止策略合规",
            score=1.0,
            raw_score=1.0,
            details=[],
            source="rule",
            warnings=["本场景未观察到终止策略触发信号。"],
        )
    expected_terminations = (
        "sut_goodbye",
        "user_end_call",
    )
    last_assistant = next(
        (t.content for t in reversed(trace.turns) if t.role == "assistant"), ""
    )
    last_text = last_assistant or (trace.turns[-1].content if trace.turns else "")
    ended_well = trace.terminated_by in expected_terminations
    farewell_seen = bool(
        re.search(
            r"(再见|稍后再打|下次|挂断|今天先聊到这|多注意身体|祝.*顺利)", last_text
        )
    )
    if fired[0] == "out_of_scope":
        assistant_after = next(
            (t.content for t in trace.turns if t.role == "assistant" and t.index > (triggered_user_turn or -1)),
            "",
        )
        passed = bool(assistant_after) and trace.terminated_by != "max_turns"
        evidence = assistant_after[:80]
        rationale = "越权问题后SUT给出回应并继续/收束对话。" if passed else "越权问题后未观察到有效回应。"
    elif fired[0] == "interrupt":
        assistant_after = next(
            (t.content for t in trace.turns if t.role == "assistant" and t.index > (triggered_user_turn or -1)),
            "",
        )
        passed = bool(assistant_after) and trace.terminated_by != "max_turns"
        evidence = assistant_after[:80]
        rationale = "打断后SUT恢复对话并继续推进。" if passed else "打断后未观察到恢复推进。"
    else:
        passed = ended_well and farewell_seen
        evidence = last_text[:80]
        rationale = (
            f"以 {trace.terminated_by} 结束，最后一句包含告别语。"
            if passed
            else f"以 {trace.terminated_by} 结束，未观察到告别语或终止策略。"
        )
    return DimensionScore(
        id="ter",
        name="终止策略合规",
        score=1.0 if passed else 0.0,
        raw_score=1.0 if passed else 0.0,
        source="rule",
        details=[
            ScoreDetail(
                criterion_id="termination.policy",
                label="按场景规定挂断/收束",
                passed=passed,
                deduction=0.0 if passed else 1.0,
                turn_ids=[trace.turns[-1].index] if trace.turns else [],
                evidence_quote=evidence,
                rationale=rationale,
            )
        ],
    )
