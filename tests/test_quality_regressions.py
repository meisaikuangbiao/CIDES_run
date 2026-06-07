import json
from datetime import datetime, timezone

from src.core.llm_client import ChatResult
from src.core.schemas import (
    DialogueTrace,
    CaseReport,
    DimensionScore,
    FlowNode,
    HardConstraints,
    InstructionConstraints,
    InstructionSpec,
    ScenarioSpec,
    ScoreDetail,
    TurnRecord,
)
from src.evaluators.flow_judge import evaluate_flow
from src.evaluators.common import extract_json
from src.evaluators.aggregator import aggregate_run
from src.evaluators.rule_constraints import check_opening_keywords, evaluate_termination
from src.evaluators.voting import _aggregate_single_dim
from src.report.renderer import _md_cell
from src.report.renderer import render_html, render_markdown


class FakeJudgeClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def chat(self, *args, **kwargs):
        return ChatResult(
            content=json.dumps(self.payload, ensure_ascii=False),
            model="fake",
            latency_ms=0,
        )


def _trace(turns: list[tuple[str, str]], terminated_by: str = "user_end_call") -> DialogueTrace:
    return DialogueTrace(
        run_id="t",
        case_id="t__c",
        instruction_id="t",
        scenario_id="c",
        turns=[
            TurnRecord(index=i, role=role, content=content, chars=len(content))
            for i, (role, content) in enumerate(turns)
        ],
        terminated_by=terminated_by,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def test_flow_judge_uses_target_nodes_as_pcr_denominator():
    spec = InstructionSpec(
        id="t",
        role="r",
        task="t",
        flow_nodes=[FlowNode(id="S1", desc="身份确认"), FlowNode(id="S2", desc="传达升级内容")],
    )
    scenario = ScenarioSpec(
        id="cooperative",
        name="配合",
        user_goal="完成",
        behaviour="配合",
        target_nodes=["S1", "S2"],
    )
    client = FakeJudgeClient(
        {
            "items": [
                {
                    "criterion_id": "pcr.S1",
                    "label": "节点S1",
                    "passed": True,
                    "deduction": 0,
                    "turn_ids": [0],
                    "evidence_quote": "您好",
                    "rationale": "覆盖",
                    "confidence": 0.9,
                }
            ]
        }
    )
    pcr, _ = evaluate_flow(spec, scenario, _trace([("assistant", "您好")]), client=client)
    assert pcr.score == 0.5
    assert any(d.criterion_id == "pcr.S2" and not d.passed for d in pcr.details)


def test_opening_keyword_matches_removed_variable_slot():
    spec = InstructionSpec(
        id="t",
        role="r",
        task="t",
        constraints=InstructionConstraints(
            hard=HardConstraints(opening_keywords=["你好，请问是吗", "我是站长"])
        ),
    )
    trace = _trace([("assistant", "你好，请问是张师傅吗？我是站长。")])
    detail = check_opening_keywords(spec, trace)[0]
    assert detail.passed is True


def test_termination_uses_busy_driving_scenario_signals():
    spec = InstructionSpec(
        id="t",
        role="r",
        task="t",
        constraints=InstructionConstraints(termination=["开车时稍后再打后挂断"]),
    )
    scenario = ScenarioSpec(
        id="busy_driving",
        name="忙碌",
        user_goal="开车",
        behaviour="说开车",
        target_constraints=["termination"],
        expected_termination="礼貌结束并挂断",
    )
    trace = _trace(
        [("assistant", "您好"), ("user", "我刚上车，太忙了。"), ("assistant", "嗯，我明白。")],
        terminated_by="user_end_call",
    )
    dim = evaluate_termination(spec, trace, scenario)
    assert dim.score == 0.0


def test_voting_penalises_missing_criterion_samples():
    sample_a = DimensionScore(
        id="scr",
        name="软约束",
        score=1.0,
        source="llm",
        details=[
            ScoreDetail(
                criterion_id="soft.0",
                label="不重复",
                passed=True,
                deduction=0.0,
                rationale="ok",
            )
        ],
    )
    sample_b = DimensionScore(id="scr", name="软约束", score=1.0, details=[], source="fallback", warnings=["LLM failed"])
    merged = _aggregate_single_dim([sample_a, sample_b])
    assert merged.details[0].deduction == 0.5
    assert merged.details[0].disagreement == 0.5
    assert merged.source == "fallback"
    assert "LLM failed" in merged.warnings


def test_markdown_cell_escapes_table_breaking_chars():
    assert _md_cell("a|b\nc") == "a\\|b c"


def test_extract_json_repairs_trailing_commas():
    payload = extract_json('```json\n{"items":[{"passed": true,}],}\n```')
    assert payload == {"items": [{"passed": True}]}


def test_quality_gate_ignores_informational_warnings():
    trace = _trace([("assistant", "您好")])
    dim = DimensionScore(
        id="hcr",
        name="硬约束",
        score=1.0,
        source="rule",
        warnings=["已按场景target_constraints过滤硬约束: termination"],
    )
    case = CaseReport(trace=trace, dimensions=[dim], weighted_total=1.0)
    report = aggregate_run("t", [case])
    assert report.aggregate["quality_gates"]["fallback_or_warning_count"] == 0
    assert report.aggregate["informational_warnings"]


def test_html_report_uses_compact_turn_spacing():
    trace = _trace([("assistant", "您好，请问方便沟通吗？"), ("user", "可以。")])
    dim = DimensionScore(id="hcr", name="硬约束", score=1.0, details=[], source="rule")
    case = CaseReport(trace=trace, dimensions=[dim], weighted_total=1.0)
    report = aggregate_run("t", [case])

    html = render_html(report)

    assert ".turns { display: flex; flex-direction: column; gap: 3px;" in html
    assert ".turn { display: grid; grid-template-columns: 72px 1fr;" in html
    assert "line-height: 1.35" in html
    assert ".turn .body > div { margin: 0; }" in html


def test_html_report_preserves_whitespace_only_on_turn_text():
    trace = _trace([("assistant", "您好，请问方便沟通吗？")])
    dim = DimensionScore(id="hcr", name="硬约束", score=1.0, details=[], source="rule")
    case = CaseReport(trace=trace, dimensions=[dim], weighted_total=1.0)
    report = aggregate_run("t", [case])

    html = render_html(report)

    assert ".turn .body { white-space: pre-wrap;" not in html
    assert ".turn-text { white-space: pre-wrap;" in html
    assert '<div class="body"><div class="turn-text">您好，请问方便沟通吗？</div>' in html


def test_reports_label_real_dialogue_cases_by_id_and_task_number():
    trace = DialogueTrace(
        run_id="real",
        case_id="001",
        instruction_id="1",
        scenario_id="real",
        turns=[TurnRecord(index=0, role="assistant", content="您好", chars=2)],
        terminated_by="imported",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    dim = DimensionScore(id="hcr", name="硬约束", score=1.0, details=[], source="rule")
    case = CaseReport(trace=trace, dimensions=[dim], weighted_total=1.0)
    report = aggregate_run("real", [case], config={"run_mode": "real_dialogue"})

    html = render_html(report)
    markdown = render_markdown(report)

    assert "对话 001 · 任务1" in html
    assert "### 对话 001 · 任务1" in markdown
    assert "1 · real" not in html
    assert "### 1 · real" not in markdown
