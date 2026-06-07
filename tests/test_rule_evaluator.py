from datetime import datetime, timezone

from src.core.schemas import (
    DialogueTrace,
    HardConstraints,
    InstructionConstraints,
    InstructionSpec,
    TurnRecord,
)
from src.evaluators.rule_constraints import (
    check_max_chars,
    check_no_discount,
    check_opening_keywords,
    check_out_of_scope_reply,
    evaluate_hard_constraints,
)


def _make_spec(**overrides) -> InstructionSpec:
    constraints = InstructionConstraints(
        hard=HardConstraints(
            max_chars_per_reply=overrides.get("max_chars", 30),
            opening_keywords=overrides.get("opening_keywords", ["你好", "我是站长"]),
            no_discount_promise=overrides.get("no_discount", True),
            required_out_of_scope_reply=overrides.get("ooo_reply"),
        ),
    )
    return InstructionSpec(
        id="t",
        role="r",
        task="t",
        opening_line_template="你好，我是站长，请问是张三吗？",
        constraints=constraints,
    )


def _make_trace(turns: list[tuple[str, str]]) -> DialogueTrace:
    return DialogueTrace(
        run_id="t",
        case_id="t__c",
        instruction_id="t",
        scenario_id="c",
        variables={},
        turns=[
            TurnRecord(
                index=i,
                role=role,
                content=content,
                chars=len(content),
            )
            for i, (role, content) in enumerate(turns)
        ],
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def test_max_chars_flags_overlength():
    spec = _make_spec(max_chars=10)
    trace = _make_trace([("assistant", "这是一句很长的话超过限制了"), ("user", "嗯")])
    details = check_max_chars(spec, trace)
    assert details and details[0].passed is False
    assert details[0].deduction > 0


def test_no_discount_catches_promise():
    spec = _make_spec()
    trace = _make_trace([("assistant", "我可以给你优惠券。"), ("user", "好")])
    details = check_no_discount(spec, trace)
    assert any(not d.passed for d in details)


def test_opening_keywords_partial_credit():
    spec = _make_spec(opening_keywords=["你好", "我是站长", "你已报名"])
    trace = _make_trace([("assistant", "你好，我是站长，请问是张三吗？")])
    details = check_opening_keywords(spec, trace)
    assert details and not details[0].passed
    assert 0 < details[0].deduction <= 1


def test_out_of_scope_reply_required():
    spec = _make_spec(ooo_reply="我向同事确认后再回电给你。")
    trace = _make_trace(
        [
            ("assistant", "你好"),
            ("user", "你们公司年终奖多少？"),
            ("assistant", "我向同事确认后再回电给你。"),
        ]
    )
    details = check_out_of_scope_reply(spec, trace)
    assert details[0].passed is True


def test_hcr_aggregate_score_in_range():
    spec = _make_spec(max_chars=30)
    trace = _make_trace(
        [
            ("assistant", "你好，我是站长，请问是张三吗？"),
            ("user", "嗯"),
            ("assistant", "好的"),
        ]
    )
    dim = evaluate_hard_constraints(spec, trace)
    assert 0.0 <= dim.score <= 1.0
    assert dim.id == "hcr"
