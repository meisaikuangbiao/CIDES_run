import json

import pytest

from src.core.real_dialogue_importer import (
    RealDialogueImportError,
    build_real_scenario,
    load_real_dialogues,
)
from src.core.schemas import (
    FlowNode,
    HardConstraints,
    InstructionConstraints,
    InstructionSpec,
    KnowledgePoint,
)


def _spec(spec_id: str = "1") -> InstructionSpec:
    return InstructionSpec(
        id=spec_id,
        role="站长",
        task="通知骑手合同生效",
        flow_nodes=[FlowNode(id="S1", desc="开场确认"), FlowNode(id="S2", desc="说明合同")],
        knowledge=[KnowledgePoint(topic="合同规则", key_points=["完成 X 单"])],
        constraints=InstructionConstraints(
            hard=HardConstraints(
                max_chars_per_reply=30,
                required_out_of_scope_reply="我向同事确认后再回电给你。",
                opening_keywords=["你好", "我是站长"],
            ),
            soft=["避免重复"],
            termination=["完成任务后礼貌挂断"],
        ),
    )


def test_load_real_dialogues_converts_upload_json_to_traces(tmp_path) -> None:
    upload = [
        {
            "id": "001",
            "任务编号": 1,
            "多轮对话": [
                {"index": 0, "role": "assistant", "content": "您好"},
                {"index": 1, "role": "user", "content": "你好"},
            ],
        }
    ]
    path = tmp_path / "dialogues.json"
    path.write_text(json.dumps(upload, ensure_ascii=False), encoding="utf-8")

    traces = load_real_dialogues(path, {"1": _spec()}, run_id="real_run")

    assert len(traces) == 1
    trace = traces[0]
    assert trace.run_id == "real_run"
    assert trace.case_id == "001"
    assert trace.instruction_id == "1"
    assert trace.scenario_id == "real"
    assert trace.terminated_by == "imported"
    assert trace.turns[0].chars == 2
    assert trace.turns[1].role == "user"


@pytest.mark.parametrize(
    "payload,error_text",
    [
        ([{"任务编号": 1, "多轮对话": []}], "id"),
        ([{"id": "001", "多轮对话": []}], "任务编号"),
        ([{"id": "001", "任务编号": 1}], "多轮对话"),
        ([{"id": "001", "任务编号": 9, "多轮对话": []}], "未找到任务编号"),
        ([{"id": "001", "任务编号": 1, "多轮对话": [{"index": 0, "role": "bot", "content": "x"}]}], "role"),
        ([{"id": "001", "任务编号": 1, "多轮对话": [{"index": 0, "role": "assistant", "content": ""}]}], "content"),
    ],
)
def test_load_real_dialogues_rejects_invalid_upload(tmp_path, payload, error_text) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RealDialogueImportError, match=error_text):
        load_real_dialogues(path, {"1": _spec()}, run_id="real_run")


def test_build_real_scenario_targets_full_instruction_scope() -> None:
    scenario = build_real_scenario(_spec())

    assert scenario.id == "real"
    assert scenario.name == "真实对话"
    assert scenario.target_nodes == ["S1", "S2"]
    assert scenario.target_knowledge == ["合同规则"]
    assert "hard.max_chars_per_reply" in scenario.target_constraints
    assert "hard.required_out_of_scope_reply" in scenario.target_constraints
    assert "hard.opening_keywords" in scenario.target_constraints
    assert "soft.repetition" in scenario.target_constraints
    assert "termination" in scenario.target_constraints
