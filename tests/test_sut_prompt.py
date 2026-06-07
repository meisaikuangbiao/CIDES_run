"""Tests for SUT prompt overlay and reply char-limit enforcement."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.schemas import InstructionSpec, ScenarioSpec
from src.core.sut_client import (
    enforce_reply_char_limit,
    render_sut_system_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = ROOT / "data" / "instructions"


def _load_spec(instr_id: str) -> InstructionSpec:
    data = json.loads((INSTRUCTIONS / f"{instr_id}.json").read_text(encoding="utf-8"))
    return InstructionSpec.model_validate(data)


def test_render_prompt_includes_target_nodes():
    spec = _load_spec("1")
    scenario = ScenarioSpec(
        id="resistant",
        name="抗拒型",
        user_goal="试探性拒绝",
        behaviour="多次表示无法配送",
        target_nodes=["S1", "S3"],
    )
    prompt = render_sut_system_prompt(spec, {"rider_name": "陈师傅"}, scenario=scenario)
    assert "必须覆盖的流程节点" in prompt
    assert "**S1**" in prompt
    assert "合同已生效" in prompt
    assert "抗拒" in prompt or "resistant" in prompt
    assert "每轮回复不超过 30" in prompt


def test_render_prompt_gsr_closure_for_hesitant():
    spec = _load_spec("2")
    scenario = ScenarioSpec(
        id="hesitant",
        name="犹豫型",
        user_goal="犹豫是否操作",
        behaviour="反复确认",
        target_nodes=["S1", "S2"],
    )
    prompt = render_sut_system_prompt(spec, {}, scenario=scenario)
    assert "犹豫" in prompt
    assert "不可直接放弃" in prompt


def test_instruction_hard_constraints_parsed():
    expected = {"1": 30, "2": 20, "3": 30}
    for instr_id, limit in expected.items():
        spec = _load_spec(instr_id)
        assert spec.constraints.hard.max_chars_per_reply == limit


def test_enforce_reply_char_limit_at_sentence():
    text = "今天飞毛腿合同已生效。请问您能开始配送吗？另外记得高峰期上线。"
    trimmed = enforce_reply_char_limit(text, 18)
    assert len(trimmed) <= 18
    assert trimmed.endswith("。") or len(trimmed) == 18


def test_enforce_reply_char_limit_noop_when_within():
    text = "合同已生效，能配送吗？"
    assert enforce_reply_char_limit(text, 30) == text
