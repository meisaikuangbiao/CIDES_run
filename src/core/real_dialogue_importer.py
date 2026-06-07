from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import DialogueTrace, InstructionSpec, ScenarioSpec, TurnRecord


class RealDialogueImportError(ValueError):
    """Raised when an uploaded real-dialogue JSON file is invalid."""


def build_real_scenario(spec: InstructionSpec) -> ScenarioSpec:
    target_constraints: list[str] = []
    hard = spec.constraints.hard
    if hard.max_chars_per_reply is not None:
        target_constraints.append("hard.max_chars_per_reply")
    if hard.forbidden_words:
        target_constraints.append("hard.forbidden_words")
    if hard.no_discount_promise:
        target_constraints.append("hard.no_discount_promise")
    if hard.required_out_of_scope_reply:
        target_constraints.append("hard.required_out_of_scope_reply")
    if hard.opening_keywords:
        target_constraints.append("hard.opening_keywords")
    if hard.required_replies:
        target_constraints.append("hard.required_replies")
    if spec.constraints.soft:
        target_constraints.append("soft.repetition")
    if spec.constraints.termination:
        target_constraints.append("termination")
    if spec.knowledge:
        target_constraints.append("knowledge.coverage")

    return ScenarioSpec(
        id="real",
        name="真实对话",
        user_goal="真实上传对话，按完整外呼任务规则评测。",
        behaviour="真实用户对话，不使用模拟用户类型。",
        target_nodes=[node.id for node in spec.flow_nodes],
        target_constraints=target_constraints,
        target_knowledge=[kp.topic for kp in spec.knowledge if kp.topic],
        expected_termination="按任务配置判断真实对话是否合理结束",
    )


def load_real_dialogues(
    path: str | Path,
    specs_by_id: dict[str, InstructionSpec],
    *,
    run_id: str,
) -> list[DialogueTrace]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RealDialogueImportError(f"上传 JSON 解析失败：{exc}") from exc
    if not isinstance(payload, list):
        raise RealDialogueImportError("上传 JSON 顶层必须是数组")

    traces: list[DialogueTrace] = []
    seen_ids: set[str] = set()
    for row_idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise RealDialogueImportError(f"第 {row_idx} 条记录必须是对象")
        case_id = _required_str(item, "id", row_idx)
        if case_id in seen_ids:
            raise RealDialogueImportError(f"id 重复：{case_id}")
        seen_ids.add(case_id)

        instruction_id = str(_required_value(item, "任务编号", row_idx))
        if instruction_id not in specs_by_id:
            raise RealDialogueImportError(f"未找到任务编号 {instruction_id} 对应的任务配置")
        turns_raw = _required_value(item, "多轮对话", row_idx)
        if not isinstance(turns_raw, list) or not turns_raw:
            raise RealDialogueImportError(f"第 {row_idx} 条记录的 多轮对话 必须是非空数组")

        turns = [_parse_turn(turn, row_idx, turn_idx) for turn_idx, turn in enumerate(turns_raw)]
        traces.append(
            DialogueTrace(
                run_id=run_id,
                case_id=case_id,
                instruction_id=instruction_id,
                scenario_id="real",
                turns=turns,
                terminated_by="imported",
                created_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
    return traces


def _required_value(item: dict[str, Any], key: str, row_idx: int) -> Any:
    if key not in item:
        raise RealDialogueImportError(f"第 {row_idx} 条记录缺少 {key}")
    return item[key]


def _required_str(item: dict[str, Any], key: str, row_idx: int) -> str:
    value = _required_value(item, key, row_idx)
    if value is None or str(value).strip() == "":
        raise RealDialogueImportError(f"第 {row_idx} 条记录的 {key} 不能为空")
    return str(value).strip()


def _parse_turn(turn: Any, row_idx: int, turn_idx: int) -> TurnRecord:
    if not isinstance(turn, dict):
        raise RealDialogueImportError(f"第 {row_idx} 条记录第 {turn_idx} 轮对话必须是对象")
    role = str(turn.get("role", "")).strip()
    if role not in {"assistant", "user", "system"}:
        raise RealDialogueImportError(f"第 {row_idx} 条记录第 {turn_idx} 轮 role 非法：{role}")
    content = str(turn.get("content", ""))
    if not content.strip():
        raise RealDialogueImportError(f"第 {row_idx} 条记录第 {turn_idx} 轮 content 不能为空")
    index = turn.get("index", turn_idx)
    try:
        index_int = int(index)
    except (TypeError, ValueError) as exc:
        raise RealDialogueImportError(f"第 {row_idx} 条记录第 {turn_idx} 轮 index 必须是整数") from exc
    return TurnRecord(index=index_int, role=role, content=content, chars=len(content))
