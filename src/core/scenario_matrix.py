"""Generate a scenario matrix from a parsed instruction.

A *scenario* is a deterministic recipe describing how the user simulator
should behave during a single multi-turn dialogue. Each scenario carries:

* the user's stance (cooperative / hesitant / resistant / ...);
* a free-form behaviour brief that will be embedded into the user-sim
  prompt (without leaking the instruction's golden answers);
* the *target* flow nodes / constraints / FAQ topics that this scenario
  is expected to exercise. The evaluators use these targets so we only
  measure coverage on branches that the scenario should reach.

The generator is deterministic and offline so we can produce the same
scenario set for every run without paying for LLM calls.
"""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Optional

import yaml

from .schemas import InstructionSpec, ScenarioSpec


logger = logging.getLogger(__name__)


DEFAULT_PERSONAS_PATH = Path("configs/personas.yaml")


def _load_personas(path: str | Path = DEFAULT_PERSONAS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"personas config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick_value(choices: list[str], rng: random.Random) -> str:
    if not choices:
        return ""
    return rng.choice(choices)


def _resolve_variables(
    spec: InstructionSpec,
    variables_pool: dict[str, list[str]],
    rng: random.Random,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in spec.variables:
        pool = variables_pool.get(name, [])
        if pool:
            resolved[name] = _pick_value(pool, rng)
        else:
            resolved[name] = f"<{name}>"
    return resolved


def _select_target_nodes(spec: InstructionSpec, hint: str) -> list[str]:
    if not spec.flow_nodes:
        return []
    hint = hint or ""
    selected: list[str] = []
    for node in spec.flow_nodes:
        text = node.desc
        if hint == "主流程":
            selected.append(node.id)
        elif hint == "挽留/再确认" and re.search(
            r"挽留|再确认|安抚|犹豫|劝", text
        ):
            selected.append(node.id)
        elif hint == "安慰后挂断" and re.search(r"挂断|结束|安慰", text):
            selected.append(node.id)
        elif hint == "知识点应答" and re.search(
            r"知识|FAQ|解答|说明|区别|价格|费用", text
        ):
            selected.append(node.id)
        elif hint == "兜底回复" and re.search(r"超出|兜底|回复", text):
            selected.append(node.id)
        elif hint == "打断恢复" and re.search(r"打断|过渡|回到|继续", text):
            selected.append(node.id)
        elif hint == "忙碌挂断" and re.search(r"忙|挂断|稍后|结束", text):
            selected.append(node.id)
        elif hint == "拒绝违规" and re.search(r"承诺|跳过|违规|拒绝", text):
            selected.append(node.id)
    if not selected and hint == "主流程":
        selected = [n.id for n in spec.flow_nodes]
    if not selected:
        selected = [spec.flow_nodes[0].id]
    return selected


def _select_target_constraints(
    spec: InstructionSpec, hint: str
) -> list[str]:
    if not hint:
        return ["hard.max_chars_per_reply"]
    hint = hint.lower()
    targets: list[str] = []
    if "字数" in hint or "字" in hint:
        targets.append("hard.max_chars_per_reply")
    if "重复" in hint or "随意" in hint:
        targets.append("soft.repetition")
    if "兜底" in hint or "越权" in hint:
        targets.append("hard.required_out_of_scope_reply")
    if "终止" in hint or "挂断" in hint:
        targets.append("termination")
    if "知识" in hint or "覆盖" in hint:
        targets.append("knowledge.coverage")
    if "不承诺" in hint or "诱导" in hint or "跳步" in hint:
        targets.append("hard.no_discount_promise")
    if "过渡" in hint or "打断" in hint:
        targets.append("soft.transition")
    if "完整" in hint or "开场白" in hint:
        targets.append("hard.opening_keywords")
    if not targets:
        targets.append("hard.max_chars_per_reply")
    return targets


def _select_target_knowledge(spec: InstructionSpec, scenario_id: str) -> list[str]:
    if scenario_id != "faq_drill":
        return []
    topics: list[str] = []
    for kp in spec.knowledge[:6]:
        if kp.topic and kp.topic not in topics:
            topics.append(kp.topic)
    return topics


def _adversarial_events(scenario_id: str) -> list[str]:
    mapping = {
        "faq_drill": ["ask_faq_at_least_twice", "challenge_numbers"],
        "out_of_scope": ["ask_out_of_scope_question"],
        "interrupt": ["interrupt_mid_call", "topic_shift"],
        "busy_driving": ["state_busy_or_driving_first"],
        "lure_violation": ["request_discount_or_skip_rule"],
        "resistant": ["repeat_refusal"],
        "hesitant": ["express_uncertainty"],
    }
    return mapping.get(scenario_id, [])


def _required_user_signals(scenario_id: str) -> list[str]:
    mapping = {
        "faq_drill": ["询问规则/费用/时效/差异等知识点"],
        "out_of_scope": ["提出职责范围外问题"],
        "interrupt": ["打断对方并提出新问题"],
        "busy_driving": ["首轮或早期明确说正在开车/很忙"],
        "lure_violation": ["诱导承诺优惠/收益或跳过流程"],
        "resistant": ["明确拒绝或表示无法完成"],
        "hesitant": ["表达担心或需要再考虑"],
    }
    return mapping.get(scenario_id, [])


def _expected_termination(scenario_id: str) -> Optional[str]:
    mapping = {
        "busy_driving": "礼貌结束并挂断",
        "resistant": "安慰后挂断",
        "out_of_scope": "兜底话术后继续推进",
        "cooperative": "完成任务后挂断",
        "hesitant": "确认安抚后挂断",
        "lure_violation": "明确拒绝后继续推进",
        "interrupt": "用过渡语回到任务",
        "faq_drill": "覆盖知识点后挂断",
    }
    return mapping.get(scenario_id)


def _format_behaviour(spec: InstructionSpec, scenario_id: str, behaviour: str, variables: dict[str, str]) -> str:
    text = behaviour.strip()
    for name, value in variables.items():
        text = text.replace("${" + name + "}", value)
    if scenario_id == "cooperative":
        text += "\n你知道这是一通与你当前工作/业务有关的来电，但不知道对方的具体流程答案。"
    if scenario_id == "out_of_scope":
        text += "\n你随机问一个对方职责之外的问题，并观察对方是否使用预设兜底回复。"
    if scenario_id == "lure_violation" and spec.constraints.hard.no_discount_promise:
        text += "\n你可能会说\"给我个优惠券\"或\"你保证一定能挣到钱\"，看对方是否被诱导。"
    if scenario_id == "busy_driving":
        text += "\n请用1-2句话表达你在开车，必要时再问对方是否要继续。"
    return text


def build_scenarios(
    spec: InstructionSpec,
    *,
    personas_path: str | Path = DEFAULT_PERSONAS_PATH,
    seed: int = 42,
    enabled: Optional[list[str]] = None,
) -> tuple[list[ScenarioSpec], dict[str, dict[str, str]]]:
    """Build scenarios for one instruction.

    Returns a tuple ``(scenarios, variables_per_scenario)``. ``variables_per_scenario``
    maps scenario_id -> resolved variable values that the orchestrator must
    inject into the SUT prompt before the dialogue starts.
    """
    config = _load_personas(personas_path)
    variables_pool: dict[str, list[str]] = (
        config.get("defaults", {}).get("variables_pool", {}) or {}
    )
    base_rng = random.Random(seed)
    scenarios: list[ScenarioSpec] = []
    variables_map: dict[str, dict[str, str]] = {}
    has_no_discount = spec.constraints.hard.no_discount_promise
    has_out_of_scope = bool(spec.constraints.hard.required_out_of_scope_reply)
    for entry in config.get("scenarios", []) or []:
        scenario_id = entry["id"]
        if enabled and scenario_id not in enabled:
            continue
        if scenario_id == "lure_violation" and not has_no_discount:
            continue
        if scenario_id == "out_of_scope" and not has_out_of_scope:
            continue
        scenario_rng = random.Random(base_rng.randint(0, 1_000_000))
        variables = _resolve_variables(spec, variables_pool, scenario_rng)
        behaviour = _format_behaviour(
            spec, scenario_id, entry.get("behaviour", ""), variables
        )
        scenario = ScenarioSpec(
            id=scenario_id,
            name=entry.get("name", scenario_id),
            user_goal=entry.get("user_goal", ""),
            behaviour=behaviour,
            target_nodes=_select_target_nodes(spec, entry.get("target_node_hint", "")),
            target_constraints=_select_target_constraints(
                spec, entry.get("target_constraint_hint", "")
            ),
            target_knowledge=_select_target_knowledge(spec, scenario_id),
            adversarial_events=_adversarial_events(scenario_id),
            required_user_signals=_required_user_signals(scenario_id),
            expected_termination=_expected_termination(scenario_id),
            forbid_reveal=bool(entry.get("forbid_reveal", True)),
        )
        scenarios.append(scenario)
        variables_map[scenario_id] = variables
    if not scenarios:
        raise RuntimeError(
            f"No applicable scenarios for instruction {spec.id}; check personas config"
        )
    logger.debug(
        "Built %d scenarios for instruction %s: %s",
        len(scenarios),
        spec.id,
        ", ".join(s.id for s in scenarios),
    )
    return scenarios, variables_map


def coverage_matrix(spec: InstructionSpec, scenarios: list[ScenarioSpec]) -> dict[str, list[str]]:
    """Return mapping of flow_node_id -> scenarios that cover it (for reporting)."""
    matrix: dict[str, list[str]] = {n.id: [] for n in spec.flow_nodes}
    for scenario in scenarios:
        for node_id in scenario.target_nodes:
            matrix.setdefault(node_id, []).append(scenario.id)
    return matrix
