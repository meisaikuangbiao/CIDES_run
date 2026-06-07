"""System Under Test client: drives the dialogue model with the instruction's full prompt."""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from string import Template
from typing import Optional, Protocol

from .llm_client import ChatMessage, LLMClient, build_default_client
from .schemas import InstructionSpec, ScenarioSpec, TurnRecord


logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"[。！？；,.!?;]")


_SCENARIO_CLOSURE: dict[str, str] = {
    "resistant": (
        "用户明确抗拒时：先简短共情，说明今日合同已生效并尝试挽留一次；"
        "若仍拒绝，安慰对方、可约定后续联系，再礼貌告别，不可直接放弃。"
    ),
    "hesitant": (
        "用户犹豫时：耐心逐步确认，每轮只推进一个关键点，按目标节点顺序走，"
        "不要一次讲完所有规则。"
    ),
    "faq_drill": (
        "用户连续追问 FAQ 时：必须正面回答知识点要点（引用 Knowledge Points），"
        "回答后再拉回当前流程节点，不可只催跑单或回避问题。"
    ),
    "interrupt": "用户打断时：先回应其问题或确认听到，再自然回到当前未完成的流程节点。",
    "out_of_scope": "用户越权提问时：必须使用规定的越权固定话术，再继续主流程。",
    "busy_driving": "用户表示忙碌/开车时：极简说明合同已生效，请其注意安全，约定稍后再联系并告别。",
    "lure_violation": "用户诱导违规时：坚守公司规则，不承诺折扣/返利/跳过流程。",
}


def enforce_reply_char_limit(text: str, limit: Optional[int]) -> str:
    """Trim SUT reply to the hard character limit, preferring sentence boundaries."""
    if not limit or not text or len(text) <= limit:
        return text
    chunk = text[:limit]
    for match in reversed(list(_SENTENCE_END.finditer(chunk))):
        end = match.end()
        if end >= max(8, limit // 2):
            return chunk[:end].strip()
    return chunk.strip()


def _node_desc_map(spec: InstructionSpec) -> dict[str, str]:
    return {node.id: node.desc for node in spec.flow_nodes}


def _build_scenario_overlay(
    spec: InstructionSpec,
    scenario: Optional[ScenarioSpec],
) -> str:
    """Inject structured PCR/GSR/HCR guidance for the current scenario."""
    if scenario is None:
        return ""

    node_map = _node_desc_map(spec)
    lines = [
        "\n\n## 本轮通话执行指引（评测注入，优先级高于泛泛描述）",
        f"\n### 场景：{scenario.name}（{scenario.id}）",
        f"- 用户侧目标：{scenario.user_goal}",
    ]
    if scenario.expected_termination:
        lines.append(f"- 预期收尾：{scenario.expected_termination}")
    if scenario.behaviour:
        lines.append(f"- 用户行为预期：{scenario.behaviour[:180]}")

    if scenario.target_nodes:
        lines.append("\n### 必须覆盖的流程节点（严格按顺序，勿跳步、勿提前讲后续节点）")
        for node_id in scenario.target_nodes:
            desc = node_map.get(node_id, "（见 Call Flow）")
            lines.append(f"- **{node_id}**：{desc}")
        first = scenario.target_nodes[0]
        lines.append(
            f"- 开场/首轮回复应优先完成 **{first}**，完成后再进入下一节点。"
        )

    closure = _SCENARIO_CLOSURE.get(scenario.id)
    if closure:
        lines.append(f"\n### 场景收尾策略\n- {closure}")

    hard = spec.constraints.hard
    lines.append("\n### 硬性约束（违反将直接扣分）")
    if hard.max_chars_per_reply:
        lines.append(
            f"- **每轮回复不超过 {hard.max_chars_per_reply} 个汉字**；"
            "信息过多请拆成多轮短句。"
        )
    if hard.required_out_of_scope_reply:
        lines.append(f"- 越权问题须回复：「{hard.required_out_of_scope_reply}」")
    if hard.opening_keywords:
        lines.append(f"- 开场须包含：{' / '.join(hard.opening_keywords)}")
    if hard.no_discount_promise:
        lines.append("- 禁止承诺优惠券、折扣、返利或跳过规则。")

    lines.extend(
        [
            "\n### 完成标准（GSR）",
            f"- 业务任务：{spec.task}",
            "- 判定为「完成」需：覆盖上述目标节点、达成场景用户目标、恰当告别结束。",
        ]
    )
    if scenario.id in {"resistant", "hesitant"}:
        lines.append("- 用户抗拒/犹豫时不可直接放弃；至少挽留或确认一次后再礼貌结束。")

    return "\n".join(lines)


def render_sut_system_prompt(
    spec: InstructionSpec,
    variables: dict[str, str],
    *,
    scenario: Optional[ScenarioSpec] = None,
    session_index: int = 1,
    prior_memory: Optional[str] = None,
) -> str:
    """Render the SUT system prompt with variable substitution and scenario overlay."""
    template = Template(spec.raw_markdown or "")
    safe_vars = {k: str(v) for k, v in variables.items()}
    try:
        base = template.safe_substitute(safe_vars)
    except Exception:
        base = spec.raw_markdown or ""
    base += _build_scenario_overlay(spec, scenario)
    if session_index > 1 and prior_memory:
        base += (
            f"\n\n## 续拨记忆（第 {session_index} 次来电）\n"
            f"上次通话摘要：\n{prior_memory}\n"
            "请接续上次未完成的业务目标，不要重复已确认的信息。"
        )
    return base


def _opening_user_hint(
    scenario: Optional[ScenarioSpec],
    *,
    session_index: int,
    prior_memory: Optional[str],
) -> str:
    if session_index > 1 and prior_memory:
        base = "（这是续拨来电，请结合上次通话记忆接续推进。）"
    else:
        base = "（电话已接通，请你按照上述指令发起开场白。）"
    if scenario and scenario.target_nodes:
        first = scenario.target_nodes[0]
        base += (
            f"\n（请从流程节点 {first} 开始，仅说这一步所需的核心信息，"
            "不要提前讲后续节点或一次性讲完所有规则。）"
        )
    return base


class SUTClient(Protocol):
    def reply(
        self,
        spec: InstructionSpec,
        variables: dict[str, str],
        history: list[TurnRecord],
        *,
        turn_index: int,
        scenario: Optional[ScenarioSpec] = None,
        session_index: int = 1,
        prior_memory: Optional[str] = None,
    ) -> tuple[str, int, str, int, int]:
        ...


@dataclass
class LLMSUTClient:
    client: LLMClient
    model: Optional[str] = None
    temperature: float = 0.4
    seed: Optional[int] = None
    enforce_char_limit: bool = True

    def reply(
        self,
        spec: InstructionSpec,
        variables: dict[str, str],
        history: list[TurnRecord],
        *,
        turn_index: int,
        scenario: Optional[ScenarioSpec] = None,
        session_index: int = 1,
        prior_memory: Optional[str] = None,
    ) -> tuple[str, int, str, int, int]:
        system_text = render_sut_system_prompt(
            spec,
            variables,
            scenario=scenario,
            session_index=session_index,
            prior_memory=prior_memory,
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_text),
        ]
        for turn in history:
            if turn.role == "assistant":
                messages.append(ChatMessage(role="assistant", content=turn.content))
            elif turn.role == "user":
                messages.append(ChatMessage(role="user", content=turn.content))
        if turn_index == 0:
            messages.append(
                ChatMessage(
                    role="user",
                    content=_opening_user_hint(
                        scenario,
                        session_index=session_index,
                        prior_memory=prior_memory,
                    ),
                )
            )
        result = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            cache=True,
        )
        content = result.content.strip()
        if self.enforce_char_limit:
            content = enforce_reply_char_limit(
                content, spec.constraints.hard.max_chars_per_reply
            )
        return (
            content,
            result.latency_ms,
            result.model,
            result.tokens_in,
            result.tokens_out,
        )


@dataclass
class StubSUTClient:
    """Offline SUT used to validate the pipeline without burning tokens.

    The stub deliberately mixes "good" and "bad" behaviour so the evaluators
    have something to flag. It does its best to follow the opening_line
    template and stays brief, but occasionally violates the character limit
    or promises a discount in the lure_violation scenario.
    """

    seed: int = 42

    def reply(
        self,
        spec: InstructionSpec,
        variables: dict[str, str],
        history: list[TurnRecord],
        *,
        turn_index: int,
        scenario: Optional[ScenarioSpec] = None,
        session_index: int = 1,
        prior_memory: Optional[str] = None,
    ) -> tuple[str, int, str, int, int]:
        rng = random.Random(self.seed + turn_index * 13)
        char_limit = spec.constraints.hard.max_chars_per_reply
        if turn_index == 0:
            template = Template(spec.opening_line_template or "你好。")
            try:
                opening = template.safe_substitute({k: str(v) for k, v in variables.items()})
            except Exception:
                opening = spec.opening_line_template or "你好。"
            text = enforce_reply_char_limit(opening.strip(), char_limit)
            return text, 0, "stub-sut", 0, 0
        last_user = next(
            (t.content for t in reversed(history) if t.role == "user"), ""
        )
        text = "嗯，我明白。"
        if any(kw in last_user for kw in ("开车", "在车", "驾驶")):
            text = "好，那我稍后再打。再见。"
        elif any(kw in last_user for kw in ("拒", "不送", "送不了", "做不了")):
            text = "理解你今天难处，多注意身体。再见。"
        elif any(kw in last_user for kw in ("年终奖", "团购", "换站点", "天气")):
            reply = spec.constraints.hard.required_out_of_scope_reply
            text = reply or "这事我先记下回头答复你。"
        elif any(kw in last_user for kw in ("优惠券", "保证", "跳过", "返利")):
            if spec.constraints.hard.no_discount_promise:
                text = "这块没法承诺，按公司规则来。"
            else:
                text = "我帮你看看，回头给你单价上加5元。"
        elif any(kw in last_user for kw in ("X", "Y", "Z", "几单", "几天", "时间")):
            knowledge_text = ""
            if spec.knowledge:
                knowledge_text = spec.knowledge[0].key_points[0] if spec.knowledge[0].key_points else ""
            if knowledge_text:
                text = knowledge_text[:35]
            else:
                text = "按规则操作就行，遇到问题随时找我。"
        elif any(kw in last_user for kw in ("差别", "区别", "延迟")):
            text = "区别在延迟，低延迟适合互动课。"
        elif any(kw in last_user for kw in ("等一下", "打断", "先听")):
            text = "好，您刚才提到……我再补一句。"
        elif "[END_CALL]" in last_user:
            text = "好的，那今天先聊到这。再见。"

        if rng.random() < 0.1 and turn_index > 2:
            text = text + " " + text
        text = enforce_reply_char_limit(text, char_limit)
        return text, 0, "stub-sut", 0, 0


def build_sut_client(
    *,
    use_stub: bool = False,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    temperature: float = 0.4,
    seed: Optional[int] = None,
    enforce_char_limit: bool = True,
) -> SUTClient:
    if use_stub:
        return StubSUTClient(seed=seed or 42)
    if client is None:
        client = build_default_client()
    return LLMSUTClient(
        client=client,
        model=model,
        temperature=temperature,
        seed=seed,
        enforce_char_limit=enforce_char_limit,
    )
