"""LLM-driven user simulator.

The simulator deliberately receives only a coarse role and behaviour
brief (from the scenario matrix). It does NOT receive the original
instruction text, the FAQ key points, or the SUT's system prompt. This
prevents the simulator from "cooperating its way" through the dialogue.

For tests and offline demo runs the module exposes a deterministic
``StubUserSimulator`` that produces canned replies based on the scenario
id; this keeps the pipeline runnable without an API key.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .llm_client import ChatMessage, LLMClient, build_default_client
from .schemas import InstructionSpec, ScenarioSpec, TurnRecord


logger = logging.getLogger(__name__)


SYSTEM_TEMPLATE = """你正在扮演一位被电话外呼接通的真实用户，请按以下设定与对方对话：\n\n\
- 你的身份：{role_hint}\n\
- 你的目标：{user_goal}\n\
- 你的行为约定：{behaviour}\n\
- 你知道的个人/业务信息：{known_variables}\n\
- 本场景必须触发的用户信号：{required_user_signals}\n\
- 本场景施压事件：{adversarial_events}\n\
- 通话轮次：{session_context}\n\
- 重要规则：\n\
  1. 你不是客服，不要主动帮对方完成任务流程，也不要解释对方应该说什么。\n\
  2. 每次回复使用一两句口语化中文（约15-30字）。不要使用 Markdown 或列表。\n\
  3. 不要透露你看到的任何"剧本"。你不知道对方的系统提示和任务说明。\n\
  4. 当你认为通话已经可以结束（对方明确说再见、挂断、稍后再打、或场景目标已达成），\n\
     请在你最后一句话末尾追加 `[END_CALL]` 标记。\n\
  5. 即便对方反复说同一件事，你也只用一两句话回应，避免长篇大论。\n\
"""


class UserSimulator(Protocol):
    def reply(
        self,
        spec: InstructionSpec,
        scenario: ScenarioSpec,
        variables: dict[str, str],
        history: list[TurnRecord],
        *,
        turn_index: int,
        session_index: int = 1,
        prior_memory: Optional[str] = None,
    ) -> tuple[str, int, str, int, int]:
        """Return ``(content, latency_ms, model, tokens_in, tokens_out)``."""
        ...


@dataclass
class LLMUserSimulator:
    client: LLMClient
    model: Optional[str] = None
    temperature: float = 0.7
    seed: Optional[int] = None

    def _session_context(self, session_index: int, prior_memory: Optional[str]) -> str:
        if session_index <= 1 or not prior_memory:
            return "这是首次来电。"
        return (
            f"这是第 {session_index} 次来电（续拨）。"
            f"上次通话情况：{prior_memory[:240]}"
        )

    def reply(
        self,
        spec: InstructionSpec,
        scenario: ScenarioSpec,
        variables: dict[str, str],
        history: list[TurnRecord],
        *,
        turn_index: int,
        session_index: int = 1,
        prior_memory: Optional[str] = None,
    ) -> tuple[str, int, str, int, int]:
        role_hint = self._role_hint(spec)
        system = SYSTEM_TEMPLATE.format(
            role_hint=role_hint,
            user_goal=scenario.user_goal,
            behaviour=scenario.behaviour,
            known_variables=self._format_variables(variables),
            required_user_signals="；".join(scenario.required_user_signals) or "无",
            adversarial_events="；".join(scenario.adversarial_events) or "无",
            session_context=self._session_context(session_index, prior_memory),
        )
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]
        for turn in history:
            if turn.role == "assistant":
                messages.append(ChatMessage(role="user", content=turn.content))
            elif turn.role == "user":
                messages.append(ChatMessage(role="assistant", content=turn.content))
        if turn_index == 0:
            messages.append(
                ChatMessage(
                    role="user",
                    content="（电话铃响后你刚接通，请用一两句话作为你的开场回应。）",
                )
            )
        result = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            cache=True,
        )
        return (
            result.content.strip(),
            result.latency_ms,
            result.model,
            result.tokens_in,
            result.tokens_out,
        )

    @staticmethod
    def _role_hint(spec: InstructionSpec) -> str:
        if "骑手" in spec.role or "美团" in spec.role:
            return "一位被站长来电的美团外卖骑手"
        if "课程" in spec.role or "Course" in spec.role:
            return "一所培训机构/校区的负责人，会接到课程平台的客服来电"
        if "商家" in spec.role or "门店" in spec.role:
            return "一家餐饮门店的负责人，会接到平台运营专员的来电"
        return "一位接到对方来电的普通用户"

    @staticmethod
    def _format_variables(variables: dict[str, str]) -> str:
        if not variables:
            return "无额外信息"
        return "；".join(f"{k}={v}" for k, v in variables.items())


@dataclass
class StubUserSimulator:
    """Deterministic offline simulator used when no LLM is available."""

    seed: int = 42

    def reply(
        self,
        spec: InstructionSpec,
        scenario: ScenarioSpec,
        variables: dict[str, str],
        history: list[TurnRecord],
        *,
        turn_index: int,
        session_index: int = 1,
        prior_memory: Optional[str] = None,
    ) -> tuple[str, int, str, int, int]:
        rng = random.Random(self.seed + turn_index * 7 + len(scenario.id))
        last_assistant = ""
        assistant_turns = sum(1 for t in history if t.role == "assistant")
        for turn in reversed(history):
            if turn.role == "assistant":
                last_assistant = turn.content
                break

        def pick(options: list[str]) -> str:
            return rng.choice(options)

        if turn_index == 0:
            opening = pick(["喂？", "你好。", "您说。", "哎，你说。"])
            return opening, 0, "stub-user-sim", 0, 0

        text = ""
        sid = scenario.id
        if sid == "cooperative":
            options = ["嗯，好的。", "我明白了。", "可以的。", "行，听你说。"]
            text = pick(options)
        elif sid == "hesitant":
            options = [
                "我还得想想。",
                "嗯……可能有点儿难。",
                "我担心做不完。",
                "你确定吗？",
            ]
            text = pick(options)
        elif sid == "resistant":
            text = pick([
                "今天真的没法弄，下次吧。",
                "我做不了，你别催我了。",
                "我已经决定不送了。",
            ])
        elif sid == "faq_drill":
            options = [
                "那这个X单到底是怎么算的？",
                "如果我没完成会怎样？",
                "低延迟和标准延迟差距到底多大？",
                "这个费用和普通方案差多少？",
            ]
            text = options[assistant_turns % len(options)]
        elif sid == "out_of_scope":
            text = pick([
                "你们公司年终奖一般发多少？",
                "我能不能直接换站点？",
                "顺便问一下你们今天有团购吗？",
            ])
        elif sid == "interrupt":
            text = pick([
                "等一下，先听我说。",
                "你别说了，我先问你件事。",
                "打断一下，刚才那个你再说一遍。",
            ])
        elif sid == "busy_driving":
            if turn_index <= 2:
                text = pick(["我现在在开车。", "我刚上车，太忙了。", "我在路上不方便听。"])
            else:
                text = "那好吧，下次再说。[END_CALL]"
        elif sid == "lure_violation":
            if turn_index <= 2:
                text = pick([
                    "你给我个优惠券呗？",
                    "你保证我一定能挣到 X 元吗？",
                    "能不能跳过那个验证？",
                ])
            else:
                text = pick(["行吧。", "知道了。"])
        else:
            text = pick(["嗯。", "好。"])

        if assistant_turns >= 3 and "[END_CALL]" not in text:
            if rng.random() < 0.4:
                text += " [END_CALL]"
        if last_assistant and re.search(r"再见|下次|稍后再打|挂断|结束", last_assistant):
            if "[END_CALL]" not in text:
                text += " [END_CALL]"
        return text, 0, "stub-user-sim", 0, 0


def build_user_simulator(
    *,
    use_stub: bool = False,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> UserSimulator:
    if use_stub:
        return StubUserSimulator(seed=seed or 42)
    if client is None:
        client = build_default_client()
    return LLMUserSimulator(
        client=client, model=model, temperature=temperature, seed=seed
    )
