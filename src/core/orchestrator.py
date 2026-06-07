"""Dialogue orchestrator: runs a multi-turn conversation between SUT and UserSimulator."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schemas import DialogueTrace, InstructionSpec, ScenarioSpec, TurnRecord
from .sut_client import SUTClient
from .user_simulator import UserSimulator


logger = logging.getLogger(__name__)


END_CALL_PATTERN = re.compile(r"\[END_CALL\]")
GOODBYE_PATTERN = re.compile(r"(再见|拜拜|挂断|稍后再打|今天先聊到这|祝.*顺利)")


def _strip_end_marker(text: str) -> str:
    return END_CALL_PATTERN.sub("", text).strip()


def _sut_ends_call(text: str) -> bool:
    return bool(GOODBYE_PATTERN.search(text))


def summarize_trace_for_memory(trace: DialogueTrace, *, max_chars: int = 500) -> str:
    """Build a short summary of a prior call for multi-session retry."""
    lines: list[str] = []
    for turn in trace.turns[:14]:
        role = "客服" if turn.role == "assistant" else "用户"
        content = (turn.content or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:72]}")
    if trace.terminated_by:
        lines.append(f"终止原因: {trace.terminated_by}")
    summary = "\n".join(lines)
    return summary[:max_chars]


def run_dialogue(
    spec: InstructionSpec,
    scenario: ScenarioSpec,
    sut: SUTClient,
    user_sim: UserSimulator,
    *,
    variables: dict[str, str],
    run_id: str,
    max_turns: int = 16,
    max_invalid_replies: int = 3,
    seed: Optional[int] = None,
    trace_dir: Optional[str | Path] = None,
    session_index: int = 1,
    max_sessions: int = 1,
    prior_memory: Optional[str] = None,
    prior_sessions: Optional[list[dict]] = None,
) -> DialogueTrace:
    case_id = f"{spec.id}__{scenario.id}"
    trace = DialogueTrace(
        run_id=run_id,
        case_id=case_id,
        instruction_id=spec.id,
        scenario_id=scenario.id,
        variables=variables,
        turns=[],
        seed=seed,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        session_index=session_index,
        max_sessions=max_sessions,
        prior_memory=prior_memory,
        prior_sessions=list(prior_sessions or []),
    )
    invalid_count = 0
    for turn_index in range(max_turns):
        try:
            sut_text, sut_latency, sut_model, sut_in, sut_out = sut.reply(
                spec,
                variables,
                trace.turns,
                turn_index=turn_index,
                scenario=scenario,
                session_index=session_index,
                prior_memory=prior_memory,
            )
        except Exception as exc:
            logger.exception("SUT failure at turn %s: %s", turn_index, exc)
            trace.turns.append(
                TurnRecord(
                    index=len(trace.turns),
                    role="assistant",
                    content="",
                    chars=0,
                    error=str(exc),
                )
            )
            trace.terminated_by = "sut_error"
            break
        sut_text = sut_text.strip()
        trace.sut_model = sut_model or trace.sut_model
        trace.turns.append(
            TurnRecord(
                index=len(trace.turns),
                role="assistant",
                content=sut_text,
                chars=len(sut_text),
                latency_ms=sut_latency,
                tokens_in=sut_in,
                tokens_out=sut_out,
                model=sut_model,
            )
        )
        if not sut_text:
            invalid_count += 1
            if invalid_count >= max_invalid_replies:
                trace.terminated_by = "sut_invalid"
                break
            continue
        if _sut_ends_call(sut_text):
            trace.terminated_by = "sut_goodbye"
            try:
                user_text, user_latency, user_model, u_in, u_out = user_sim.reply(
                    spec,
                    scenario,
                    variables,
                    trace.turns,
                    turn_index=turn_index,
                    session_index=session_index,
                    prior_memory=prior_memory,
                )
            except Exception as exc:
                logger.exception("UserSim failure at terminal turn: %s", exc)
                user_text = "[END_CALL]"
                user_latency = 0
                user_model = "user-sim-error"
                u_in = u_out = 0
            trace.user_sim_model = user_model or trace.user_sim_model
            content = _strip_end_marker(user_text)
            trace.turns.append(
                TurnRecord(
                    index=len(trace.turns),
                    role="user",
                    content=content or user_text,
                    chars=len(content or user_text),
                    latency_ms=user_latency,
                    tokens_in=u_in,
                    tokens_out=u_out,
                    model=user_model,
                )
            )
            break
        try:
            user_text, user_latency, user_model, u_in, u_out = user_sim.reply(
                spec,
                scenario,
                variables,
                trace.turns,
                turn_index=turn_index,
                session_index=session_index,
                prior_memory=prior_memory,
            )
        except Exception as exc:
            logger.exception("UserSim failure at turn %s: %s", turn_index, exc)
            trace.turns.append(
                TurnRecord(
                    index=len(trace.turns),
                    role="user",
                    content="",
                    chars=0,
                    error=str(exc),
                )
            )
            trace.terminated_by = "user_sim_error"
            break
        trace.user_sim_model = user_model or trace.user_sim_model
        ends = bool(END_CALL_PATTERN.search(user_text))
        content = _strip_end_marker(user_text)
        trace.turns.append(
            TurnRecord(
                index=len(trace.turns),
                role="user",
                content=content if content else user_text,
                chars=len(content if content else user_text),
                latency_ms=user_latency,
                tokens_in=u_in,
                tokens_out=u_out,
                model=user_model,
            )
        )
        if ends:
            trace.terminated_by = "user_end_call"
            break
    else:
        trace.terminated_by = "max_turns"

    if trace_dir is not None:
        save_trace(trace, trace_dir)
    return trace


def save_trace(trace: DialogueTrace, trace_dir: str | Path) -> Path:
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trace.case_id}.json"
    out_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_trace(path: str | Path) -> DialogueTrace:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DialogueTrace.model_validate(data)
