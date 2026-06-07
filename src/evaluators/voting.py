"""Multi-sample voting wrapper for LLM judges.

A *voted judge* runs the underlying judge ``samples`` times with a
diverse seed and averages the resulting dimension scores. We also
compute per-criterion disagreement and aggregate confidence based on
how often the samples agreed on the ``passed`` field.

For deterministic offline mode (no client) we still call the judge once
and just attach a low confidence so the report reflects the uncertainty.
"""
from __future__ import annotations

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.llm_client import LLMClient
from ..core.schemas import (
    DialogueTrace,
    DimensionScore,
    InstructionSpec,
    ScenarioSpec,
    ScoreDetail,
)


_DEFAULT_SAMPLES_WORKERS_CAP = 4


logger = logging.getLogger(__name__)


JudgeFn = Callable[..., DimensionScore]
DualJudgeFn = Callable[..., tuple[DimensionScore, DimensionScore]]


@dataclass
class VotingConfig:
    samples: int = 3
    base_seed: int = 13
    temperature: float = 0.2


def _aggregate_single_dim(samples: list[DimensionScore]) -> DimensionScore:
    if not samples:
        raise ValueError("empty samples")
    if len(samples) == 1:
        sole = samples[0]
        return sole
    scores = [s.score for s in samples]
    mean_score = round(statistics.fmean(scores), 4)
    stdev = round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0
    confidence = max(0.0, 1.0 - stdev * 1.5)
    base = samples[0]
    sources = [s.source for s in samples if s.source]
    source = "fallback" if "fallback" in sources else (sources[0] if sources else base.source)
    warnings: list[str] = []
    for sample in samples:
        for warning in sample.warnings:
            if warning not in warnings:
                warnings.append(warning)
    detail_map: dict[str, list[ScoreDetail]] = {}
    for sample in samples:
        for d in sample.details:
            detail_map.setdefault(d.criterion_id, []).append(d)
    merged_details: list[ScoreDetail] = []
    for key, group in detail_map.items():
        passed_votes = sum(1 for d in group if d.passed)
        total_votes = len(samples)
        missing_votes = total_votes - len(group)
        passed_majority = passed_votes >= (total_votes + 1) // 2
        deductions = [d.deduction for d in group] + [1.0] * missing_votes
        deduction_avg = round(statistics.fmean(deductions), 4)
        vote_disagreement = 1.0 - max(passed_votes, total_votes - passed_votes) / total_votes
        missing_ratio = missing_votes / total_votes
        disagreement = round(
            max(vote_disagreement, missing_ratio),
            4,
        )
        evidence = next(
            (d.evidence_quote for d in group if d.evidence_quote), ""
        )
        rationale = next((d.rationale for d in group if d.rationale), "")
        turn_ids: list[int] = []
        for d in group:
            for t in d.turn_ids:
                if t not in turn_ids:
                    turn_ids.append(t)
        merged_details.append(
            ScoreDetail(
                criterion_id=key,
                label=group[0].label,
                passed=passed_majority,
                deduction=deduction_avg,
                turn_ids=turn_ids,
                evidence_quote=evidence,
                rationale=rationale,
                confidence=round(1.0 - disagreement, 4),
                disagreement=disagreement,
            )
        )
        if missing_votes:
            merged_details[-1].rationale = (
                merged_details[-1].rationale
                + f"（{missing_votes}/{total_votes}个采样未返回该criterion）"
            ).strip()
            warning = f"{key}: {missing_votes}/{total_votes}个采样未返回criterion"
            if warning not in warnings:
                warnings.append(warning)
    return DimensionScore(
        id=base.id,
        name=base.name,
        score=mean_score,
        raw_score=mean_score,
        confidence=round(confidence, 4),
        details=merged_details,
        source=source,
        warnings=warnings,
    )


def _resolve_samples_workers(
    samples: int, samples_workers: Optional[int]
) -> int:
    """Return the effective worker count for parallel sampling.

    Defaults to ``min(samples, _DEFAULT_SAMPLES_WORKERS_CAP)``; explicit
    ``samples_workers <= 1`` keeps the legacy serial path.
    """
    if samples_workers is None:
        return max(1, min(samples, _DEFAULT_SAMPLES_WORKERS_CAP))
    return max(1, min(samples_workers, samples))


def run_voted_single(
    judge_fn: JudgeFn,
    *,
    spec: InstructionSpec,
    scenario: Optional[ScenarioSpec],
    trace: DialogueTrace,
    client: Optional[LLMClient],
    model: Optional[str],
    samples: int = 3,
    base_seed: int = 13,
    temperature: float = 0.2,
    samples_workers: Optional[int] = None,
    **extra,
) -> DimensionScore:
    if client is None:
        kwargs = dict(spec=spec, trace=trace, client=None, model=None, **extra)
        if scenario is not None:
            kwargs["scenario"] = scenario
        return judge_fn(**kwargs)

    n = max(1, samples)

    def _one(i: int) -> DimensionScore:
        seed = base_seed + i * 1009
        kwargs = dict(
            spec=spec,
            trace=trace,
            client=client,
            model=model,
            seed=seed,
            temperature=temperature,
            **extra,
        )
        if scenario is not None:
            kwargs["scenario"] = scenario
        return judge_fn(**kwargs)

    workers = _resolve_samples_workers(n, samples_workers)
    if workers <= 1 or n <= 1:
        runs = [_one(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(_one, range(n)))
    return _aggregate_single_dim(runs)


def run_voted_dual(
    judge_fn: DualJudgeFn,
    *,
    spec: InstructionSpec,
    scenario: ScenarioSpec,
    trace: DialogueTrace,
    client: Optional[LLMClient],
    model: Optional[str],
    samples: int = 3,
    base_seed: int = 13,
    temperature: float = 0.2,
    samples_workers: Optional[int] = None,
    **extra,
) -> tuple[DimensionScore, DimensionScore]:
    if client is None:
        a, b = judge_fn(
            spec=spec,
            scenario=scenario,
            trace=trace,
            client=None,
            model=None,
            **extra,
        )
        return a, b

    n = max(1, samples)

    def _one(i: int) -> tuple[DimensionScore, DimensionScore]:
        seed = base_seed + i * 1009
        return judge_fn(
            spec=spec,
            scenario=scenario,
            trace=trace,
            client=client,
            model=model,
            seed=seed,
            temperature=temperature,
            **extra,
        )

    workers = _resolve_samples_workers(n, samples_workers)
    if workers <= 1 or n <= 1:
        pairs = [_one(i) for i in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(_one, range(n)))
    first_runs = [p[0] for p in pairs]
    second_runs = [p[1] for p in pairs]
    return _aggregate_single_dim(first_runs), _aggregate_single_dim(second_runs)
