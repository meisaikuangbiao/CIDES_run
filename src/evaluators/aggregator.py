"""Aggregate dimension scores into a CaseReport / RunReport.

Responsibilities:
- Compose all evaluators (rule + LLM) for a single case.
- Apply configured weights to derive ``weighted_total``.
- Provide bootstrap confidence intervals at the run level.
- Produce a top-K failure attribution list.
"""
from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from ..core.llm_client import LLMClient
from ..core.schemas import (
    CaseReport,
    DialogueTrace,
    DimensionScore,
    InstructionSpec,
    RunReport,
    ScenarioSpec,
    ScoreDetail,
)
from .constraint_judge import evaluate_constraints
from .flow_judge import evaluate_flow
from .goal_judge import evaluate_goal
from .knowledge_judge import evaluate_knowledge
from .rule_constraints import evaluate_hard_constraints, evaluate_termination
from .voting import run_voted_dual, run_voted_single


logger = logging.getLogger(__name__)


DEFAULT_CASE_GATE: dict[str, Any] = {
    "weighted_total": 0.75,
    "gsr_min": 0.85,
    "hcr_min": 0.80,
    "require_hcr_details_pass": True,
}


DEFAULT_WEIGHTS: dict[str, float] = {
    "gsr": 0.20,
    "pcr": 0.20,
    "bca": 0.10,
    "kar": 0.15,
    "hcr": 0.15,
    "scr": 0.10,
    "ter": 0.05,
    "rob": 0.05,
}

DIMENSION_NAMES: dict[str, str] = {
    "gsr": "目标完成度",
    "pcr": "路径覆盖率",
    "bca": "分支条件适配",
    "kar": "知识准确率",
    "hcr": "硬约束合规",
    "scr": "软约束合规",
    "ter": "终止策略合规",
    "rob": "稳健性",
}


INFO_WARNING_PREFIXES = (
    "已按场景target_constraints过滤硬约束",
    "该场景未声明termination为目标约束",
    "本场景未观察到终止策略触发信号",
)


def _is_informational_warning(warning: str) -> bool:
    return any(warning.startswith(prefix) for prefix in INFO_WARNING_PREFIXES)


def _attach_weight(dim: DimensionScore, weights: dict[str, float]) -> DimensionScore:
    dim.weight = weights.get(dim.id, 0.0)
    return dim


def assess_case_pass(
    case_report: CaseReport,
    *,
    gate: Optional[dict[str, Any]] = None,
) -> tuple[bool, list[str]]:
    """Return whether a single case passes and human-readable failure reasons."""
    gate = gate or DEFAULT_CASE_GATE
    reasons: list[str] = []
    total_threshold = float(gate.get("weighted_total", 0.75))
    if case_report.weighted_total < total_threshold:
        reasons.append(
            f"总分 {case_report.weighted_total:.3f} 低于门槛 {total_threshold:.2f}"
        )
    dim_mins = {
        "gsr": float(gate.get("gsr_min", 0.85)),
        "hcr": float(gate.get("hcr_min", 0.80)),
    }
    for dim in case_report.dimensions:
        min_score = dim_mins.get(dim.id)
        if min_score is not None and dim.score < min_score:
            reasons.append(
                f"{dim.name}({dim.id}) {dim.score:.3f} 低于门槛 {min_score:.2f}"
            )
        if dim.id == "hcr" and gate.get("require_hcr_details_pass", True):
            for detail in dim.details:
                if not detail.passed and detail.deduction > 0:
                    reasons.append(f"硬约束未通过: {detail.label}")
    return len(reasons) == 0, reasons


def evaluate_case(
    spec: InstructionSpec,
    scenario: ScenarioSpec,
    trace: DialogueTrace,
    *,
    client: Optional[LLMClient] = None,
    judge_model: Optional[str] = None,
    judge_samples: int = 3,
    judge_temperature: float = 0.2,
    judge_seed: int = 13,
    judge_require_evidence: bool = False,
    judge_workers: int = 1,
    samples_workers: Optional[int] = None,
    weights: Optional[dict[str, float]] = None,
    case_gate: Optional[dict[str, Any]] = None,
    sessions_used: int = 1,
) -> CaseReport:
    weights = weights or DEFAULT_WEIGHTS

    def eval_flow_pair() -> tuple[DimensionScore, DimensionScore]:
        return run_voted_dual(
            evaluate_flow,
            spec=spec,
            scenario=scenario,
            trace=trace,
            client=client,
            model=judge_model,
            samples=judge_samples,
            base_seed=judge_seed,
            temperature=judge_temperature,
            samples_workers=samples_workers,
            require_evidence=judge_require_evidence,
        )

    def eval_goal_dim() -> DimensionScore:
        return run_voted_single(
            evaluate_goal,
            spec=spec,
            scenario=scenario,
            trace=trace,
            client=client,
            model=judge_model,
            samples=judge_samples,
            base_seed=judge_seed + 1,
            temperature=judge_temperature,
            samples_workers=samples_workers,
            require_evidence=judge_require_evidence,
        )

    def eval_soft_dim() -> DimensionScore:
        return run_voted_single(
            evaluate_constraints,
            spec=spec,
            scenario=scenario,
            trace=trace,
            client=client,
            model=judge_model,
            samples=judge_samples,
            base_seed=judge_seed + 2,
            temperature=judge_temperature,
            samples_workers=samples_workers,
            require_evidence=judge_require_evidence,
        )

    def eval_knowledge_dim() -> DimensionScore:
        return run_voted_single(
            evaluate_knowledge,
            spec=spec,
            scenario=None,
            trace=trace,
            client=client,
            model=judge_model,
            samples=judge_samples,
            base_seed=judge_seed + 3,
            temperature=judge_temperature,
            samples_workers=samples_workers,
            require_evidence=judge_require_evidence,
        )

    if client is not None and judge_workers > 1:
        with ThreadPoolExecutor(max_workers=judge_workers) as pool:
            flow_future = pool.submit(eval_flow_pair)
            goal_future = pool.submit(eval_goal_dim)
            soft_future = pool.submit(eval_soft_dim)
            knowledge_future = pool.submit(eval_knowledge_dim)
            pcr_score, bca_score = flow_future.result()
            gsr_score = goal_future.result()
            scr_score = soft_future.result()
            kar_score = knowledge_future.result()
    else:
        pcr_score, bca_score = eval_flow_pair()
        gsr_score = eval_goal_dim()
        scr_score = eval_soft_dim()
        kar_score = eval_knowledge_dim()

    hcr_score = evaluate_hard_constraints(spec, trace, scenario)
    ter_score = evaluate_termination(spec, trace, scenario)
    rob_score = DimensionScore(
        id="rob",
        name=DIMENSION_NAMES["rob"],
        score=0.0,
        raw_score=0.0,
        details=[],
        source="derived",
    )

    dimensions = [
        _attach_weight(gsr_score, weights),
        _attach_weight(pcr_score, weights),
        _attach_weight(bca_score, weights),
        _attach_weight(kar_score, weights),
        _attach_weight(hcr_score, weights),
        _attach_weight(scr_score, weights),
        _attach_weight(ter_score, weights),
        rob_score,
    ]

    weighted_total = sum(d.score * (weights.get(d.id, 0.0)) for d in dimensions)
    weighted_total = round(min(1.0, max(0.0, weighted_total)), 4)
    failure_attribution = build_failure_attribution(dimensions)
    case_report = CaseReport(
        trace=trace,
        dimensions=dimensions,
        weighted_total=weighted_total,
        sessions_used=sessions_used,
        failure_attribution=failure_attribution,
    )
    passed, fail_reasons = assess_case_pass(case_report, gate=case_gate)
    case_report.passed = passed
    case_report.fail_reasons = fail_reasons
    return case_report


def build_failure_attribution(dimensions: list[DimensionScore], top_k: int = 8) -> list[dict[str, Any]]:
    items: list[tuple[float, ScoreDetail, DimensionScore]] = []
    for dim in dimensions:
        if dim.id == "rob":
            continue
        for detail in dim.details:
            if detail.deduction <= 0:
                continue
            weighted = detail.deduction * dim.weight
            items.append((weighted, detail, dim))
    items.sort(key=lambda x: -x[0])
    out: list[dict[str, Any]] = []
    for weighted, detail, dim in items[:top_k]:
        out.append(
            {
                "dim_id": dim.id,
                "dim_name": dim.name,
                "criterion_id": detail.criterion_id,
                "label": detail.label,
                "deduction": detail.deduction,
                "weighted_loss": round(weighted, 4),
                "turn_ids": detail.turn_ids,
                "evidence_quote": detail.evidence_quote,
                "rationale": detail.rationale,
            }
        )
    return out


def attach_robustness(cases: list[CaseReport], weights: Optional[dict[str, float]] = None) -> None:
    """After all cases for an instruction are evaluated, fill in the ROB dimension.

    ROB is computed per-instruction as ``1 - normalised_stdev(weighted_totals)``.
    Each case's ROB dimension is updated in place; the weighted total is rerun.
    """
    weights = weights or DEFAULT_WEIGHTS
    if not cases:
        return
    grouped: dict[str, list[CaseReport]] = {}
    for c in cases:
        grouped.setdefault(c.trace.instruction_id, []).append(c)
    for instruction_id, group in grouped.items():
        scores = [c.weighted_total for c in group]
        if len(scores) < 2:
            rob = 1.0
        else:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = math.sqrt(var)
            rob = max(0.0, 1.0 - min(1.0, std * 2))
        for c in group:
            for dim in c.dimensions:
                if dim.id == "rob":
                    dim.score = round(rob, 4)
                    dim.raw_score = round(rob, 4)
                    dim.weight = weights.get("rob", 0.0)
                    dim.details = [
                        ScoreDetail(
                            criterion_id="rob.score_variance",
                            label=f"指令{instruction_id}下场景分数稳健性",
                            passed=rob >= 0.7,
                            deduction=round(1.0 - rob, 4),
                            turn_ids=[],
                            evidence_quote=f"scores={scores}",
                            rationale=f"基于 {len(scores)} 个场景的标准差派生",
                            confidence=0.6,
                        )
                    ]
            c.weighted_total = round(
                sum(d.score * weights.get(d.id, 0.0) for d in c.dimensions), 4
            )


def bootstrap_confidence(scores: list[float], iters: int = 200, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    if len(scores) == 1:
        return scores[0], scores[0]
    rng = random.Random(seed)
    means: list[float] = []
    n = len(scores)
    for _ in range(iters):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[max(0, int(iters * (alpha / 2)))]
    hi = means[min(iters - 1, int(iters * (1 - alpha / 2)))]
    return round(lo, 4), round(hi, 4)


def aggregate_run(
    run_id: str,
    cases: list[CaseReport],
    *,
    config: Optional[dict[str, Any]] = None,
    weights: Optional[dict[str, float]] = None,
    case_gate: Optional[dict[str, Any]] = None,
    bootstrap_iters: int = 200,
    bootstrap_alpha: float = 0.05,
    top_k_failures: int = 10,
) -> RunReport:
    weights = weights or DEFAULT_WEIGHTS
    gate = case_gate or (config or {}).get("case_gate") or DEFAULT_CASE_GATE
    attach_robustness(cases, weights)
    for c in cases:
        passed, reasons = assess_case_pass(c, gate=gate)
        c.passed = passed
        c.fail_reasons = reasons
    totals = [c.weighted_total for c in cases]
    overall = round(sum(totals) / len(totals), 4) if totals else 0.0
    ci = bootstrap_confidence(totals, iters=bootstrap_iters, alpha=bootstrap_alpha)
    by_dim: dict[str, list[float]] = {}
    for c in cases:
        for dim in c.dimensions:
            by_dim.setdefault(dim.id, []).append(dim.score)
    dim_summary = {
        dim_id: {
            "name": DIMENSION_NAMES.get(dim_id, dim_id),
            "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
        }
        for dim_id, scores in by_dim.items()
    }
    by_instruction: dict[str, list[float]] = {}
    for c in cases:
        by_instruction.setdefault(c.trace.instruction_id, []).append(c.weighted_total)
    instruction_summary = {
        iid: {
            "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
            "n": len(scores),
        }
        for iid, scores in by_instruction.items()
    }
    failures: list[dict[str, Any]] = []
    for c in cases:
        for f in c.failure_attribution:
            failures.append(
                {
                    "instruction_id": c.trace.instruction_id,
                    "scenario_id": c.trace.scenario_id,
                    **f,
                }
            )
    failures.sort(key=lambda x: -x["weighted_loss"])
    low_confidence: list[dict[str, Any]] = []
    fallback_dims: list[dict[str, Any]] = []
    informational_warnings: list[dict[str, Any]] = []
    high_disagreement: list[dict[str, Any]] = []
    for c in cases:
        for dim in c.dimensions:
            if dim.confidence is not None and dim.confidence < 0.6:
                low_confidence.append(
                    {
                        "case_id": c.trace.case_id,
                        "dim_id": dim.id,
                        "confidence": dim.confidence,
                    }
                )
            blocking_warnings = [
                warning
                for warning in dim.warnings
                if not _is_informational_warning(warning)
            ]
            info_warnings = [
                warning for warning in dim.warnings if _is_informational_warning(warning)
            ]
            if info_warnings:
                informational_warnings.append(
                    {
                        "case_id": c.trace.case_id,
                        "dim_id": dim.id,
                        "warnings": info_warnings,
                    }
                )
            if dim.source == "fallback" or blocking_warnings:
                fallback_dims.append(
                    {
                        "case_id": c.trace.case_id,
                        "dim_id": dim.id,
                        "source": dim.source,
                        "warnings": blocking_warnings,
                    }
                )
            for detail in dim.details:
                if detail.disagreement is not None and detail.disagreement >= 0.34:
                    high_disagreement.append(
                        {
                            "case_id": c.trace.case_id,
                            "dim_id": dim.id,
                            "criterion_id": detail.criterion_id,
                            "disagreement": detail.disagreement,
                            "label": detail.label,
                        }
                    )
    passed_cases = [c for c in cases if c.passed]
    failed_cases = [c for c in cases if not c.passed]
    case_pass_rate = round(len(passed_cases) / len(cases), 4) if cases else 0.0
    aggregate = {
        "overall_mean": overall,
        "confidence_interval": list(ci),
        "n_cases": len(cases),
        "n_passed": len(passed_cases),
        "n_failed": len(failed_cases),
        "case_pass_rate": case_pass_rate,
        "dimensions": dim_summary,
        "by_instruction": instruction_summary,
        "weights": weights,
        "top_failures": failures[:top_k_failures],
        "failed_case_ids": [c.trace.case_id for c in failed_cases],
        "quality_gates": {
            "passed": case_pass_rate >= 1.0 and not fallback_dims,
            "run_pass_mode": "all_cases",
            "thresholds": {
                "case_pass_rate": 1.0,
                "overall_mean": 0.75,
                "dimension_min_mean": 0.6,
            },
            "low_confidence_count": len(low_confidence),
            "fallback_or_warning_count": len(fallback_dims),
            "high_disagreement_count": len(high_disagreement),
        },
        "low_confidence": low_confidence[:top_k_failures],
        "fallback_or_warnings": fallback_dims[:top_k_failures],
        "informational_warnings": informational_warnings[:top_k_failures],
        "high_disagreement": high_disagreement[:top_k_failures],
    }
    return RunReport(
        run_id=run_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        config=config or {},
        cases=cases,
        aggregate=aggregate,
    )
