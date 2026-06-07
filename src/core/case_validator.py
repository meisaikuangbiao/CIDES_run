"""Validate instruction × scenario case matrix coverage."""
from __future__ import annotations

from pathlib import Path

from .instruction_parser import load_specs
from .scenario_matrix import build_scenarios


def validate_case_matrix(
    instructions_dir: str | Path = "data/instructions",
    *,
    seed: int = 42,
) -> dict:
    """Return expected cases and any coverage issues."""
    specs = load_specs(instructions_dir)
    cases: list[dict] = []
    issues: list[str] = []
    for spec in specs:
        scenarios, _ = build_scenarios(spec, seed=seed)
        if not scenarios:
            issues.append(f"指令 {spec.id} 未生成任何场景")
        for sc in scenarios:
            case_id = f"{spec.id}__{sc.id}"
            case_issues: list[str] = []
            if not sc.target_nodes and spec.flow_nodes:
                case_issues.append("target_nodes 为空")
            if not sc.behaviour.strip():
                case_issues.append("behaviour 为空")
            if sc.id == "out_of_scope" and not spec.constraints.hard.required_out_of_scope_reply:
                case_issues.append("out_of_scope 场景但指令无越权兜底话术")
            if sc.id == "lure_violation" and not spec.constraints.hard.no_discount_promise:
                case_issues.append("lure_violation 场景但指令未声明 no_discount_promise")
            cases.append(
                {
                    "case_id": case_id,
                    "instruction_id": spec.id,
                    "scenario_id": sc.id,
                    "scenario_name": sc.name,
                    "target_nodes": sc.target_nodes,
                    "issues": case_issues,
                }
            )
            issues.extend(f"{case_id}: {msg}" for msg in case_issues)
    return {
        "n_instructions": len(specs),
        "n_cases": len(cases),
        "cases": cases,
        "issues": issues,
        "valid": len(issues) == 0,
    }
