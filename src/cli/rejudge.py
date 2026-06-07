"""Re-evaluate a saved trace JSON without rerunning the dialogue.

This is useful when iterating on judges or weights: you already have an
expensive dialogue trace and only want to refresh the scoring.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..core.config import load_config
from ..core.instruction_parser import load_specs
from ..core.llm_client import build_default_client
from ..core.orchestrator import load_trace
from ..core.scenario_matrix import build_scenarios
from ..evaluators.aggregator import evaluate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge an existing trace JSON file.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--instructions-dir", type=Path, default=Path("data/instructions"))
    parser.add_argument(
        "--judge", choices=["llm", "offline"], default="offline",
        help="Judge backend.",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = load_config(args.config)
    judge_samples = args.judge_samples or cfg.get("judge", {}).get("samples", 3)
    judge_temp = cfg.get("judge", {}).get("temperature", 0.2)
    judge_require_evidence = bool(cfg.get("judge", {}).get("require_evidence", False))
    seed = args.seed if args.seed is not None else cfg.get("runtime", {}).get("seed", 42)
    judge_client = (
        build_default_client(cache_dir=cfg.get("runtime", {}).get("cache_dir", "data/.llm_cache"))
        if args.judge == "llm"
        else None
    )
    weights = cfg.get("metrics", {}).get("weights")

    trace = load_trace(args.trace)
    specs = {s.id: s for s in load_specs(args.instructions_dir)}
    spec = specs.get(trace.instruction_id)
    if spec is None:
        raise SystemExit(
            f"Instruction {trace.instruction_id} not found in {args.instructions_dir}"
        )
    scenarios, _ = build_scenarios(spec, seed=seed)
    scenario = next((s for s in scenarios if s.id == trace.scenario_id), None)
    if scenario is None:
        raise SystemExit(f"Scenario {trace.scenario_id} not found for instruction {spec.id}")

    case = evaluate_case(
        spec,
        scenario,
        trace,
        client=judge_client,
        judge_model=args.judge_model or cfg.get("models", {}).get("judge"),
        judge_samples=judge_samples,
        judge_temperature=judge_temp,
        judge_require_evidence=judge_require_evidence,
        judge_seed=seed + 17,
        weights=weights,
    )
    console = Console()
    table = Table(title=f"Rejudge {trace.case_id}")
    table.add_column("Dim")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    for d in case.dimensions:
        table.add_row(d.id, f"{d.score:.3f}", f"{(d.confidence or 0):.2f}")
    console.print(table)
    console.print(f"[bold green]Weighted total:[/] {case.weighted_total:.3f}")
    if case.failure_attribution:
        console.print("Top failures:")
        for f in case.failure_attribution[:5]:
            console.print(
                f"  - [{f['dim_id']}] {f['label']} (loss={f['weighted_loss']:.3f}) — {f['rationale']}"
            )


if __name__ == "__main__":
    main()
