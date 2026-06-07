"""End-to-end CLI: parse instructions, run dialogues, evaluate, and write reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..core.config import load_config
from ..core.instruction_parser import load_specs, parse_workbook
from ..core.llm_client import build_default_client
from ..core.orchestrator import run_dialogue, summarize_trace_for_memory
from ..core.real_dialogue_importer import build_real_scenario, load_real_dialogues
from ..core.run_store import RunStore
from ..core.scenario_matrix import build_scenarios
from ..core.schemas import CaseReport, DialogueTrace, ScenarioSpec
from ..core.sut_client import build_sut_client
from ..core.user_simulator import build_user_simulator
from ..evaluators.aggregator import aggregate_run, evaluate_case
from ..report.renderer import write_run_artifacts


@dataclass
class CaseTask:
    spec_id: str
    scenario_id: str
    spec: object
    scenario: ScenarioSpec
    variables: dict[str, str]
    trace: Optional[DialogueTrace] = None


@dataclass
class CaseResult:
    case_report: Optional[CaseReport]
    trace_path: Path
    failed: bool = False
    error: str = ""


def _build_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_filter(values: Optional[list[str]]) -> Optional[set[str]]:
    if not values:
        return None
    expanded: list[str] = []
    for v in values:
        for piece in str(v).split(","):
            piece = piece.strip()
            if piece:
                expanded.append(piece)
    if not expanded:
        return None
    if expanded == ["all"]:
        return None
    return set(expanded)


def _case_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dialogue evaluation pipeline.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--xlsx", type=Path,
        default=Path("data/source/命题二：外呼任务对话模型指令示例.xlsx"),
        help="Source instruction Excel file (only used when --parse).",
    )
    parser.add_argument(
        "--instructions-dir", type=Path, default=Path("data/instructions"),
        help="Directory containing parsed instruction JSON specs.",
    )
    parser.add_argument(
        "--instructions", nargs="*", default=["all"],
        help="Instruction ids to run, or 'all'.",
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=["all"],
        help="Scenario ids to run, or 'all'.",
    )
    parser.add_argument(
        "--sut", choices=["llm", "stub"], default="llm",
        help="SUT backend.",
    )
    parser.add_argument(
        "--user-sim", choices=["llm", "stub"], default="llm",
        help="User simulator backend.",
    )
    parser.add_argument(
        "--judge", choices=["llm", "offline"], default="llm",
        help="Judge backend (offline = heuristic).",
    )
    parser.add_argument("--sut-model", default=None)
    parser.add_argument("--user-sim-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--judge-workers", type=int, default=None)
    parser.add_argument(
        "--samples-workers", type=int, default=None,
        help="Per-dimension parallel sampling worker count (default auto = min(samples, 4)).",
    )
    parser.add_argument(
        "--max-in-flight", type=int, default=None,
        help="Global cap on concurrent LLM requests across SUT/UserSim/Judge "
             "(default 32; set 0 to disable).",
    )
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--judge-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory; defaults to reports/{run_id}.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Optional run identifier (defaults to a timestamp).",
    )
    parser.add_argument(
        "--parse", action="store_true",
        help="Re-parse the Excel before running (always uses --judge mode for parsing).",
    )
    parser.add_argument(
        "--reuse-traces", type=Path, default=None,
        help="Optional directory of pre-generated traces; skips dialogue generation.",
    )
    parser.add_argument(
        "--real-dialogues", type=Path, default=None,
        help="Uploaded real dialogue JSON file; skips dialogue generation and judges imported conversations.",
    )
    parser.add_argument(
        "--dialogue-only",
        action="store_true",
        help="Only generate dialogue traces; skip judging and report rendering.",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Only judge existing traces. Requires --reuse-traces.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = load_config(args.config)
    sut_model = args.sut_model or config.get("models", {}).get("sut")
    judge_model = args.judge_model or config.get("models", {}).get("judge")
    user_sim_model = args.user_sim_model or config.get("models", {}).get("user_sim")
    workers = (
        args.workers
        or config.get("runtime", {}).get("case_workers")
        or config.get("runtime", {}).get("workers", 4)
    )
    judge_workers = args.judge_workers or config.get("runtime", {}).get("judge_workers", 1)
    samples_workers = (
        args.samples_workers
        if args.samples_workers is not None
        else config.get("runtime", {}).get("samples_workers")
    )
    max_in_flight_cfg = config.get("runtime", {}).get("max_in_flight", 32)
    max_in_flight = (
        args.max_in_flight if args.max_in_flight is not None else max_in_flight_cfg
    )
    max_in_flight = int(max_in_flight) if max_in_flight else 0
    max_turns = args.max_turns or config.get("runtime", {}).get("max_turns", 16)
    seed = args.seed if args.seed is not None else config.get("runtime", {}).get("seed", 42)
    judge_samples = args.judge_samples or config.get("judge", {}).get("samples", 3)
    judge_temp = config.get("judge", {}).get("temperature", 0.2)
    sut_temp = config.get("generation", {}).get("sut_temperature", 0.4)
    user_sim_temp = config.get("generation", {}).get("user_sim_temperature", 0.7)
    judge_require_evidence = bool(config.get("judge", {}).get("require_evidence", False))
    judge_thinking = bool(config.get("judge", {}).get("thinking", False))
    judge_reasoning = config.get("judge", {}).get("reasoning_effort", "high")
    cache_dir = config.get("runtime", {}).get("cache_dir", "data/.llm_cache")
    max_invalid_replies = config.get("orchestrator", {}).get("max_invalid_replies", 3)
    max_sessions = int(config.get("orchestrator", {}).get("max_sessions", 1))
    retry_on_fail = bool(config.get("orchestrator", {}).get("retry_on_fail", False))
    case_gate = config.get("case_gate")
    weights = config.get("metrics", {}).get("weights")
    bootstrap_iters = config.get("metrics", {}).get("bootstrap_iters", 200)
    bootstrap_alpha = config.get("metrics", {}).get("bootstrap_alpha", 0.05)
    top_k_failures = config.get("report", {}).get("top_k_failures", 10)

    run_id = args.run_id or _build_run_id()
    out_dir = args.out or Path("reports") / run_id
    trace_dir = Path("traces") / run_id
    console = Console()
    console.print(f"[bold cyan]Run ID:[/] {run_id}")
    console.print(f"[bold cyan]Output:[/] {out_dir}")
    if args.judge_only and args.reuse_traces is None:
        console.print("[red]--judge-only requires --reuse-traces.[/]")
        return
    if args.dialogue_only and args.reuse_traces is not None:
        console.print("[red]--dialogue-only cannot be combined with --reuse-traces.[/]")
        return
    if args.real_dialogues and (args.dialogue_only or args.reuse_traces or args.judge_only):
        console.print("[red]--real-dialogues cannot be combined with dialogue-only, judge-only, or reuse-traces.[/]")
        return

    if args.parse:
        if not args.xlsx.exists():
            root_xlsx = Path("命题二：外呼任务对话模型指令示例.xlsx")
            if root_xlsx.exists():
                args.xlsx = root_xlsx
        console.print(f"[yellow]Parsing instructions from {args.xlsx} ...[/]")
        client = build_default_client(cache_dir=cache_dir) if args.judge == "llm" else None
        parse_workbook(
            args.xlsx,
            output_dir=args.instructions_dir,
            client=client,
            mode="llm" if args.judge == "llm" else "offline",
        )

    specs = load_specs(args.instructions_dir)
    if not specs:
        console.print(
            f"[red]No instruction specs found in {args.instructions_dir}. Run with --parse first.[/]"
        )
        return

    instr_filter = _resolve_filter(args.instructions)
    scenario_filter = _resolve_filter(args.scenarios)

    specs_by_id = {spec.id: spec for spec in specs}
    case_tasks: list[CaseTask] = []
    if args.real_dialogues:
        traces = load_real_dialogues(args.real_dialogues, specs_by_id, run_id=run_id)
        for trace in traces:
            spec = specs_by_id[trace.instruction_id]
            sc = build_real_scenario(spec)
            case_tasks.append(
                CaseTask(
                    spec_id=spec.id,
                    scenario_id=sc.id,
                    spec=spec,
                    scenario=sc,
                    variables={},
                    trace=trace,
                )
            )
    else:
        for spec in specs:
            if instr_filter and spec.id not in instr_filter:
                continue
            scenarios, vmap = build_scenarios(spec, seed=seed)
            for sc in scenarios:
                if scenario_filter and sc.id not in scenario_filter:
                    continue
                case_tasks.append(
                    CaseTask(
                        spec_id=spec.id,
                        scenario_id=sc.id,
                        spec=spec,
                        scenario=sc,
                        variables=vmap[sc.id],
                    )
                )
    if not case_tasks:
        console.print("[red]No cases produced after filtering.[/]")
        return

    console.print(
        f"Total cases to evaluate: {len(case_tasks)} "
        f"(workers={workers}, judge_workers={judge_workers}, "
        f"samples={judge_samples}, max_in_flight={max_in_flight or 'unlimited'})"
    )

    # 跨所有 LLMClient 共用一个全局 BoundedSemaphore，使 SUT/UserSim/Judge 三个 client
    # 的 in-flight 请求数共用同一上限，避免 case×judge_workers×samples 同时打爆 API。
    in_flight_semaphore: Optional[threading.BoundedSemaphore] = (
        threading.BoundedSemaphore(max_in_flight) if max_in_flight > 0 else None
    )

    sut_llm = (
        build_default_client(
            cache_dir=cache_dir,
            max_in_flight=max_in_flight if max_in_flight > 0 else None,
            in_flight_semaphore=in_flight_semaphore,
        )
        if args.sut == "llm"
        else None
    )
    user_sim_llm = (
        build_default_client(
            cache_dir=cache_dir,
            max_in_flight=max_in_flight if max_in_flight > 0 else None,
            in_flight_semaphore=in_flight_semaphore,
        )
        if args.user_sim == "llm"
        else None
    )
    judge_llm = (
        build_default_client(
            cache_dir=cache_dir,
            thinking=judge_thinking,
            reasoning_effort=judge_reasoning if judge_thinking else None,
            max_in_flight=max_in_flight if max_in_flight > 0 else None,
            in_flight_semaphore=in_flight_semaphore,
        )
        if args.judge == "llm"
        else None
    )

    def process(task: CaseTask) -> CaseResult:
        try:
            case_id = task.trace.case_id if task.trace is not None else f"{task.spec_id}__{task.scenario_id}"
            case_seed = _case_seed(seed, case_id)
            if task.trace is not None:
                from ..core.orchestrator import save_trace

                trace = task.trace
                trace_path = save_trace(trace, trace_dir)
                case = None
            elif args.reuse_traces is not None:
                from ..core.orchestrator import load_trace

                trace_path = (
                    Path(args.reuse_traces) / f"{case_id}.json"
                )
                trace = load_trace(trace_path)
            else:
                sut = build_sut_client(
                    use_stub=(args.sut == "stub"),
                    client=sut_llm,
                    model=sut_model,
                    temperature=sut_temp,
                    seed=case_seed,
                )
                user_sim = build_user_simulator(
                    use_stub=(args.user_sim == "stub"),
                    client=user_sim_llm,
                    model=user_sim_model,
                    temperature=user_sim_temp,
                    seed=case_seed,
                )
                prior_memory = None
                prior_sessions: list[dict] = []
                trace = None
                case = None
                sessions_to_run = max_sessions if retry_on_fail else 1
                for session_idx in range(1, sessions_to_run + 1):
                    trace = run_dialogue(
                        task.spec,
                        task.scenario,
                        sut,
                        user_sim,
                        variables=task.variables,
                        run_id=run_id,
                        max_turns=max_turns,
                        max_invalid_replies=max_invalid_replies,
                        seed=case_seed + session_idx - 1,
                        trace_dir=None,
                        session_index=session_idx,
                        max_sessions=sessions_to_run,
                        prior_memory=prior_memory,
                        prior_sessions=prior_sessions,
                    )
                    if args.dialogue_only:
                        break
                    case = evaluate_case(
                        task.spec,
                        task.scenario,
                        trace,
                        client=judge_llm,
                        judge_model=judge_model,
                        judge_samples=judge_samples,
                        judge_temperature=judge_temp,
                        judge_require_evidence=judge_require_evidence,
                        judge_workers=judge_workers,
                        samples_workers=samples_workers,
                        judge_seed=case_seed + 17 + session_idx,
                        weights=weights,
                        case_gate=case_gate,
                        sessions_used=session_idx,
                    )
                    if case.passed or session_idx >= sessions_to_run:
                        break
                    prior_sessions.append(
                        {
                            "session": session_idx,
                            "score": case.weighted_total,
                            "passed": case.passed,
                            "summary": summarize_trace_for_memory(trace),
                        }
                    )
                    prior_memory = prior_sessions[-1]["summary"]
                if trace is not None:
                    from ..core.orchestrator import save_trace

                    trace.prior_sessions = prior_sessions
                    trace_path = save_trace(trace, trace_dir)
                else:
                    trace_path = trace_dir / f"{case_id}.json"
            if args.dialogue_only:
                return CaseResult(case_report=None, trace_path=trace_path)
            if case is None:
                case = evaluate_case(
                    task.spec,
                    task.scenario,
                    trace,
                    client=judge_llm,
                    judge_model=judge_model,
                    judge_samples=judge_samples,
                    judge_temperature=judge_temp,
                    judge_require_evidence=judge_require_evidence,
                    judge_workers=judge_workers,
                    samples_workers=samples_workers,
                    judge_seed=case_seed + 17,
                    weights=weights,
                    case_gate=case_gate,
                )
            return CaseResult(case_report=case, trace_path=trace_path)
        except Exception as exc:
            logging.exception("Case %s failed", f"{task.spec_id}__{task.scenario_id}")
            return CaseResult(case_report=None, trace_path=Path(""), failed=True, error=str(exc))

    results: list[CaseResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[label]}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task(
            description="cases", total=len(case_tasks), label="Evaluating"
        )
        if workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process, t): t for t in case_tasks}
                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    progress.advance(progress_task)
        else:
            for t in case_tasks:
                results.append(process(t))
                progress.advance(progress_task)

    valid_cases = [r.case_report for r in results if not r.failed and r.case_report is not None]
    failed = [r for r in results if r.failed]
    if args.dialogue_only:
        console.print(f"[bold green]Dialogue traces generated:[/] {len(results) - len(failed)}")
        if failed:
            console.print(f"[red]Failed cases:[/] {len(failed)}")
            for f in failed[:5]:
                console.print(f"  - {f.error}")
        console.print(f"[bold cyan]Traces:[/] {trace_dir}")
        return
    if not valid_cases:
        console.print("[red]All cases failed; aborting report generation.[/]")
        return

    run_report = aggregate_run(
        run_id,
        valid_cases,
        config={
            "models": {
                "sut": sut_model,
                "user_sim": user_sim_model,
                "judge": judge_model,
            },
            "judge_samples": judge_samples,
            "judge_workers": judge_workers,
            "case_workers": workers,
            "samples_workers": samples_workers,
            "max_in_flight": max_in_flight,
            "max_turns": max_turns,
            "max_invalid_replies": max_invalid_replies,
            "max_sessions": max_sessions,
            "retry_on_fail": retry_on_fail,
            "case_gate": case_gate,
            "seed": seed,
            "judge_backend": args.judge,
            "sut_backend": args.sut,
            "user_sim_backend": args.user_sim,
            "judge_require_evidence": judge_require_evidence,
            "generation": {
                "sut_temperature": sut_temp,
                "user_sim_temperature": user_sim_temp,
                "judge_temperature": judge_temp,
            },
            "trace_dir": str(trace_dir),
            "output_dir": str(out_dir),
            "reuse_traces": str(args.reuse_traces) if args.reuse_traces else None,
            "real_dialogues": str(args.real_dialogues) if args.real_dialogues else None,
            "source": "upload" if args.real_dialogues else "generated",
            "run_mode": "real_dialogue" if args.real_dialogues else ("judge_only" if args.judge_only else "full"),
        },
        weights=weights,
        case_gate=case_gate,
        bootstrap_iters=bootstrap_iters,
        bootstrap_alpha=bootstrap_alpha,
        top_k_failures=top_k_failures,
    )
    paths = write_run_artifacts(run_report, out_dir)

    store = RunStore(root="reports")
    store.register_run(
        run_id=run_id,
        config=run_report.config,
        sut_model=sut_model,
        judge_model=judge_model,
        user_sim_model=user_sim_model,
    )
    for r in results:
        if r.failed or r.case_report is None:
            continue
        store.add_case(
            case_id=r.case_report.trace.case_id,
            run_id=run_id,
            instruction_id=r.case_report.trace.instruction_id,
            scenario_id=r.case_report.trace.scenario_id,
            trace_path=str(r.trace_path),
            weighted_total=r.case_report.weighted_total,
            passed=r.case_report.passed,
            report_path=str(paths["html"]),
        )
    store.save_report(
        run_id=run_id,
        aggregate=run_report.aggregate,
        html_path=str(paths["html"]),
        md_path=str(paths["md"]),
    )

    table = Table(title="Run summary", show_lines=True)
    table.add_column("指令")
    table.add_column("场景")
    table.add_column("总分", justify="right")
    table.add_column("通过")
    table.add_column("会话")
    table.add_column("终止")
    for c in valid_cases:
        table.add_row(
            c.trace.instruction_id,
            c.trace.scenario_id,
            f"{c.weighted_total:.3f}",
            "Y" if c.passed else "N",
            str(c.sessions_used),
            c.trace.terminated_by or "-",
        )
    console.print(table)
    agg = run_report.aggregate
    console.print(
        f"[bold green]Case 通过率:[/] {agg.get('case_pass_rate', 0):.1%} "
        f"({agg.get('n_passed', 0)}/{agg.get('n_cases', 0)})"
    )
    console.print(
        f"[bold green]Overall mean:[/] {agg['overall_mean']:.3f} "
        f"CI=[{agg['confidence_interval'][0]:.3f}, "
        f"{agg['confidence_interval'][1]:.3f}]"
    )
    if failed:
        console.print(f"[red]Failed cases:[/] {len(failed)}")
        for f in failed[:5]:
            console.print(f"  - {f.error}")
    console.print(f"[bold cyan]HTML:[/] {paths['html']}")
    console.print(f"[bold cyan]Markdown:[/] {paths['md']}")
    console.print(f"[bold cyan]Traces:[/] {trace_dir}")


if __name__ == "__main__":
    main()
